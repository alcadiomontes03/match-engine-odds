"""
Pulsematch — run manifest, shared by every agent.

Agents hand work to each other through files, never chat. Each agent finishes by
writing a manifest.json into the run folder, so the next agent and the Checker can read
what was produced, from what, and with what warnings. This module is the one place that
defines that format.

Run folders live in Drive ("Match Engine — Runs"), one per run, named by UTC start time to the
second plus a 4-character random suffix (v6.2; earlier runs used minute resolution, e.g. 2026-10-05T1600Z):

    2026-10-05T160012Z-3fa9/
        manifest.json           written last by each agent (see below)
        collector/              the Data collector's outputs
            manifest.json
            completeness.json

manifest.json fields
    schema      format version (int)
    agent       who wrote it ("collector", "pricer", ...)
    run_id      the run folder name
    status      "OK" | "PARTIAL" | "FAILED"
                  OK       finished, nothing to flag
                  PARTIAL  finished, but something is incomplete or stale; the reason is in `warnings`
                  FAILED   could not produce usable output; downstream agents must not run
    started, finished   UTC ISO timestamps
    inputs      what was read: [{"path": ..., "sha256": ..., "rows": ...}]
    outputs     what was written, hashed automatically from the agent's folder
    row_counts  {name: rows} for anything countable
    warnings    [str]  things a person should look at
    notes       [str]  facts that are not problems (e.g. "international break")

A manifest is a statement of what happened, not a grade. The Checker is the only agent
that rules on whether the output may be published.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 1
STATUSES = ("OK", "PARTIAL", "FAILED")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id(now: datetime | None = None) -> str:
    """UTC start time to the second plus a short random suffix, e.g. 2026-10-05T192907Z-3fa9.
    v6.2: the old minute-resolution id let two runs started in the same minute share a folder."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(2)}"


def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` so a crash can never leave a half-written file: write a temp file
    in the same folder, flush it to disk, then atomically replace the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{secrets.token_hex(4)}")
    try:
        with open(tmp, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(path: Path, rows: int | None = None) -> dict:
    d = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
    if rows is not None:
        d["rows"] = rows
    return d


def write_manifest(folder: Path, *, agent: str, run_id: str, status: str, started: str,
                   inputs: list | None = None, row_counts: dict | None = None,
                   warnings: list | None = None, notes: list | None = None) -> dict:
    """Write folder/manifest.json. Outputs are every other file already in the folder,
    hashed here so the manifest cannot drift from what is on disk."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    folder.mkdir(parents=True, exist_ok=True)
    outputs = [{"path": p.name, "sha256": sha256(p), "bytes": p.stat().st_size}
               for p in sorted(folder.iterdir()) if p.is_file() and p.name != "manifest.json"]
    m = {
        "schema": SCHEMA,
        "agent": agent,
        "run_id": run_id,
        "status": status,
        "started": started,
        "finished": utc_now(),
        "inputs": inputs or [],
        "outputs": outputs,
        "row_counts": row_counts or {},
        "warnings": warnings or [],
        "notes": notes or [],
    }
    (folder / "manifest.json").write_text(json.dumps(m, indent=2))
    return m


def read_manifest(folder: Path) -> dict | None:
    p = folder / "manifest.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def verify_outputs(folder: Path, strict: bool = False) -> list[str]:
    """Return a list of problems if the files on disk no longer match the manifest.
    A downstream agent calls this before trusting anything the previous agent wrote.
    strict=True also reports files sitting in the folder that the manifest does not list
    (the same top-level files write_manifest hashes), so nothing unmanifested can ride along."""
    m = read_manifest(folder)
    if m is None:
        return [f"{folder} has no manifest.json"]
    problems = []
    listed = set()
    for o in m.get("outputs", []):
        listed.add(o["path"])
        p = folder / o["path"]
        if not p.exists():
            problems.append(f"{o['path']} is listed in the manifest but missing")
        elif sha256(p) != o["sha256"]:
            problems.append(f"{o['path']} changed after the manifest was written")
    if strict:
        for p in sorted(folder.iterdir()):
            if p.is_symlink():
                problems.append(f"{p.name} is a symlink; agent folders hold plain files only")
            elif p.is_dir():
                # manifests hash top-level files only, so nothing nested could be verified
                problems.append(f"{p.name}/ is a sub-folder; agent folders hold no sub-folders")
            elif p.name != "manifest.json" and p.name not in listed:
                problems.append(f"{p.name} is in the folder but not in the manifest")
    return problems


def check_manifest(folder: Path, *, agent: str, run_id: str, strict: bool = True) -> list[str]:
    """Everything a downstream agent must confirm before trusting `folder`: a manifest exists,
    it was written by `agent` for `run_id`, its status is legal, and the files match it."""
    m = read_manifest(folder)
    if m is None:
        return [f"{folder.name}: no manifest.json"]
    problems = []
    if m.get("agent") != agent:
        problems.append(f"{folder.name}: manifest written by {m.get('agent')!r}, expected {agent!r}")
    if m.get("run_id") != run_id:
        problems.append(f"{folder.name}: manifest is for run {m.get('run_id')!r}, not {run_id!r}")
    if m.get("status") not in STATUSES:
        problems.append(f"{folder.name}: manifest status {m.get('status')!r} is not one of {STATUSES}")
    problems += [f"{folder.name}: {p}" for p in verify_outputs(folder, strict=strict)]
    return problems
