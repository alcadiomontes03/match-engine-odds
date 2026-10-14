"""
Pulsematch — Publisher agent.

The only agent allowed to act on the Checker's verdict. It publishes in two steps, because
the writes themselves (the dashboard's database, Drive) can only be made by the Claude
session's tools, not by a script:

  1. PREPARE (the orchestrator's publisher stage):
         python publish.py --root <repo> --run-dir <runs/run_id>
     * refuses unless checker/verdict.json exists, is PASS or WARN, and was made for THIS run;
     * v6.2: refuses unless collector, pricer, runner, banker AND checker each have a manifest
       written for this run, none FAILED, outputs unchanged and no unlisted file in their folder;
     * builds the dashboard's state/v5 document with the v5 builder (pipeline/v5/dashboard_doc.py)
       from this run's runner outputs, never the repo's data/v5;
     * writes publisher/state_v5.json, publisher/plan.json (what the session must do), manifest.

  2. The session APPLIES plan.json: writes state_v5.json to the dashboard (ArtifactData set,
     pinned to the version it read) and uploads the run folder to Drive "Match Engine — Runs".

  3. CONFIRM, only after both writes succeeded:
         python publish.py --confirm --root <repo> --run-dir <runs/run_id> \\
             --dashboard-version <new version> --drive-folder <Drive folder id of this run>
     * re-verifies the prepared files are unchanged;
     * records publisher/confirm/confirmed.json (+ manifest);
     * writes the row-count baseline from the COLLECTOR's manifest (the counts the Checker
       actually judged), to <runs-dir>/baseline/rowcounts.json. This is the only place the
       baseline is written, and only after a real publish, so a truncated fetch can never
       become the new normal by clearing a block.
     * v6.2: re-verifies the collector's manifest first, refuses to drop any season that is in the
       current baseline (retire one by editing the baseline by hand), and writes atomically.

The "Prediction Model" ledger doc is never touched.
Exit codes: 0 done, 1 refused / FAILED, 2 crashed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import manifest as M  # noqa: E402

DASHBOARD_URL = "https://claude.ai/artifact/L5teSNXiRwc9eUxRdmZngQ"
DASHBOARD_DOC = {"collection": "state", "doc_id": "v5"}
DRIVE_RUNS_FOLDER = "1GpY8W50T-7XtJS8Zi2yalq9Bat7uujbm"      # "Match Engine — Runs"
UPSTREAM = ("collector", "pricer", "runner", "banker", "checker")
# v6.2: the only top-level entries a run folder may hold when it is uploaded
RUN_FOLDER_ALLOWED = set(UPSTREAM) | {"publisher", "orchestrator", "bankroll"}


def baseline_path(run_dir: Path) -> Path:
    return run_dir.parent / "baseline" / "rowcounts.json"


def _commit(root: Path):
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out or None
    except Exception:  # noqa: BLE001
        return None


def build_state(root: Path, run_dir: Path) -> dict:
    """Run the v5 dashboard builder against this run's runner outputs.

    dashboard_doc.build() reads backtest.json and checks.json next to itself, and the paper
    summary and picks from ROOT/data/v5. ROOT is pointed at a scratch folder holding copies of
    this run's runner/summary.json and runner/paper_picks.csv, so the builder is reused as is
    and the repo is never written."""
    v5 = root / "pipeline" / "v5"
    for f in ("dashboard_doc.py", "out/backtest.json", "out/checks.json"):
        if not (v5 / f).exists():
            raise FileNotFoundError(f"pipeline/v5/{f} not found")
    sys.path.insert(0, str(v5))
    import dashboard_doc as dd  # noqa: E402

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "data" / "v5"
        stage.mkdir(parents=True)
        for f in ("summary.json", "paper_picks.csv"):
            src = run_dir / "runner" / f
            if src.exists():
                shutil.copy(src, stage / f)
        old_root = dd.ROOT
        dd.ROOT = Path(tmp)
        try:
            doc = dd.build()
        finally:
            dd.ROOT = old_root
    doc["code_commit"] = _commit(root)
    return doc


def prepare(root: Path, run_dir: Path, break_until: str | None = None) -> dict:
    started = M.utc_now()
    out = run_dir / "publisher"
    out.mkdir(parents=True, exist_ok=True)
    rid = run_dir.name

    def refuse(msg):
        for f in ("state_v5.json", "plan.json"):          # never leave a publishable bundle behind
            (out / f).unlink(missing_ok=True)
        return M.write_manifest(out, agent="publisher", run_id=rid, status="FAILED", started=started,
                                warnings=[f"refused to publish: {msg}"])

    vp = run_dir / "checker" / "verdict.json"
    if not vp.exists():
        return refuse("no Checker verdict for this run")
    verdict = json.loads(vp.read_text())
    if verdict.get("verdict") not in ("PASS", "WARN"):
        return refuse(f"Checker verdict is {verdict.get('verdict')!r}")
    vrun = verdict.get("run_dir")
    if not vrun or Path(vrun).resolve() != run_dir.resolve():
        return refuse(f"the verdict was made for {vrun!r}, not this run")

    # v6.2: every upstream agent must have a manifest written for THIS run, with unchanged outputs
    # and no unlisted files (before, a missing manifest was skipped and only the runner's was required)
    problems = []
    for agent in UPSTREAM:
        problems += M.check_manifest(run_dir / agent, agent=agent, run_id=rid, strict=True)
        m = M.read_manifest(run_dir / agent)
        if m is not None and m.get("status") == "FAILED":
            problems.append(f"{agent}: status FAILED")
    if problems:
        return refuse("upstream outputs are missing, from another run or changed: " + "; ".join(problems))
    stray = sorted(p.name for p in run_dir.iterdir() if p.name not in RUN_FOLDER_ALLOWED)
    if stray:                                 # everything uploaded must come from a known stage
        return refuse(f"unexpected entries in the run folder: {stray}")

    doc = build_state(root, run_dir)
    doc["pulsematch_run"] = {
        "run_id": rid,
        "verdict": verdict["verdict"],
        "checker_counts": verdict.get("counts"),
        "warnings": [f"{w['id']}: {w['detail']}" for w in verdict.get("warnings", [])],
        "break_until": break_until,
    }
    bp, bk = run_dir / "banker" / "plan.json", run_dir / "banker" / "bankroll.json"
    if bp.exists() and bk.exists():            # the Banker's view, shown read-only on the dashboard
        doc["banker"] = {"bankroll": json.loads(bk.read_text()), "plan": json.loads(bp.read_text())}
    (out / "state_v5.json").write_text(json.dumps(doc, indent=1, default=str))

    upload = sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*")
                    if p.is_file() and not p.relative_to(run_dir).parts[0] == "publisher")
    plan = {
        "run_id": rid,
        "verdict": verdict["verdict"],
        "steps": [
            {"step": 1, "do": "dashboard",
             "how": "ArtifactData get, then set pinned with if_version to the version just read",
             "url": DASHBOARD_URL, **DASHBOARD_DOC, "file": str(out / "state_v5.json")},
            {"step": 2, "do": "drive",
             "how": f"create folder '{rid}' in Drive folder {DRIVE_RUNS_FOLDER}, one sub-folder per agent, "
                    "upload each file, check each uploaded size equals the local size",
             "parent_folder_id": DRIVE_RUNS_FOLDER, "files": upload + [
                 "publisher/state_v5.json", "publisher/plan.json", "publisher/manifest.json"]},
            {"step": 3, "do": "confirm", "only_if": "steps 1 and 2 both succeeded",
             "command": f"python {HERE / 'publish.py'} --confirm --root {root} --run-dir {run_dir} "
                        "--dashboard-version <version returned by the set> --drive-folder <id of the run folder>"},
        ],
        "never": ["write the 'Prediction Model' ledger doc", "run --confirm if a step failed",
                  "edit any file in the run folder"],
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    notes = [f"verdict {verdict['verdict']}: bundle ready; apply plan.json, then run --confirm",
             f"dashboard doc: {len(doc['dk_paper']['picks'])} picks, gate open={doc['gate']['open']}"]
    return M.write_manifest(out, agent="publisher", run_id=rid, status="OK", started=started,
                            inputs=[{**M.describe(vp), "path": "checker/verdict.json"}],
                            row_counts={"files_to_upload": len(upload) + 3,
                                        "dashboard_picks": len(doc["dk_paper"]["picks"])},
                            notes=notes)


def confirm(root: Path, run_dir: Path, dashboard_version: int, drive_folder: str) -> dict:
    started = M.utc_now()
    out = run_dir / "publisher" / "confirm"
    rid = run_dir.name

    def refuse(msg):
        return M.write_manifest(out, agent="publisher-confirm", run_id=rid, status="FAILED",
                                started=started, warnings=[f"refused to confirm: {msg}"])

    pm = M.read_manifest(run_dir / "publisher")
    if pm is None or pm["status"] != "OK":
        return refuse("there is no successfully prepared publish for this run")
    changed = M.verify_outputs(run_dir / "publisher")
    if changed:
        return refuse("prepared files changed after preparing: " + "; ".join(changed))
    if not drive_folder or dashboard_version is None or dashboard_version < 1:
        return refuse("needs the dashboard version returned by the write and the Drive folder id")

    cm = M.read_manifest(run_dir / "collector")
    if cm is None:
        return refuse("no collector manifest, so there are no checked row counts for the baseline")
    cp = M.check_manifest(run_dir / "collector", agent="collector", run_id=rid, strict=True)
    if cp:                                       # v6.2: re-verify the counts the baseline is built from
        return refuse("collector output does not verify: " + "; ".join(cp))
    counts = {k: int(v) for k, v in cm.get("row_counts", {}).items() if k != "odds_snapshots"}
    if not counts:
        return refuse("collector manifest has no row counts")

    out.mkdir(parents=True, exist_ok=True)
    bp = baseline_path(run_dir)
    bp.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(bp.read_text()) if bp.exists() else None
    shrunk = [k for k, v in (previous or {}).items() if k in counts and counts[k] < v]
    gone = [k for k in (previous or {}) if k not in counts]
    if shrunk or gone:                           # the Checker should have blocked this; never lower the bar
        return refuse(f"would lower the baseline for {shrunk + gone}"
                      + (f" ({gone} would be dropped; retire a season by editing the baseline by hand)"
                         if gone else ""))
    M.atomic_write_text(bp, json.dumps(counts, indent=1, sort_keys=True))   # v6.2: never half-written
    (out / "confirmed.json").write_text(json.dumps({
        "run_id": rid, "confirmed_at": M.utc_now(), "dashboard": {"url": DASHBOARD_URL, **DASHBOARD_DOC,
                                                                 "version": dashboard_version},
        "drive_folder": drive_folder, "baseline": str(bp), "baseline_files": len(counts),
        "baseline_rows": sum(counts.values()), "previous_baseline": previous is not None,
    }, indent=1))
    return M.write_manifest(out, agent="publisher-confirm", run_id=rid, status="OK", started=started,
                            row_counts={"baseline_files": len(counts), "baseline_rows": sum(counts.values())},
                            notes=[f"published: dashboard state/v5 version {dashboard_version}, "
                                   f"Drive folder {drive_folder}; baseline written to {bp}"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pulsematch Publisher agent.")
    ap.add_argument("--root", required=True, help="checkout of the odds repo")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--break-until", default=None)
    ap.add_argument("--confirm", action="store_true", help="step 3: record the publish and write the baseline")
    ap.add_argument("--dashboard-version", type=int, default=None)
    ap.add_argument("--drive-folder", default=None)
    a = ap.parse_args(argv)
    root, run_dir = Path(a.root).resolve(), Path(a.run_dir).resolve()
    try:
        m = (confirm(root, run_dir, a.dashboard_version, a.drive_folder) if a.confirm
             else prepare(root, run_dir, a.break_until))
    except Exception as e:  # noqa: BLE001
        print(f"PUBLISHER CRASHED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    for w in m["warnings"]:
        print(f"  WARN  {w}")
    for n in m["notes"]:
        print(f"  note  {n}")
    print(f"{m['status']}  {m['agent']}")
    return 1 if m["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
