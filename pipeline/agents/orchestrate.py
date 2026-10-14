"""
Pulsematch — Orchestrator.

Runs the agents in order, stops at the first hard failure, and writes one run folder:

    runs/<run_id>/
        collector/      manifest.json, completeness.json      (collect.py)
        pricer/         priced.csv                            (pricer.py)
        runner/         paper_picks.csv, summary.json, ...    (runner.py)
        banker/         bankroll.json, plan.json              (banker.py; stakes only with the gate open)
        checker/        verdict.json, manifest.json           (checker.py)
        publisher/      state_v5.json, plan.json              (publish.py)
        orchestrator/   manifest.json, summary.md, <stage>.log

It computes nothing itself. Policy (decided 2026-10-02):
  * collector FAILED            -> stop. Nothing downstream runs, nothing is published.
  * collector OK or PARTIAL     -> continue. Its warnings go in the summary and the Checker rules.
  * any stage crashes / times out / exits with an error -> stop, nothing published.
  * Checker BLOCK (or Checker crashed) -> stop before the Publisher. Previous state stays up.
  * Checker PASS or WARN        -> the Publisher may run.
A stage whose script is not in the agents folder yet is recorded as NOT_BUILT and skipped, so the
orchestrator picks up new agents as they are added.
v6.2: refuses to start if the run folder already exists (exit 1), and the Publisher runs only if
the Checker exited 0 AND wrote a PASS/WARN verdict.json for this run folder.

Exit codes: 0 run completed and the Checker allowed publishing; 1 run stopped or blocked;
2 the orchestrator itself crashed (treat as 1).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import manifest as M  # noqa: E402

# (stage name, script). Order matters.
STAGES = [
    ("collector", "collect.py"),
    ("pricer", "pricer.py"),
    ("runner", "runner.py"),
    ("banker", "banker.py"),
    ("checker", "checker.py"),
    ("publisher", "publish.py"),
]

# stage outcomes
DONE, NOT_BUILT, FAILED, TIMEOUT, CRASHED, SKIPPED = "DONE", "NOT_BUILT", "FAILED", "TIMEOUT", "CRASHED", "SKIPPED"


def _args(stage: str, ctx: dict) -> list[str]:
    root, runs, rid, brk = ctx["root"], ctx["runs_dir"], ctx["run_id"], ctx["break_until"]
    run = Path(runs) / rid
    if stage == "collector":
        a = ["--root", str(root), "--runs-dir", str(runs), "--run-id", rid]
    elif stage == "checker":
        (run / "checker").mkdir(parents=True, exist_ok=True)
        a = ["--root", str(root), "--run-dir", str(run), "--out", str(run / "checker" / "verdict.json")]
        if ctx.get("baseline"):
            a += ["--baseline", str(ctx["baseline"])]
    else:  # pricer, runner, banker, publisher share one calling convention
        a = ["--root", str(root), "--run-dir", str(run)]
    if brk:
        a += ["--break-until", brk]
    return a


def run_stage(stage: str, script: Path, ctx: dict, timeout: float, log: Path) -> dict:
    t0 = time.time()
    cmd = [sys.executable, str(script), *_args(stage, ctx)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"")
        out = out.decode() if isinstance(out, bytes) else out
        log.write_text(f"TIMED OUT after {timeout:.0f}s\n{out}")
        return {"stage": stage, "outcome": TIMEOUT, "exit": None, "seconds": round(time.time() - t0, 1),
                "detail": f"exceeded the {timeout:.0f}s budget"}
    log.write_text(f"$ {' '.join(cmd)}\n--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}\nexit {p.returncode}\n")
    tail = [ln for ln in (p.stdout.strip().splitlines() or p.stderr.strip().splitlines()) if ln.strip()]
    last = tail[-1] if tail else ""
    if p.returncode == 0:
        outcome = DONE
    elif p.returncode == 1:
        outcome = FAILED          # the agent ran and said no (FAILED collector, BLOCK verdict, ...)
    else:
        outcome = CRASHED         # exit 2 or anything unexpected: the agent itself broke
    return {"stage": stage, "outcome": outcome, "exit": p.returncode,
            "seconds": round(time.time() - t0, 1), "detail": last}


def _sub_status(run: Path, name: str) -> tuple[str | None, list[str]]:
    m = M.read_manifest(run / name)
    return (m["status"], m.get("warnings", [])) if m else (None, [])


def _verdict_problem(run: Path) -> str | None:
    """None if checker/verdict.json exists, says PASS or WARN, and was made for this run folder."""
    v = run / "checker" / "verdict.json"
    if not v.exists():
        return "wrote no verdict.json"
    try:
        rep = json.loads(v.read_text())
    except Exception as e:  # noqa: BLE001
        return f"verdict.json is unreadable ({e})"
    if rep.get("verdict") not in ("PASS", "WARN"):
        return f"the verdict is {rep.get('verdict')!r}"
    if not rep.get("run_dir") or Path(rep["run_dir"]).resolve() != run.resolve():
        return f"the verdict was made for {rep.get('run_dir')!r}, not this run"
    return None


def _checker_manifest(run: Path, rid: str, started: str, exit_code: int) -> dict | None:
    """The Checker is read-only and writes only verdict.json; the orchestrator records its manifest."""
    v = run / "checker" / "verdict.json"
    if not v.exists():
        return None
    rep = json.loads(v.read_text())
    status = {"PASS": "OK", "WARN": "PARTIAL"}.get(rep.get("verdict"), "FAILED")
    notes = [f"verdict {rep.get('verdict')}", f"exit {exit_code}"]
    warns = [f"{c['id']}: {c['detail']}" for c in rep.get("checks", []) if c["severity"] in ("BLOCK", "WARN")]
    return M.write_manifest(run / "checker", agent="checker", run_id=rid, status=status,
                            started=started, warnings=warns, notes=notes)


def orchestrate(root: Path, runs_dir: Path, run_id: str | None = None, break_until: str | None = None,
                agents_dir: Path | None = None, stage_timeout: float = 600, total_budget: float = 1800,
                baseline: Path | None = None) -> dict:
    agents_dir = agents_dir or HERE
    baseline = baseline or runs_dir / "baseline" / "rowcounts.json"   # written only by publish.py --confirm
    rid = run_id or M.new_run_id()
    started = M.utc_now()
    run = runs_dir / rid
    if run.exists():
        # v6.2: never reuse a run folder; stale files from an earlier run could ride into this one
        raise FileExistsError(f"run folder {run} already exists; use a new run id")
    (run / "orchestrator").mkdir(parents=True)
    ctx = {"root": root, "runs_dir": runs_dir, "run_id": rid, "break_until": break_until, "baseline": baseline}

    results: list[dict] = []
    warnings: list[str] = []
    notes: list[str] = []
    outcome = "COMPLETED"          # COMPLETED | BLOCKED | STOPPED | CRASHED
    t_start = time.time()
    stop_reason = None
    checker_allowed = False

    for stage, script in STAGES:
        if stop_reason:
            results.append({"stage": stage, "outcome": SKIPPED, "detail": f"not run: {stop_reason}"})
            continue
        path = agents_dir / script
        if not path.exists():
            results.append({"stage": stage, "outcome": NOT_BUILT, "detail": f"{script} not built yet"})
            continue
        if stage == "publisher" and not checker_allowed:
            results.append({"stage": stage, "outcome": SKIPPED, "detail": "Checker did not allow publishing"})
            continue
        left = total_budget - (time.time() - t_start)
        if left <= 0:
            stop_reason = f"total time budget of {total_budget:.0f}s used up"
            outcome = "STOPPED"
            warnings.append(stop_reason)
            results.append({"stage": stage, "outcome": SKIPPED, "detail": f"not run: {stop_reason}"})
            continue

        stage_started = M.utc_now()
        r = run_stage(stage, path, ctx, min(stage_timeout, left), run / "orchestrator" / f"{stage}.log")
        results.append(r)

        if stage == "collector":
            status, cw = _sub_status(run, "collector")
            r["status"] = status
            warnings += [f"collector: {w}" for w in cw]
            if r["outcome"] == DONE and status in ("OK", "PARTIAL"):
                if status == "PARTIAL":
                    notes.append("collector reported PARTIAL; continuing, the Checker rules on it")
                continue
            stop_reason = f"collector {status or r['outcome']}: {r['detail']}"
            outcome = "STOPPED"
            warnings.append(stop_reason)
            continue

        if stage == "checker":
            cm = _checker_manifest(run, rid, stage_started, r["exit"] if r["exit"] is not None else -1)
            vproblem = _verdict_problem(run) if r["outcome"] == DONE else None
            if r["outcome"] == DONE and vproblem:
                # v6.2: exit 0 is not enough; a PASS/WARN verdict for this run must exist on disk
                outcome = "BLOCKED"
                stop_reason = f"Checker exited 0 but {vproblem}"
                warnings.append(stop_reason)
                continue
            if r["outcome"] == DONE:
                checker_allowed = True
                warnings += [f"checker: {w}" for w in (cm or {}).get("warnings", [])]
                continue
            if r["outcome"] == FAILED:
                outcome = "BLOCKED"
                stop_reason = f"Checker BLOCK: {r['detail']}"
            else:
                outcome = "BLOCKED"
                stop_reason = f"Checker {r['outcome'].lower()}: an unexplained checker is not a pass"
            warnings.append(stop_reason)
            continue

        if r["outcome"] == DONE:                  # pricer, runner, banker, publisher: carry their warnings up
            st, sw = _sub_status(run, stage)
            r["status"] = st
            warnings += [f"{stage}: {w}" for w in sw]
            continue
        if r["outcome"] != DONE:
            outcome = "CRASHED" if r["outcome"] == CRASHED else "STOPPED"
            stop_reason = f"{stage} {r['outcome'].lower()}: {r['detail']}"
            warnings.append(stop_reason)

    built = [r for r in results if r["outcome"] not in (NOT_BUILT,)]
    missing = [r["stage"] for r in results if r["outcome"] == NOT_BUILT]
    if missing:
        notes.append("not built yet, skipped: " + ", ".join(missing))
    publish_ready = any(r["stage"] == "publisher" and r["outcome"] == DONE for r in results)
    published = False            # only publish.py --confirm, after the session's writes, makes it published
    if outcome == "COMPLETED" and publish_ready:
        notes.append("publish bundle ready in publisher/: apply plan.json (dashboard, Drive), "
                     "then run publish.py --confirm")
    elif outcome == "COMPLETED":
        notes.append("Checker allowed publishing but no publish bundle was prepared")

    status = "OK" if outcome == "COMPLETED" and not warnings else ("PARTIAL" if outcome == "COMPLETED" else "FAILED")
    summary = _summary(rid, outcome, published, results, warnings, notes, break_until)
    (run / "orchestrator" / "summary.md").write_text(summary)
    (run / "orchestrator" / "stages.json").write_text(json.dumps(results, indent=2))
    m = M.write_manifest(run / "orchestrator", agent="orchestrator", run_id=rid, status=status, started=started,
                         warnings=warnings, notes=notes + [f"outcome {outcome}", f"publish_ready {publish_ready}"],
                         row_counts={"stages_run": len(built)})
    m["outcome"] = outcome
    m["published"] = published
    m["publish_ready"] = publish_ready
    return m


def _summary(rid, outcome, published, results, warnings, notes, brk) -> str:
    head = {"COMPLETED": "Run completed; the Checker allowed publishing (nothing is published until the "
                         "plan is applied and confirmed).",
            "BLOCKED": "Run BLOCKED by the Checker. Nothing was published; the previous state stays up.",
            "STOPPED": "Run STOPPED at a hard failure. Nothing was published.",
            "CRASHED": "A stage crashed. Nothing was published."}[outcome]
    lines = [f"# Pulsematch run {rid}", "", head, ""]
    if brk:
        lines += [f"International break until {brk}: no fixtures or results expected.", ""]
    lines += ["| Stage | Outcome | Detail |", "|---|---|---|"]
    for r in results:
        lines.append(f"| {r['stage']} | {r['outcome']} | {str(r.get('detail', '')).replace('|', '/')[:140]} |")
    if warnings:
        lines += ["", "**Needs a look**"] + [f"- {w}" for w in warnings]
    if notes:
        lines += ["", "**Notes**"] + [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pulsematch Orchestrator.")
    ap.add_argument("--root", required=True, help="checkout of the odds repo")
    ap.add_argument("--runs-dir", required=True, help="folder that holds the per-run folders")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--break-until", default=None)
    ap.add_argument("--agents-dir", default=None, help="folder holding the agent scripts (default: this folder)")
    ap.add_argument("--stage-timeout", type=float, default=600, help="seconds per stage")
    ap.add_argument("--total-budget", type=float, default=1800, help="seconds for the whole run")
    ap.add_argument("--baseline", default=None, help="row-count baseline for the Checker")
    a = ap.parse_args(argv)
    try:
        m = orchestrate(Path(a.root).resolve(), Path(a.runs_dir).resolve(), a.run_id, a.break_until,
                        Path(a.agents_dir).resolve() if a.agents_dir else None, a.stage_timeout,
                        a.total_budget, Path(a.baseline).resolve() if a.baseline else None)
    except FileExistsError as e:
        print(f"ORCHESTRATOR REFUSED: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"ORCHESTRATOR CRASHED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    run = Path(a.runs_dir).resolve() / m["run_id"]
    print((run / "orchestrator" / "summary.md").read_text())
    return 0 if m["outcome"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
