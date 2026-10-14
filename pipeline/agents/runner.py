"""
Pulsematch — Model runner agent.

Runs the v5 model steps that produce this week's output, by calling the v5 code
(pipeline/v5/paper_picks.py, runtime.py, out/backtest.json). It recomputes nothing:

    paper picks ... paper_picks.picks_from   (DraftKings' own grid prices its side markets)
    CLV ........... paper_picks.add_clv
    gate state .... pipeline/v5/out/backtest.json  (written by backtest.py, the only thing
                    allowed to open the gate) and runtime.GATE_OPEN

Differences from running paper_picks.py directly, all deliberate:
  * Writes to the run folder, never to the repo's data/v5 (the repo is public and read-only here).
  * Only the four modelled leagues. The snapshots still hold old Championship captures;
    paper_picks.py filters them too since v5.5 (commit 3112a2e), and this stays as a second guard.
  * Publishes the blend weight (0 while the gate is closed) and the gate state so the Checker
    can verify them, and never adds a stake column.

Output (run folder):
    runner/paper_picks.csv   the full paper ledger, rebuilt from all in-scope snapshots
    runner/summary.json      counts, CLV, mode, blend_weight, gate
    runner/gate.json         gate rule and state, per-season shrinkage of the adjustment model
    runner/ratings.json      current Dixon-Coles team ratings per league (features.py settings:
                             dc4.fit, same decay and window, newcomer prior), so the Checker can
                             bound-check them (rho within +/-0.25, ratings within +/-1.0)
    runner/manifest.json

v6.2: GATE_OPEN in runtime.py must be a single plain True/False assignment, or the runner FAILS.

Status: FAILED if the v5 code, the snapshots or backtest.json are missing, or if the gate
sources disagree (runtime.GATE_OPEN vs backtest.json); PARTIAL if the pricer was PARTIAL or no
snapshot has the side markets the picks need; OK otherwise.

    python runner.py --root <repo checkout> --run-dir <runs/run_id> [--break-until]
Exit codes: 0 usable, 1 FAILED, 2 crashed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import manifest as M  # noqa: E402

SCOPE = ("PL", "LaLiga", "Bundesliga", "Ligue1")
SIDE_MARKETS = ("spreads", "totals", "btts")


def load_v5(root: Path):
    v5 = root / "pipeline" / "v5"
    for f in ("paper_picks.py", "market.py", "runtime.py"):
        if not (v5 / f).exists():
            raise FileNotFoundError(f"{v5 / f} not found")
    sys.path.insert(0, str(v5))
    import paper_picks  # noqa: E402
    gate_open = _runtime_gate(v5 / "runtime.py")
    return paper_picks, gate_open


_GATE_LINE = re.compile(r"^GATE_OPEN\s*=\s*(True|False)\s*(#.*)?$")


def _runtime_gate(path: Path) -> bool:
    """runtime.py uses a relative import, so read its GATE_OPEN constant instead of importing it.
    v6.2: strict. Exactly one top-level assignment, and it must be the literal True or False;
    anything else (a computed value, a second assignment, a typo) raises, so the runner fails
    rather than guessing the gate."""
    hits = []
    for line in path.read_text().splitlines():
        if re.match(r"^GATE_OPEN\b", line):
            m = _GATE_LINE.match(line.rstrip())
            if not m:
                raise ValueError(f"runtime.py GATE_OPEN is not a plain True/False literal: {line.strip()!r}")
            hits.append(m.group(1) == "True")
    if len(hits) != 1:
        raise ValueError(f"runtime.py must assign GATE_OPEN exactly once at top level, found {len(hits)}")
    return hits[0]


def current_ratings(root: Path, as_of) -> dict:
    """One Dixon-Coles fit per league as of `as_of`, with the same function and settings the v5
    gap features use (dc4.fit, XI, WINDOW, LAM_PRIOR, newcomer prior from the season-start fit).
    Only clubs in the current season are reported. Nothing here is new maths."""
    v5 = root / "pipeline" / "v5"
    sys.path.insert(0, str(v5))
    import features  # noqa: E402
    import build_dataset as bd  # noqa: E402
    dc4 = features.dc4
    bd.FD = root / "data" / "fd"          # always read this repo's history, even if the module was cached
    as_of = pd.Timestamp(as_of).tz_localize(None) if pd.Timestamp(as_of).tzinfo else pd.Timestamp(as_of)
    out = {}
    for code in bd.LEAGUES:
        df = bd.load(code)
        seasons = sorted(df.Season.unique())
        cur = seasons[-1]
        cur_teams = set(df[df.Season == cur].HomeTeam) | set(df[df.Season == cur].AwayTeam)
        prev_teams = (set(df[df.Season == seasons[-2]].HomeTeam) | set(df[df.Season == seasons[-2]].AwayTeam)
                      if len(seasons) > 1 else set())
        new = cur_teams - prev_teams if prev_teams else set()
        start = df[df.Season == cur].Date.min()
        has_history = bool((df.Date < start).any())
        r0 = dc4.fit(df, start, features.XI, window_days=features.WINDOW) if has_history else None
        fb = features._newcomer_fallback(r0) if r0 else None    # no earlier season: no newcomer prior
        r = dc4.fit(df, as_of, features.XI, warm=r0, window_days=features.WINDOW,
                    priors={t: fb for t in new} if fb else None, lam_prior=features.LAM_PRIOR)
        out[code] = {"season": cur, "rho": r["rho"], "home": r["home"], "n_matches": r["n_matches"],
                     "converged": r["converged"], "newcomers": sorted(new),
                     "attack": {t: round(v, 4) for t, v in sorted(r["att"].items()) if t in cur_teams},
                     "defence": {t: round(v, 4) for t, v in sorted(r["def"].items()) if t in cur_teams}}
    return {"as_of": str(as_of.date()), "source": "pipeline/v5 features.py settings via dc4.fit",
            "note": "strength = attack + defence (higher is stronger)", "leagues": out}


def shrinkage(bt: dict) -> dict:
    """Per season: the chosen L2 penalty and the largest coefficient, for 1X2 and totals."""
    out = {}
    for season, c in (bt.get("coefficients") or {}).items():
        row = {}
        for part in ("1x2", "ou"):
            body = c.get(part) or {}
            vals = []

            def walk(x):
                if isinstance(x, dict):
                    for k, v in x.items():
                        if k != "lambda":
                            walk(v)
                elif isinstance(x, (int, float)):
                    vals.append(abs(float(x)))
            walk(body)
            row[part] = {"lambda": body.get("lambda"), "max_abs_coef": round(max(vals), 4) if vals else None}
        out[season] = row
    return out


def run(root: Path, run_dir: Path, break_until: str | None = None) -> dict:
    started = M.utc_now()
    out = run_dir / "runner"
    out.mkdir(parents=True, exist_ok=True)
    rid = run_dir.name
    warnings, notes, inputs = [], [], []

    def fail(msg):
        return M.write_manifest(out, agent="runner", run_id=rid, status="FAILED", started=started,
                                inputs=inputs, warnings=[msg])

    try:
        pp, gate_runtime = load_v5(root)
    except Exception as e:  # noqa: BLE001
        return fail(f"v5 code unavailable: {e}")
    snap_p = root / "data" / "odds" / "snapshots.csv"
    bt_p = root / "pipeline" / "v5" / "out" / "backtest.json"
    if not snap_p.exists():
        return fail("data/odds/snapshots.csv not found")
    if not bt_p.exists():
        return fail("pipeline/v5/out/backtest.json not found: the gate state is unknown")

    bt = json.loads(bt_p.read_text())
    inputs.append({**M.describe(bt_p), "path": "pipeline/v5/out/backtest.json"})
    gate_bt = bool((bt.get("gate") or {}).get("open", False))
    if gate_bt != gate_runtime:
        return fail(f"gate sources disagree: backtest.json open={gate_bt}, runtime.GATE_OPEN={gate_runtime}")
    gate_open = gate_bt

    pricer = M.read_manifest(run_dir / "pricer")
    if pricer is None:
        notes.append("no pricer output in this run")
    elif pricer["status"] == "PARTIAL":
        warnings.append("pricer reported PARTIAL: " + "; ".join(pricer.get("warnings", []))[:300])

    snap = pd.read_csv(snap_p)
    inputs.append({**M.describe(snap_p, len(snap)), "path": "data/odds/snapshots.csv"})
    snap = snap[snap.bookmaker == pp.BOOK].copy()
    snap["point"] = pd.to_numeric(snap.point, errors="coerce")
    dropped = sorted(set(snap.comp) - set(SCOPE))
    if dropped:
        notes.append(f"snapshots from competitions not modelled were left out: {dropped}")
    snap = snap[snap.comp.isin(SCOPE)]

    side = snap[snap.market.isin(SIDE_MARKETS)]
    if snap.empty:
        notes.append("no in-scope DraftKings snapshots yet")
    elif side.empty:
        warnings.append("no in-scope snapshot carries spreads, totals or BTTS yet, so no paper picks can be "
                        "made (the v5 method prices DraftKings' side markets from its own main total line)")

    picks = pp.add_clv(pp.picks_from(snap), snap) if not snap.empty else pd.DataFrame()
    if picks.empty:
        picks = pd.DataFrame(columns=pp.COLUMNS)
    for col in ("stake", "stake_units", "size", "bankroll_fraction"):
        if col in picks.columns:                      # never: the gate decides stakes, not this agent
            return fail(f"paper picks carry a '{col}' column")
    picks.to_csv(out / "paper_picks.csv", index=False)

    c = pd.to_numeric(picks.clv, errors="coerce").dropna() if len(picks) else pd.Series(dtype=float)
    summary = {
        "generated": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
        "mode": "paper (DraftKings only)",
        "edge_threshold": pp.EDGE,
        "scope": list(SCOPE),
        "picks": int(len(picks)),
        "picks_with_close": int(len(c)),
        "mean_clv": round(float(c.mean()), 4) if len(c) else None,
        "clv_t": (round(float(c.mean() / (c.std(ddof=1) / np.sqrt(len(c)))), 2)
                  if len(c) > 2 and c.std() > 0 else None),
        "by_market": {m: int(n) for m, n in picks.market.value_counts().items()} if len(picks) else {},
        "by_comp": {m: int(n) for m, n in picks.comp.value_counts().items()} if len(picks) else {},
        "clv_basis": {b: int(n) for b, n in picks.clv_basis.value_counts().items()} if len(picks) else {},
        "blend_weight": 1.0 if gate_open else 0.0,
        "gate_open": gate_open,
        "gate": "open only with >= 300 picks with a closing price, mean CLV > 0 and CLV t >= 2",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    sh = shrinkage(bt)
    max_shrunk = [s for s, r in sh.items()
                  if all((r[p]["max_abs_coef"] or 0) < 0.01 for p in ("1x2", "ou"))]
    (out / "gate.json").write_text(json.dumps({
        "open": gate_open, "rule": (bt.get("gate") or {}).get("rule"),
        "recent": (bt.get("gate") or {}).get("recent"), "backtest_generated": bt.get("generated"),
        "shrinkage_by_season": sh,
        "seasons_fully_shrunk": max_shrunk,
    }, indent=2))
    try:
        rt = current_ratings(root, pd.Timestamp.now("UTC"))
        (out / "ratings.json").write_text(json.dumps(rt, indent=1))
        unconverged = [c for c, b in rt["leagues"].items() if not b["converged"]]
        if unconverged:
            warnings.append(f"rating fit did not converge for {unconverged}")
        notes.append("current ratings fitted for " + ", ".join(
            f"{c} (rho {b['rho']:+.3f})" for c, b in rt["leagues"].items()))
    except Exception as e:  # noqa: BLE001
        warnings.append(f"current ratings not produced ({type(e).__name__}: {e}); the Checker will skip them")
    notes.append(f"gate {'OPEN' if gate_open else 'closed'}; adjustment model fully shrunk in "
                 f"{len(max_shrunk)} of {len(sh)} seasons, so v5 prices equal the market")
    if not len(picks) and break_until:
        notes.append(f"international break until {break_until}: no new picks expected")

    status = "PARTIAL" if warnings else "OK"
    return M.write_manifest(out, agent="runner", run_id=rid, status=status, started=started, inputs=inputs,
                            row_counts={"snapshot_rows_in_scope": int(len(snap)),
                                        "side_market_rows": int(len(side)), "paper_picks": int(len(picks)),
                                        "picks_with_close": int(len(c))},
                            warnings=warnings, notes=notes)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pulsematch Model runner agent.")
    ap.add_argument("--root", required=True, help="checkout of the odds repo")
    ap.add_argument("--run-dir", required=True, help="this run's folder")
    ap.add_argument("--break-until", default=None)
    a = ap.parse_args(argv)
    try:
        m = run(Path(a.root).resolve(), Path(a.run_dir).resolve(), a.break_until)
    except Exception as e:  # noqa: BLE001
        print(f"RUNNER CRASHED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    for w in m["warnings"]:
        print(f"  WARN  {w}")
    for n in m["notes"]:
        print(f"  note  {n}")
    print(f"{m['status']}  runner  ({m['row_counts'].get('paper_picks', 0)} paper picks, "
          f"{m['row_counts'].get('picks_with_close', 0)} with a closing price)")
    return 1 if m["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
