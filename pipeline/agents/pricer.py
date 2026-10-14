"""
Pulsematch — Market pricer agent.

Prices every upcoming in-scope fixture from DraftKings' latest pre-kickoff snapshot, using the
v5 market code only (pipeline/v5/market.py). Nothing here re-derives the maths:

    Shin de-vig ............ market.shin
    implied goals .......... market.implied_lambdas
    goals grid ............. market.grid / market.grid_probs / market.p_over_line
    derivative fair prices . market.price_derivatives

Output (run folder, never the repo):
    pricer/priced.csv     one row per fixture: odds, overround, fair 1X2, main total line,
                          implied goals (lam, mu), fair totals / spreads / BTTS, fit error, flags
    pricer/manifest.json

Rules:
  * Goals are only implied when DraftKings shows a usable total line (half or whole line), the
    same rule as v5 paper_picks: no total line -> total goals unknown -> no derivative prices.
    The fixture keeps its fair 1X2 and is flagged no_total_line.
  * Flags: no_total_line, overround (outside 1.00-1.15), fit (grid misses the fair prices by more
    than FIT_TOL), missing_h2h (fixture skipped).
  * Status: FAILED if the snapshots file is missing or the v5 code cannot be imported;
    PARTIAL if any fixture is flagged; OK otherwise. No fixtures (a break week) is OK.

    python pricer.py --root <repo checkout> --run-dir <runs/run_id> [--as-of ISO] [--break-until]

--as-of prices the fixtures that were upcoming at that moment, from the snapshots taken before
it. Used for testing against past captures; a live run leaves it unset (now).
Exit codes: 0 usable, 1 FAILED, 2 crashed.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import manifest as M  # noqa: E402

SCOPE = ("PL", "LaLiga", "Bundesliga", "Ligue1")      # odds-capture names of the four leagues
BOOK = "draftkings"
OVERROUND_RANGE = (1.00, 1.15)
FIT_TOL = 0.005
TOTALS = (1.5, 2.5, 3.5)
SPREADS = (-1.5, -0.5, 0.5, 1.5)
COLUMNS = (["event_id", "comp", "commence", "home", "away", "pulled_at", "odds_home", "odds_draw", "odds_away",
            "overround", "p_home", "p_draw", "p_away", "total_line", "p_over_main", "lam", "mu", "goals_basis"]
           + [f"p_over_{t}" for t in TOTALS] + [f"p_home_cover_{s:+g}" for s in SPREADS]
           + ["p_btts", "fit_max_err", "flags"])


def load_market(root: Path):
    v5 = root / "pipeline" / "v5"
    if not (v5 / "market.py").exists():
        raise FileNotFoundError(f"{v5 / 'market.py'} not found")
    sys.path.insert(0, str(v5))
    import market  # noqa: E402
    return market


def _ts(s):
    return pd.to_datetime(s, utc=True, errors="coerce")


def price_event(market, g: pd.DataFrame) -> dict:
    """g: one event's rows from one snapshot."""
    r0 = g.iloc[0]
    home, away = r0.home, r0.away
    row = {"event_id": r0.event_id, "comp": r0.comp, "commence": r0.commence, "home": home, "away": away,
           "pulled_at": r0.pulled_at}
    flags = []
    h2h = g[g.market == "h2h"].drop_duplicates("outcome", keep="last").set_index("outcome").price
    if not {home, away, "Draw"} <= set(h2h.index):
        row["flags"] = "missing_h2h"
        return row
    odds = [float(h2h[home]), float(h2h["Draw"]), float(h2h[away])]
    row["odds_home"], row["odds_draw"], row["odds_away"] = odds
    row["overround"] = round(float(sum(1.0 / o for o in odds)), 4)
    if not (OVERROUND_RANGE[0] <= row["overround"] <= OVERROUND_RANGE[1]):
        flags.append("overround")
    pH, pD, pA = market.shin(odds)
    row["p_home"], row["p_draw"], row["p_away"] = float(pH), float(pD), float(pA)

    # main total: the half/whole line priced closest to 50/50 (as v5 paper_picks chooses it)
    best = None
    for L, t in g[g.market == "totals"].groupby("point"):
        t = t.drop_duplicates("outcome", keep="last").set_index("outcome").price
        if not {"Over", "Under"} <= set(t.index) or round((float(L) * 4) % 2) == 1:
            continue
        pO = float(market.shin([t["Over"], t["Under"]])[0])
        if best is None or abs(pO - 0.5) < abs(best[1] - 0.5):
            best = (float(L), pO)
    if best is None:
        flags.append("no_total_line")
        row["goals_basis"] = "none"
        row["flags"] = ";".join(flags)
        return row

    line, pO = best
    row["total_line"], row["p_over_main"] = line, pO
    lam, mu = market.implied_lambdas(pH, pD, pA, pO, line=line)
    row["lam"], row["mu"], row["goals_basis"] = lam, mu, "1x2+total"
    fair = market.price_derivatives(lam, mu, totals=TOTALS, spreads=SPREADS)
    for t in TOTALS:
        row[f"p_over_{t}"] = fair["totals"][str(t)]["over"]
    for s in SPREADS:
        row[f"p_home_cover_{s:+g}"] = fair["spreads"][str(s)]
    row["p_btts"] = fair["btts"]
    G = market.grid(lam, mu)
    gp = market.grid_probs(G)
    err = max(abs(gp["H"] - pH), abs(gp["D"] - pD), abs(gp["A"] - pA), abs(market.p_over_line(G, line) - pO))
    row["fit_max_err"] = round(float(err), 5)
    if err > FIT_TOL:
        flags.append("fit")
    row["flags"] = ";".join(flags)
    return row


def price(root: Path, run_dir: Path, as_of: datetime | None = None, break_until: str | None = None) -> dict:
    started = M.utc_now()
    out = run_dir / "pricer"
    out.mkdir(parents=True, exist_ok=True)
    rid = run_dir.name
    warnings, notes, inputs = [], [], []

    snap_p = root / "data" / "odds" / "snapshots.csv"
    try:
        market = load_market(root)
    except Exception as e:  # noqa: BLE001
        pd.DataFrame(columns=COLUMNS).to_csv(out / "priced.csv", index=False)
        return M.write_manifest(out, agent="pricer", run_id=rid, status="FAILED", started=started,
                                warnings=[f"v5 market code unavailable: {e}"])
    if not snap_p.exists():
        pd.DataFrame(columns=COLUMNS).to_csv(out / "priced.csv", index=False)
        return M.write_manifest(out, agent="pricer", run_id=rid, status="FAILED", started=started,
                                warnings=["data/odds/snapshots.csv not found"])

    snap = pd.read_csv(snap_p)
    inputs.append({**M.describe(snap_p, len(snap)), "path": "data/odds/snapshots.csv"})
    now = as_of or datetime.now(timezone.utc)
    snap = snap[snap.bookmaker == BOOK].copy()
    snap["point"] = pd.to_numeric(snap.point, errors="coerce")
    snap["_pulled"], snap["_ko"] = _ts(snap.pulled_at), _ts(snap.commence)

    out_scope = sorted(set(snap.comp) - set(SCOPE))
    if out_scope:
        notes.append(f"captures from competitions not modelled are ignored: {out_scope}")
    snap = snap[snap.comp.isin(SCOPE)]
    up = snap[(snap._ko > now) & (snap._pulled <= now)]

    rows = []
    for eid, g in up.groupby("event_id"):
        latest = g[g._pulled == g._pulled.max()]
        rows.append(price_event(market, latest))
    d = pd.DataFrame(rows, columns=COLUMNS) if rows else pd.DataFrame(columns=COLUMNS)
    d = d.sort_values(["commence", "comp", "home"]) if len(d) else d
    d.to_csv(out / "priced.csv", index=False)

    # exact flag names (flags are ";"-joined), not substring matches
    flag_sets = d["flags"].fillna("").astype(str).str.split(";").apply(set) if len(d) else None
    counts = {f: int(flag_sets.apply(lambda s, f=f: f in s).sum()) for f in
              ("no_total_line", "overround", "fit", "missing_h2h")} if len(d) else {}
    for f, n in counts.items():
        if n:
            warnings.append({
                "no_total_line": f"{n} fixture(s) have no DraftKings total line: fair 1X2 only, "
                                 "no implied goals or derivative prices",
                "overround": f"{n} fixture(s) with DraftKings overround outside "
                             f"{OVERROUND_RANGE[0]:.2f}-{OVERROUND_RANGE[1]:.2f}",
                "fit": f"{n} fixture(s) where the goals grid misses the fair prices by more than {FIT_TOL}",
                "missing_h2h": f"{n} fixture(s) skipped: incomplete moneyline (kept in priced.csv with "
                               "flag missing_h2h and no prices, so the Checker can list them)",
            }[f])
    if not len(d):
        in_break = False
        if break_until:
            try:
                in_break = now.date() <= datetime.fromisoformat(break_until).date()
            except ValueError:
                pass
        notes.append("no upcoming in-scope fixtures in the snapshots"
                     + (f" (international break until {break_until}: expected)" if in_break else ""))
    by_comp = d.comp.value_counts().to_dict() if len(d) else {}
    notes.append(f"priced as of {now.isoformat(timespec='seconds')}: {len(d)} fixture(s) {by_comp}")
    status = "PARTIAL" if warnings else "OK"
    return M.write_manifest(out, agent="pricer", run_id=rid, status=status, started=started, inputs=inputs,
                            row_counts={"fixtures": int(len(d)),
                                        "with_goals": int(d.goals_basis.eq("1x2+total").sum()) if len(d) else 0,
                                        **{f"flag_{k}": v for k, v in counts.items()}},
                            warnings=warnings, notes=notes)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pulsematch Market pricer agent.")
    ap.add_argument("--root", required=True, help="checkout of the odds repo (pipeline/v5, data/odds)")
    ap.add_argument("--run-dir", required=True, help="this run's folder, e.g. runs/2026-10-05T1600Z")
    ap.add_argument("--as-of", default=None, help="ISO timestamp to price as of (default: now)")
    ap.add_argument("--break-until", default=None)
    a = ap.parse_args(argv)
    try:
        as_of = datetime.fromisoformat(a.as_of.replace("Z", "+00:00")) if a.as_of else None
        if as_of and as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        m = price(Path(a.root).resolve(), Path(a.run_dir).resolve(), as_of, a.break_until)
    except Exception as e:  # noqa: BLE001
        print(f"PRICER CRASHED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    for w in m["warnings"]:
        print(f"  WARN  {w}")
    for n in m["notes"]:
        print(f"  note  {n}")
    print(f"{m['status']}  pricer  ({m['row_counts'].get('fixtures', 0)} fixtures, "
          f"{m['row_counts'].get('with_goals', 0)} with implied goals)")
    return 1 if m["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
