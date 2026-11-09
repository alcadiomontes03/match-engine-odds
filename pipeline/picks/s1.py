"""
Rule S1 — priced side-market picks (pre-registered in docs/prereg-forward-2026-11.md).

Runs in the odds workflow straight after a match's side-market pull and uses only DraftKings prices from that
pull. For every match it appends exactly one row to data/picks/s1.csv, either a pick or "no priced pick" with
the reason. Rows are never changed or deleted; git history is the audit trail.

  1. Shin de-vig of DraftKings' 1X2 (the moneyline from the same pull).
  2. Main total = the totals line whose Shin over-probability is closest to 0.5 (half or whole line).
  3. Solve (lam, mu) so the Dixon-Coles grid (rho = -0.05) matches that 1X2 and main total
     (BTTS as the anchor when no total exists; neither = no anchor).
  4. Price every other half-goal line from the grid: totals over/under, home/away spreads, BTTS yes/no.
     Quarter lines, whole lines and the anchor line are not eligible.
  5. EV = p_grid x DK price - 1 for each eligible side priced 1.40-3.50. The best line is always recorded;
     it is a pick only if EV >= 0.03. At most one pick per match.

Usage:  python pipeline/picks/s1.py            (append rows for matches not yet decided)
        python pipeline/picks/s1.py --dry-run  (print, write nothing)
Nothing here touches the dashboard, prices or stakes. Paper only.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
from v5 import market as M  # noqa: E402

RULE = "S1"
# Registration sample start. Only matches whose side pull is after this go into data/picks/s1.csv.
# Replays of earlier weeks are exploration: write them elsewhere with --out (see test_s1.py).
SAMPLE_START = "2026-11-10T00:00:00+00:00"
THRESHOLD = 0.03
PRICE_MIN, PRICE_MAX = 1.40, 3.50
RHO = M.RHO_MKT                 # -0.05
COMPS = {"PL": "E0", "LaLiga": "SP1", "Bundesliga": "D1", "Ligue1": "F1"}
SNAP = ROOT / "data" / "odds" / "snapshots.csv"
OUT = ROOT / "data" / "picks" / "s1.csv"

COLS = ["event_id", "league", "home", "away", "kickoff", "side_pulled_at", "mins_to_ko", "h2h_pulled_at",
        "dk_home", "dk_draw", "dk_away", "p_home", "p_draw", "p_away", "anchor", "anchor_line", "lam", "mu",
        "n_eligible", "market", "dk_market", "line", "side", "label", "dk_price", "dk_price_other", "p_grid",
        "p_dk_shin", "ev", "decision", "reason", "rule", "threshold", "price_min", "price_max", "code_commit",
        "decided_at"]

TOTAL_KEYS = ("totals", "alternate_totals")
SPREAD_KEYS = ("spreads", "alternate_spreads")


# ---------------------------------------------------------------- small helpers
def _half(x: float) -> bool:
    return abs((x * 2) % 2 - 1) < 1e-9            # x.5 only


def _half_or_whole(x: float) -> bool:
    return abs((x * 2) - round(x * 2)) < 1e-9


def _commit() -> str:
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha[:12]
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=12", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def lambdas_btts(pH, pD, pA, pB):
    def resid(x):
        lam, mu = np.exp(x)
        G = M.grid(lam, mu, RHO)
        p = M.grid_probs(G)
        return np.array([p["H"] - pH, p["D"] - pD, p["A"] - pA, 2.0 * (float(G[1:, 1:].sum()) - pB)])
    sup = np.log(max(pH, 1e-3) / max(pA, 1e-3)) * 0.6
    x0 = np.array([np.log(1.35) + sup / 2, np.log(1.35) - sup / 2])
    r = least_squares(resid, x0, bounds=([np.log(0.05)] * 2, [np.log(6.0)] * 2))
    return tuple(float(v) for v in np.exp(r.x))


# ---------------------------------------------------------------- lines from one pull
def side_lines(rows: pd.DataFrame, home: str, away: str) -> dict:
    """{('tot', L): {'over': (price, dk_market), 'under': ...}, ('sp', h): {'home': ..., 'away': ...},
    ('btts', None): {'yes': ..., 'no': ...}} from one pull's side rows. Plain markets win over alternates.
    Spread keys are the HOME handicap h; the away side of key h is the away team at -h."""
    out: dict = {}
    for mk in (TOTAL_KEYS[1], TOTAL_KEYS[0]):          # alternate first, plain overwrites
        for _, r in rows[rows.market == mk].iterrows():
            if pd.isna(r.point):
                continue
            side = str(r.outcome).lower()
            if side in ("over", "under"):
                out.setdefault(("tot", float(r.point)), {})[side] = (float(r.price), mk)
    for mk in (SPREAD_KEYS[1], SPREAD_KEYS[0]):
        for _, r in rows[rows.market == mk].iterrows():
            if pd.isna(r.point):
                continue
            if r.outcome == home:
                out.setdefault(("sp", float(r.point)), {})["home"] = (float(r.price), mk)
            elif r.outcome == away:
                out.setdefault(("sp", -float(r.point)), {})["away"] = (float(r.price), mk)
    for _, r in rows[rows.market == "btts"].iterrows():
        side = str(r.outcome).lower()
        if side in ("yes", "no"):
            out.setdefault(("btts", None), {})[side] = (float(r.price), "btts")
    return {k: v for k, v in out.items() if len(v) == 2}


def p_grid(G, kind, line, side):
    I, J = np.indices(G.shape)
    if kind == "tot":
        p_over = float(G[(I + J) > line].sum())
        return p_over if side == "over" else 1.0 - p_over
    if kind == "sp":
        p_home = float(G[(I - J + line) > 0].sum())
        return p_home if side == "home" else 1.0 - p_home
    p_yes = float(G[1:, 1:].sum())
    return p_yes if side == "yes" else 1.0 - p_yes


def label(kind, line, side, home, away):
    if kind == "tot":
        return f"{side.title()} {line:g} goals"
    if kind == "btts":
        return f"BTTS {side.title()}"
    h = line if side == "home" else -line
    return f"{home if side == 'home' else away} {h:+g}"


# ---------------------------------------------------------------- the rule
def decide(ev_rows: pd.DataFrame, h2h_rows: pd.DataFrame, meta: dict) -> dict:
    """One decision row for one match. ev_rows: that match's side rows from ONE pull.
    h2h_rows: its moneyline rows from the same pull (or the latest earlier pull)."""
    row = {c: "" for c in COLS}
    row.update(meta, rule=RULE, threshold=THRESHOLD, price_min=PRICE_MIN, price_max=PRICE_MAX)
    home, away = meta["home"], meta["away"]

    def no(reason):
        row.update(decision="no priced pick", reason=reason)
        return row

    px = {}
    for _, r in h2h_rows.iterrows():
        px["home" if r.outcome == home else "away" if r.outcome == away else "draw"] = float(r.price)
    if not all(px.get(k, 0) > 1 for k in ("home", "draw", "away")):
        return no("no complete DraftKings moneyline")
    pH, pD, pA = M.shin([px["home"], px["draw"], px["away"]])
    row.update(dk_home=px["home"], dk_draw=px["draw"], dk_away=px["away"],
               p_home=round(pH, 6), p_draw=round(pD, 6), p_away=round(pA, 6))

    lines = side_lines(ev_rows, home, away)
    if not lines:
        return no("no side prices")
    main = None
    for (kind, L), d in lines.items():
        if kind != "tot" or not _half_or_whole(L):
            continue
        po = float(M.shin([d["over"][0], d["under"][0]])[0])
        if main is None or abs(po - 0.5) < abs(main[1] - 0.5):
            main = (L, po)
    if main:
        lam, mu = M.implied_lambdas(pH, pD, pA, main[1], line=main[0])
        anchor, anchor_key = "total", ("tot", main[0])
    elif ("btts", None) in lines:
        b = lines[("btts", None)]
        lam, mu = lambdas_btts(pH, pD, pA, float(M.shin([b["yes"][0], b["no"][0]])[0]))
        anchor, anchor_key = "btts", ("btts", None)
    else:
        return no("no anchor (no total or BTTS)")
    if not (np.isfinite(lam) and np.isfinite(mu)):
        return no("grid fit failed")
    G = M.grid(lam, mu, RHO)
    row.update(anchor=anchor, anchor_line="" if anchor_key[1] is None else anchor_key[1],
               lam=round(lam, 4), mu=round(mu, 4))

    cands = []
    for (kind, L), d in lines.items():
        if (kind, L) == anchor_key or (kind != "btts" and not _half(L)):
            continue
        sides = list(d)
        shin2 = M.shin([d[sides[0]][0], d[sides[1]][0]])
        for i, s in enumerate(sides):
            price, mk = d[s]
            if not (PRICE_MIN <= price <= PRICE_MAX):
                continue
            p = p_grid(G, kind, L, s)
            cands.append(dict(market=kind, dk_market=mk, line="" if L is None else L, side=s,
                              label=label(kind, L, s, home, away), dk_price=price,
                              dk_price_other=d[sides[1 - i]][0], p_grid=round(p, 6),
                              p_dk_shin=round(float(shin2[i]), 6), ev=round(p * price - 1.0, 6)))
    row["n_eligible"] = len(cands)
    if not cands:
        if all(k == anchor_key for k in lines):
            return no(f"only the anchor ({anchor}) was priced, nothing left to test")
        return no("no eligible line (quarter/whole lines only, or priced outside 1.40-3.50)")
    best = max(cands, key=lambda c: (c["ev"], -abs(c["p_grid"] - 0.5)))
    row.update(best)
    if best["ev"] >= THRESHOLD:
        row.update(decision="pick", reason=f"largest EV {best['ev']:+.4f} >= {THRESHOLD}")
    else:
        row.update(decision="no priced pick", reason=f"no line with EV >= {THRESHOLD} (best {best['ev']:+.4f})")
    return row


# ---------------------------------------------------------------- driver
def decided_ids(path: Path = OUT) -> set:
    if not path.exists():
        return set()
    with path.open() as fh:
        return {r["event_id"] for r in csv.DictReader(fh)}


def build(snap: pd.DataFrame, done: set, now: datetime | None = None) -> list[dict]:
    """Decision rows for every in-scope match whose first side pull is not yet decided."""
    now = now or datetime.now(timezone.utc)
    d = snap[(snap.bookmaker == "draftkings") & (snap.comp.isin(COMPS))]
    side = d[d.market != "h2h"]
    h2h = d[d.market == "h2h"]
    out = []
    for eid, g in side.groupby("event_id"):
        if eid in done:
            continue
        first = g.pulled_at.min()
        rows = g[g.pulled_at == first]
        r0 = rows.iloc[0]
        ko = pd.Timestamp(r0.commence)
        pulled = pd.Timestamp(first)
        mins = (ko - pulled).total_seconds() / 60
        hh = h2h[(h2h.event_id == eid) & (h2h.pulled_at <= first)]
        h_at = hh.pulled_at.max() if len(hh) else ""
        meta = dict(event_id=eid, league=COMPS[r0.comp], home=r0.home, away=r0.away, kickoff=r0.commence,
                    side_pulled_at=first, mins_to_ko=round(mins, 1), h2h_pulled_at=h_at,
                    code_commit=_commit(), decided_at=now.isoformat(timespec="seconds"))
        if mins <= 0:
            row = {c: "" for c in COLS}
            row.update(meta, rule=RULE, threshold=THRESHOLD, price_min=PRICE_MIN, price_max=PRICE_MAX,
                       decision="no priced pick", reason="side pull was after kickoff")
        else:
            row = decide(rows, hh[hh.pulled_at == h_at] if h_at else hh, meta)
        out.append(row)
    return out


def append(rows: list[dict], path: Path = OUT):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--snap", default=str(SNAP))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--since", default=None, help="only matches kicking off after this ISO time")
    a = ap.parse_args()
    snap = pd.read_csv(a.snap, dtype={"point": float})
    out = Path(a.out)
    since = a.since
    if out.resolve() == OUT.resolve():
        since = max(since or SAMPLE_START, SAMPLE_START)     # the registered file never gets pre-sample rows
    if since:
        snap = snap[snap.pulled_at > since]
    rows = build(snap, decided_ids(out))
    picks = [r for r in rows if r["decision"] == "pick"]
    for r in rows:
        print(f"S1 {r['league']} {r['home']} v {r['away']}: {r['decision']}"
              + (f" {r['label']} @ {r['dk_price']} EV {float(r['ev']):+.3f}" if r["decision"] == "pick" else
                 f" ({r['reason']})"))
    print(f"S1: {len(rows)} new decisions, {len(picks)} picks")
    if not a.dry_run:
        append(rows, out)


if __name__ == "__main__":
    main()
