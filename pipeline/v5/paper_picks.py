"""
Pulsematch v5 — DraftKings paper picks (DraftKings-only mode).

With DraftKings as the only book, the fair line comes from DraftKings itself: its
moneyline and main total line are de-vigged (Shin) and turned into expected goals, and that
goals grid prices DraftKings' own spreads, other total lines and BTTS. A price that
beats the grid by >= EDGE is logged as a paper pick. This tests whether DraftKings
prices its derivative markets consistently with its own main lines; it cannot find
edges on the moneyline or main total themselves (those are the reference).

CLV: each pick is compared with DraftKings' last pre-kickoff price for the same
selection (two-way Shin de-vig of that market). Where that snapshot doesn't carry the
selection (BTTS is pulled once per match; lines move), the fair closing price comes
from that snapshot's goals grid instead; clv_basis records which. The main total line
can be any half or whole line, not just 2.5.

Rebuilt in full from data/odds/snapshots.csv on every run:
  python pipeline/v5/paper_picks.py  ->  data/v5/paper_picks.csv, data/v5/summary.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import market  # noqa: E402

ROOT = HERE.parent.parent
SNAP = ROOT / "data" / "odds" / "snapshots.csv"
OUT = ROOT / "data" / "v5"
EDGE = 0.02
BOOK = "draftkings"
COLUMNS = ["pulled_at", "kind", "comp", "event_id", "commence", "home", "away", "market", "selection",
           "point", "price", "ev", "exp_goals", "close_price", "clv", "clv_basis"]


def _fair_grid(g: pd.DataFrame, home: str, away: str):
    """DraftKings' own fair goals grid from its moneyline and main total line.
    The main line is the half/whole line priced closest to 50/50 (DraftKings' bulk
    feed usually carries just one). Quarter lines can't anchor the fit and are skipped.
    Returns (grid, lam, mu, main_line) or None."""
    h2h = g[g.market == "h2h"].set_index("outcome").price
    if not {home, away, "Draw"} <= set(h2h.index):
        return None
    best = None
    for L, t in g[g.market == "totals"].groupby("point"):
        t = t.set_index("outcome").price
        if not {"Over", "Under"} <= set(t.index) or round((L * 4) % 2) == 1:
            continue
        pO = market.shin([t["Over"], t["Under"]])[0]
        if best is None or abs(pO - 0.5) < abs(best[1] - 0.5):
            best = (float(L), pO)
    if best is None:
        return None                       # no usable total line -> total goals unknown, don't price
    pH, pD, pA = market.shin([h2h[home], h2h["Draw"], h2h[away]])
    lam, mu = market.implied_lambdas(pH, pD, pA, best[1], line=best[0])
    return market.grid(lam, mu), lam, mu, best[0]


def _ev(G, market_, outcome, point, price, home):
    if market_ == "spreads":
        side = "home" if outcome == home else "away"
        return market.ev_spread(G, point if side == "home" else -point, price, side)
    if market_ == "totals":
        return market.ev_total(G, point, price, over=outcome == "Over")
    if market_ == "btts":
        return market.ev_btts(G, price, yes=outcome == "Yes")
    return None


def picks_from(snap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (eid, pulled), g in snap.groupby(["event_id", "pulled_at"]):
        r0 = g.iloc[0]
        fg = _fair_grid(g, r0.home, r0.away)
        if fg is None:
            continue
        G, lam, mu, main_line = fg
        for o in g.itertuples():
            if o.market not in ("spreads", "totals", "btts") or (o.market == "totals" and o.point == main_line):
                continue                  # the moneyline and main total are the reference, not bets
            ev = _ev(G, o.market, o.outcome, o.point, o.price, r0.home)
            if ev >= EDGE:
                rows.append({"pulled_at": pulled, "kind": o.kind, "comp": o.comp, "event_id": eid,
                             "commence": o.commence, "home": r0.home, "away": r0.away, "market": o.market,
                             "selection": o.outcome, "point": o.point, "price": o.price,
                             "ev": round(ev, 4), "exp_goals": f"{lam:.2f}-{mu:.2f}"})
    return pd.DataFrame(rows)


def add_clv(p: pd.DataFrame, snap: pd.DataFrame) -> pd.DataFrame:
    """CLV against DraftKings' last pre-kickoff snapshot of the match.
    'market': that snapshot has the same selection -> two-way Shin de-vig of it.
    'grid':   it doesn't (BTTS is pulled once per match; lines move) -> fair price
              from the closing snapshot's own goals grid (its moneyline + main total)."""
    if p.empty:
        return p
    pre = snap[snap.pulled_at < snap.commence]
    close_at = pre.groupby("event_id").pulled_at.max()
    out = []
    for r in p.itertuples():
        ca = close_at.get(r.event_id)
        if ca is None or ca <= r.pulled_at:
            out.append((np.nan, np.nan, None)); continue
        cs = pre[(pre.event_id == r.event_id) & (pre.pulled_at == ca)]
        m = cs[cs.market == r.market]
        if r.market == "spreads":
            m = m[(m.point == r.point) | (m.point == -r.point)]
        elif r.market == "totals":
            m = m[m.point == r.point]
        if len(m) == 2 and r.selection in set(m.outcome):
            mine = m[m.outcome == r.selection].price.iloc[0]; other = m[m.outcome != r.selection].price.iloc[0]
            out.append((mine, round(r.price * market.shin([mine, other])[0] - 1, 4), "market")); continue
        fg = _fair_grid(cs, r.home, r.away)
        if fg is None:
            out.append((np.nan, np.nan, None)); continue
        ev = _ev(fg[0], r.market, r.selection, r.point, r.price, r.home)
        out.append((np.nan, round(ev, 4) if ev is not None else np.nan, "grid"))
    p = p.copy()
    p["close_price"], p["clv"], p["clv_basis"] = zip(*out)
    return p


def run():
    snap = pd.read_csv(SNAP)
    snap = snap[snap.bookmaker == BOOK].copy()
    snap["point"] = pd.to_numeric(snap.point, errors="coerce")
    p = add_clv(picks_from(snap), snap)
    if p.empty:                       # keep a header so readers never hit an empty file
        p = pd.DataFrame(columns=COLUMNS)
    OUT.mkdir(parents=True, exist_ok=True)
    p.to_csv(OUT / "paper_picks.csv", index=False)
    c = p.clv.dropna() if not p.empty else pd.Series(dtype=float)
    summary = {"generated": pd.Timestamp.now("UTC").isoformat(timespec="seconds"), "mode": "paper (DraftKings only)",
               "edge_threshold": EDGE, "picks": int(len(p)), "picks_with_close": int(len(c)),
               "mean_clv": round(float(c.mean()), 4) if len(c) else None,
               "clv_t": round(float(c.mean() / (c.std(ddof=1) / np.sqrt(len(c)))), 2) if len(c) > 2 and c.std() > 0 else None,
               "by_market": ({m: int(n) for m, n in p.market.value_counts().items()} if not p.empty else {}),
               "clv_basis": ({b: int(n) for b, n in p.clv_basis.value_counts().items()} if not p.empty else {}),
               "gate": "open only with >= 300 picks with a closing price, mean CLV > 0 and CLV t >= 2"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
