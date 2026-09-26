"""
Pulsematch v5 — DraftKings paper picks (DraftKings-only mode).

With DraftKings as the only book, the fair line comes from DraftKings itself: its
moneyline and 2.5 total are de-vigged (Shin) and turned into expected goals, and that
goals grid prices DraftKings' own spreads, other total lines and BTTS. A price that
beats the grid by >= EDGE is logged as a paper pick. This tests whether DraftKings
prices its derivative markets consistently with its own main lines; it cannot find
edges on the moneyline or 2.5 total themselves (those are the reference).

CLV: each pick is compared with DraftKings' last pre-kickoff price for the same
selection (two-way Shin de-vig of that market). That is the paper-mode scorecard.

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


def _fair_grid(g: pd.DataFrame, home: str, away: str):
    h2h = g[g.market == "h2h"].set_index("outcome").price
    if not {home, away, "Draw"} <= set(h2h.index):
        return None
    tot = g[(g.market == "totals") & (g.point == 2.5)].set_index("outcome").price
    pO = market.shin([tot["Over"], tot["Under"]])[0] if {"Over", "Under"} <= set(tot.index) else None
    if pO is None:
        return None                       # no 2.5 line -> total goals unknown, don't price
    pH, pD, pA = market.shin([h2h[home], h2h["Draw"], h2h[away]])
    lam, mu = market.implied_lambdas(pH, pD, pA, pO)
    return market.grid(lam, mu), lam, mu


def picks_from(snap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (eid, pulled), g in snap.groupby(["event_id", "pulled_at"]):
        r0 = g.iloc[0]
        fg = _fair_grid(g, r0.home, r0.away)
        if fg is None:
            continue
        G, lam, mu = fg
        for o in g.itertuples():
            if o.market == "spreads":
                side = "home" if o.outcome == r0.home else "away"
                home_line = o.point if side == "home" else -o.point
                ev = market.ev_spread(G, home_line, o.price, side)
            elif o.market == "totals" and o.point != 2.5:
                ev = market.ev_total(G, o.point, o.price, over=o.outcome == "Over")
            elif o.market == "btts":
                ev = market.ev_btts(G, o.price, yes=o.outcome == "Yes")
            else:
                continue
            if ev >= EDGE:
                rows.append({"pulled_at": pulled, "kind": o.kind, "comp": o.comp, "event_id": eid,
                             "commence": o.commence, "home": r0.home, "away": r0.away, "market": o.market,
                             "selection": o.outcome, "point": o.point, "price": o.price,
                             "ev": round(ev, 4), "exp_goals": f"{lam:.2f}-{mu:.2f}"})
    return pd.DataFrame(rows)


def add_clv(p: pd.DataFrame, snap: pd.DataFrame) -> pd.DataFrame:
    if p.empty:
        return p
    pre = snap[snap.pulled_at < snap.commence]
    close_at = pre.groupby("event_id").pulled_at.max()
    closes, clvs = [], []
    for r in p.itertuples():
        ca = close_at.get(r.event_id)
        if ca is None or ca <= r.pulled_at:
            closes.append(np.nan); clvs.append(np.nan); continue
        m = pre[(pre.event_id == r.event_id) & (pre.pulled_at == ca) & (pre.market == r.market)]
        if r.market != "btts":
            m = m[(m.point == r.point) | (m.point == -r.point)] if r.market == "spreads" else m[m.point == r.point]
        if len(m) != 2 or r.selection not in set(m.outcome):
            closes.append(np.nan); clvs.append(np.nan); continue
        mine = m[m.outcome == r.selection].price.iloc[0]; other = m[m.outcome != r.selection].price.iloc[0]
        pc = market.shin([mine, other])[0]
        closes.append(mine); clvs.append(round(r.price * pc - 1, 4))
    p = p.copy()
    p["close_price"], p["clv"] = closes, clvs
    return p


def run():
    snap = pd.read_csv(SNAP)
    snap = snap[snap.bookmaker == BOOK].copy()
    snap["point"] = pd.to_numeric(snap.point, errors="coerce")
    p = add_clv(picks_from(snap), snap)
    OUT.mkdir(parents=True, exist_ok=True)
    p.to_csv(OUT / "paper_picks.csv", index=False)
    c = p.clv.dropna() if not p.empty else pd.Series(dtype=float)
    summary = {"generated": pd.Timestamp.now("UTC").isoformat(timespec="seconds"), "mode": "paper (DraftKings only)",
               "edge_threshold": EDGE, "picks": int(len(p)), "picks_with_close": int(len(c)),
               "mean_clv": round(float(c.mean()), 4) if len(c) else None,
               "clv_t": round(float(c.mean() / (c.std(ddof=1) / np.sqrt(len(c)))), 2) if len(c) > 2 and c.std() > 0 else None,
               "by_market": ({m: int(n) for m, n in p.market.value_counts().items()} if not p.empty else {}),
               "gate": "open only with >= 300 picks with a closing price, mean CLV > 0 and CLV t >= 2"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
