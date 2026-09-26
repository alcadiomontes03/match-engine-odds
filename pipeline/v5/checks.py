"""
Pulsematch v5 — two market-structure checks (ideas 4-6 from the reading list).

1. Soft vs sharp: treat the Pinnacle pre-match price as the fair line and look for
   prices at the *market-average* book (a stand-in for a recreational book like
   DraftKings) that beat it by >= EDGE. Scored on CLV against the Pinnacle close.
2. Derivative pricing: take the closing 1X2 + over/under 2.5, back out (lam, mu),
   price the Asian handicap from the goals grid, and compare with Pinnacle's own
   closing handicap price (half-goal lines only). If these agree, the engine can
   price spreads/totals/BTTS from the main lines and compare them with DraftKings.

  python pipeline/v5/checks.py   ->  pipeline/v5/out/checks.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import market  # noqa: E402
from backtest import EDGE, summarize, _tstat  # noqa: E402


def soft_vs_sharp(d: pd.DataFrame) -> dict:
    d = d[(d.open_src == "pinnacle") & (d.close_src == "pinnacle")]
    rows = []
    for r in d.itertuples():
        for sel, p, o, pc, won in (
            ("H", r.open_H, r.AvgH, r.close_H, r.y == 0), ("D", r.open_D, r.AvgD, r.close_D, r.y == 1),
            ("A", r.open_A, r.AvgA, r.close_A, r.y == 2),
            ("O2.5", r.open_O25, r.AvgO25, r.close_O25, r.y_over == 1),
        ):
            if pd.notna(o) and pd.notna(p) and pd.notna(pc) and p * o - 1 >= EDGE:
                rows.append((r.Season, sel, o, p, pc, won))
    b = pd.DataFrame(rows, columns=["Season", "sel", "price", "p_model", "p_close", "won"])
    if b.empty:
        return {"n": 0, "matches": int(len(d))}
    b["clv"] = b.price * b.p_close - 1
    b["pnl"] = np.where(b.won, b.price - 1, -1.0)
    out = summarize(b)
    out["matches_scanned"] = int(len(d))
    out["bets_per_100_matches"] = round(100 * len(b) / len(d), 2)
    return out


def derivative_pricing(d: pd.DataFrame, max_rows: int = 4000) -> dict:
    d = d[(d.close_src == "pinnacle") & d.AHCh.notna() & d.PCAHH.notna() & d.PCAHA.notna() & d.close_O25.notna()]
    d = d[(d.AHCh * 2) % 2 == 1]                                   # half-goal lines only (…, -0.5, 0.5, 1.5, …)
    d = d.sort_values("Date").tail(max_rows)
    ours, pins, outc = [], [], []
    for r in d.itertuples():
        lam, mu = market.implied_lambdas(r.close_H, r.close_D, r.close_A, r.close_O25)
        G = market.grid(lam, mu)
        I, J = np.indices(G.shape)
        ours.append(float(G[(I - J + r.AHCh) > 0].sum()))
        pins.append(market.shin([r.PCAHH, r.PCAHA])[0])
        outc.append(float(r.FTHG - r.FTAG + r.AHCh > 0))
    ours, pins, outc = map(np.array, (ours, pins, outc))
    return {"n": int(len(d)), "mean_abs_diff_vs_pinnacle": round(float(np.abs(ours - pins).mean()), 4),
            "bias_vs_pinnacle": round(float((ours - pins).mean()), 4),
            "p90_abs_diff": round(float(np.quantile(np.abs(ours - pins), 0.9)), 4),
            "brier_engine": round(float(((ours - outc) ** 2).mean()), 4),
            "brier_pinnacle": round(float(((pins - outc) ** 2).mean()), 4),
            "brier_diff_t": round(_tstat((ours - outc) ** 2 - (pins - outc) ** 2), 2)}


def run():
    d = pd.read_csv(HERE / "out" / "dataset.csv", parse_dates=["Date"], dtype={"Season": str})
    d = d.rename(columns={"Avg>2.5": "AvgO25"})
    out = {"soft_vs_sharp_all_seasons": soft_vs_sharp(d),
           "soft_vs_sharp_2324_on": soft_vs_sharp(d[d.Season >= "2324"]),
           "derivative_pricing_asian_handicap": derivative_pricing(d)}
    (HERE / "out" / "checks.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
