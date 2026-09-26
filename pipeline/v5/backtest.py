"""
Pulsematch v5 — walk-forward backtest against the closing line.

For each test season S: train the adjustment model on every earlier season
(penalty chosen on S-1), predict S, and score:

  * log loss: pre-match market (baseline) vs v5 adjusted vs closing market
  * share of the baseline->close gap the model closes
  * paper bets where the adjusted probability beats the pre-match price by >= EDGE:
      CLV   = price_taken x fair closing probability - 1   (the v5 gate metric)
      ROI   = flat-stake profit per unit
  * the same bets for the pure ratings model (v4 style, blend weight 1) for reference
  * quarter-Kelly bankroll path with a per-match exposure cap
  * Monte Carlo: seasons re-simulated with the closing line as the true probability

  python pipeline/v5/backtest.py   ->  pipeline/v5/out/backtest.json, backtest.md
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src"))
import dc4  # noqa: E402
from features import FEATURES_1X2, FEATURES_OU  # noqa: E402
from model import AdjustModel  # noqa: E402

TEST_SEASONS = ["2021", "2122", "2223", "2324", "2425", "2526", "2627"]
FIRST_TRAIN = "1819"
EDGE = 0.02
KELLY_FRAC, BET_CAP, MATCH_CAP = 0.25, 0.02, 0.03
GATE_T, GATE_MIN_BETS = 2.0, 300
LEAGUE_NAME = {"E0": "Premier League", "SP1": "La Liga", "D1": "Bundesliga", "F1": "Ligue 1"}


def _ll(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))


def _llb(p, y):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _tstat(x):
    x = np.asarray(x, float)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else float("nan")


def dc_probs(df):
    G = dc4.grids(df.dc_lam.to_numpy(), df.dc_mu.to_numpy(), -0.05)
    return dc4.probs_1x2(G), dc4.p_over(G)


def bets_from(df, P, Pou, tag):
    """One row per paper bet (1X2 outcomes + over/under 2.5)."""
    rows = []
    specs = [("H", 0, "open_oH", "close_H", lambda r: r.y == 0), ("D", 1, "open_oD", "close_D", lambda r: r.y == 1),
             ("A", 2, "open_oA", "close_A", lambda r: r.y == 2)]
    for i, r in enumerate(df.itertuples()):
        for sel, k, oc, cc, win in specs:
            o, pc = getattr(r, oc), getattr(r, cc)
            if pd.notna(o) and pd.notna(pc) and P[i, k] * o - 1 >= EDGE:
                rows.append((r.Date, r.Div, r.Season, r.Index, "1X2", sel, o, P[i, k], pc, win(r)))
        for sel, p, oc, pc_, win in (("O2.5", Pou[i], "open_oO", r.close_O25, r.y_over == 1),
                                     ("U2.5", 1 - Pou[i], "open_oU", 1 - r.close_O25, r.y_over == 0)):
            o = getattr(r, oc)
            if pd.notna(o) and pd.notna(pc_) and pd.notna(p) and p * o - 1 >= EDGE:
                rows.append((r.Date, r.Div, r.Season, r.Index, "OU", sel, o, p, pc_, win))
    b = pd.DataFrame(rows, columns=["Date", "Div", "Season", "match", "market", "sel", "price", "p_model", "p_close", "won"])
    b["model"] = tag
    b["clv"] = b.price * b.p_close - 1
    b["pnl"] = np.where(b.won, b.price - 1, -1.0)
    return b


def kelly_path(b):
    """Quarter Kelly, 2% cap per bet, 3% cap per match, bankroll updated by match date."""
    if b.empty:
        return {"final": 1.0, "max_drawdown": 0.0}
    b = b.assign(Date=pd.to_datetime(b.Date)).sort_values("Date")
    b["f"] = (KELLY_FRAC * (b.p_model * b.price - 1) / (b.price - 1)).clip(0, BET_CAP)
    tot = b.groupby("match").f.transform("sum")
    b["f"] = np.where(tot > MATCH_CAP, b.f * MATCH_CAP / tot, b.f)       # correlated same-match bets = one position
    bank, peak, mdd = 1.0, 1.0, 0.0
    for _, day in b.groupby(b.Date.dt.date):
        stake = day.f.to_numpy() * bank
        bank += float((stake * day.pnl.to_numpy()).sum())
        peak = max(peak, bank); mdd = max(mdd, 1 - bank / peak)
    return {"final": round(bank, 3), "max_drawdown": round(mdd, 3)}


def monte_carlo(b, n=5000, seed=7):
    """Re-simulate the bet list with the fair closing probability as the truth: what a
    season looks like for an edge of this size, by luck alone."""
    if b.empty:
        return {}
    rng = np.random.default_rng(seed)
    wins = rng.random((n, len(b))) < b.p_close.to_numpy()
    roi = np.where(wins, b.price.to_numpy() - 1, -1).mean(1)
    return {"expected_roi": round(float(b.clv.mean()), 4), "p_losing_season": round(float((roi < 0).mean()), 3),
            "roi_5th_pct": round(float(np.quantile(roi, 0.05)), 4), "roi_95th_pct": round(float(np.quantile(roi, 0.95)), 4)}


def summarize(b):
    if b.empty:
        return {"n": 0}
    return {"n": int(len(b)), "mean_clv": round(float(b.clv.mean()), 4), "clv_t": round(_tstat(b.clv), 2),
            "pct_beat_close": round(float((b.clv > 0).mean()), 3), "roi": round(float(b.pnl.mean()), 4),
            "roi_t": round(_tstat(b.pnl), 2)}


def run():
    d = pd.read_csv(HERE / "out" / "dataset.csv", parse_dates=["Date"], dtype={"Season": str})
    d = d[d.Season >= FIRST_TRAIN]
    ok = d.dc_lam.notna() & d[["open_H", "open_D", "open_A", "close_H", "close_D", "close_A"]].notna().all(axis=1)
    d = d[ok].reset_index(drop=True)
    d_ou = d.open_O25.notna() & d.close_O25.notna()

    per_season, all_bets, coefs = [], [], {}
    for S in TEST_SEASONS:
        tr, te = d[d.Season < S], d[d.Season == S]
        if te.empty or tr.Season.nunique() < 2:
            continue
        prev = sorted(tr.Season.unique())[-1]
        m1 = AdjustModel("1x2", FEATURES_1X2).fit(tr, val_season=prev)
        mo = AdjustModel("ou", FEATURES_OU).fit(tr[d_ou[tr.index]], val_season=prev)
        coefs[S] = {"1x2": m1.coefficients(), "ou": mo.coefficients()}
        P = m1.predict(te)
        Pou = np.full(len(te), np.nan)
        mask = d_ou[te.index].to_numpy()
        Pou[mask] = mo.predict(te[mask])
        Pdc, Pdc_ou = dc_probs(te)
        y = te.y.to_numpy(); Po = te[["open_H", "open_D", "open_A"]].to_numpy(); Pc = te[["close_H", "close_D", "close_A"]].to_numpy()
        l_open, l_adj, l_close, l_dc = _ll(Po, y), _ll(P, y), _ll(Pc, y), _ll(Pdc, y)
        tm = te[mask]; yo = tm.y_over.to_numpy(float)
        o_open, o_adj, o_close = _llb(tm.open_O25.to_numpy(), yo), _llb(Pou[mask], yo), _llb(tm.close_O25.to_numpy(), yo)
        gap = l_open.mean() - l_close.mean()
        row = {"season": S, "n": int(len(te)), "close_source": te.close_src.value_counts().to_dict(),
               "ll_1x2": {"open": round(l_open.mean(), 4), "v5": round(l_adj.mean(), 4), "close": round(l_close.mean(), 4),
                          "ratings_only": round(l_dc.mean(), 4), "v5_minus_open_t": round(_tstat(l_adj - l_open), 2),
                          "share_of_gap_to_close": round((l_open.mean() - l_adj.mean()) / gap, 2) if gap > 0 else None},
               "ll_ou": {"open": round(o_open.mean(), 4), "v5": round(o_adj.mean(), 4), "close": round(o_close.mean(), 4),
                         "v5_minus_open_t": round(_tstat(o_adj - o_open), 2)}}
        bv5 = bets_from(te, P, Pou, "v5"); bdc = bets_from(te, Pdc, Pdc_ou, "ratings_only")
        row["bets_v5"] = summarize(bv5); row["bets_ratings_only"] = summarize(bdc)
        per_season.append(row); all_bets += [bv5, bdc]
        print(S, row["ll_1x2"], row["bets_v5"], flush=True)

    B = pd.concat(all_bets, ignore_index=True)
    hist = B[B.Season != "2627"]
    v5, dcb = hist[hist.model == "v5"], hist[hist.model == "ratings_only"]
    recent = v5[v5.Season.isin(["2425", "2526"])]
    gate_open = bool(len(recent) >= GATE_MIN_BETS and recent.clv.mean() > 0 and _tstat(recent.clv) >= GATE_T)
    report = {
        "generated": pd.Timestamp.now("UTC").isoformat(), "edge_threshold": EDGE,
        "seasons": per_season, "coefficients": coefs,
        "pooled_2021_2526": {"v5": summarize(v5), "ratings_only": summarize(dcb),
                              "v5_by_market": {m: summarize(g) for m, g in v5.groupby("market")},
                              "v5_by_league": {LEAGUE_NAME[k]: summarize(g) for k, g in v5.groupby("Div")},
                              "v5_kelly": kelly_path(v5), "v5_monte_carlo": monte_carlo(v5)},
        "live_2627": summarize(B[(B.Season == "2627") & (B.model == "v5")]),
        "gate": {"rule": f"2024-25 + 2025-26 v5 bets: n >= {GATE_MIN_BETS}, mean CLV > 0, CLV t >= {GATE_T}",
                 "recent": summarize(recent), "open": gate_open},
    }
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "backtest.json").write_text(json.dumps(report, indent=2, default=str))
    B.to_csv(HERE / "out" / "bets.csv", index=False)
    return report


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: r[k] for k in ("pooled_2021_2526", "live_2627", "gate")}, indent=2, default=str))
