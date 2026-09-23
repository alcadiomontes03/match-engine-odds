"""
Match Engine v4 — validation gate.

What changed vs v3 backtest.py, and why:
  1. OUT-OF-SAMPLE SIZE. v3 tested only the current season (~40-70 matches in
     mid-September). A real edge over the market is a Brier/log-loss gap of a few
     thousandths; on n=50 the noise is ~10x that, so the v3 gate could never
     open, whatever the model's quality. v4 walks forward over three full past
     seasons (2023/24-2025/26), refitting before every match date.
  2. BENCHMARK. v3 used AvgH/AvgD/AvgA, which in football-data.co.uk are the
     PRE-closing snapshot, while its docstring called them closing. v4 benchmarks
     against the de-vigged market-average CLOSE (AvgC*), with Pinnacle close
     (PSC*) as a sensitivity check where present.
  3. BLEND WEIGHT CHOSEN FORWARD. v3 picked the blend weight on the same matches
     it scored. v4 picks it on earlier seasons only (2022/23 is tuning-only).
  4. SIGNIFICANCE. v3 opened on any improvement. v4 needs the 95% bootstrap CI
     (resampled by match date) of the log-loss difference to sit below zero.
  5. PER MARKET. v3 scored 1X2 only; the bets are also totals and spreads.
     v4 gates 1X2, Over/Under 2.5, and Asian handicap at half-point closing lines.
  6. LINE-MOVE TEST. Does (model - opening price) predict (closing - opening)?
     A model with real information should anticipate where the line goes.

Data: data/hist/<league-slug>_<YYYY>.csv, full football-data.co.uk columns.
Source used 15 Sep 2026: github.com/huhao930422-debug/football-odds-mirror
(data/<league>/season-YYYY.csv) — PL and La Liga only, verified vs
datasets/football-datasets (760/760 scores matched).
"""
from __future__ import annotations
import json, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import dc4

warnings.filterwarnings("ignore")
HIST = Path(__file__).resolve().parent.parent / "data" / "hist"
SEASONS = ["2122", "2223", "2324", "2425", "2526"]
TUNE, TEST = ["2223"], ["2324", "2425", "2526"]
EPS = 1e-12


def load(league_slug):
    fr = []
    for s in SEASONS:
        d = pd.read_csv(HIST / f"{league_slug}_{s}.csv", low_memory=False)
        d["Season"] = s
        fr.append(d)
    df = pd.concat(fr, ignore_index=True)
    df["Date"] = pd.to_datetime(df.Date, dayfirst=True)
    return df.dropna(subset=["FTHG", "FTAG"]).sort_values("Date").reset_index(drop=True)


def devig(cols, df):
    raw = np.stack([1.0 / df[c].to_numpy(float) for c in cols], 1)
    return raw / raw.sum(1, keepdims=True)


def walk(df, xi, lam_prior, seasons):
    """Predict every match in `seasons`, fitting on strictly earlier dates."""
    out, warm = [], None
    prev_teams, priors = None, {}
    for s in SEASONS:
        teams = set(df.loc[df.Season == s, "HomeTeam"])
        if prev_teams is not None and warm is not None:
            relegated = prev_teams - teams
            newcomers = teams - prev_teams
            ra = np.mean([warm["att"][t] for t in relegated if t in warm["att"]] or [0])
            rd = np.mean([warm["def"][t] for t in relegated if t in warm["def"]] or [0])
            priors = {t: (ra, rd) for t in newcomers}
        prev_teams = teams
        if s not in seasons:
            # still advance the warm start through non-evaluated seasons
            end = df.loc[df.Season == s, "Date"].max() + pd.Timedelta(days=1)
            warm = dc4.fit(df, end, xi, warm, priors, lam_prior)
            continue
        sd = df[df.Season == s]
        for day, block in sd.groupby("Date"):
            warm = dc4.fit(df, day, xi, warm, priors, lam_prior)
            lam, mu = dc4.lambdas(warm, block.HomeTeam.to_numpy(), block.AwayTeam.to_numpy())
            G = dc4.grids(lam, mu, warm["rho"])
            b = block.copy()
            b[["m_H", "m_D", "m_A"]] = dc4.probs_1x2(G)
            b["m_O25"] = dc4.p_over(G, 2.5)
            ah = b.AHCh.to_numpy(float)
            half = np.isfinite(ah) & (np.abs(np.mod(ah, 1)) == 0.5)
            b["ah_half"] = half
            b["m_AHH"] = np.where(half, dc4.p_home_cover_half(G, np.nan_to_num(ah)), np.nan)
            out.append(b)
    return pd.concat(out, ignore_index=True)


def attach_market(p):
    p[["c_H", "c_D", "c_A"]] = devig(["AvgCH", "AvgCD", "AvgCA"], p)
    p[["o_H", "o_D", "o_A"]] = devig(["AvgH", "AvgD", "AvgA"], p)
    if "PSCH" in p:
        p[["pc_H", "pc_D", "pc_A"]] = devig(["PSCH", "PSCD", "PSCA"], p)
    p["c_O25"] = devig(["AvgC>2.5", "AvgC<2.5"], p)[:, 0]
    p["o_O25"] = devig(["Avg>2.5", "Avg<2.5"], p)[:, 0]
    p["c_AHH"] = devig(["AvgCAHH", "AvgCAHA"], p)[:, 0]
    p["y"] = p.FTR.map({"H": 0, "D": 1, "A": 2})
    p["y_O25"] = (p.FTHG + p.FTAG > 2.5).astype(float)
    p["y_AHH"] = ((p.FTHG - p.FTAG + p.AHCh) > 0).astype(float)
    return p


def ll_multi(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], EPS, 1))


def ll_bin(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def pool_multi(Pm, Pk, w):
    L = w * np.log(np.clip(Pm, EPS, 1)) + (1 - w) * np.log(np.clip(Pk, EPS, 1))
    E = np.exp(L - L.max(1, keepdims=True)); return E / E.sum(1, keepdims=True)


def pool_bin(pm, pk, w):
    lg = lambda q: np.log(np.clip(q, EPS, 1 - EPS)) - np.log(np.clip(1 - q, EPS, 1))
    return 1 / (1 + np.exp(-(w * lg(pm) + (1 - w) * lg(pk))))


def boot_ci(diff, clusters, reps=2000, seed=7):
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(clusters)
    sums = np.bincount(codes, diff); cnts = np.bincount(codes)
    k = len(uniq); m = []
    for _ in range(reps):
        s = rng.integers(0, k, k)
        m.append(sums[s].sum() / cnts[s].sum())
    return float(diff.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


WGRID = np.linspace(0, 1, 21)


def market_gate(p, kind):
    """kind in {'1x2','o25','ah'}; returns dict with forward-chosen blend result."""
    if kind == "1x2":
        mask = p[["c_H", "c_D", "c_A"]].notna().all(1)
        d = p[mask]
        Pm = d[["m_H", "m_D", "m_A"]].to_numpy(); Pk = d[["c_H", "c_D", "c_A"]].to_numpy(); y = d.y.to_numpy()
        score = lambda w, sel: ll_multi(pool_multi(Pm[sel], Pk[sel], w), y[sel])
        base = lambda sel: ll_multi(Pk[sel], y[sel])
    else:
        mcol, kcol, ycol = {"o25": ("m_O25", "c_O25", "y_O25"), "ah": ("m_AHH", "c_AHH", "y_AHH")}[kind]
        mask = p[mcol].notna() & p[kcol].notna()
        if kind == "ah":
            mask &= p.ah_half
        d = p[mask]
        pm, pk, y = d[mcol].to_numpy(), d[kcol].to_numpy(), d[ycol].to_numpy()
        score = lambda w, sel: ll_bin(pool_bin(pm[sel], pk[sel], w), y[sel])
        base = lambda sel: ll_bin(pk[sel], y[sel])

    seasons = d.Season.to_numpy()
    diffs, model_diffs, clus, wts = [], [], [], {}
    for i, s in enumerate(TEST):
        train = np.isin(seasons, TUNE + TEST[:i]); test = seasons == s
        w = WGRID[np.argmin([score(w, train).mean() for w in WGRID])]
        wts[s] = float(w)
        diffs.append(score(w, test) - base(test))
        model_diffs.append(score(1.0, test) - base(test))
        clus.append(d.Date.to_numpy()[test])
    diff = np.concatenate(diffs); mdiff = np.concatenate(model_diffs); cl = np.concatenate(clus)
    mean, lo, hi = boot_ci(diff, cl)
    mm, mlo, mhi = boot_ci(mdiff, cl)
    return {"n_test": int(len(diff)), "blend_w_by_season": wts,
            "blend_minus_market_logloss": [round(mean, 5), round(lo, 5), round(hi, 5)],
            "model_minus_market_logloss": [round(mm, 5), round(mlo, 5), round(mhi, 5)],
            "gate_open": bool(hi < 0 and all(v > 0 for v in wts.values()))}


def line_move(p):
    """Slope of (close-open) on (model-open), home-win and over-2.5, test seasons."""
    t = p[p.Season.isin(TEST)]
    res = {}
    for name, m, o, c in [("home_1x2", "m_H", "o_H", "c_H"), ("over25", "m_O25", "o_O25", "c_O25")]:
        d = t[[m, o, c, "Date"]].dropna()
        x = (d[m] - d[o]).to_numpy(); y = (d[c] - d[o]).to_numpy()
        codes, _ = pd.factorize(d.Date); k = codes.max() + 1
        rng = np.random.default_rng(3); sl = []
        slope = float(np.polyfit(x, y, 1)[0])
        groups = [np.where(codes == g)[0] for g in range(k)]
        for _ in range(1000):
            ix = np.concatenate([groups[g] for g in rng.integers(0, k, k)])
            sl.append(np.polyfit(x[ix], y[ix], 1)[0])
        res[name] = [round(slope, 4), round(float(np.percentile(sl, 2.5)), 4), round(float(np.percentile(sl, 97.5)), 4)]
    return res


def run(league_slug, xis=(0.001, 0.002, 0.003, 0.005, 0.007), lams=(0.0, 3.0)):
    df = load(league_slug)
    tune = {}
    for xi in xis:
        for lp in lams:
            p = attach_market(walk(df, xi, lp, TUNE))
            tune[(xi, lp)] = float(ll_multi(p[["m_H", "m_D", "m_A"]].to_numpy(), p.y.to_numpy()).mean())
    xi, lp = min(tune, key=tune.get)
    p = attach_market(walk(df, xi, lp, TUNE + TEST))
    out = {"league": league_slug, "chosen_xi": xi, "chosen_newcomer_prior": lp,
           "tuning_logloss_1x2": {f"xi={k[0]},prior={k[1]}": round(v, 5) for k, v in tune.items()},
           "1x2": market_gate(p, "1x2"), "over_under_2_5": market_gate(p, "o25"),
           "asian_handicap_half_lines": market_gate(p, "ah"), "line_move_slope": line_move(p)}
    t = p[p.Season.isin(TEST)]
    out["raw_logloss_test_1x2"] = {
        "model": round(float(ll_multi(t[["m_H", "m_D", "m_A"]].to_numpy(), t.y.to_numpy()).mean()), 5),
        "market_open": round(float(ll_multi(t[["o_H", "o_D", "o_A"]].to_numpy(), t.y.to_numpy()).mean()), 5),
        "market_close": round(float(ll_multi(t[["c_H", "c_D", "c_A"]].to_numpy(), t.y.to_numpy()).mean()), 5)}
    pc = t.dropna(subset=["pc_H"])
    out["pinnacle_close_subset"] = {
        "n": int(len(pc)),
        "model": round(float(ll_multi(pc[["m_H", "m_D", "m_A"]].to_numpy(), pc.y.to_numpy()).mean()), 5),
        "pinnacle_close": round(float(ll_multi(pc[["pc_H", "pc_D", "pc_A"]].to_numpy(), pc.y.to_numpy()).mean()), 5)}
    return out, p


if __name__ == "__main__":
    results = {}
    for lg in ["premier-league", "la-liga"]:
        r, p = run(lg)
        results[lg] = r
        p.to_csv(HIST.parent / f"wf_{lg}.csv", index=False)
    print(json.dumps(results, indent=2))
    (HIST.parent / "gate4_results.json").write_text(json.dumps(results, indent=2))
