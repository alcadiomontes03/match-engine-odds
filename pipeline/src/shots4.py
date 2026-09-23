"""
Match Engine v4 — shots layer.

Two uses of football-data shot columns (HS/AS = shots, HST/AST = on target):

1. MODEL INPUT. Shots on target are a less noisy signal of team strength than
   goals. We fit a second Poisson rating model on shots on target (same
   structure as Dixon-Coles, rho fixed at 0), convert it to goal scale with the
   training-window conversion rate (goals per shot on target, home and away
   separately), and blend in log space:
       log lam = w * log lam_goals + (1 - w) * log(conv * lam_sot)
   w is chosen on the tuning season only; the blended model then goes through
   the same closing-line gate as gate4.py.

2. SHOTS MARKETS (team / match level only — player props are out of scope).
   The shots-on-target and total-shots fits give expected counts per team.
   Shot counts are overdispersed relative to Poisson, so over/under
   probabilities use a negative binomial whose dispersion is estimated on the
   tuning season. No free historical odds exist for these markets, so they are
   scored against base rates only: predictions-only.

RESULTS (15 Sep 2026, test seasons 2023/24-2025/26, see shots4_results.json)
  Model input: NO GAIN. Forward-chosen goal weight, blend minus goals-only
  1X2 log loss: PL -0.001 (CI -0.009 to +0.007); La Liga +0.0045 (CI 0.000
  to +0.009). No early-season benefit either. Tuning on 2022/23 alone picked
  heavy shots weight (w=0.2/0.3) because that season had only one year of
  history; it did not carry forward. -> USE_SOT_IN_MODEL = False.
  Shots markets vs base rate (log loss, lower is better):
    home-team SOT o3.5  PL 0.570 vs 0.623 | LL 0.574 vs 0.628  (corr ~0.40-0.43)
    home-team SOT o4.5  PL 0.647 vs 0.693 | LL 0.636 vs 0.693
    match SOT o8.5      PL 0.688 vs 0.690 | LL 0.676 vs 0.692  (corr ~0.18-0.25)
    match shots o24.5   PL 0.677 vs 0.683 | LL 0.660 vs 0.696
  Team-level SOT carries real signal; match totals very little. Mild
  overdispersion (var/mean 1.05-1.44). La Liga totals predicted slightly low.
  Beating base rates is not beating a bookmaker; no shots odds history exists
  in the free stack.

Requires dc4.fit(..., rho_bounds=...) — see dc4_rho_bounds.patch.txt.
"""
from __future__ import annotations
import json, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import nbinom, poisson

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import dc4, gate4 as g

warnings.filterwarnings("ignore")
USE_SOT_IN_MODEL = False
SETTINGS = {"premier-league": (0.003, 3.0), "la-liga": (0.001, 3.0)}  # (xi, newcomer prior)
LINES = {"SOT_total": [7.5, 8.5, 9.5], "SOT_team": [3.5, 4.5], "Shots_total": [22.5, 24.5, 26.5]}


def _as(df, h, a):
    d = df.copy(); d["FTHG"] = d[h]; d["FTAG"] = d[a]
    return d.dropna(subset=["FTHG", "FTAG"])


def walk3(df, xi, lp, seasons):
    """Like gate4.walk but with parallel goals / shots-on-target / shots fits."""
    views = {"g": df, "s": _as(df, "HST", "AST"), "a": _as(df, "HS", "AS")}
    rb = {"g": (-0.25, 0.25), "s": (0.0, 0.0), "a": (0.0, 0.0)}
    warm = {k: None for k in views}; priors = {k: {} for k in views}
    prev, out = None, []
    for s in g.SEASONS:
        teams = set(df.loc[df.Season == s, "HomeTeam"])
        if prev is not None and warm["g"] is not None:
            rel, new = prev - teams, teams - prev
            for k in views:
                ra = np.mean([warm[k]["att"][t] for t in rel if t in warm[k]["att"]] or [0])
                rd = np.mean([warm[k]["def"][t] for t in rel if t in warm[k]["def"]] or [0])
                priors[k] = {t: (ra, rd) for t in new}
        prev = teams
        if s not in seasons:
            end = df.loc[df.Season == s, "Date"].max() + pd.Timedelta(days=1)
            for k, v in views.items():
                warm[k] = dc4.fit(v, end, xi, warm[k], priors[k], lp, rho_bounds=rb[k])
            continue
        for day, block in df[df.Season == s].groupby("Date"):
            b = block.copy()
            for k, v in views.items():
                warm[k] = dc4.fit(v, day, xi, warm[k], priors[k], lp, rho_bounds=rb[k])
                lam, mu = dc4.lambdas(warm[k], b.HomeTeam.to_numpy(), b.AwayTeam.to_numpy())
                b[f"lh_{k}"] = lam; b[f"la_{k}"] = mu
            b["rho"] = warm["g"]["rho"]
            # decay-weighted conversion (goals per shot on target) in the training window
            tr = views["s"][(views["s"].Date < day) & (views["s"].Date >= day - pd.Timedelta(days=1100))]
            w = np.exp(-xi * (day - tr.Date).dt.days.to_numpy())
            b["conv_h"] = np.sum(w * tr.FTHG_goals) / np.sum(w * tr.HST) if len(tr) else 0.33
            b["conv_a"] = np.sum(w * tr.FTAG_goals) / np.sum(w * tr.AST) if len(tr) else 0.33
            out.append(b)
    return pd.concat(out, ignore_index=True)


def blended_probs(p, w):
    lh = np.exp(w * np.log(p.lh_g) + (1 - w) * np.log(p.conv_h * p.lh_s))
    la = np.exp(w * np.log(p.la_g) + (1 - w) * np.log(p.conv_a * p.la_s))
    # low-score correction uses the goals fit's rho for that date
    G = dc4.grids(lh.to_numpy(), la.to_numpy(), p.rho.to_numpy())
    return dc4.probs_1x2(G), dc4.p_over(G, 2.5)


def nb_over(mean, k, line):
    """P(X > line) for negative binomial with mean `mean`, size k (var = m + m^2/k)."""
    pr = k / (k + mean)
    return 1 - nbinom.cdf(np.floor(line), k, pr)


def fit_dispersion(y, m):
    """Method-of-moments NB size from actual vs predicted means."""
    excess = np.mean((y - m) ** 2 - m)
    return float(np.mean(m ** 2) / excess) if excess > 0 else 1e6


def run(league):
    xi, lp = SETTINGS[league]
    df = g.load(league)
    df["FTHG_goals"] = df.FTHG; df["FTAG_goals"] = df.FTAG
    p = walk3(df, xi, lp, g.TUNE + g.TEST)
    p = g.attach_market(p)
    out = {"league": league, "xi": xi}

    # --- 1. shots as model input -----------------------------------------
    tune = p.Season.isin(g.TUNE)
    grid = np.round(np.linspace(0, 1, 11), 2)
    tl = {}
    for w in grid:
        P, _ = blended_probs(p[tune], w)
        tl[float(w)] = float(g.ll_multi(P, p.y[tune].to_numpy()).mean())
    w_best = min(tl, key=tl.get)
    out["tuning_logloss_by_goal_weight"] = {k: round(v, 5) for k, v in tl.items()}
    out["chosen_goal_weight"] = w_best
    res = {}
    for label, w in [("goals_only", 1.0), ("goals_plus_sot", w_best)]:
        q = p.copy()
        P, o25 = blended_probs(q, w)
        q[["m_H", "m_D", "m_A"]] = P; q["m_O25"] = o25
        q["m_AHH"] = np.nan  # AH not re-scored here
        t = q[q.Season.isin(g.TEST)]
        res[label] = {
            "test_logloss_1x2": round(float(g.ll_multi(t[["m_H", "m_D", "m_A"]].to_numpy(), t.y.to_numpy()).mean()), 5),
            "gate_1x2": g.market_gate(q, "1x2"),
            "gate_ou25": g.market_gate(q, "o25"),
        }
    t = p[p.Season.isin(g.TEST)]
    res["market_close_logloss_1x2"] = round(float(g.ll_multi(t[["c_H", "c_D", "c_A"]].to_numpy(), t.y.to_numpy()).mean()), 5)
    out["model_input"] = res

    # --- 2. shots markets (predictions only) -----------------------------
    mk = {}
    tn, ts = p[tune], p[p.Season.isin(g.TEST)]
    specs = {
        "SOT_total": (lambda d: d.lh_s + d.la_s, lambda d: d.HST + d.AST),
        "SOT_home_team": (lambda d: d.lh_s, lambda d: d.HST),
        "Shots_total": (lambda d: d.lh_a + d.la_a, lambda d: d.HS + d.AS),
    }
    for name, (mfun, yfun) in specs.items():
        k = fit_dispersion(yfun(tn).to_numpy(float), mfun(tn).to_numpy(float))
        m, y = mfun(ts).to_numpy(float), yfun(ts).to_numpy(float)
        lines = LINES["SOT_team"] if "team" in name else LINES["SOT_total" if "SOT" in name else "Shots_total"]
        rows = []
        for L in lines:
            pm = nb_over(m, k, L); pp = 1 - poisson.cdf(np.floor(L), m)
            yy = (y > L).astype(float)
            base = float((yfun(tn) > L).mean())
            rows.append({"line": L, "actual_over_rate": round(float(yy.mean()), 3),
                         "pred_over_rate_nb": round(float(pm.mean()), 3),
                         "pred_over_rate_poisson": round(float(pp.mean()), 3),
                         "logloss_nb": round(float(g.ll_bin(pm, yy).mean()), 4),
                         "logloss_poisson": round(float(g.ll_bin(pp, yy).mean()), 4),
                         "logloss_base_rate": round(float(g.ll_bin(np.full(len(yy), base), yy).mean()), 4)})
        corr = float(np.corrcoef(m, y)[0, 1])
        mk[name] = {"nb_size": round(k, 2), "mean_pred": round(float(m.mean()), 2),
                    "mean_actual": round(float(y.mean()), 2),
                    "var_actual_over_mean": round(float(y.var() / y.mean()), 2),
                    "corr_pred_actual": round(corr, 3), "lines": rows}
    out["shots_markets_test"] = mk
    p.to_csv(g.HIST.parent / f"wf_shots_{league}.csv", index=False)
    return out


if __name__ == "__main__":
    R = {lg: run(lg) for lg in SETTINGS}
    (g.HIST.parent / "shots4_results.json").write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))
