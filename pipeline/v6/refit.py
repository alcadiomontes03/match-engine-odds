"""
Alpha Lines v6.1 - Dixon-Coles refit with a centred defence ridge ("goal-rate fix").

Why this exists
---------------
The live v4.10 fit (pipeline/src/dixon_coles.py) shrinks attack AND defence toward 0 with a
ridge penalty. Attack is pinned to mean 0, so that is harmless there. Defence is not: its
mean carries the league's goal level (log of the away goal rate = att[away] - def[home], so a
league that averages 1.4 away goals needs mean(def) of about -0.33). Pulling defence toward
0 therefore drags every team's expected goals DOWN, away goals most of all. On the Sep 25
fit the mean defence was -0.14 where the data wants about -0.33, so the model expected 1.21
away goals in the Bundesliga against 1.46 actual, and 2.90 total against 3.2.

The fix keeps the same likelihood, window and decay, and shrinks each team's defence toward
the LEAGUE MEAN defence instead of toward 0:

    penalty = ridge * (sum(att^2) + sum((def - mean(def))^2))

Nothing else changes. Goals-only target (the xG blend needs the Drive xG history, which is not
in the repo), analytic gradient, rho bounded to [-0.25, 0.25].

  python pipeline/v6/refit.py backtest [--leagues D1,E0]        walk-forward bias and log loss
  python pipeline/v6/refit.py fit D1 [--asof 2026-10-12] [--extra extra.json] [--out out.json]
        extra.json: [{"Date": "2026-10-09", "HomeTeam": ..., "AwayTeam": ..., "FTHG": n, "FTAG": n}]
        for results the football-data CSV does not carry yet.

Display and paper tracking only. Blend weight stays 0, so this never sets a price.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[2]
FD = ROOT / "data" / "fd"
LEAGUES = ["E0", "SP1", "D1", "F1"]
XI = 0.002
WINDOW = 1100
RIDGE = 5.0
MAXG = 10


def load(code: str, extra: list | None = None) -> pd.DataFrame:
    out = []
    for f in sorted(FD.glob(f"{code}_*.csv")):
        d = pd.read_csv(f).dropna(subset=["FTHG", "FTAG", "HomeTeam", "AwayTeam"])
        d["Date"] = pd.to_datetime(d.Date, dayfirst=True, errors="coerce", format="mixed")
        out.append(d[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]])
    df = pd.concat(out, ignore_index=True).dropna(subset=["Date"])
    if extra:
        e = pd.DataFrame(extra)
        e["Date"] = pd.to_datetime(e.Date)
        have = set(zip(df.Date, df.HomeTeam, df.AwayTeam))
        e = e[[(a, b, c) not in have for a, b, c in zip(e.Date, e.HomeTeam, e.AwayTeam)]]
        df = pd.concat([df, e[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]]], ignore_index=True)
    df[["FTHG", "FTAG"]] = df[["FTHG", "FTAG"]].astype(int)
    return df.sort_values("Date").reset_index(drop=True)


def _nll_grad(p, hi, ai, hg, ag, w, n, ridge, center):
    att, dfn = p[:n], p[n:2 * n]
    h, rho = p[2 * n], p[2 * n + 1]
    ll_ = h + att[hi] - dfn[ai]
    lm_ = att[ai] - dfn[hi]
    lam, mu = np.exp(ll_), np.exp(lm_)
    tau = np.ones_like(lam); t_l = np.zeros_like(lam); t_m = np.zeros_like(lam); t_r = np.zeros_like(lam)
    m00 = (hg == 0) & (ag == 0); m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0); m11 = (hg == 1) & (ag == 1)
    lm = lam * mu
    tau[m00] = 1 - lm[m00] * rho; t_l[m00] = -lm[m00] * rho; t_m[m00] = -lm[m00] * rho; t_r[m00] = -lm[m00]
    tau[m01] = 1 + lam[m01] * rho; t_l[m01] = lam[m01] * rho; t_r[m01] = lam[m01]
    tau[m10] = 1 + mu[m10] * rho; t_m[m10] = mu[m10] * rho; t_r[m10] = mu[m10]
    tau[m11] = 1 - rho; t_r[m11] = -1.0
    tau = np.clip(tau, 1e-9, None)
    f = -np.sum(w * (hg * ll_ - lam + ag * lm_ - mu + np.log(tau) - gammaln(hg + 1) - gammaln(ag + 1)))
    gl = -w * (hg - lam + t_l / tau); gm = -w * (ag - mu + t_m / tau)
    g_att = np.bincount(hi, gl, n) + np.bincount(ai, gm, n)
    g_def = -np.bincount(ai, gl, n) - np.bincount(hi, gm, n)
    g_h = gl.sum(); g_r = -np.sum(w * t_r / tau)
    ma = att.mean(); f += 1000.0 * ma ** 2; g_att += 2000.0 * ma / n
    dref = dfn - dfn.mean() if center else dfn
    f += ridge * (np.sum(att ** 2) + np.sum(dref ** 2))
    g_att += 2 * ridge * att; g_def += 2 * ridge * dref
    return f, np.concatenate([g_att, g_def, [g_h, g_r]])


def fit(df: pd.DataFrame, asof, xi: float = XI, ridge: float = RIDGE, center: bool = True,
        window: int = WINDOW) -> dict:
    asof = pd.Timestamp(asof)
    d = df[(df.Date < asof) & (df.Date >= asof - pd.Timedelta(days=window))]
    teams = sorted(set(d.HomeTeam) | set(d.AwayTeam)); n = len(teams)
    ix = {t: i for i, t in enumerate(teams)}
    hi = d.HomeTeam.map(ix).to_numpy(); ai = d.AwayTeam.map(ix).to_numpy()
    hg = d.FTHG.to_numpy(float); ag = d.FTAG.to_numpy(float)
    w = np.exp(-xi * (asof - d.Date).dt.days.to_numpy())
    x0 = np.zeros(2 * n + 2)
    x0[n:2 * n] = -0.3; x0[2 * n] = 0.25; x0[2 * n + 1] = -0.05
    bounds = [(None, None)] * (2 * n + 1) + [(-0.25, 0.25)]
    res = minimize(_nll_grad, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                   args=(hi, ai, hg, ag, w, n, ridge, center), options={"maxiter": 2000})
    x = res.x
    return {"teams": teams, "att": dict(zip(teams, x[:n].tolist())), "def": dict(zip(teams, x[n:2 * n].tolist())),
            "home_adv": float(x[2 * n]), "rho": float(x[2 * n + 1]), "n_matches": int(len(d)),
            "as_of": str(asof.date()), "xi": xi, "ridge": ridge, "center": center, "converged": bool(res.success)}


def level_factor(df: pd.DataFrame, r: dict, half_life: float = 90.0, max_days: int = 400) -> float:
    """Recent goal-level correction: time-weighted actual total goals / predicted total goals over the
    last `max_days` of the fit window. Multiply both lambdas by it. Walk-forward 2023-27: halves the
    remaining total-goals bias in all four leagues (La Liga -0.097 to -0.049) without touching ratings,
    so 1X2 is unchanged. The 1,100-day window lags a league whose goal rate is rising; this is the lag.
    Display and paper tracking only; not wired into any price (blend weight 0)."""
    asof = pd.Timestamp(r["as_of"])
    tr = df[(df.Date < asof) & (df.Date >= asof - pd.Timedelta(days=max_days))]
    if tr.empty:
        return 1.0
    age = (asof - tr.Date).dt.days.to_numpy()
    w = np.exp(-np.log(2) / half_life * age)
    pt = np.array([sum(lambdas(r, h, a)) for h, a in zip(tr.HomeTeam, tr.AwayTeam)])
    act = (tr.FTHG + tr.FTAG).to_numpy()
    return float((w * act).sum() / (w * pt).sum())


def lambdas(r: dict, home: str, away: str):
    a, dd = r["att"], r["def"]
    ma, md = float(np.mean(list(a.values()))), float(np.mean(list(dd.values())))
    ah, dh = a.get(home, ma), dd.get(home, md)
    aa, da = a.get(away, ma), dd.get(away, md)
    return float(np.exp(r["home_adv"] + ah - da)), float(np.exp(aa - dh))


def grid(lam, mu, rho):
    g = np.arange(MAXG + 1)
    from scipy.stats import poisson
    G = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    G[0, 0] *= 1 - lam * mu * rho; G[0, 1] *= 1 + lam * rho; G[1, 0] *= 1 + mu * rho; G[1, 1] *= 1 - rho
    return G / G.sum()


def probs(r, home, away):
    lam, mu = lambdas(r, home, away)
    G = grid(lam, mu, r["rho"])
    i, j = np.indices(G.shape)
    return {"lam": lam, "mu": mu, "H": float(G[i > j].sum()), "D": float(G[i == j].sum()), "A": float(G[i < j].sum()),
            "o25": float(G[(i + j) > 2].sum()), "btts": float(G[(i > 0) & (j > 0)].sum())}


def _ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def backtest(leagues, variants, start="2023-08-01"):
    rows = []
    for code in leagues:
        df = load(code)
        test = df[df.Date >= start]
        cuts = pd.date_range(start, df.Date.max() + pd.Timedelta(days=28), freq="28D")
        for a, b in zip(cuts[:-1], cuts[1:]):
            chunk = test[(test.Date >= a) & (test.Date < b)]
            if chunk.empty:
                continue
            for name, kw in variants.items():
                r = fit(df, a, **kw)
                for m in chunk.itertuples():
                    p = probs(r, m.HomeTeam, m.AwayTeam)
                    res = "H" if m.FTHG > m.FTAG else ("D" if m.FTHG == m.FTAG else "A")
                    rows.append({"lg": code, "v": name, "pred_tot": p["lam"] + p["mu"], "act_tot": m.FTHG + m.FTAG,
                                 "pred_home": p["lam"], "act_home": m.FTHG, "pred_away": p["mu"], "act_away": m.FTAG,
                                 "o25": p["o25"], "y_o25": int(m.FTHG + m.FTAG > 2), "btts": p["btts"],
                                 "y_btts": int(m.FTHG > 0 and m.FTAG > 0), "p_res": p[res]})
    bt = pd.DataFrame(rows)
    out = {}
    for (lg, v), g in bt.groupby(["lg", "v"]):
        out.setdefault(lg, {})[v] = {
            "n": int(len(g)),
            "bias_total": round(float((g.pred_tot - g.act_tot).mean()), 3),
            "bias_home": round(float((g.pred_home - g.act_home).mean()), 3),
            "bias_away": round(float((g.pred_away - g.act_away).mean()), 3),
            "ll_1x2": round(float(-np.log(g.p_res).mean()), 4),
            "ll_o25": round(float(_ll(g.o25, g.y_o25).mean()), 4),
            "ll_btts": round(float(_ll(g.btts, g.y_btts).mean()), 4)}
    for v, g in bt.groupby("v"):
        out.setdefault("ALL", {})[v] = {
            "n": int(len(g)), "bias_total": round(float((g.pred_tot - g.act_tot).mean()), 3),
            "ll_1x2": round(float(-np.log(g.p_res).mean()), 4),
            "ll_o25": round(float(_ll(g.o25, g.y_o25).mean()), 4),
            "ll_btts": round(float(_ll(g.btts, g.y_btts).mean()), 4)}
    return out


VARIANTS = {"old_ridge_to_0": dict(center=False, ridge=5.0), "centred_ridge5": dict(center=True, ridge=5.0),
            "centred_ridge2": dict(center=True, ridge=2.0), "centred_ridge10": dict(center=True, ridge=10.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["backtest", "fit"])
    ap.add_argument("league", nargs="?")
    ap.add_argument("--leagues", default=",".join(LEAGUES))
    ap.add_argument("--asof", default=None)
    ap.add_argument("--extra", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "backtest":
        print(json.dumps(backtest(a.leagues.split(","), VARIANTS), indent=1))
        return
    extra = json.loads(Path(a.extra).read_text()) if a.extra else None
    df = load(a.league, extra)
    asof = a.asof or str((df.Date.max() + pd.Timedelta(days=1)).date())
    r = fit(df, asof)
    r["goal_level_factor"] = round(level_factor(df, r), 4)
    txt = json.dumps(r, indent=1)
    Path(a.out).write_text(txt) if a.out else print(txt)


if __name__ == "__main__":
    main()
