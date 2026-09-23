"""
Match Engine v4 — Dixon-Coles fitter.
(15 Sep 2026, second revision: adds fit_joint / lambdas_joint for cups.
 Supersedes dc4.py — the first section is unchanged. Rename to dc4.py.)
(23 Sep 2026, v4.7: applied the two pending edits that were never merged when this
 was promoted from dc4.py.new — fit(..., rho_bounds=) from dc4_rho_bounds.patch.txt
 and lambdas(..., fallback=) from V4_AUDIT.md. price4.py and shots4.py call both and
 raised TypeError against the promoted file.)

Changes vs v3 dixon_coles.py:
  * analytic gradient (v3 used finite differences: ~2n+2 likelihood evals per
    step) -> a multi-season, week-by-week walk-forward is now tractable
  * warm start from the previous fit, keyed by team name
  * optional Gaussian prior per team (used for promoted "newcomer" teams, which
    v3 fit from a handful of matches with no shrinkage at all)
  * rho bounded to [-0.25, 0.25]
  * vectorised market probabilities for 1X2, totals at any half line, and
    Asian handicap at half lines
Same model otherwise: log(lam) = home + att[h] - def[a], log(mu) = att[a] - def[h].
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

MAXG = 10


def _nll_grad(p, hi, ai, hg, ag, w, n, prior_idx, prior_a, prior_d, lam_prior):
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

    ll = hg * ll_ - lam + ag * lm_ - mu + np.log(tau)
    f = -np.sum(w * ll)

    gl = -w * (hg - lam + t_l / tau)
    gm = -w * (ag - mu + t_m / tau)
    gr = -np.sum(w * t_r / tau)
    g_att = np.bincount(hi, gl, n) + np.bincount(ai, gm, n)
    g_def = -np.bincount(ai, gl, n) - np.bincount(hi, gm, n)
    g_h = gl.sum()

    # identifiability: mean(att) = 0
    ma = att.mean()
    f += 1000.0 * ma ** 2
    g_att += 2000.0 * ma / n
    # tiny global ridge for numerical stability
    f += 1e-3 * (np.sum(att ** 2) + np.sum(dfn ** 2))
    g_att += 2e-3 * att; g_def += 2e-3 * dfn
    # newcomer prior
    if lam_prior > 0 and len(prior_idx):
        da = att[prior_idx] - prior_a; dd = dfn[prior_idx] - prior_d
        f += lam_prior * (np.sum(da ** 2) + np.sum(dd ** 2))
        g_att[prior_idx] += 2 * lam_prior * da
        g_def[prior_idx] += 2 * lam_prior * dd

    return f, np.concatenate([g_att, g_def, [g_h, gr]])


def fit(df: pd.DataFrame, as_of, xi: float, warm: dict | None = None,
        priors: dict | None = None, lam_prior: float = 0.0, window_days: int = 1100,
        rho_bounds=(-0.25, 0.25)):
    """df needs Date, HomeTeam, AwayTeam, FTHG, FTAG. Uses only rows with Date < as_of.
    priors: {team: (att_mean, def_mean)}. Returns dict ratings."""
    as_of = pd.Timestamp(as_of)
    d = df[(df.Date < as_of) & (df.Date >= as_of - pd.Timedelta(days=window_days))]
    d = d.dropna(subset=["FTHG", "FTAG"])
    teams = sorted(set(d.HomeTeam) | set(d.AwayTeam))
    n = len(teams); idx = {t: i for i, t in enumerate(teams)}
    hi = d.HomeTeam.map(idx).to_numpy(); ai = d.AwayTeam.map(idx).to_numpy()
    hg = d.FTHG.to_numpy(float); ag = d.FTAG.to_numpy(float)
    w = np.exp(-xi * (as_of - d.Date).dt.days.to_numpy())

    x0 = np.zeros(2 * n + 2); x0[2 * n] = 0.25; x0[2 * n + 1] = float(np.clip(-0.05, *rho_bounds))
    if warm:
        for t, i in idx.items():
            if t in warm["att"]:
                x0[i] = warm["att"][t]; x0[n + i] = warm["def"][t]
        x0[2 * n] = warm["home"]; x0[2 * n + 1] = float(np.clip(warm["rho"], *rho_bounds))
    priors = priors or {}
    pt = [t for t in teams if t in priors]
    prior_idx = np.array([idx[t] for t in pt], dtype=int)
    prior_a = np.array([priors[t][0] for t in pt]); prior_d = np.array([priors[t][1] for t in pt])
    for t in pt:
        if not warm or t not in warm["att"]:
            x0[idx[t]], x0[n + idx[t]] = priors[t]

    bounds = [(None, None)] * (2 * n + 1) + [tuple(rho_bounds)]
    res = minimize(_nll_grad, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                   args=(hi, ai, hg, ag, w, n, prior_idx, prior_a, prior_d, lam_prior),
                   options={"maxiter": 1000})
    x = res.x
    return {"att": {t: float(x[idx[t]]) for t in teams},
            "def": {t: float(x[n + idx[t]]) for t in teams},
            "home": float(x[2 * n]), "rho": float(x[2 * n + 1]),
            "as_of": str(as_of.date()), "xi": xi, "n_matches": int(len(d)), "converged": bool(res.success)}


def lambdas(r, home, away, fallback=None):
    """Unseen clubs fall back to `fallback[team]` = (att, def) (the newcomer prior),
    else to the league-mean att/def. Previously they defaulted to (0, 0), which is not
    league average because defence is not mean-zero (V4_AUDIT.md fix a, applied 2026-09-23)."""
    a = r["att"]; d = r["def"]; fb = fallback or {}
    ma = float(np.mean(list(a.values()))) if a else 0.0
    md = float(np.mean(list(d.values()))) if d else 0.0
    ga = lambda t: a[t] if t in a else fb.get(t, (ma, md))[0]
    gd = lambda t: d[t] if t in d else fb.get(t, (ma, md))[1]
    lam = np.exp(r["home"] + np.array([ga(h) for h in home]) - np.array([gd(t) for t in away]))
    mu = np.exp(np.array([ga(t) for t in away]) - np.array([gd(h) for h in home]))
    return lam, mu


def grids(lam, mu, rho):
    g = np.arange(MAXG + 1)
    ph = poisson.pmf(g[None, :], lam[:, None]); pa = poisson.pmf(g[None, :], mu[:, None])
    G = ph[:, :, None] * pa[:, None, :]
    G[:, 0, 0] *= 1 - lam * mu * rho
    G[:, 0, 1] *= 1 + lam * rho
    G[:, 1, 0] *= 1 + mu * rho
    G[:, 1, 1] *= 1 - rho
    return G / G.sum(axis=(1, 2), keepdims=True)


_I, _J = np.indices((MAXG + 1, MAXG + 1))
_DIFF = _I - _J; _TOT = _I + _J


def probs_1x2(G):
    return np.stack([(G * (_DIFF > 0)).sum((1, 2)), (G * (_DIFF == 0)).sum((1, 2)),
                     (G * (_DIFF < 0)).sum((1, 2))], 1)


def p_over(G, line=2.5):
    return (G * (_TOT > line)).sum((1, 2))


def p_home_cover_half(G, hcap):
    """P(home covers) for half-point handicaps only (e.g. -0.5, +1.5). hcap = home line."""
    hcap = np.asarray(hcap)[:, None, None]
    return (G * ((_DIFF[None] + hcap) > 0)).sum((1, 2))


# ---------------------------------------------------------------------------
# Joint multi-competition fit (v4 cups work)
#   log(lam) = h[comp]*(1-neutral) + att[home] - def[away]
#   log(mu)  =                       att[away] - def[home]
# Teams in a prior group k are shrunk toward a LEARNED group mean (ga[k], gd[k]),
# e.g. "English non-league" or "UEFA club with no domestic data".
# ---------------------------------------------------------------------------
def _nll_grad_joint(p, hi, ai, ci, neu, hg, ag, w, n, C, G, grp, lam_grp,
                    gpar=None, H=0, lam_hyp=10.0):
    att, dfn = p[:n], p[n:2 * n]
    h = p[2 * n:2 * n + C]; rho = p[2 * n + C]
    o = 2 * n + C + 1
    ga = p[o:o + G]; gd = p[o + G:o + 2 * G]
    Ha = p[o + 2 * G:o + 2 * G + H]; Hd = p[o + 2 * G + H:]
    hv = h[ci] * (1 - neu)
    ll_ = hv + att[hi] - dfn[ai]; lm_ = att[ai] - dfn[hi]
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
    f = -np.sum(w * (hg * ll_ - lam + ag * lm_ - mu + np.log(tau)))
    gl = -w * (hg - lam + t_l / tau); gm = -w * (ag - mu + t_m / tau)
    g_att = np.bincount(hi, gl, n) + np.bincount(ai, gm, n)
    g_def = -np.bincount(ai, gl, n) - np.bincount(hi, gm, n)
    g_h = np.bincount(ci, gl * (1 - neu), C)
    g_r = -np.sum(w * t_r / tau)
    ma = att.mean(); f += 1000.0 * ma ** 2; g_att += 2000.0 * ma / n
    f += 1e-3 * (np.sum(att ** 2) + np.sum(dfn ** 2)); g_att += 2e-3 * att; g_def += 2e-3 * dfn
    g_ga = np.zeros(G); g_gd = np.zeros(G)
    m = grp >= 0
    if G and m.any():
        k = grp[m]
        da = att[m] - ga[k]; dd = dfn[m] - gd[k]
        f += lam_grp * (np.sum(da ** 2) + np.sum(dd ** 2))
        g_att[m] += 2 * lam_grp * da; g_def[m] += 2 * lam_grp * dd
        g_ga = -2 * lam_grp * np.bincount(k, da, G); g_gd = -2 * lam_grp * np.bincount(k, dd, G)
    g_Ha = np.zeros(H); g_Hd = np.zeros(H)
    if H and gpar is not None and (gpar >= 0).any():
        mk = gpar >= 0; kk = gpar[mk]
        ea = ga[mk] - Ha[kk]; ed = gd[mk] - Hd[kk]
        f += lam_hyp * (np.sum(ea ** 2) + np.sum(ed ** 2))
        g_ga[mk] += 2 * lam_hyp * ea; g_gd[mk] += 2 * lam_hyp * ed
        g_Ha = -2 * lam_hyp * np.bincount(kk, ea, H); g_Hd = -2 * lam_hyp * np.bincount(kk, ed, H)
    return f, np.concatenate([g_att, g_def, g_h, [g_r], g_ga, g_gd, g_Ha, g_Hd])


def fit_joint(df, as_of, xi, comps, groups=(), team_group=None, lam_grp=3.0,
              warm=None, window_days=1100, parents=None, lam_hyp=10.0):
    """df: Date, HomeTeam, AwayTeam, FTHG, FTAG, Comp, Neutral(bool).
    comps: list of competition codes (each gets its own home advantage).
    team_group: callable team -> group name in `groups` or None.
    parents: dict or callable group -> hyper name; groups with a parent have their
    mean shrunk toward a learned hyper-mean (every "unc:<country>" -> "uefa_uncovered").
    Groups returned by team_group but not listed in `groups` are added automatically."""
    as_of = pd.Timestamp(as_of)
    d = df[(df.Date < as_of) & (df.Date >= as_of - pd.Timedelta(days=window_days))]
    teams = sorted(set(d.HomeTeam) | set(d.AwayTeam))
    n = len(teams); idx = {t: i for i, t in enumerate(teams)}
    cidx = {c: i for i, c in enumerate(comps)}; C = len(comps)
    team_list_groups = [team_group(t) if team_group else None for t in teams]
    groups = list(groups) + sorted({g for g in team_list_groups if g and g not in groups})
    gidx = {g: i for i, g in enumerate(groups)}; G = len(groups)
    parent_of = parents if callable(parents) else (lambda g, _p=(parents or {}): _p.get(g))
    par = {g: parent_of(g) for g in groups}
    hypers = sorted({v for v in par.values() if v})
    hidx = {hname: i for i, hname in enumerate(hypers)}; H = len(hypers)
    gpar = np.array([hidx.get(par[g], -1) for g in groups], dtype=int)
    grp = np.array([gidx.get(g, -1) if g else -1 for g in team_list_groups])
    hi = d.HomeTeam.map(idx).to_numpy(); ai = d.AwayTeam.map(idx).to_numpy()
    ci = d.Comp.map(cidx).to_numpy(); neu = d.Neutral.to_numpy(float)
    hg = d.FTHG.to_numpy(float); ag = d.FTAG.to_numpy(float)
    w = np.exp(-xi * (as_of - d.Date).dt.days.to_numpy())
    o = 2 * n + C + 1
    x0 = np.zeros(o + 2 * G + 2 * H); x0[2 * n:2 * n + C] = 0.25; x0[2 * n + C] = -0.05
    x0[o:] = -0.3
    if warm:
        for t, i in idx.items():
            if t in warm["att"]:
                x0[i] = warm["att"][t]; x0[n + i] = warm["def"][t]
            elif grp[i] >= 0 and groups[grp[i]] in warm["group_att"]:
                x0[i] = warm["group_att"][groups[grp[i]]]; x0[n + i] = warm["group_def"][groups[grp[i]]]
        for c, i in cidx.items():
            x0[2 * n + i] = warm["home"].get(c, 0.25)
        x0[2 * n + C] = warm["rho"]
        for g, i in gidx.items():
            x0[o + i] = warm["group_att"].get(g, warm.get("hyper_att", {}).get(par[g], -0.3))
            x0[o + G + i] = warm["group_def"].get(g, warm.get("hyper_def", {}).get(par[g], -0.3))
        for hname, i in hidx.items():
            x0[o + 2 * G + i] = warm.get("hyper_att", {}).get(hname, -0.3)
            x0[o + 2 * G + H + i] = warm.get("hyper_def", {}).get(hname, -0.3)
    bounds = [(None, None)] * (2 * n + C) + [(-0.25, 0.25)] + [(None, None)] * (2 * G + 2 * H)
    res = minimize(_nll_grad_joint, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                   args=(hi, ai, ci, neu, hg, ag, w, n, C, G, grp, lam_grp, gpar, H, lam_hyp),
                   options={"maxiter": 3000})
    x = res.x
    neff = np.bincount(hi, w, n) + np.bincount(ai, w, n)
    return {"att": {t: float(x[idx[t]]) for t in teams}, "def": {t: float(x[n + idx[t]]) for t in teams},
            "n_eff": {t: float(neff[idx[t]]) for t in teams},
            "home": {c: float(x[2 * n + i]) for c, i in cidx.items()}, "rho": float(x[2 * n + C]),
            "group_att": {g: float(x[o + i]) for g, i in gidx.items()},
            "group_def": {g: float(x[o + G + i]) for g, i in gidx.items()},
            "hyper_att": {k: float(x[o + 2 * G + i]) for k, i in hidx.items()},
            "hyper_def": {k: float(x[o + 2 * G + H + i]) for k, i in hidx.items()},
            "parents": {g: v for g, v in par.items() if v},
            "as_of": str(as_of.date()), "n_matches": int(len(d)), "n_teams": n, "converged": bool(res.success)}


def lambdas_joint(r, home, away, comp, neutral, team_group=None):
    """Unseen team -> its group mean; unseen group -> the group's hyper-mean;
    otherwise league average (0)."""
    def get(t, which):
        v = r[which].get(t)
        if v is None and team_group is not None:
            g = team_group(t)
            if g:
                v = r["group_" + which].get(g)
                if v is None:
                    hyp = r.get("parents", {}).get(g) or (team_group.parent(g) if hasattr(team_group, "parent") else None)
                    v = r.get("hyper_" + which, {}).get(hyp)
        return 0.0 if v is None else v
    ha = np.array([r["home"].get(c, 0.0) for c in comp]) * (1 - np.asarray(neutral, float))
    lam = np.exp(ha + np.array([get(t, "att") for t in home]) - np.array([get(t, "def") for t in away]))
    mu = np.exp(np.array([get(t, "att") for t in away]) - np.array([get(t, "def") for t in home]))
    return lam, mu
