"""
Alpha Lines v8 fixes - walk-forward research (2026-11-11). Exploration only: nothing here prices a bet, writes to the
dashboard or touches a registered test (docs/prereg-forward-2026-11.md is unchanged).

Three fixes to the frozen v7/v8 rating method (pipeline/v7/candidate.py, CHOSEN = shots 30%, ridge 1, goal level),
from the 2026-11-11 weak-spots review:

  nnls     The shot proxy goals ~ a*SOT + b*(other shots) is fitted with a, b >= 0 (scipy nnls). The live fit had
           b = -0.015 on Nov 3: a negative weight on a shot is noise.
  prom     Newly promoted teams are pulled toward the level of the teams relegated the season before, instead of
           toward the league average. Promoted = in this season's league file but not last season's; relegated =
           the reverse. Two-stage: fit as usual, read the relegated teams' mean attack and mean defence (relative
           to the league mean), then refit with an extra penalty
               P * k/(k + n) * ((att - a_rel)^2 + (def - d_rel)^2)
           on each promoted team, n = its league matches this season before the fit date, so the pull fades as the
           team's own results arrive. A promoted team with no match yet in the window gets (a_rel, d_rel) directly.
           Relegated-team ratings are known at the fit date, so there is no look-ahead.
  stretch  Favourites rated too close to their opponents. After the fit, the neutral-ground strength difference
           d = log(lam) - log(mu) - home_adv is multiplied by s, keeping log(lam) + log(mu) fixed. s is fitted
           walk-forward: at each 28-day cut, the s in 0.80-1.50 that minimises 1X2 log loss over the previous 730
           days of this same backtest's out-of-sample predictions (all four leagues pooled). s = 1 until 300 such
           matches exist.

Variants and every constant (P = 5 and 20, k = 10, the s grid and window) were fixed before the first run.
Scored on 2023-08-01 onward against the closing price (Pinnacle where football-data has it, else the market
average; Shin de-vig). Folds before 2023-08 are run only to warm up the stretch fit and are not scored.

  python pipeline/v8/fixes.py            writes pipeline/v8/out/fixes_rows.csv and fixes_summary.md
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline" / "v7"))
sys.path.insert(0, str(ROOT / "pipeline" / "v6"))
sys.path.insert(0, str(ROOT / "pipeline"))
import candidate as C          # noqa: E402
import refit as R              # noqa: E402
from v5 import market as M     # noqa: E402

FD = ROOT / "data" / "fd"
OUT = Path(__file__).resolve().parent / "out"
LEAGUES = ["E0", "SP1", "D1", "F1"]
SCORE_FROM = pd.Timestamp("2023-08-01")
WARM_FROM = pd.Timestamp("2022-08-01")
K_FADE = 10.0
S_GRID = np.round(np.arange(0.80, 1.501, 0.01), 2)
S_WINDOW_DAYS, S_MIN_ROWS = 730, 300
VARIANTS = {   # name: (nnls, P)
    "v8_live": (False, 0.0),
    "nnls": (True, 0.0),
    "nnls_prom5": (True, 5.0),
    "nnls_prom20": (True, 20.0),
}
G = np.arange(R.MAXG + 1)


# ---------------------------------------------------------------- seasons
def season_of(t: pd.Timestamp) -> str:
    y = t.year % 100 if t.month >= 7 else t.year % 100 - 1
    return f"{y:02d}{y + 1:02d}"


def prev_season(s: str) -> str:
    y = int(s[:2]) - 1
    return f"{y:02d}{y + 1:02d}"


_TEAMS: dict = {}


def season_teams(code: str, s: str) -> set:
    key = (code, s)
    if key not in _TEAMS:
        p = FD / f"{code}_{s}.csv"
        _TEAMS[key] = set(pd.read_csv(p).HomeTeam.dropna()) if p.exists() else set()
    return _TEAMS[key]


def promoted_relegated(code: str, asof: pd.Timestamp):
    s = season_of(asof)
    cur, prev = season_teams(code, s), season_teams(code, prev_season(s))
    if not cur or not prev:
        return set(), set()
    return cur - prev, prev - cur


# ---------------------------------------------------------------- fit (candidate.fit plus nnls and the promoted prior)
def fit(df, asof, ridge=1.0, shots=0.3, use_nnls=False, P=0.0, code=None):
    asof = pd.Timestamp(asof)
    d = df[(df.Date < asof) & (df.Date >= asof - pd.Timedelta(days=R.WINDOW))]
    teams = sorted(set(d.HomeTeam) | set(d.AwayTeam)); n = len(teams)
    ix = {t: i for i, t in enumerate(teams)}
    hi = d.HomeTeam.map(ix).to_numpy(); ai = d.AwayTeam.map(ix).to_numpy()
    hg = d.FTHG.to_numpy(float); ag = d.FTAG.to_numpy(float)
    w = np.exp(-R.XI * (asof - d.Date).dt.days.to_numpy())
    sot = np.concatenate([d.HST.to_numpy(float), d.AST.to_numpy(float)])
    off = np.clip(np.concatenate([d.HS.to_numpy(float), d.AS.to_numpy(float)]) - sot, 0, None)
    g = np.concatenate([hg, ag]); ok = np.isfinite(sot) & np.isfinite(off)
    X = np.column_stack([sot[ok], off[ok]])
    coef = nnls(X, g[ok])[0] if use_nnls else np.linalg.lstsq(X, g[ok], rcond=None)[0]
    xh = coef[0] * d.HST.to_numpy(float) + coef[1] * np.clip((d.HS - d.HST).to_numpy(float), 0, None)
    xa = coef[0] * d.AST.to_numpy(float) + coef[1] * np.clip((d.AS - d.AST).to_numpy(float), 0, None)
    miss = ~(np.isfinite(xh) & np.isfinite(xa)); xh[miss], xa[miss] = hg[miss], ag[miss]
    wg = 1.0 - shots

    def make_obj(tgt_idx, tgt_a, tgt_d, tgt_w):
        def obj(p):
            f, gr = R._nll_grad(p, hi, ai, hg, ag, w * wg, n, ridge, True)
            f2, g2 = C._proxy_nll_grad(p, hi, ai, xh, xa, w * shots, n)
            f, gr = f + f2, gr + g2
            if len(tgt_idx):
                att, dfn = p[:n], p[n:2 * n]
                ea = att[tgt_idx] - tgt_a; ed = dfn[tgt_idx] - tgt_d
                f = f + float(np.sum(tgt_w * (ea ** 2 + ed ** 2)))
                gr = gr.copy(); gr[tgt_idx] += 2 * tgt_w * ea; gr[n + tgt_idx] += 2 * tgt_w * ed
            return f, gr
        return obj

    x0 = np.zeros(2 * n + 2); x0[n:2 * n] = -0.3; x0[2 * n] = 0.25; x0[2 * n + 1] = -0.05
    bounds = [(None, None)] * (2 * n + 1) + [(-0.25, 0.25)]
    e = np.array([], int)
    res = minimize(make_obj(e, 0, 0, 0), x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000})
    x = res.x
    prom_info = None
    if P > 0 and code:
        promo, releg = promoted_relegated(code, asof)
        rel = [ix[t] for t in releg if t in ix]
        if promo and rel:
            md = x[n:2 * n].mean()
            a_rel = float(x[:n][rel].mean()); d_rel_off = float(x[n:2 * n][rel].mean() - md)
            s = season_of(asof)
            cur = d[d.Date.map(season_of) == s]
            pidx, pw = [], []
            for t in promo:
                if t in ix:
                    nm = int(((cur.HomeTeam == t) | (cur.AwayTeam == t)).sum())
                    pidx.append(ix[t]); pw.append(P * K_FADE / (K_FADE + nm))
            if pidx:
                res = minimize(make_obj(np.array(pidx), a_rel, md + d_rel_off, np.array(pw)), x, jac=True,
                               method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000})
                x = res.x
            prom_info = {"promoted": sorted(promo), "a_rel": a_rel, "d_rel_off": d_rel_off}
    r = {"teams": teams, "att": dict(zip(teams, x[:n].tolist())), "def": dict(zip(teams, x[n:2 * n].tolist())),
         "home_adv": float(x[2 * n]), "rho": float(x[2 * n + 1]), "as_of": str(asof.date()),
         "shot_coef": [float(c) for c in coef], "converged": bool(res.success), "prom": prom_info}
    if prom_info:   # promoted team not in the window yet: give it the relegated-team level
        md = float(np.mean(list(r["def"].values())))
        for t in prom_info["promoted"]:
            if t not in r["att"]:
                r["att"][t] = prom_info["a_rel"]; r["def"][t] = md + prom_info["d_rel_off"]
    return r


# ---------------------------------------------------------------- grids, vectorised over matches
def probs_vec(lam, mu, rho):
    """1X2, over 2.5 and BTTS for arrays of (lam, mu, rho), Dixon-Coles with the rho clamp of src/guards."""
    lam, mu, rho = map(np.asarray, (lam, mu, rho))
    lo = np.maximum(-1 / lam, -1 / mu); hi = np.minimum(1 / (lam * mu), 1.0)
    rho = np.clip(rho, lo, hi)
    ph = poisson.pmf(G[None, :], lam[:, None]); pa = poisson.pmf(G[None, :], mu[:, None])
    Gr = ph[:, :, None] * pa[:, None, :]
    Gr[:, 0, 0] *= 1 - lam * mu * rho; Gr[:, 0, 1] *= 1 + lam * rho; Gr[:, 1, 0] *= 1 + mu * rho; Gr[:, 1, 1] *= 1 - rho
    Gr /= Gr.sum(axis=(1, 2), keepdims=True)
    I, J = np.indices((len(G), len(G)))
    return (Gr[:, I > J].sum(1), Gr[:, I == J].sum(1), Gr[:, I < J].sum(1),
            Gr[:, (I + J) > 2].sum(1), Gr[:, 1:, 1:].sum((1, 2)))


def stretch(lam, mu, ha, s):
    ll, lm = np.log(lam), np.log(mu)
    c = (ll + lm) / 2; dd = (ll - lm - ha) * s
    return np.exp(c + (ha + dd) / 2), np.exp(c - (ha + dd) / 2)


def ll1x2(pH, pD, pA, y):
    p = np.where(y == 0, pH, np.where(y == 1, pD, pA))
    return -np.log(np.clip(p, 1e-9, 1))


def fit_s(rows: pd.DataFrame, cut: pd.Timestamp) -> float:
    h = rows[(rows.Date < cut) & (rows.Date >= cut - pd.Timedelta(days=S_WINDOW_DAYS))]
    if len(h) < S_MIN_ROWS:
        return 1.0
    best, bs = None, 1.0
    for s in S_GRID:
        lam, mu = stretch(h.lam.to_numpy(), h.mu.to_numpy(), h.ha.to_numpy(), s)
        pH, pD, pA, _, _ = probs_vec(lam, mu, h.rho.to_numpy())
        v = ll1x2(pH, pD, pA, h.y.to_numpy()).mean()
        if best is None or v < best:
            best, bs = v, float(s)
    return bs


# ---------------------------------------------------------------- backtest
def market_close(m):
    for c in ("PSC", "AvgC"):
        h, dd, a = (getattr(m, c + x, np.nan) for x in "HDA")
        if all(np.isfinite([h, dd, a])) and min(h, dd, a) > 1:
            return M.shin([h, dd, a])
    return None


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for code in LEAGUES:
        df = C.load(code)
        cuts = pd.date_range(WARM_FROM, df.Date.max() + pd.Timedelta(days=28), freq="28D")
        for a, b in zip(cuts[:-1], cuts[1:]):
            chunk = df[(df.Date >= a) & (df.Date < b)]
            if chunk.empty:
                continue
            promo, _ = promoted_relegated(code, a)
            for name, (use_nnls, P) in VARIANTS.items():
                r = fit(df, a, use_nnls=use_nnls, P=P, code=code)
                k = R.level_factor(df, r)
                for m in chunk.itertuples():
                    lam, mu = R.lambdas(r, m.HomeTeam, m.AwayTeam)
                    q = market_close(m)
                    rows.append({"lg": code, "v": name, "fold": str(a.date()), "Date": m.Date, "home": m.HomeTeam,
                                 "away": m.AwayTeam, "lam": lam * k, "mu": mu * k, "rho": r["rho"], "ha": r["home_adv"],
                                 "y": 0 if m.FTHG > m.FTAG else (1 if m.FTHG == m.FTAG else 2),
                                 "tot": m.FTHG + m.FTAG, "btts_y": int(m.FTHG > 0 and m.FTAG > 0),
                                 "prom": bool({m.HomeTeam, m.AwayTeam} & promo),
                                 "qH": np.nan if q is None else q[0], "qD": np.nan if q is None else q[1],
                                 "qA": np.nan if q is None else q[2],
                                 "coef_off": r["shot_coef"][1]})
            print(code, a.date(), flush=True)
    d = pd.DataFrame(rows)
    # walk-forward stretch per variant (pooled leagues), fitted on that variant's own earlier out-of-sample rows
    out = []
    for name in VARIANTS:
        v = d[d.v == name].copy()
        v["s"] = 1.0
        for f in sorted(v.fold.unique()):
            v.loc[v.fold == f, "s"] = fit_s(v, pd.Timestamp(f))
        base = v.copy(); out.append(base)
        st = v.copy(); st["v"] = name + "+stretch"
        st["lam"], st["mu"] = stretch(v.lam.to_numpy(), v.mu.to_numpy(), v.ha.to_numpy(), v.s.to_numpy())
        out.append(st)
    d = pd.concat(out, ignore_index=True)
    pH, pD, pA, po, pb = probs_vec(d.lam.to_numpy(), d.mu.to_numpy(), d.rho.to_numpy())
    d["pH"], d["pD"], d["pA"], d["po25"], d["pbtts"] = pH, pD, pA, po, pb
    d["ll"] = ll1x2(pH, pD, pA, d.y.to_numpy())
    d["ll_mkt"] = ll1x2(d.qH.to_numpy(), d.qD.to_numpy(), d.qA.to_numpy(), d.y.to_numpy())
    d = d[d.Date >= SCORE_FROM].reset_index(drop=True)
    d.to_csv(OUT / "fixes_rows.csv", index=False)
    return d


# ---------------------------------------------------------------- summary
def _boot(diff_by_fold: pd.DataFrame, n=4000, seed=7):
    """Mean per-match difference with a 95% interval, resampling whole 28-day folds (league x fold blocks)."""
    g = diff_by_fold.groupby(["lg", "fold"]).agg(s=("dd", "sum"), c=("dd", "size")).reset_index()
    rng = np.random.default_rng(seed); S, Cn = g.s.to_numpy(), g.c.to_numpy()
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(g), len(g)); bs.append(S[i].sum() / Cn[i].sum())
    return S.sum() / Cn.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def summarise(d: pd.DataFrame) -> str:
    mk = d.dropna(subset=["qH"])
    ref = mk[mk.v == "v8_live"].set_index(["lg", "Date", "home", "away"])
    names = list(dict.fromkeys(mk.v))
    L = ["# v8 fixes, walk-forward (2023-08 to 2026-11)", "",
         f"{ref.shape[0]:,} matches with a closing price; 28-day refits; lower log loss is better. Gap = model minus closing "
         "market (Shin). Change = variant minus the live v8 method, with a 95% interval from resampling whole 28-day folds.", "",
         "| Variant | 1X2 log loss | Gap to market | Change vs live | 95% | Promoted-team matches: gap | Others: gap | "
         "Heavy favs: market / model | Over 2.5 | BTTS | Goals bias |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    fav = ref[np.maximum(ref.qH, ref.qA) >= 0.6].index
    for nm in names:
        v = mk[mk.v == nm].set_index(["lg", "Date", "home", "away"])
        j = v.join(ref[["ll"]], rsuffix="_ref"); j["dd"] = j.ll - j.ll_ref
        m, lo, hi = _boot(j.reset_index())
        pr = v[v.prom]; npr = v[~v.prom]
        hv = v.loc[v.index.intersection(fav)]
        fm = np.where(hv.qH >= hv.qA, hv.pH, hv.pA).mean(); fq = np.maximum(hv.qH, hv.qA).mean()
        o25 = -np.mean(np.where(v.tot > 2, np.log(v.po25), np.log(1 - v.po25)))
        bt = -np.mean(np.where(v.btts_y == 1, np.log(v.pbtts), np.log(1 - v.pbtts)))
        bias = ((v.lam + v.mu) - v.tot).mean()
        L.append(f"| {nm} | {v.ll.mean():.4f} | {(v.ll - v.ll_mkt).mean():+.4f} | {m:+.4f} | {lo:+.4f} to {hi:+.4f} | "
                 f"{(pr.ll - pr.ll_mkt).mean():+.4f} ({len(pr)}) | {(npr.ll - npr.ll_mkt).mean():+.4f} | "
                 f"{fq:.3f} / {fm:.3f} | {o25:.4f} | {bt:.4f} | {bias:+.3f} |")
    L += ["", "## By league, gap to market", "", "| Variant | " + " | ".join(LEAGUES) + " |", "|---" * 5 + "|"]
    for nm in names:
        v = mk[mk.v == nm]
        L.append(f"| {nm} | " + " | ".join(f"{(v[v.lg == c].ll - v[v.lg == c].ll_mkt).mean():+.4f}" for c in LEAGUES) + " |")
    s = d[d.v.str.endswith("+stretch")].groupby("v").s.agg(["min", "median", "max"]).round(2)
    L += ["", "## Fitted stretch s (walk-forward)", "", s.to_markdown() if hasattr(s, "to_markdown") else s.to_string()]
    co = d[d.v == "v8_live"].coef_off
    L += ["", f"Live-method off-target shot coefficient across folds: min {co.min():+.4f}, median {co.median():+.4f}, "
          f"share negative {float((co < 0).mean()):.0%}."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    d = run()
    txt = summarise(d)
    (OUT / "fixes_summary.md").write_text(txt)
    print(txt)
