"""
Alpha Lines v8 "remaining gap" research - walk-forward (2026-11-15). Exploration only: nothing here prices a bet,
writes to the dashboard or touches a registered test.

Builds on fixes.py (same folds, same scoring: 28-day refits, scored 2023-08-01 onward against the closing price,
Shin de-vig, 95% intervals from resampling whole league x 28-day folds).

Stage 1, rating fits. Each variant is a mix of three targets inside one Dixon-Coles fit, plus a decay rate:
    goals    actual goals (Dixon-Coles likelihood, centred ridge 1)                       weight wg
    shots    the shot proxy a*SOT + b*(other shots), a, b >= 0 (nnls)                     weight ws
    market   the goals the closing market implied for that past match: (lam, mu) whose
             grid reproduces the Shin 1X2 and the fair over-2.5 (market.implied_lambdas)  weight wm
  xi is the per-day decay (0.002 = 347-day half-life, live; 0.006 = 116 days; 0.010 = 69 days).
  P is the promoted-team prior strength from fixes.py (0 = off, 5 = the Nov 11 pick).
  A past match with no closing price uses its shot proxy (or goals) in place of the market target.
  The closing price of a match is known before any later fit date, so there is no look-ahead.

Stage 2, adjustments to the neutral-ground strength difference d = log(lam) - log(mu) - home_adv, fitted
walk-forward on the previous 730 days of the same variant's own out-of-sample rows (pooled leagues):
    stretch  d' = s*d
    rest     d' = s*d + b*r          r = clip(home rest days, 2, 10) - clip(away rest days, 2, 10), league matches only
    elo      d' = s*d + g*e          e = (Club Elo home - Club Elo away)/400, latest snapshot before the match
                                     (needs --elo <csv with Club,Elo,date>; the free mirror stops 2026-01-14)
  log(lam) + log(mu) is kept fixed. Coefficients are 0 (s = 1) until 300 earlier rows exist.

Variants, weights, grids and windows were fixed before the first run.

  python pipeline/v8/gap.py [--elo path/to/elo4.csv]     writes out/gap_rows.csv (not committed) and out/gap_summary.md
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fixes as F              # noqa: E402
from fixes import C, R, M      # noqa: E402

OUT = HERE / "out"
LEAGUES = F.LEAGUES
VARIANTS = {   # name: (wg, ws, wm, xi, P)
    "v8_live":        (0.70, 0.30, 0.00, 0.002, 0.0),   # nnls off, exactly the live method
    "fix_nov11":      (0.70, 0.30, 0.00, 0.002, 5.0),   # nnls + prom5 (PR #8's pick, before stretch)
    "goals_x6":       (0.70, 0.30, 0.00, 0.006, 5.0),   # same, faster decay only
    "mkt_x2":         (0.00, 0.00, 1.00, 0.002, 0.0),
    "mkt_x6":         (0.00, 0.00, 1.00, 0.006, 0.0),
    "mkt_x10":        (0.00, 0.00, 1.00, 0.010, 0.0),
    "mkt_x6_prom5":   (0.00, 0.00, 1.00, 0.006, 5.0),
    "mix_x6_prom5":   (0.15, 0.15, 0.70, 0.006, 5.0),
}
VARIANTS2 = {  # round 2 (2026-11-15, after round 1 put the best decay at the edge of its grid): decay and mix only
    "v8_live":         (0.70, 0.30, 0.00, 0.002, 0.0),
    "mkt_x10_prom5":   (0.00, 0.00, 1.00, 0.010, 5.0),
    "mkt_x15_prom5":   (0.00, 0.00, 1.00, 0.015, 5.0),
    "mkt_x20_prom5":   (0.00, 0.00, 1.00, 0.020, 5.0),
    "mkt_x30_prom5":   (0.00, 0.00, 1.00, 0.030, 5.0),
    "mix_x10_prom5":   (0.15, 0.15, 0.70, 0.010, 5.0),
    "mix85_x10_prom5": (0.075, 0.075, 0.85, 0.010, 5.0),
}
ELO_ALIAS = {"Ath Bilbao": "Bilbao", "Ath Madrid": "Atletico", "Bayern Munich": "Bayern", "Ein Frankfurt": "Frankfurt",
             "Espanol": "Espanyol", "FC Koln": "Koeln", "Holstein Kiel": "Holstein", "La Coruna": "Depor",
             "M'gladbach": "Gladbach", "Nott'm Forest": "Forest", "Schalke 04": "Schalke", "St Etienne": "Saint-Etienne",
             "Vallecano": "Rayo Vallecano", "Werder Bremen": "Werder"}


# ---------------------------------------------------------------- data
def load(code: str) -> pd.DataFrame:
    """candidate.load plus the closing over/under 2.5 prices and the market-implied goals of every match."""
    df = C.load(code)
    ou = []
    for f in sorted(C.FD.glob(f"{code}_*.csv")):
        d = pd.read_csv(f).dropna(subset=["FTHG", "FTAG", "HomeTeam", "AwayTeam"])
        d["Date"] = pd.to_datetime(d.Date, dayfirst=True, errors="coerce", format="mixed")
        for c in ["PC>2.5", "PC<2.5", "AvgC>2.5", "AvgC<2.5"]:
            if c not in d:
                d[c] = np.nan
        ou.append(d[["Date", "HomeTeam", "AwayTeam", "PC>2.5", "PC<2.5", "AvgC>2.5", "AvgC<2.5"]])
    ou = pd.concat(ou, ignore_index=True).dropna(subset=["Date"]).drop_duplicates(["Date", "HomeTeam", "AwayTeam"])
    df = df.merge(ou, on=["Date", "HomeTeam", "AwayTeam"], how="left")
    ml, mm = [], []
    for m in df.itertuples(index=False):
        lam, mu = market_goals(m)
        ml.append(lam); mm.append(mu)
    df["mlam"], df["mmu"] = ml, mm
    return df


def _two_way(o, u):
    if np.isfinite(o) and np.isfinite(u) and min(o, u) > 1:
        return (1 / o) / (1 / o + 1 / u)
    return np.nan


def market_goals(m):
    q = None
    for c in ("PSC", "AvgC"):
        h, dd, a = (getattr(m, c + x, np.nan) for x in "HDA")
        if all(np.isfinite([h, dd, a])) and min(h, dd, a) > 1:
            q = M.shin([h, dd, a]); break
    if q is None:
        return np.nan, np.nan
    d = m._asdict()
    po = _two_way(d.get("PC>2.5", np.nan), d.get("PC<2.5", np.nan))
    if not np.isfinite(po):
        po = _two_way(d.get("AvgC>2.5", np.nan), d.get("AvgC<2.5", np.nan))
    return M.implied_lambdas(q[0], q[1], q[2], po if np.isfinite(po) else None)


# ---------------------------------------------------------------- fit
def fit(df, asof, wg=0.7, ws=0.3, wm=0.0, xi=R.XI, ridge=1.0, use_nnls=True, P=0.0, code=None, xg=False):
    """fixes.fit generalised: three targets (goals / shot proxy / market-implied goals) and a decay rate."""
    asof = pd.Timestamp(asof)
    d = df[(df.Date < asof) & (df.Date >= asof - pd.Timedelta(days=R.WINDOW))]
    teams = sorted(set(d.HomeTeam) | set(d.AwayTeam)); n = len(teams)
    ix = {t: i for i, t in enumerate(teams)}
    hi = d.HomeTeam.map(ix).to_numpy(); ai = d.AwayTeam.map(ix).to_numpy()
    hg = d.FTHG.to_numpy(float); ag = d.FTAG.to_numpy(float)
    w = np.exp(-xi * (asof - d.Date).dt.days.to_numpy())
    sot = np.concatenate([d.HST.to_numpy(float), d.AST.to_numpy(float)])
    off = np.clip(np.concatenate([d.HS.to_numpy(float), d.AS.to_numpy(float)]) - sot, 0, None)
    g = np.concatenate([hg, ag]); ok = np.isfinite(sot) & np.isfinite(off)
    X = np.column_stack([sot[ok], off[ok]])
    coef = nnls(X, g[ok])[0] if use_nnls else np.linalg.lstsq(X, g[ok], rcond=None)[0]
    xh = coef[0] * d.HST.to_numpy(float) + coef[1] * np.clip((d.HS - d.HST).to_numpy(float), 0, None)
    xa = coef[0] * d.AST.to_numpy(float) + coef[1] * np.clip((d.AS - d.AST).to_numpy(float), 0, None)
    miss = ~(np.isfinite(xh) & np.isfinite(xa)); xh[miss], xa[miss] = hg[miss], ag[miss]
    if xg and "xgh" in d:   # xg.py: real xG (Understat archive) replaces the shot proxy where the archive has the match
        gh, ga = d.xgh.to_numpy(float), d.xga.to_numpy(float); has = np.isfinite(gh) & np.isfinite(ga)
        xh, xa = xh.copy(), xa.copy(); xh[has], xa[has] = gh[has], ga[has]
    if wm > 0:
        mh = d.mlam.to_numpy(float).copy(); ma = d.mmu.to_numpy(float).copy()
        nm = ~(np.isfinite(mh) & np.isfinite(ma)); mh[nm], ma[nm] = xh[nm], xa[nm]
        n_mkt = int((~nm).sum())
    else:
        mh = ma = None; n_mkt = 0

    def make_obj(tgt_idx, tgt_a, tgt_d, tgt_w):
        def obj(p):
            f, gr = R._nll_grad(p, hi, ai, hg, ag, w * wg, n, ridge, True)   # ridge applies whatever wg is
            if ws > 0:
                f2, g2 = C._proxy_nll_grad(p, hi, ai, xh, xa, w * ws, n); f, gr = f + f2, gr + g2
            if wm > 0:
                f3, g3 = C._proxy_nll_grad(p, hi, ai, mh, ma, w * wm, n); f, gr = f + f3, gr + g3
            if len(tgt_idx):
                att, dfn = p[:n], p[n:2 * n]
                ea = att[tgt_idx] - tgt_a; ed = dfn[tgt_idx] - tgt_d
                f = f + float(np.sum(tgt_w * (ea ** 2 + ed ** 2)))
                gr = gr.copy(); gr[tgt_idx] += 2 * tgt_w * ea; gr[n + tgt_idx] += 2 * tgt_w * ed
            return f, gr
        return obj

    x0 = np.zeros(2 * n + 2); x0[n:2 * n] = -0.3; x0[2 * n] = 0.25; x0[2 * n + 1] = -0.05
    if wg > 0:
        bounds = [(None, None)] * (2 * n + 1) + [(-0.25, 0.25)]
    else:   # no goals term: rho is not identified, hold it at the market's fixed value
        x0[2 * n + 1] = M.RHO_MKT
        bounds = [(None, None)] * (2 * n + 1) + [(M.RHO_MKT, M.RHO_MKT)]
    e = np.array([], int)
    res = minimize(make_obj(e, 0, 0, 0), x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000})
    x = res.x
    prom_info = None
    if P > 0 and code:
        promo, releg = F.promoted_relegated(code, asof)
        rel = [ix[t] for t in releg if t in ix]
        if promo and rel:
            md = x[n:2 * n].mean()
            a_rel = float(x[:n][rel].mean()); d_rel_off = float(x[n:2 * n][rel].mean() - md)
            s = F.season_of(asof)
            cur = d[d.Date.map(F.season_of) == s]
            pidx, pw = [], []
            for t in promo:
                if t in ix:
                    k = int(((cur.HomeTeam == t) | (cur.AwayTeam == t)).sum())
                    pidx.append(ix[t]); pw.append(P * F.K_FADE / (F.K_FADE + k))
            if pidx:
                res = minimize(make_obj(np.array(pidx), a_rel, md + d_rel_off, np.array(pw)), x, jac=True,
                               method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000})
                x = res.x
            prom_info = {"promoted": sorted(promo), "a_rel": a_rel, "d_rel_off": d_rel_off}
    r = {"teams": teams, "att": dict(zip(teams, x[:n].tolist())), "def": dict(zip(teams, x[n:2 * n].tolist())),
         "home_adv": float(x[2 * n]), "rho": float(x[2 * n + 1]), "as_of": str(asof.date()),
         "shot_coef": [float(c) for c in coef], "converged": bool(res.success), "prom": prom_info,
         "n_matches": int(len(d)), "n_market": n_mkt}
    if prom_info:
        md = float(np.mean(list(r["def"].values())))
        for t in prom_info["promoted"]:
            if t not in r["att"]:
                r["att"][t] = prom_info["a_rel"]; r["def"][t] = md + prom_info["d_rel_off"]
    return r


# ---------------------------------------------------------------- extra features
def rest_days(df: pd.DataFrame) -> pd.DataFrame:
    last, rh, ra = {}, [], []
    for m in df.itertuples():
        rh.append((m.Date - last[m.HomeTeam]).days if m.HomeTeam in last else np.nan)
        ra.append((m.Date - last[m.AwayTeam]).days if m.AwayTeam in last else np.nan)
        last[m.HomeTeam] = last[m.AwayTeam] = m.Date
    out = df[["Date", "HomeTeam", "AwayTeam"]].copy()
    c = lambda v: np.clip(np.nan_to_num(np.array(v, float), nan=7.0), 2, 10)   # noqa: E731
    out["r"] = c(rh) - c(ra)
    return out


def elo_table(path):
    e = pd.read_csv(path, usecols=["Club", "Elo", "date"]).dropna()
    e["date"] = pd.to_datetime(e.date)
    return {k: (v.date.to_numpy(), v.Elo.to_numpy()) for k, v in e.sort_values("date").groupby("Club")}


def elo_at(tab, team, when, max_age=60):
    t = tab.get(ELO_ALIAS.get(team, team))
    if t is None:
        return np.nan
    i = np.searchsorted(t[0], np.datetime64(when), side="left") - 1     # latest snapshot strictly before the match day
    if i < 0 or (np.datetime64(when) - t[0][i]) / np.timedelta64(1, "D") > max_age:
        return np.nan
    return float(t[1][i])


# ---------------------------------------------------------------- stage 2: walk-forward adjustment of d
def adjust(lam, mu, ha, s, add):
    ll, lm = np.log(lam), np.log(mu)
    c = (ll + lm) / 2; dd = (ll - lm - ha) * s + add
    return np.exp(c + (ha + dd) / 2), np.exp(c - (ha + dd) / 2)


def fit_coef(h: pd.DataFrame, feat):
    """(s, coef) minimising 1X2 log loss on h; feat is a column name or None (stretch only)."""
    lam, mu, ha, rho, y = (h[c].to_numpy() for c in ("lam", "mu", "ha", "rho", "y"))
    x = h[feat].to_numpy() if feat else None

    def loss(p):
        add = p[1] * x if feat else 0.0
        l2, m2 = adjust(lam, mu, ha, p[0], add)
        pH, pD, pA, _, _ = F.probs_vec(l2, m2, rho)
        return F.ll1x2(pH, pD, pA, y).mean()

    p0 = [1.0, 0.0] if feat else [1.0]
    res = minimize(loss, p0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-7, "maxiter": 400})
    return (float(res.x[0]), float(res.x[1]) if feat else 0.0)


def stage2(v: pd.DataFrame, feat, tag):
    """Walk-forward: per fold, fit on the previous 730 days of v's own rows that have the feature."""
    v = v.copy(); v["s"], v["coef"] = 1.0, 0.0
    usable = v if not feat else v[np.isfinite(v[feat])]
    for f in sorted(v.fold.unique()):
        cut = pd.Timestamp(f)
        h = usable[(usable.Date < cut) & (usable.Date >= cut - pd.Timedelta(days=F.S_WINDOW_DAYS))]
        if len(h) >= F.S_MIN_ROWS:
            s, cf = fit_coef(h, feat)
            v.loc[v.fold == f, ["s", "coef"]] = s, cf
    x = np.nan_to_num(v[feat].to_numpy(), nan=0.0) if feat else 0.0
    if feat:   # rows without the feature get the stretch-only adjustment of their fold
        pass
    v["lam"], v["mu"] = adjust(v.lam.to_numpy(), v.mu.to_numpy(), v.ha.to_numpy(), v.s.to_numpy(), v.coef.to_numpy() * x)
    v["v"] = v.v + tag
    return v


# ---------------------------------------------------------------- backtest
def run(elo_path=None, rnd=1):
    global VARIANTS
    if rnd == 2:
        VARIANTS = VARIANTS2
    OUT.mkdir(parents=True, exist_ok=True)
    tab = elo_table(elo_path) if elo_path else None
    rows = []
    for code in LEAGUES:
        df = load(code)
        rd = rest_days(df).set_index(["Date", "HomeTeam", "AwayTeam"]).r
        cuts = pd.date_range(F.WARM_FROM, df.Date.max() + pd.Timedelta(days=28), freq="28D")
        for a, b in zip(cuts[:-1], cuts[1:]):
            chunk = df[(df.Date >= a) & (df.Date < b)]
            if chunk.empty:
                continue
            promo, _ = F.promoted_relegated(code, a)
            for name, (wg, ws, wm, xi, P) in VARIANTS.items():
                r = fit(df, a, wg=wg, ws=ws, wm=wm, xi=xi, use_nnls=(name != "v8_live"), P=P, code=code)
                k = R.level_factor(df, r)
                for m in chunk.itertuples():
                    lam, mu = R.lambdas(r, m.HomeTeam, m.AwayTeam)
                    q = F.market_close(m)
                    eh = elo_at(tab, m.HomeTeam, m.Date) if tab else np.nan
                    ea = elo_at(tab, m.AwayTeam, m.Date) if tab else np.nan
                    rows.append({"lg": code, "v": name, "fold": str(a.date()), "Date": m.Date, "home": m.HomeTeam,
                                 "away": m.AwayTeam, "lam": lam * k, "mu": mu * k, "rho": r["rho"], "ha": r["home_adv"],
                                 "y": 0 if m.FTHG > m.FTAG else (1 if m.FTHG == m.FTAG else 2),
                                 "tot": m.FTHG + m.FTAG, "btts_y": int(m.FTHG > 0 and m.FTAG > 0),
                                 "prom": bool({m.HomeTeam, m.AwayTeam} & promo),
                                 "qH": np.nan if q is None else q[0], "qD": np.nan if q is None else q[1],
                                 "qA": np.nan if q is None else q[2],
                                 "r": float(rd.get((m.Date, m.HomeTeam, m.AwayTeam), 0.0)),
                                 "e": (eh - ea) / 400.0})
            print(code, a.date(), flush=True)
    d = pd.DataFrame(rows)
    out = []
    for name in VARIANTS:
        v = d[d.v == name]
        out.append(v.assign(s=1.0, coef=0.0))
        out.append(stage2(v, None, "+stretch"))
        if rnd == 1:
            out.append(stage2(v, "r", "+rest"))
        if tab:
            out.append(stage2(v, "e", "+elo"))
    d = pd.concat(out, ignore_index=True)
    pH, pD, pA, po, pb = F.probs_vec(d.lam.to_numpy(), d.mu.to_numpy(), d.rho.to_numpy())
    d["pH"], d["pD"], d["pA"], d["po25"], d["pbtts"] = pH, pD, pA, po, pb
    d["ll"] = F.ll1x2(pH, pD, pA, d.y.to_numpy())
    d["ll_mkt"] = F.ll1x2(d.qH.to_numpy(), d.qD.to_numpy(), d.qA.to_numpy(), d.y.to_numpy())
    d = d[d.Date >= F.SCORE_FROM].reset_index(drop=True)
    d.to_csv(OUT / ("gap_rows.csv" if rnd == 1 else "gap_rows_round2.csv"), index=False)
    return d


# ---------------------------------------------------------------- summary
KEY = ["lg", "Date", "home", "away"]


def _cmp(mk, a, b, mask=None):
    """Mean per-match log-loss difference a - b with a fold-block 95% interval, on rows where both exist."""
    A = mk[mk.v == a].set_index(KEY); B = mk[mk.v == b].set_index(KEY)
    j = A.join(B[["ll"]], rsuffix="_b", how="inner")
    if mask is not None:
        j = j[mask(j)]
    j["dd"] = j.ll - j.ll_b
    if j.empty:
        return np.nan, np.nan, np.nan, 0
    m, lo, hi = F._boot(j.reset_index())
    return m, lo, hi, len(j)


def summarise(d: pd.DataFrame, has_elo: bool) -> str:
    mk = d.dropna(subset=["qH"])
    names = list(dict.fromkeys(mk.v))
    n = int((mk.v == "v8_live").sum())
    L = ["# v8 remaining-gap research, walk-forward (2023-08 to 2026-11)", "",
         f"{n:,} matches with a closing price; 28-day refits; lower log loss is better. Gap = model minus closing market "
         "(Shin). Change = variant minus the live v8 method; 95% interval from resampling whole 28-day folds.", "",
         "| Variant | 1X2 log loss | Gap to market | Change vs live | 95% | Promoted: gap | Heavy favs: market / model | "
         "Over 2.5 | BTTS | Goals bias | 2026-27 gap |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    ref = mk[mk.v == "v8_live"].set_index(KEY)
    fav = ref[np.maximum(ref.qH, ref.qA) >= 0.6].index
    for nm in names:
        if nm.endswith("+elo"):
            continue
        v = mk[mk.v == nm].set_index(KEY)
        m, lo, hi, _ = _cmp(mk, nm, "v8_live")
        pr = v[v.prom]; hv = v.loc[v.index.intersection(fav)]
        fm = np.where(hv.qH >= hv.qA, hv.pH, hv.pA).mean(); fq = np.maximum(hv.qH, hv.qA).mean()
        o25 = -np.mean(np.where(v.tot > 2, np.log(v.po25), np.log(1 - v.po25)))
        bt = -np.mean(np.where(v.btts_y == 1, np.log(v.pbtts), np.log(1 - v.pbtts)))
        cur = v[v.index.get_level_values("Date") >= pd.Timestamp("2026-07-01")]
        L.append(f"| {nm} | {v.ll.mean():.4f} | {(v.ll - v.ll_mkt).mean():+.4f} | {m:+.4f} | {lo:+.4f} to {hi:+.4f} | "
                 f"{(pr.ll - pr.ll_mkt).mean():+.4f} | {fq:.3f} / {fm:.3f} | {o25:.4f} | {bt:.4f} | "
                 f"{((v.lam + v.mu) - v.tot).mean():+.3f} | {(cur.ll - cur.ll_mkt).mean():+.4f} ({len(cur)}) |")
    L += ["", "## By league, gap to market", "", "| Variant | " + " | ".join(LEAGUES) + " |", "|---" * 5 + "|"]
    for nm in names:
        if "+" in nm and not nm.endswith("+stretch"):
            continue
        v = mk[mk.v == nm]
        L.append(f"| {nm} | " + " | ".join(f"{(v[v.lg == c].ll - v[v.lg == c].ll_mkt).mean():+.4f}" for c in LEAGUES) + " |")
    L += ["", "## Rest days (league matches only): +rest minus +stretch on the same ratings", "",
          "| Ratings | Change | 95% | Median fitted b |", "|---|---|---|---|"]
    for base in [b for b in VARIANTS if (mk.v == b + "+rest").any()]:
        m, lo, hi, _ = _cmp(mk, base + "+rest", base + "+stretch")
        b = d[d.v == base + "+rest"].coef.median()
        L.append(f"| {base} | {m:+.4f} | {lo:+.4f} to {hi:+.4f} | {b:+.4f} |")
    if has_elo:
        L += ["", "## Club Elo: +elo minus +stretch on the same ratings, matches where both teams have an Elo", "",
              "| Ratings | Matches | Change | 95% | Median fitted s | Median fitted g | Promoted-team matches: change |",
              "|---|---|---|---|---|---|---|"]
        has = lambda j: np.isfinite(j.e)        # noqa: E731
        for base in VARIANTS:
            m, lo, hi, k = _cmp(mk, base + "+elo", base + "+stretch", has)
            mp, _, _, kp = _cmp(mk, base + "+elo", base + "+stretch", lambda j: np.isfinite(j.e) & j.prom)
            x = d[(d.v == base + "+elo") & (d.coef != 0)]
            L.append(f"| {base} | {k:,} | {m:+.4f} | {lo:+.4f} to {hi:+.4f} | {x.s.median():.2f} | {x.coef.median():.2f} | "
                     f"{mp:+.4f} ({kp}) |")
    s = d[d.v.str.endswith("+stretch")].groupby("v").s.agg(["min", "median", "max"]).round(2)
    L += ["", "## Fitted stretch s (walk-forward)", "", s.to_markdown()]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--elo", default=None)
    ap.add_argument("--round", type=int, default=1, choices=[1, 2])
    a = ap.parse_args()
    d = run(a.elo, a.round)
    txt = summarise(d, bool(a.elo))
    (OUT / ("gap_summary.md" if a.round == 1 else "gap_summary_round2.md")).write_text(txt)
    print(txt)
