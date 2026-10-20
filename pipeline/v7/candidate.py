"""
Alpha Lines v7 candidate - shadow research only (2026-10-20). Nothing here prices a bet or touches the dashboard.

Question: can the Dixon-Coles ratings stop under-rating favourites (week 7: market favourites >55% averaged 65.5%,
the model 55.9%) without losing what the centred ridge gained on goal level?

Ideas tested walk-forward (28-day refits, 2023-08 onward, same protocol as pipeline/v6/refit.py backtest):
  ridge      lighter ridge (less pull of every team toward average)
  shots      fit the ratings on a blend of goals and a shot-based expected-goals proxy. football-data has shots and
             shots on target for every match in all four leagues, so this is the free xG stand-in. Inside each
             fit window the proxy is  a * shots_on_target + b * (shots - shots_on_target), with a, b from a
             no-intercept least-squares fit of goals on those two counts in that same window (no look-ahead).
  level      v6.1's level_factor (recent actual/predicted goal ratio) applied to both lambdas

Scored on every test match against the closing market (football-data average closing 1X2, proportional de-vig):
1X2 log loss, model minus market, over 2.5 and BTTS log loss, total-goals bias, and "fav_gap": mean of model
minus market probability on the market favourite when the market gives it more than 55%.

  python pipeline/v7/candidate.py backtest [--round 1|2] [--leagues E0,SP1,D1,F1] [--out pipeline/v7/out/backtest.json]
  python pipeline/v7/candidate.py shadow --db <ArtifactData out_dir> --asof YYYY-MM-DD
      fits CHOSEN as of that date (football-data CSVs plus newer dashboard results) and writes
      out/shadow_<date>.csv (unplayed fixtures) and out/fit_<date>.json. Commit before kickoff.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline" / "v6"))
import refit as R  # noqa: E402  (v6.1 code: DC likelihood, grid, level_factor)

FD = ROOT / "data" / "fd"
LEAGUES = ["E0", "SP1", "D1", "F1"]


def load(code: str) -> pd.DataFrame:
    out = []
    for f in sorted(FD.glob(f"{code}_*.csv")):
        d = pd.read_csv(f).dropna(subset=["FTHG", "FTAG", "HomeTeam", "AwayTeam"])
        d["Date"] = pd.to_datetime(d.Date, dayfirst=True, errors="coerce", format="mixed")
        for c in ["HS", "AS", "HST", "AST", "AvgCH", "AvgCD", "AvgCA", "PSCH", "PSCD", "PSCA"]:
            if c not in d:
                d[c] = np.nan
        out.append(d[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST",
                      "AvgCH", "AvgCD", "AvgCA", "PSCH", "PSCD", "PSCA"]])
    df = pd.concat(out, ignore_index=True).dropna(subset=["Date"])
    df[["FTHG", "FTAG"]] = df[["FTHG", "FTAG"]].astype(int)
    return df.sort_values("Date").reset_index(drop=True)


def _proxy_nll_grad(p, hi, ai, xh, xa, w, n):
    """Plain Poisson (no Dixon-Coles low-score term) quasi-likelihood of the shot proxy, no ridge."""
    att, dfn, h = p[:n], p[n:2 * n], p[2 * n]
    ll_, lm_ = h + att[hi] - dfn[ai], att[ai] - dfn[hi]
    lam, mu = np.exp(ll_), np.exp(lm_)
    f = -np.sum(w * (xh * ll_ - lam + xa * lm_ - mu))
    gl, gm = -w * (xh - lam), -w * (xa - mu)
    g_att = np.bincount(hi, gl, n) + np.bincount(ai, gm, n)
    g_def = -np.bincount(ai, gl, n) - np.bincount(hi, gm, n)
    return f, np.concatenate([g_att, g_def, [gl.sum(), 0.0]])


def fit(df: pd.DataFrame, asof, ridge=5.0, shots=0.0, xi=R.XI, window=R.WINDOW) -> dict:
    asof = pd.Timestamp(asof)
    d = df[(df.Date < asof) & (df.Date >= asof - pd.Timedelta(days=window))]
    teams = sorted(set(d.HomeTeam) | set(d.AwayTeam)); n = len(teams)
    ix = {t: i for i, t in enumerate(teams)}
    hi = d.HomeTeam.map(ix).to_numpy(); ai = d.AwayTeam.map(ix).to_numpy()
    hg = d.FTHG.to_numpy(float); ag = d.FTAG.to_numpy(float)
    w = np.exp(-xi * (asof - d.Date).dt.days.to_numpy())
    coef = None
    if shots > 0:
        sot = np.concatenate([d.HST.to_numpy(float), d.AST.to_numpy(float)])
        off = np.concatenate([d.HS.to_numpy(float), d.AS.to_numpy(float)]) - sot
        g = np.concatenate([hg, ag]); ok = np.isfinite(sot) & np.isfinite(off)
        X = np.column_stack([sot[ok], np.clip(off[ok], 0, None)])
        coef = np.linalg.lstsq(X, g[ok], rcond=None)[0]
        xh = coef[0] * d.HST.to_numpy(float) + coef[1] * np.clip((d.HS - d.HST).to_numpy(float), 0, None)
        xa = coef[0] * d.AST.to_numpy(float) + coef[1] * np.clip((d.AS - d.AST).to_numpy(float), 0, None)
        miss = ~(np.isfinite(xh) & np.isfinite(xa))
        xh[miss], xa[miss] = hg[miss], ag[miss]          # no shots row: fall back to goals
    wg = 1.0 - shots

    def obj(p):
        f, g = R._nll_grad(p, hi, ai, hg, ag, w * wg, n, ridge, True)
        if shots > 0:
            f2, g2 = _proxy_nll_grad(p, hi, ai, xh, xa, w * shots, n)
            f, g = f + f2, g + g2
        return f, g
    x0 = np.zeros(2 * n + 2); x0[n:2 * n] = -0.3; x0[2 * n] = 0.25; x0[2 * n + 1] = -0.05
    bounds = [(None, None)] * (2 * n + 1) + [(-0.25, 0.25)]
    res = minimize(obj, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000})
    x = res.x
    return {"teams": teams, "att": dict(zip(teams, x[:n].tolist())), "def": dict(zip(teams, x[n:2 * n].tolist())),
            "home_adv": float(x[2 * n]), "rho": float(x[2 * n + 1]), "as_of": str(asof.date()),
            "ridge": ridge, "shots": shots, "shot_coef": None if coef is None else [float(c) for c in coef],
            "converged": bool(res.success)}


VARIANTS = {
    "v61_base":      dict(ridge=5.0, shots=0.0, level=False),
    "v61_base_lvl":  dict(ridge=5.0, shots=0.0, level=True),
    "ridge2":        dict(ridge=2.0, shots=0.0, level=False),
    "shots50":       dict(ridge=5.0, shots=0.5, level=False),
    "shots50_r2":    dict(ridge=2.0, shots=0.5, level=False),
    "shots70_r2":    dict(ridge=2.0, shots=0.7, level=False),
    "shots50_r2_lvl": dict(ridge=2.0, shots=0.5, level=True),
}
VARIANTS_R2 = {   # round 2 (same day): push the ridge lighter, keep the goal-level factor
    "v61_base":       dict(ridge=5.0, shots=0.0, level=False),
    "ridge1_lvl":     dict(ridge=1.0, shots=0.0, level=True),
    "ridge0.5_lvl":   dict(ridge=0.5, shots=0.0, level=True),
    "shots30_r1_lvl": dict(ridge=1.0, shots=0.3, level=True),
    "shots50_r1_lvl": dict(ridge=1.0, shots=0.5, level=True),
    "shots50_r0.5_lvl": dict(ridge=0.5, shots=0.5, level=True),
}


def _market(m):
    for c in ("AvgC", "PSC"):
        h, dd, a = getattr(m, c + "H"), getattr(m, c + "D"), getattr(m, c + "A")
        if all(np.isfinite([h, dd, a])) and min(h, dd, a) > 1:
            q = np.array([1 / h, 1 / dd, 1 / a]); return q / q.sum()
    return None


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
                kw = dict(kw); lvl = kw.pop("level")
                r = fit(df, a, **kw)
                k = R.level_factor(df, r) if lvl else 1.0
                for m in chunk.itertuples():
                    lam, mu = R.lambdas(r, m.HomeTeam, m.AwayTeam)
                    lam, mu = lam * k, mu * k
                    G = R.grid(lam, mu, r["rho"]); i, j = np.indices(G.shape)
                    p = np.array([G[i > j].sum(), G[i == j].sum(), G[i < j].sum()])
                    y = 0 if m.FTHG > m.FTAG else (1 if m.FTHG == m.FTAG else 2)
                    q = _market(m)
                    fav_gap = None
                    if q is not None and q.max() > 0.55 and q.argmax() != 1:
                        fav_gap = float(p[q.argmax()] - q.max())
                    rows.append({"lg": code, "v": name, "p_res": float(p[y]),
                                 "q_res": None if q is None else float(q[y]),
                                 "pred_tot": lam + mu, "act_tot": m.FTHG + m.FTAG,
                                 "o25": float(G[(i + j) > 2].sum()), "y_o25": int(m.FTHG + m.FTAG > 2),
                                 "btts": float(G[(i > 0) & (j > 0)].sum()), "y_btts": int(m.FTHG > 0 and m.FTAG > 0),
                                 "fav_gap": fav_gap})
            print(code, a.date(), flush=True)
    return pd.DataFrame(rows)


def summarise(bt: pd.DataFrame) -> dict:
    out = {}
    for keys, g in list(bt.groupby(["lg", "v"])) + [(("ALL", v), g) for v, g in bt.groupby("v")]:
        lg, v = keys
        mk = g[g.q_res.notna()]
        fg = g.fav_gap.dropna()
        out.setdefault(lg, {})[v] = {
            "n": int(len(g)), "n_market": int(len(mk)),
            "ll_1x2": round(float(-np.log(g.p_res).mean()), 4),
            "ll_1x2_minus_market": round(float((-np.log(mk.p_res) + np.log(mk.q_res.astype(float))).mean()), 4),
            "ll_o25": round(float(R._ll(g.o25, g.y_o25).mean()), 4),
            "ll_btts": round(float(R._ll(g.btts, g.y_btts).mean()), 4),
            "bias_total": round(float((g.pred_tot - g.act_tot).mean()), 3),
            "fav_gap": round(float(fg.mean()), 4) if len(fg) else None, "n_fav": int(len(fg))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["backtest", "shadow"])
    ap.add_argument("--db", help="shadow: ArtifactData out_dir with fixtures/ and results/")
    ap.add_argument("--asof", help="shadow: date, YYYY-MM-DD")
    ap.add_argument("--leagues", default=",".join(LEAGUES))
    ap.add_argument("--out", default=str(ROOT / "pipeline" / "v7" / "out" / "backtest.json"))
    ap.add_argument("--round", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "shadow":
        recs, _ = shadow(Path(a.db), a.asof, ROOT / "pipeline" / "v7" / "out")
        print(f"{len(recs)} shadow predictions")
        return
    bt = backtest(a.leagues.split(","), VARIANTS if a.round == 1 else VARIANTS_R2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    bt.to_csv(Path(a.out).with_suffix(".rows.csv"), index=False)
    Path(a.out).write_text(json.dumps(summarise(bt), indent=1))
    print(json.dumps(summarise(bt)["ALL"], indent=1))



# ------------------------------------------------------------------ shadow predictions (2026-10-20)
CHOSEN = dict(ridge=1.0, shots=0.3, level=True)   # "shots30_r1_lvl", best of rounds 1-2 (out/backtest_round2.json)


def extras_from_db(db: Path, code: str, after: pd.Timestamp) -> pd.DataFrame:
    """Dashboard results newer than the football-data CSV (with sportradar shots when present)."""
    rows = []
    for f in (db / "fixtures").glob("*.json"):
        fx = json.loads(f.read_text())
        rp = db / "results" / f.name
        if fx.get("league") != code or not rp.exists():
            continue
        r = json.loads(rp.read_text())
        if "home_score" not in r or pd.Timestamp(fx["kickoff"]).tz_localize(None) <= after:
            continue
        st = r.get("stats") or {}
        h, a = st.get("h") or {}, st.get("a") or {}
        rows.append({"Date": pd.Timestamp(fx["kickoff"]).tz_localize(None).normalize(), "HomeTeam": fx["home"],
                     "AwayTeam": fx["away"], "FTHG": int(r["home_score"]), "FTAG": int(r["away_score"]),
                     "HS": h.get("shots", np.nan), "AS": a.get("shots", np.nan),
                     "HST": h.get("sot", np.nan), "AST": a.get("sot", np.nan)})
    return pd.DataFrame(rows)


def shadow(db: Path, asof: str, out: Path):
    asof_ts = pd.Timestamp(asof)
    recs, fits = [], {}
    for code in LEAGUES:
        df = load(code)
        ex = extras_from_db(db, code, df.Date.max())
        if len(ex):
            df = pd.concat([df, ex], ignore_index=True).sort_values("Date").reset_index(drop=True)
        kw = dict(CHOSEN); lvl = kw.pop("level")
        r = fit(df, asof_ts + pd.Timedelta(days=1), **kw)
        k = R.level_factor(df, r) if lvl else 1.0
        fits[code] = {**r, "level_factor": k, "n_extra_results": int(len(ex)), "fd_last": str(load(code).Date.max().date())}
        for f in (db / "fixtures").glob("*.json"):
            fx = json.loads(f.read_text())
            if fx.get("league") != code or pd.Timestamp(fx["kickoff"]).tz_localize(None) <= asof_ts:
                continue
            if (db / "results" / f.name).exists():
                continue
            unknown = [t for t in (fx["home"], fx["away"]) if t not in r["att"]]
            lam, mu = R.lambdas(r, fx["home"], fx["away"]); lam, mu = lam * k, mu * k
            G = R.grid(lam, mu, r["rho"]); i, j = np.indices(G.shape)
            recs.append({"fixture_id": fx["id"], "league": code, "kickoff": fx["kickoff"], "home": fx["home"],
                         "away": fx["away"], "xg_home": round(lam, 4), "xg_away": round(mu, 4),
                         "p_home": round(float(G[i > j].sum()), 5), "p_draw": round(float(G[i == j].sum()), 5),
                         "p_away": round(float(G[i < j].sum()), 5), "p_o15": round(float(G[(i + j) > 1].sum()), 5),
                         "p_o25": round(float(G[(i + j) > 2].sum()), 5), "p_o35": round(float(G[(i + j) > 3].sum()), 5),
                         "p_btts": round(float(G[(i > 0) & (j > 0)].sum()), 5), "unknown_teams": ";".join(unknown),
                         "model": "v7-candidate shots30_r1_lvl", "made_at": pd.Timestamp.utcnow().isoformat(timespec="seconds")})
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).sort_values(["kickoff", "league"]).to_csv(out / f"shadow_{asof}.csv", index=False)
    (out / f"fit_{asof}.json").write_text(json.dumps(fits, indent=1))
    return recs, fits


if __name__ == "__main__":
    main()
