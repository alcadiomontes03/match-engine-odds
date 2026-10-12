"""
Pulsematch v6.1 - corners model.

Per league, each side's corners are modelled as
    log E[home corners] = c + h + win[home] + give[away]
    log E[away corners] = c     + win[away] + give[home]
fitted by time-weighted Poisson likelihood (half-life HALF_LIFE days) with a ridge penalty
that pulls every team to the league average. Counts are slightly over-dispersed, so each
side is priced as a negative binomial with the dispersion measured on the same data, and
the total is the convolution of the two sides.

Data: data/fd/<code>_<season>.csv (football-data.co.uk HC/AC columns, from fetch_fd.py).
Team names are football-data's, the same short names the dashboard's fixtures use.

  python pipeline/v6/corners.py backtest            -> walk-forward check on 2025-26
  python pipeline/v6/corners.py predict FIXTURES.json OUT_DIR
        FIXTURES.json: [{id, league, home, away, kickoff}, ...]  (dashboard fixtures)
        writes OUT_DIR/<id>.json = one dashboard corners/<id> doc per fixture

Display and paper tracking only. DraftKings corner prices are not captured, so these are
fair prices with no edge measured, and they never feed the CLV gate.
"""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson

ROOT = Path(__file__).resolve().parents[2]
FD = ROOT / "data" / "fd"
LEAGUES = ["E0", "SP1", "D1", "F1"]
HALF_LIFE = 240.0           # days
RIDGE = 80.0                # pull toward the league average (tuned on the 2025-26 walk-forward)
YEARS_BACK = 3              # seasons of history before the target date
LINES = (7.5, 8.5, 9.5, 10.5, 11.5, 12.5)
MAXC = 30
MODEL = "v6.1-corners-nb"


def load(code: str) -> pd.DataFrame:
    out = []
    for f in sorted(FD.glob(f"{code}_*.csv")):
        d = pd.read_csv(f)
        if not {"HC", "AC"} <= set(d.columns):
            continue
        d = d.dropna(subset=["HC", "AC", "HomeTeam", "AwayTeam"])
        d["date"] = pd.to_datetime(d.Date, dayfirst=True, errors="coerce", format="mixed")
        out.append(d[["date", "HomeTeam", "AwayTeam", "HC", "AC", "FTHG", "FTAG"]])
    df = pd.concat(out, ignore_index=True).dropna(subset=["date"]).sort_values("date")
    df[["HC", "AC"]] = df[["HC", "AC"]].astype(int)
    return df.reset_index(drop=True)


def fit(df: pd.DataFrame, asof: pd.Timestamp) -> dict:
    tr = df[(df.date < asof) & (df.date >= asof - pd.Timedelta(days=365 * YEARS_BACK))]
    teams = sorted(set(tr.HomeTeam) | set(tr.AwayTeam))
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi, ai = tr.HomeTeam.map(ix).to_numpy(), tr.AwayTeam.map(ix).to_numpy()
    yh, ya = tr.HC.to_numpy(float), tr.AC.to_numpy(float)
    w = np.exp(-np.log(2) * (asof - tr.date).dt.days.to_numpy() / HALF_LIFE)

    def unpack(x):
        return x[0], x[1], x[2:2 + n], x[2 + n:]

    def f(x):
        c, h, win, give = unpack(x)
        lh = c + h + win[hi] + give[ai]
        la = c + win[ai] + give[hi]
        mh, ma = np.exp(lh), np.exp(la)
        nll = np.sum(w * (mh - yh * lh)) + np.sum(w * (ma - ya * la))
        nll += RIDGE * (np.sum(win ** 2) + np.sum(give ** 2))
        rh, ra = w * (mh - yh), w * (ma - ya)
        g = np.zeros_like(x)
        g[0] = rh.sum() + ra.sum()
        g[1] = rh.sum()
        gw = np.bincount(hi, rh, n) + np.bincount(ai, ra, n) + 2 * RIDGE * win
        gg = np.bincount(ai, rh, n) + np.bincount(hi, ra, n) + 2 * RIDGE * give
        g[2:2 + n], g[2 + n:] = gw, gg
        return nll, g

    x0 = np.zeros(2 + 2 * n); x0[0] = np.log(max(np.average(np.r_[yh, ya], weights=np.r_[w, w]), 1))
    res = minimize(f, x0, jac=True, method="L-BFGS-B")
    c, h, win, give = unpack(res.x)
    mh = np.exp(c + h + win[hi] + give[ai]); ma = np.exp(c + win[ai] + give[hi])
    # dispersion, measured on the match TOTAL (the two sides' counts are negatively correlated,
    # so per-side over-dispersion overstates the spread of the total): Pearson phi of the total,
    # then the per-side NB size k that gives the total that variance, var(T) = muT + (mh^2+ma^2)/k
    mt, yt = mh + ma, yh + ya
    phi = float(np.sum(w * (yt - mt) ** 2 / mt) / w.sum())
    sq = float(np.average(mh ** 2 + ma ** 2, weights=w)); mtb = float(np.average(mt, weights=w))
    k = sq / ((phi - 1) * mtb) if phi > 1.02 else None
    return {"c": float(c), "h": float(h), "win": dict(zip(teams, win.tolist())),
            "give": dict(zip(teams, give.tolist())), "phi": float(phi), "k": k,
            "n_train": int(len(tr)), "asof": str(asof.date())}


def means(m: dict, home: str, away: str):
    W, G = m["win"], m["give"]
    lh = m["c"] + m["h"] + W.get(home, 0.0) + G.get(away, 0.0)
    la = m["c"] + W.get(away, 0.0) + G.get(home, 0.0)
    return float(np.exp(lh)), float(np.exp(la))


def pmf(mu: float, k):
    x = np.arange(MAXC + 1)
    p = poisson.pmf(x, mu) if not k else nbinom.pmf(x, k, k / (k + mu))
    return p / p.sum()


def price(m: dict, home: str, away: str) -> dict:
    eh, ea = means(m, home, away)
    ph, pa = pmf(eh, m["k"]), pmf(ea, m["k"])
    tot = np.convolve(ph, pa)[:MAXC + 1]; tot /= tot.sum()
    t = np.arange(MAXC + 1)
    totals = {f"{L:.1f}": {"over": round(float(tot[t > L].sum()), 4), "under": round(float(tot[t < L].sum()), 4)}
              for L in LINES}
    diff = np.subtract.outer(np.arange(MAXC + 1), np.arange(MAXC + 1))
    J = np.outer(ph, pa)
    most = {"home": round(float(J[diff > 0].sum()), 4), "tie": round(float(J[diff == 0].sum()), 4),
            "away": round(float(J[diff < 0].sum()), 4)}
    return {"exp_home": round(eh, 3), "exp_away": round(ea, 3), "exp_total": round(eh + ea, 3),
            "totals": totals, "most": most}


def backtest():
    rows = []
    for code in LEAGUES:
        df = load(code)
        test = df[(df.date >= "2025-08-01") & (df.date < "2026-07-01")]
        cuts = pd.date_range("2025-08-01", "2026-06-30", freq="28D")
        for a, b in zip(cuts[:-1], cuts[1:]):
            chunk = test[(test.date >= a) & (test.date < b)]
            if chunk.empty:
                continue
            m = fit(df, a)
            tr = df[(df.date < a) & (df.date >= a - pd.Timedelta(days=365))]
            base = (tr.HC + tr.AC).mean()
            for r in chunk.itertuples():
                pr = price(m, r.HomeTeam, r.AwayTeam)
                T = r.HC + r.AC
                for L in LINES:
                    po = pr["totals"][f"{L:.1f}"]["over"]
                    pb = float(1 - poisson.cdf(int(L), base))
                    rows.append({"lg": code, "line": L, "p": po, "pb": pb, "y": int(T > L), "exp": pr["exp_total"], "T": T})
    bt = pd.DataFrame(rows)
    ll = lambda p, y: -np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    out = {"n_matches": int(len(bt) // len(LINES)), "by_line": {}, "by_league": {}, "calibration_9.5": []}
    for L, g in bt.groupby("line"):
        out["by_line"][f"{L:.1f}"] = {"logloss_model": round(ll(g.p, g.y), 4), "logloss_league_avg": round(ll(g.pb, g.y), 4),
                                     "avg_p": round(g.p.mean(), 3), "hit": round(g.y.mean(), 3)}
    for lg, g in bt[bt.line == 9.5].groupby("lg"):
        out["by_league"][lg] = {"logloss_model": round(ll(g.p, g.y), 4), "logloss_league_avg": round(ll(g.pb, g.y), 4), "n": int(len(g))}
    g = bt[bt.line == 9.5].copy(); g["bin"] = pd.cut(g.p, [0, .3, .4, .5, .6, .7, 1])
    for b, h in g.groupby("bin", observed=True):
        out["calibration_9.5"].append({"bin": str(b), "n": int(len(h)), "said": round(h.p.mean(), 3), "hit": round(h.y.mean(), 3)})
    e = bt[bt.line == 9.5]
    out["mae_total"] = round(float(np.mean(np.abs(e.exp - e["T"]))), 3)
    out["mae_league_avg"] = round(float(np.mean(np.abs(e["T"] - e["T"].mean()))), 3)
    return out


def predict(fixtures_path: str, out_dir: str):
    fx = json.loads(Path(fixtures_path).read_text())
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz=None).normalize()
    models, written, missing = {}, [], []
    for f in fx:
        lg = f["league"]
        if lg not in LEAGUES:
            continue
        if lg not in models:
            df = load(lg)
            models[lg] = (fit(df, now + pd.Timedelta(days=1)), set(df.HomeTeam) | set(df.AwayTeam))
        m, known = models[lg]
        unknown = [t for t in (f["home"], f["away"]) if t not in m["win"]]
        doc = {"id": f["id"], "fixture_id": f["id"], "league": lg, "kickoff": f.get("kickoff"),
               "home": f["home"], "away": f["away"], **price(m, f["home"], f["away"]),
               "model": MODEL, "fit_asof": m["asof"], "n_train": m["n_train"], "dispersion_k": m["k"],
               "unknown_teams": unknown, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        (out / f"{f['id']}.json").write_text(json.dumps(doc, indent=1))
        written.append(f["id"])
        if unknown:
            missing.append(f"{f['home']} v {f['away']}: {unknown}")
    print(json.dumps({"written": len(written), "unknown_teams": missing}))


if __name__ == "__main__":
    if sys.argv[1:2] == ["backtest"]:
        print(json.dumps(backtest(), indent=1))
    elif sys.argv[1:2] == ["predict"]:
        predict(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
