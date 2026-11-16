"""
Real xG test (2026-11-15). Understat is down, so match xG comes from a public archive of Understat's per-team
match files (github.com/WanyoikeNgugi/football-intelligence, data/<league>/<start year>/teams/<Team>/<Team>.csv,
last updated 2026-10-06, matches to mid-May 2026). No 2026-27 matches: this is a backtest only.

    python3 pipeline/v8/xg.py build <path to the archive clone>   -> data/xg/understat_archive.csv
    python3 pipeline/v8/xg.py run                                 -> out/xg_summary.md

Team names are matched to football-data's automatically, per league and season, by home matches that agree on
date and score. Same folds, scoring and bootstrap as gap.py. Two questions:
  1. As a rating target: does real xG beat the shot proxy, and does mixing it into v9's market target help?
  2. As an add-on: does a team's recent xG, measured against what the market expected of it, improve v9?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gap as G                # noqa: E402
from fixes import C, R         # noqa: E402
import fixes as F              # noqa: E402

ARCH = F.ROOT / "data" / "xg" / "understat_archive.csv"
US = {"E0": "EPL", "SP1": "La_liga", "D1": "Bundesliga", "F1": "Ligue_1"}
END = pd.Timestamp("2026-06-01")     # the archive has nothing after 2025-26
VARIANTS = {   # name: (wg, ws, wm, xi, P, real xG in place of the shot proxy)
    "v8_live":          (0.70, 0.30, 0.00, 0.002, 0.0, False),
    "fix_nov11":        (0.70, 0.30, 0.00, 0.002, 5.0, False),   # goals + shot proxy
    "goals_xg":         (0.70, 0.30, 0.00, 0.002, 5.0, True),    # goals + real xG
    "xg_only_x2":       (0.00, 1.00, 0.00, 0.002, 5.0, True),
    "xg_only_x6":       (0.00, 1.00, 0.00, 0.006, 5.0, True),
    "mkt_x10_prom5":    (0.00, 0.00, 1.00, 0.010, 5.0, False),   # v9's ratings
    "mkt85_xg15_x10":   (0.00, 0.15, 0.85, 0.010, 5.0, True),
    "mkt70_xg30_x10":   (0.00, 0.30, 0.70, 0.010, 5.0, True),
}
FORM_N, FORM_MIN = 8, 4


def build(src: str) -> pd.DataFrame:
    out, rep = [], []
    for code, lg in US.items():
        fd = C.load(code)
        fd["season"] = fd.Date.map(F.season_of)
        for sdir in sorted((Path(src) / "data" / lg).glob("*")):
            if not sdir.name.isdigit():
                continue
            y = int(sdir.name) % 100
            fs = fd[fd.season == f"{y:02d}{y + 1:02d}"]
            if fs.empty:
                continue
            for tdir in sorted((sdir / "teams").glob("*")):
                f = tdir / f"{tdir.name}.csv"
                if not f.exists():
                    continue
                t = pd.read_csv(f); t = t[t.h_a == "h"].copy()
                t["d"] = pd.to_datetime(t.date).dt.normalize()
                best, hit = None, None
                for name, g in fs.groupby("HomeTeam"):
                    for sh in (0, 1, -1):
                        j = g.merge(t.assign(Date=t.d + pd.Timedelta(days=sh)), on="Date")
                        j = j[(j.FTHG == j.scored) & (j.FTAG == j.missed)]
                        if hit is None or len(j) > len(hit):
                            best, hit = name, j
                # second pass: matches a day off (kickoff near midnight) for the chosen team
                g = fs[fs.HomeTeam == best]; got = []
                for sh in (0, 1, -1):
                    j = g.merge(t.assign(Date=t.d + pd.Timedelta(days=sh)), on="Date")
                    got.append(j[(j.FTHG == j.scored) & (j.FTAG == j.missed)])
                j = pd.concat(got).drop_duplicates(["Date", "HomeTeam", "AwayTeam"])
                rep.append((code, sdir.name, tdir.name, best, len(t), len(g), len(j)))
                out.append(j.assign(lg=code)[["lg", "Date", "HomeTeam", "AwayTeam", "xG", "xGA", "npxG", "npxGA"]])
    x = pd.concat(out, ignore_index=True).rename(columns={"xG": "xgh", "xGA": "xga", "npxG": "npxgh", "npxGA": "npxga"})
    x = x.drop_duplicates(["lg", "Date", "HomeTeam", "AwayTeam"]).sort_values(["lg", "Date"])
    ARCH.parent.mkdir(parents=True, exist_ok=True)
    x.to_csv(ARCH, index=False)
    r = pd.DataFrame(rep, columns=["lg", "season", "understat", "football_data", "us_home", "fd_home", "matched"])
    r.to_csv(G.OUT / "xg_name_map.csv", index=False)
    return r


def load(code: str) -> pd.DataFrame:
    df = G.load(code)
    x = pd.read_csv(ARCH, parse_dates=["Date"])
    x = x[x.lg == code].drop(columns="lg")
    return df.merge(x, on=["Date", "HomeTeam", "AwayTeam"], how="left")


def xg_form(df: pd.DataFrame) -> pd.Series:
    """Per match: home minus away of each team's mean, over its last FORM_N league matches (all strictly earlier),
    of (xG for - xG against) - (market-expected goals for - against). Positive = playing better than it was priced.
    NaN unless both teams have FORM_MIN such matches, the latest within 60 days."""
    hist, val = {}, []
    for m in df.itertuples():
        v = []
        for t in (m.HomeTeam, m.AwayTeam):
            h = hist.get(t, [])[-FORM_N:]
            v.append(np.mean([e for _, e in h]) if len(h) >= FORM_MIN and (m.Date - h[-1][0]).days <= 60 else np.nan)
        val.append(v[0] - v[1])
        if np.isfinite(m.xgh) and np.isfinite(m.mlam):
            e = (m.xgh - m.xga) - (m.mlam - m.mmu)
            hist.setdefault(m.HomeTeam, []).append((m.Date, e)); hist.setdefault(m.AwayTeam, []).append((m.Date, -e))
    return pd.Series(val, index=pd.MultiIndex.from_frame(df[["Date", "HomeTeam", "AwayTeam"]]))


def run() -> pd.DataFrame:
    rows = []
    for code in G.LEAGUES:
        df = load(code)
        df = df[df.Date < END].reset_index(drop=True)
        xf = xg_form(df)
        cuts = pd.date_range(F.WARM_FROM, df.Date.max() + pd.Timedelta(days=28), freq="28D")
        for a, b in zip(cuts[:-1], cuts[1:]):
            chunk = df[(df.Date >= a) & (df.Date < b)]
            if chunk.empty:
                continue
            promo, _ = F.promoted_relegated(code, a)
            for name, (wg, ws, wm, xi, P, xg) in VARIANTS.items():
                r = G.fit(df, a, wg=wg, ws=ws, wm=wm, xi=xi, use_nnls=(name != "v8_live"), P=P, code=code, xg=xg)
                k = R.level_factor(df, r)
                for m in chunk.itertuples():
                    lam, mu = R.lambdas(r, m.HomeTeam, m.AwayTeam)
                    q = F.market_close(m)
                    rows.append({"lg": code, "v": name, "fold": str(a.date()), "Date": m.Date, "home": m.HomeTeam,
                                 "away": m.AwayTeam, "lam": lam * k, "mu": mu * k, "rho": r["rho"], "ha": r["home_adv"],
                                 "y": 0 if m.FTHG > m.FTAG else (1 if m.FTHG == m.FTAG else 2),
                                 "tot": m.FTHG + m.FTAG, "prom": bool({m.HomeTeam, m.AwayTeam} & promo),
                                 "qH": np.nan if q is None else q[0], "qD": np.nan if q is None else q[1],
                                 "qA": np.nan if q is None else q[2], "has_xg": bool(np.isfinite(m.xgh)),
                                 "x": float(xf.get((m.Date, m.HomeTeam, m.AwayTeam), np.nan))})
            print(code, a.date(), flush=True)
    d = pd.DataFrame(rows); out = []
    for name in VARIANTS:
        v = d[d.v == name]
        out += [v.assign(s=1.0, coef=0.0), G.stage2(v, None, "+stretch"), G.stage2(v, "x", "+xgform")]
    d = pd.concat(out, ignore_index=True)
    pH, pD, pA, _, _ = F.probs_vec(d.lam.to_numpy(), d.mu.to_numpy(), d.rho.to_numpy())
    d["pH"], d["pD"], d["pA"] = pH, pD, pA
    d["ll"] = F.ll1x2(pH, pD, pA, d.y.to_numpy())
    d["ll_mkt"] = F.ll1x2(d.qH.to_numpy(), d.qD.to_numpy(), d.qA.to_numpy(), d.y.to_numpy())
    d = d[d.Date >= F.SCORE_FROM].reset_index(drop=True)
    d.to_csv(G.OUT / "xg_rows.csv", index=False)
    return d


def summarise(d: pd.DataFrame) -> str:
    mk = d.dropna(subset=["qH"])
    n = int((mk.v == "v8_live").sum()); cov = mk[mk.v == "v8_live"].has_xg.mean()
    L = ["# Real xG test, walk-forward (2023-08 to 2026-05)", "",
         f"{n:,} matches with a closing price ({cov:.1%} have Understat xG in the archive); 28-day refits; lower is "
         "better. Gap = model minus closing market. 95% intervals resample whole 28-day folds.", "",
         "## 1. xG as a rating target (all with the walk-forward stretch)", "",
         "| Ratings | Gap to market | Change vs v9 ratings | 95% |", "|---|---|---|---|"]
    ref = "mkt_x10_prom5+stretch"
    for nm in VARIANTS:
        v = mk[mk.v == nm + "+stretch"]
        m, lo, hi, _ = G._cmp(mk, nm + "+stretch", ref)
        L.append(f"| {nm} | {(v.ll - v.ll_mkt).mean():+.4f} | {m:+.4f} | {lo:+.4f} to {hi:+.4f} |")
    m, lo, hi, _ = G._cmp(mk, "goals_xg+stretch", "fix_nov11+stretch")
    L += ["", f"Real xG against the shot proxy, same method otherwise (goals_xg minus fix_nov11): {m:+.4f} "
          f"({lo:+.4f} to {hi:+.4f}).", "",
          "## 2. xG form as an add-on (+xgform minus +stretch on the same ratings, matches that have the feature)", "",
          "| Ratings | Matches | Change | 95% | Median fitted coefficient |", "|---|---|---|---|---|"]
    has = lambda j: np.isfinite(j.x)       # noqa: E731
    for nm in VARIANTS:
        m, lo, hi, k = G._cmp(mk, nm + "+xgform", nm + "+stretch", has)
        c = d[(d.v == nm + "+xgform") & (d.coef != 0)].coef.median()
        L.append(f"| {nm} | {k:,} | {m:+.4f} | {lo:+.4f} to {hi:+.4f} | {c:+.3f} |")
    L += ["", "## v9 ratings + xG form, by league (change vs v9 ratings, matches that have the feature)", "",
          "| League | Matches | Change | 95% |", "|---|---|---|---|"]
    for c in G.LEAGUES:
        m, lo, hi, k = G._cmp(mk[mk.lg == c], "mkt_x10_prom5+xgform", ref, has)
        L.append(f"| {c} | {k:,} | {m:+.4f} | {lo:+.4f} to {hi:+.4f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    G.OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "build":
        r = build(sys.argv[2])
        print(r.groupby(["lg", "season"]).agg(teams=("matched", "size"), fd_home=("fd_home", "sum"),
                                              matched=("matched", "sum")).to_string())
        print(r[r.matched < r.fd_home - 1].to_string())
    else:
        d = run() if "--reuse" not in sys.argv else pd.read_csv(G.OUT / "xg_rows.csv", parse_dates=["Date"])
        txt = summarise(d)
        (G.OUT / "xg_summary.md").write_text(txt)
        print(txt)
