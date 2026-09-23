"""
Match Engine v4 — fair-odds generator (PAPER MODE).
(16 Sep 2026. Requires dc4.lambdas(..., fallback=) from V4_AUDIT.md and
dc4.fit(..., rho_bounds=) from dc4_rho_bounds.patch.txt.
Results data: data/ext/Matches.csv from
github.com/xgabora/Club-Football-Match-Data-2000-2025 data/Matches.csv.
Prices input: data/odds/snapshots.csv from github.com/alcadiomontes03/match-engine-odds.
Run: python src/price4.py <snapshots.csv> <as_of YYYY-MM-DD> <out.json>
First sheet published: https://claude.ai/artifact/JKyHb1eveftLLunw1JHDgf)

For each upcoming fixture captured by the DraftKings moneyline pull, produce
fair (no-vig) prices for:
  * spreads (Asian handicap, half-point lines)
  * totals (over/under 1.5, 2.5, 3.5)
  * both teams to score
  * team shot attempts (over/under)

Two goal models are reported side by side:
  MARKET-ANCHORED  expected goals (lam, mu) solved so the Dixon-Coles 1X2 matches
                   DraftKings' moneyline after Shin de-vig. The v4 gate showed the
                   market beats our ratings, so this is the primary number: it
                   carries the market's view of who is stronger and only uses the
                   model for the SHAPE of the scoreline distribution.
  MODEL-ONLY       our own Dixon-Coles ratings (for comparison only).
Team shot attempts come from a separate Poisson rating fit on shots (HS/AS)
with a negative-binomial spread; there is no market anchor for shots.

Nothing here is validated against a market for these derived lines. These are
prices to compare against, not bet recommendations.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import dc4

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent

LEAGUES = {  # odds-capture label -> (football-data division, decay xi)
    "PL": ("E0", 0.003), "LaLiga": ("SP1", 0.001), "Championship": ("E1", 0.002)}
PRIOR_LAMBDA = 3.0
SEASON_START = pd.Timestamp("2026-07-01")

NAME_MAP = {
    # Premier League
    "Brighton and Hove Albion": "Brighton", "Coventry City": "Coventry", "Hull City": "Hull",
    "Ipswich Town": "Ipswich", "Leeds United": "Leeds", "Manchester City": "Man City",
    "Manchester United": "Man United", "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest", "Tottenham Hotspur": "Tottenham",
    # La Liga
    "Alavés": "Alaves", "Athletic Bilbao": "Ath Bilbao", "Atlético Madrid": "Ath Madrid",
    "CA Osasuna": "Osasuna", "Celta Vigo": "Celta", "Deportivo La Coruña": "La Coruna",
    "Elche CF": "Elche", "Espanyol": "Espanol", "Málaga": "Malaga", "Rayo Vallecano": "Vallecano",
    "Real Betis": "Betis", "Real Racing Club de Santander": "Santander", "Real Sociedad": "Sociedad",
    # Championship
    "Birmingham City": "Birmingham", "Blackburn Rovers": "Blackburn", "Bolton Wanderers": "Bolton",
    "Cardiff City": "Cardiff", "Charlton Athletic": "Charlton", "Derby County": "Derby",
    "Lincoln City": "Lincoln", "Norwich City": "Norwich", "Preston North End": "Preston",
    "Queens Park Rangers": "QPR", "Stoke City": "Stoke", "Swansea City": "Swansea",
    "West Bromwich Albion": "West Brom", "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves", "Wrexham AFC": "Wrexham",
}


def load_results(div):
    m = pd.read_csv(ROOT / "data/ext/Matches.csv", low_memory=False)
    m = m[m.Division == div].copy()
    d = pd.DataFrame({"Date": pd.to_datetime(m.MatchDate), "HomeTeam": m.HomeTeam, "AwayTeam": m.AwayTeam,
                      "FTHG": m.FTHome, "FTAG": m.FTAway, "HS": m.HomeShots, "AS": m.AwayShots})
    d = d.dropna(subset=["FTHG", "FTAG"]).sort_values("Date")
    d["Season"] = np.where(d.Date.dt.month >= 7, d.Date.dt.year, d.Date.dt.year - 1)
    return d


def newcomer_priors(d, xi, view):
    """Promoted teams start at the mean rating of the teams they replaced."""
    cur = set(d.loc[d.Date >= SEASON_START, "HomeTeam"])
    prev = set(d.loc[(d.Date < SEASON_START) & (d.Date >= SEASON_START - pd.Timedelta(days=330)), "HomeTeam"])
    if not cur:
        return {}
    r = dc4.fit(view, SEASON_START, xi, rho_bounds=(-0.25, 0.25) if view is d else (0.0, 0.0))
    rel = [t for t in prev - cur if t in r["att"]]
    if not rel:
        return {}
    ra, rd = np.mean([r["att"][t] for t in rel]), np.mean([r["def"][t] for t in rel])
    return {t: (ra, rd) for t in cur - prev}


def shin(odds):
    pi = 1.0 / np.asarray(odds, float); Pi = pi.sum()
    lo, hi = 0.0, 0.4
    f = lambda z: (np.sqrt(z * z + 4 * (1 - z) * pi * pi / Pi) - z) / (2 * (1 - z))
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid).sum() > 1 else (lo, mid)
    p = f((lo + hi) / 2)
    return p / p.sum()


def anchor(p_home, p_away, rho, start):
    """Solve (lam, mu) so Dixon-Coles H/A probabilities match the market."""
    def loss(x):
        G = dc4.grids(np.array([np.exp(x[0])]), np.array([np.exp(x[1])]), rho)
        P = dc4.probs_1x2(G)[0]
        return (P[0] - p_home) ** 2 + (P[2] - p_away) ** 2
    res = minimize(loss, np.log(start), method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-12})
    return float(np.exp(res.x[0])), float(np.exp(res.x[1])), float(res.fun)


def markets(lam, mu, rho):
    G = dc4.grids(np.array([lam]), np.array([mu]), rho)
    H, D, A = dc4.probs_1x2(G)[0]
    out = {"H": H, "D": D, "A": A, "xg_home": lam, "xg_away": mu}
    for L in (1.5, 2.5, 3.5):
        out[f"O{L}"] = float(dc4.p_over(G, L)[0])
    for h in (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5):
        out[f"AH{h:+}"] = float(dc4.p_home_cover_half(G, [h])[0])
    g = G[0]
    out["BTTS"] = float(1 - g[0, :].sum() - g[:, 0].sum() + g[0, 0])
    # main spread: the half line whose home-cover probability is closest to 50%
    main = min((-2.5, -1.5, -0.5, 0.5, 1.5), key=lambda h: abs(out[f"AH{h:+}"] - 0.5))
    out["main_line"] = main
    return out


def nb_size(y, m):
    excess = np.mean((y - m) ** 2 - m)
    return float(np.mean(m ** 2) / excess) if excess > 0 else 1e6


def nb_over(mean, k, line):
    return float(1 - nbinom.cdf(np.floor(line), k, k / (k + mean)))


def american(p):
    if p is None or not (0 < p < 1):
        return ""
    return f"{-round(100 * p / (1 - p)):+d}" if p >= 0.5 else f"+{round(100 * (1 - p) / p)}"


def build(snapshots_csv, as_of):
    snaps = pd.read_csv(snapshots_csv)
    snaps = snaps[snaps.market == "h2h"]
    latest = snaps.sort_values("pulled_at").groupby(["event_id", "outcome"]).tail(1)
    as_of = pd.Timestamp(as_of)
    rows, fit_info = [], {}
    for label, (div, xi) in LEAGUES.items():
        d = load_results(div)
        shots = d.dropna(subset=["HS", "AS"]).copy()
        shots["FTHG"], shots["FTAG"] = shots.HS, shots.AS
        pg = newcomer_priors(d, xi, d)
        ps = newcomer_priors(d, xi, shots)
        rg = dc4.fit(d, as_of, xi, priors=pg, lam_prior=PRIOR_LAMBDA)
        rs = dc4.fit(shots, as_of, xi, priors=ps, lam_prior=PRIOR_LAMBDA, rho_bounds=(0.0, 0.0))
        # team-shots dispersion from the last ~2 seasons, in-sample
        recent = shots[shots.Date >= as_of - pd.Timedelta(days=700)]
        lh, la = dc4.lambdas(rs, recent.HomeTeam.to_numpy(), recent.AwayTeam.to_numpy(), fallback=ps)
        k_shots = nb_size(np.concatenate([recent.HS, recent.AS]).astype(float), np.concatenate([lh, la]))
        fit_info[label] = {"results_through": str(d.Date.max().date()), "rho": round(rg["rho"], 3),
                           "home_adv": round(rg["home"], 3), "shots_nb_size": round(k_shots, 1),
                           "newcomers": sorted(pg)}
        ev = latest[latest.comp == label]
        for eid, g in ev.groupby("event_id"):
            home_api, away_api = g.home.iloc[0], g.away.iloc[0]
            home, away = NAME_MAP.get(home_api, home_api), NAME_MAP.get(away_api, away_api)
            unknown = [t for t in (home, away) if t not in rg["att"] and t not in pg]
            price = {r.outcome: r.price for r in g.itertuples()}
            dk = [price.get(home_api), price.get("Draw"), price.get(away_api)]
            lam_m, mu_m = dc4.lambdas(rg, [home], [away], fallback=pg)
            lam_m, mu_m = float(lam_m[0]), float(mu_m[0])
            model = markets(lam_m, mu_m, rg["rho"])
            anch, fit_err = None, None
            if all(x and x > 1 for x in dk):
                pH, pD, pA = shin(dk)
                la_, mu_, fit_err = anchor(pH, pA, rg["rho"], (lam_m, mu_m))
                anch = markets(la_, mu_, rg["rho"])
                anch["dk"] = dk; anch["dk_fair"] = [pH, pD, pA]
            sh_h, sh_a = dc4.lambdas(rs, [home], [away], fallback=ps)
            sh_h, sh_a = float(sh_h[0]), float(sh_a[0])
            shots_out = {}
            for side, m_ in (("home", sh_h), ("away", sh_a)):
                line = np.floor(m_) + 0.5
                shots_out[side] = {"mean": m_, "line": line, "over": nb_over(m_, k_shots, line)}
            rows.append({"league": label, "event_id": eid, "kickoff": g.commence.iloc[0],
                         "home": home_api, "away": away_api, "unknown_teams": unknown,
                         "anchored": anch, "model": model, "anchor_error": fit_err, "shots": shots_out})
    rows.sort(key=lambda r: (r["kickoff"], r["league"]))
    return rows, fit_info


if __name__ == "__main__":
    rows, info = build(sys.argv[1], sys.argv[2])
    out = Path(sys.argv[3])
    out.write_text(json.dumps({"fits": info, "matches": rows}, indent=1, default=float))
    print(json.dumps(info, indent=1))
    print(len(rows), "matches priced")
