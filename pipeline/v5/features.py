"""
Pulsematch v5 — gap features.

The Dixon-Coles ratings stop being the final answer and become inputs. For every
match, using only information from before its kick-off week:

  sup_dc   = log(lam/mu) from the goals Dixon-Coles fit   - market log(lam/mu)
  tot_dc   = log(lam+mu) from the goals fit               - market log(lam+mu)
  sup_sot  = log(s_h/s_a) from a shots-on-target rating fit - market log(lam/mu)
  tot_sot  = log(s_h+s_a) - league-average log(s_h+s_a)  (shot volume vs normal)
  rest     = (home rest days - away rest days)/7, clipped to +-1  (league matches only)
  early    = 1 in a club's first 6 league matches of the season
Shots on target stand in for xG: football-data carries them for every league and
season back to 2017, while free xG only covers the PL and La Liga from 2022.

Ratings refit once per week (Monday) per league with dc4.fit, warm-started.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import dc4  # noqa: E402

XI = 0.0018          # same decay as the live model (~385-day half-life)
WINDOW = 800         # days of history per fit
LAM_PRIOR = 3.0      # strength of the newcomer prior (roughly "a few matches" of evidence)


def _newcomer_fallback(r):
    """Unseen (promoted) clubs get the average of the 3 weakest rated clubs, not the
    league mean — promoted sides are well below average."""
    if not r or not r["att"]:
        return None
    strength = {t: r["att"][t] + r["def"][t] for t in r["att"]}
    weak = sorted(strength, key=strength.get)[:3]
    return (float(np.mean([r["att"][t] for t in weak])), float(np.mean([r["def"][t] for t in weak])))


def _lams(r, home, away):
    fb = _newcomer_fallback(r)
    known = set(r["att"])
    fallback = {t: fb for t in set(home) | set(away) if t not in known} if fb else None
    return dc4.lambdas(r, list(home), list(away), fallback=fallback)


def walk_forward_ratings(df: pd.DataFrame, min_history_days: int = 300) -> pd.DataFrame:
    """df: one league, sorted, with Date/HomeTeam/AwayTeam/FTHG/FTAG/HST/AST.
    Returns dc_lam, dc_mu, sot_h, sot_a per row (NaN until enough history)."""
    df = df.sort_values("Date").reset_index(drop=True)
    wk = df.Date.dt.to_period("W-SUN").dt.start_time      # Monday of the match week
    shots = df.rename(columns={"FTHG": "g_h", "FTAG": "g_a"}).copy()
    shots["FTHG"], shots["FTAG"] = df.HST, df.AST
    shots = shots.dropna(subset=["FTHG", "FTAG"])
    start = df.Date.min() + pd.Timedelta(days=min_history_days)
    out = np.full((len(df), 4), np.nan)
    warm_g = warm_s = None
    # newcomer prior (as in the live model): clubs not in the league last season are
    # pulled toward the average of last season's three weakest clubs until they have
    # enough matches of their own. Without it a promoted side with 1-2 games can land
    # on a degenerate rating (e.g. an expected 0.01 home goals).
    teams_by_season = {s: set(g.HomeTeam) | set(g.AwayTeam) for s, g in df.groupby("Season")}
    seasons = sorted(teams_by_season)
    newcomers = {s: teams_by_season[s] - teams_by_season[seasons[i - 1]] if i else set()
                 for i, s in enumerate(seasons)}
    season_of_week = df.groupby(wk).Season.first()
    for monday in sorted(wk.unique()):
        if monday < start:
            continue
        rows = np.where(wk == monday)[0]
        new = newcomers[season_of_week[monday]]
        pg = _newcomer_fallback(warm_g); ps = _newcomer_fallback(warm_s)
        rg = dc4.fit(df, monday, XI, warm=warm_g, window_days=WINDOW,
                     priors={t: pg for t in new} if pg else None, lam_prior=LAM_PRIOR)
        rs = dc4.fit(shots, monday, XI, warm=warm_s, window_days=WINDOW, rho_bounds=(0.0, 0.0),
                     priors={t: ps for t in new} if ps else None, lam_prior=LAM_PRIOR)
        warm_g, warm_s = rg, rs
        h, a = df.HomeTeam.iloc[rows], df.AwayTeam.iloc[rows]
        out[rows, 0], out[rows, 1] = _lams(rg, h, a)
        out[rows, 2], out[rows, 3] = _lams(rs, h, a)
    df[["dc_lam", "dc_mu", "sot_h", "sot_a"]] = out
    return df


def schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rest days and early-season flag, from league fixtures only."""
    df = df.sort_values("Date").reset_index(drop=True)
    long = pd.concat([
        pd.DataFrame({"i": df.index, "team": df.HomeTeam, "Date": df.Date, "side": "h", "Season": df.Season}),
        pd.DataFrame({"i": df.index, "team": df.AwayTeam, "Date": df.Date, "side": "a", "Season": df.Season}),
    ]).sort_values(["team", "Date"])
    long["rest"] = long.groupby("team").Date.diff().dt.days.clip(upper=14)
    long["n_in_season"] = long.groupby(["team", "Season"]).cumcount()
    h = long[long.side == "h"].set_index("i"); a = long[long.side == "a"].set_index("i")
    df["rest_h"], df["rest_a"] = h.rest.reindex(df.index), a.rest.reindex(df.index)
    df["rest"] = ((df.rest_h.fillna(7) - df.rest_a.fillna(7)) / 7).clip(-1, 1)
    df["early"] = ((h.n_in_season.reindex(df.index) < 6) | (a.n_in_season.reindex(df.index) < 6)).astype(float)
    return df


def gap_features(df: pd.DataFrame) -> pd.DataFrame:
    m_sup = np.log(df.mkt_lam / df.mkt_mu)
    m_tot = np.log(df.mkt_lam + df.mkt_mu)
    df["sup_dc"] = np.log(df.dc_lam / df.dc_mu) - m_sup
    df["tot_dc"] = np.log(df.dc_lam + df.dc_mu) - m_tot
    df["sup_sot"] = np.log(df.sot_h / df.sot_a) - m_sup
    lt = np.log(df.sot_h + df.sot_a)
    # vs the league's running average of earlier matches only (no look-ahead)
    df["tot_sot"] = lt - lt.groupby(df.Div).transform(lambda s: s.expanding().mean().shift(1))
    df["early_sup_dc"] = df.early * df.sup_dc
    return df


FEATURES_1X2 = ["sup_dc", "sup_sot", "rest", "early_sup_dc"]
FEATURES_OU = ["tot_dc", "tot_sot", "early"]
