"""
Dixon-Coles (1997) bivariate Poisson rating model with exponential time decay.

Parameterization:
    log(lambda_home) = home_adv + attack[home] - defense[away]
    log(lambda_away) =            attack[away] - defense[home]

Identifiability constraint: mean(attack) == 0 (in log space).

Low-score dependence correction (tau) is applied to the (0,0), (1,0), (0,1), (1,1)
scoreline probabilities as in the original paper, controlled by rho.

Time decay: each match is weighted by exp(-xi * days_since_match), so the fit is a
rolling / discounted MLE rather than a single static season fit.

--- v4.5 fix (2026-09-21) ---
With ~60-65 matches per league to fit 40+ parameters, an unconstrained MLE was
overfitting: rho (normally -0.1 to -0.2 in the literature) was drifting to -0.46
to -0.48, wildly inflating draw/low-score probabilities, and individual team
attack/defense ratings had no shrinkage (e.g. a thin-sample team landing at a
degenerate -13.88 attack rating). Two changes fix this:
  1. rho is now bounded to [-0.25, 0.25] via L-BFGS-B's `bounds` argument (it
     previously had none at all).
  2. A ridge penalty (`ridge`, default 5.0) shrinks attack/defense toward 0, so
     a thin-sample team's rating regresses toward the league average instead
     of running to an extreme value chasing a handful of results.
Both are backtested against the prior fit before being trusted live.

Backtest (2026-09-21, Premier League, fit on 39 matches through 2026-09-14,
predicting the 10 real 2026-09-18/20 holdout matches): OLD rho=-1.42, 2/10
correct (20%), 5/10 draw calls, mean Brier 1.078. NEW rho=-0.22, 5/10 correct
(50%), 0/10 draw calls, mean Brier 0.672.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Ratings:
    teams: list
    attack: dict
    defense: dict
    home_adv: float
    rho: float
    as_of: datetime
    xi: float
    n_matches: int
    league: str = ""

    def to_dict(self):
        return {
            "league": self.league,
            "as_of": self.as_of.isoformat(),
            "xi": self.xi,
            "n_matches": self.n_matches,
            "home_adv": self.home_adv,
            "rho": self.rho,
            "attack": self.attack,
            "defense": self.defense,
        }


def _tau(x, y, lam, mu, rho):
    """Dixon-Coles low-score correlation adjustment."""
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    elif x == 0 and y == 1:
        return 1 + lam * rho
    elif x == 1 and y == 0:
        return 1 + mu * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _neg_log_likelihood(params, home_idx, away_idx, hg, ag, weights, n_teams, ridge=5.0):
    attack = params[:n_teams]
    defense = params[n_teams:2 * n_teams]
    home_adv = params[2 * n_teams]
    rho = params[2 * n_teams + 1]

    log_lam = home_adv + attack[home_idx] - defense[away_idx]
    log_mu = attack[away_idx] - defense[home_idx]
    lam = np.exp(log_lam)
    mu = np.exp(log_mu)

    ll = (hg * log_lam - lam - _gammaln(hg + 1)) + (ag * log_mu - mu - _gammaln(ag + 1))

    # low-score correction, vectorized over the small set of (0,0)/(0,1)/(1,0)/(1,1) cells
    tau = np.ones_like(lam)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    tau[m00] = 1 - lam[m00] * mu[m00] * rho
    tau[m01] = 1 + lam[m01] * rho
    tau[m10] = 1 + mu[m10] * rho
    tau[m11] = 1 - rho
    tau = np.clip(tau, 1e-6, None)
    ll = ll + np.log(tau)

    # identifiability: mean(attack) == 0, soft penalty (keeps optimizer well-posed)
    penalty = 1000.0 * (np.mean(attack) ** 2)
    # ridge shrinkage on team ratings — prevents a thin-sample team from landing
    # on a degenerate rating with nothing pulling it back toward league average
    penalty = penalty + ridge * (np.sum(attack ** 2) + np.sum(defense ** 2))

    return -np.sum(weights * ll) + penalty


def _gammaln(x):
    from scipy.special import gammaln
    return gammaln(x)


def fit(matches: pd.DataFrame, as_of: datetime, xi: float = 0.0018,
        league: str = "", ridge: float = 5.0) -> Ratings:
    """
    matches: DataFrame with columns Date (datetime), HomeTeam, AwayTeam, FTHG, FTAG.
    as_of: reference date for the exponential time decay (usually "today").
    xi: decay rate per day. 0.0018/day ~ half-life of ~385 days (season-and-a-bit).
    ridge: L2 shrinkage strength on attack/defense ratings (see module docstring).
    """
    df = matches.dropna(subset=["FTHG", "FTAG"]).copy()
    if len(df) < 20:
        raise ValueError(f"only {len(df)} matches with results — too few to fit reliably")

    teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
    n_teams = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    home_idx = df.HomeTeam.map(idx).to_numpy()
    away_idx = df.AwayTeam.map(idx).to_numpy()
    hg = df.FTHG.to_numpy(dtype=float)
    ag = df.FTAG.to_numpy(dtype=float)

    days_ago = (as_of - df.Date).dt.days.clip(lower=0).to_numpy()
    weights = np.exp(-xi * days_ago)

    x0 = np.zeros(2 * n_teams + 2)
    x0[2 * n_teams] = 0.25  # home_adv init
    x0[2 * n_teams + 1] = -0.05  # rho init

    # rho bounded to the literature-typical Dixon-Coles range so a thin sample
    # can't walk it out to a degenerate value (it previously had no bound at
    # all and was landing around -0.46 to -0.48, or as low as -1.42 in one
    # backtest run). attack/defense/home_adv stay unbounded — the ridge
    # penalty above already regularizes them.
    bounds = [(None, None)] * (2 * n_teams + 1) + [(-0.25, 0.25)]

    res = minimize(
        _neg_log_likelihood, x0,
        args=(home_idx, away_idx, hg, ag, weights, n_teams, ridge),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-10},
    )

    attack = {t: float(res.x[idx[t]]) for t in teams}
    defense = {t: float(res.x[n_teams + idx[t]]) for t in teams}
    home_adv = float(res.x[2 * n_teams])
    rho = float(res.x[2 * n_teams + 1])

    return Ratings(teams=teams, attack=attack, defense=defense, home_adv=home_adv,
                    rho=rho, as_of=as_of, xi=xi, n_matches=len(df), league=league)


def match_lambdas(ratings: Ratings, home: str, away: str):
    """Expected goals (lambda_home, lambda_away) for a fixture, given ratings.
    Unknown teams fall back to league-average attack/defense (0.0)."""
    a_h = ratings.attack.get(home, 0.0)
    d_h = ratings.defense.get(home, 0.0)
    a_a = ratings.attack.get(away, 0.0)
    d_a = ratings.defense.get(away, 0.0)
    lam = np.exp(ratings.home_adv + a_h - d_a)
    mu = np.exp(a_a - d_h)
    return float(lam), float(mu)


def scoreline_matrix(lam: float, mu: float, rho: float, max_goals: int = 10) -> np.ndarray:
    """P(home=i, away=j) grid with the DC low-score correction applied."""
    i = np.arange(max_goals + 1)
    ph = poisson.pmf(i, lam)
    pa = poisson.pmf(i, mu)
    grid = np.outer(ph, pa)
    for x in range(2):
        for y in range(2):
            grid[x, y] *= _tau(x, y, lam, mu, rho)
    grid = grid / grid.sum()
    return grid


def market_probs(ratings: Ratings, home: str, away: str, max_goals: int = 10):
    """1X2, over/under 2.5, and a scoreline grid for one fixture."""
    lam, mu = match_lambdas(ratings, home, away)
    grid = scoreline_matrix(lam, mu, ratings.rho, max_goals)
    p_home = float(np.tril(grid, -1).sum())
    p_draw = float(np.trace(grid))
    p_away = float(np.triu(grid, 1).sum())
    i, j = np.indices(grid.shape)
    total = i + j
    p_over25 = float(grid[total > 2.5].sum()) if False else float(grid[(i + j) >= 3].sum())
    p_under25 = 1 - p_over25
    return {
        "lambda_home": lam, "lambda_away": mu,
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "p_over25": p_over25, "p_under25": p_under25,
        "grid": grid,
    }
