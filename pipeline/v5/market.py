"""
Pulsematch v5 — market layer.

The market price is the baseline. This module turns bookmaker odds into
  * fair probabilities (Shin de-vig for 1X2, Shin for two-way lines),
  * market-implied expected goals (lam_home, lam_away) that reproduce the
    de-vigged 1X2 and over/under 2.5 through the Dixon-Coles grid,
  * prices for derivative markets (totals, spreads, BTTS) from any (lam, mu).

Line sources (football-data.co.uk columns, fetched by fetch_fd.py):
  pre-match ("open")  : Pinnacle PSH/PSD/PSA + P>2.5/P<2.5, fallback market average AvgH.. / Avg>2.5
  closing ("close")   : Pinnacle PSCH/PSCD/PSCA + PC>2.5/PC<2.5, fallback AvgCH.. / AvgC>2.5
Pinnacle stops appearing in football-data partway through 2025-26 and is absent
in 2026-27, so the fallback is what the live season uses. Every row records which
source it used (open_src / close_src).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.stats import poisson

MAXG = 10
RHO_MKT = -0.05          # fixed low-score correction used when inverting market prices
_I, _J = np.indices((MAXG + 1, MAXG + 1))


# ---------------------------------------------------------------- de-vig
def shin(odds) -> np.ndarray:
    """Shin (1993) de-vig. odds: decimal odds for mutually exclusive outcomes.
    Corrects the favourite-longshot bias that proportional de-vig leaves in."""
    o = np.asarray(odds, float)
    if np.any(~np.isfinite(o)) or np.any(o <= 1.0):
        return np.full(len(o), np.nan)
    pi = 1.0 / o
    S = pi.sum()
    if S <= 1.0:                       # no margin (rare) -> proportional
        return pi / S

    def probs(z):
        return (np.sqrt(z * z + 4 * (1 - z) * pi * pi / S) - z) / (2 * (1 - z))

    try:
        z = brentq(lambda z: probs(z).sum() - 1.0, 0.0, 0.5)
    except ValueError:
        return pi / S
    p = probs(z)
    return p / p.sum()


def proportional(odds) -> np.ndarray:
    pi = 1.0 / np.asarray(odds, float)
    return pi / pi.sum()


# ---------------------------------------------------------------- goals grid
def grid(lam: float, mu: float, rho: float = RHO_MKT) -> np.ndarray:
    g = np.arange(MAXG + 1)
    G = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    G[0, 0] *= 1 - lam * mu * rho
    G[0, 1] *= 1 + lam * rho
    G[1, 0] *= 1 + mu * rho
    G[1, 1] *= 1 - rho
    return G / G.sum()


def grid_probs(G: np.ndarray) -> dict:
    d, t = _I - _J, _I + _J
    return {"H": float(G[d > 0].sum()), "D": float(G[d == 0].sum()), "A": float(G[d < 0].sum()),
            "O25": float(G[t > 2.5].sum())}


def implied_lambdas(pH: float, pD: float, pA: float, pO25: float | None,
                    rho: float = RHO_MKT, tot_prior: float = 2.7):
    """(lam, mu) whose Dixon-Coles grid best reproduces the fair market probabilities.
    If the totals price is missing, a weak prior on total goals replaces it."""
    target = np.array([pH, pD, pA])

    def resid(x):
        lam, mu = np.exp(x)
        p = grid_probs(grid(lam, mu, rho))
        r = [p["H"] - pH, p["D"] - pD, p["A"] - pA]
        if pO25 is not None and np.isfinite(pO25):
            r.append(2.0 * (p["O25"] - pO25))
        else:
            r.append(0.05 * (np.log(lam + mu) - np.log(tot_prior)))
        return np.array(r)

    if not np.all(np.isfinite(target)):
        return np.nan, np.nan
    sup = np.log(max(pH, 1e-3) / max(pA, 1e-3)) * 0.6
    x0 = np.array([np.log(1.35) + sup / 2, np.log(1.35) - sup / 2])
    res = least_squares(resid, x0, bounds=([np.log(0.05)] * 2, [np.log(6.0)] * 2))
    lam, mu = np.exp(res.x)
    return float(lam), float(mu)


# ---------------------------------------------------------------- derivative pricing
def price_derivatives(lam: float, mu: float, rho: float = RHO_MKT,
                      totals=(1.5, 2.5, 3.5), spreads=(-1.5, -0.5, 0.5, 1.5)) -> dict:
    """Fair probabilities for the markets the model prices itself (spreads, totals, BTTS).
    Spread lines are the home handicap at half-goal lines."""
    G = grid(lam, mu, rho)
    d, t = _I - _J, _I + _J
    out = grid_probs(G)
    out["totals"] = {str(L): {"over": float(G[t > L].sum()), "under": float(G[t < L].sum())} for L in totals}
    out["spreads"] = {str(h): float(G[d + h > 0].sum()) for h in spreads}   # P(home covers h)
    out["btts"] = float(G[1:, 1:].sum())
    out["lam_home"], out["lam_away"] = lam, mu
    return out


# ---------------------------------------------------------------- per-row market frame
def _pick(row, cols_primary, cols_fallback):
    a = [row.get(c) for c in cols_primary]
    if all(pd.notna(x) and x > 1 for x in a):
        return a, "pinnacle"
    b = [row.get(c) for c in cols_fallback]
    if all(pd.notna(x) and x > 1 for x in b):
        return b, "average"
    return None, None


def market_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Adds fair open/close probabilities and market-implied lambdas to a match frame."""
    recs = []
    for row in df.to_dict("records"):
        r = {}
        for tag, x12, xou in (
            ("open", (["PSH", "PSD", "PSA"], ["AvgH", "AvgD", "AvgA"]),
                     (["P>2.5", "P<2.5"], ["Avg>2.5", "Avg<2.5"])),
            ("close", (["PSCH", "PSCD", "PSCA"], ["AvgCH", "AvgCD", "AvgCA"]),
                      (["PC>2.5", "PC<2.5"], ["AvgC>2.5", "AvgC<2.5"])),
        ):
            o12, src = _pick(row, *x12)
            oou, src_ou = _pick(row, *xou)
            p = shin(o12) if o12 else np.full(3, np.nan)
            po = shin(oou)[0] if oou else np.nan
            r[f"{tag}_src"] = src
            r[f"{tag}_H"], r[f"{tag}_D"], r[f"{tag}_A"] = p
            r[f"{tag}_O25"] = po
            if o12:
                r[f"{tag}_oH"], r[f"{tag}_oD"], r[f"{tag}_oA"] = o12
            if oou:
                r[f"{tag}_oO"], r[f"{tag}_oU"] = oou
        lam, mu = implied_lambdas(r["open_H"], r["open_D"], r["open_A"], r["open_O25"])
        r["mkt_lam"], r["mkt_mu"] = lam, mu
        recs.append(r)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
