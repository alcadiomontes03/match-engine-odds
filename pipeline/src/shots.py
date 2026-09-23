"""
Match Engine v4.7 — shot-rate model for the dashboard's "Goal attempts" card.

Writes the optional `shots` field on predictions/<odds_event_id>, in the shape the
dashboard already reads (see claude/match-engine-beta-spec.md):

    shots = {"home": 13.1, "away": 10.4,            # expected shot attempts
             "home_sot": 4.6, "away_sot": 3.5,      # expected shots on target
             "totals": {"23.5": {"over": .., "under": ..}, ...},   # match attempts
             # extras (not yet shown by the UI, kept because this is where the
             # backtested signal is — see shots4.py RESULTS):
             "team_sot": {"home": {"line": 4.5, "over": .., "under": ..}, "away": {...}},
             "model": "v4.7-nb-shots", "k_attempts": .., "k_sot": ..}

Method (the same one price4.py / shots4.py validated, now callable from the refit):
  * two Poisson rating fits with dc4.fit on shot counts instead of goals — one on
    attempts (HS/AS), one on shots on target (HST/AST) — rho locked at 0 because the
    Dixon-Coles low-score correction is for goals, not shots;
  * team counts are overdispersed vs Poisson, so each side is a negative binomial
    with the fitted mean and a dispersion k estimated by method of moments over the
    last ~2 seasons (in-sample residuals);
  * the match-total distribution is the exact convolution of the two team NBs.

Honest status (from shots4.py, test seasons 2023/24-2025/26): team-level shots on
target beats the base rate clearly (home SOT o3.5 log loss 0.570 vs 0.623 PL); match
attempt totals only barely (o24.5 0.677 vs 0.683 PL). There is no free bookmaker
history for shot markets, so none of this has been checked against a market price:
predictions only, never bets.

Inputs: a football-data.co.uk frame with Date, HomeTeam, AwayTeam, HS, AS, HST, AST
(loader.load_league already keeps those columns; they are simply unused by the goals fit).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import nbinom

try:
    from src import dc4
except ImportError:  # running from inside src/
    import dc4

MODEL = "v4.7-nb-shots"
MAX_SHOTS = 70          # support for the per-team pmf (P(X > 70) is negligible)
DISPERSION_DAYS = 700   # window for the NB dispersion estimate (~2 seasons)
K_CAP = 1e4             # "no overdispersion" -> effectively Poisson


def _as_counts(df, h, a):
    d = df[["Date", "HomeTeam", "AwayTeam"]].copy()
    d["FTHG"] = pd.to_numeric(df[h], errors="coerce")
    d["FTAG"] = pd.to_numeric(df[a], errors="coerce")
    return d.dropna(subset=["FTHG", "FTAG"])


def _nb_size(y, m):
    """Method-of-moments NB size k (var = m + m^2/k); large k = Poisson-like."""
    excess = float(np.mean((y - m) ** 2 - m))
    return float(min(np.mean(m ** 2) / excess, K_CAP)) if excess > 0 else K_CAP


def _pmf(mean, k):
    x = np.arange(MAX_SHOTS + 1)
    p = nbinom.pmf(x, k, k / (k + mean))
    return p / p.sum()


def _over(pmf, line):
    return float(pmf[np.arange(len(pmf)) > line].sum())


def fit(df: pd.DataFrame, as_of, xi: float, priors: dict | None = None, lam_prior: float = 0.0):
    """Fit attempts + on-target rating models as of `as_of` (only rows before it are used).
    `priors` = {team: (att, def)} on the SHOT scale is optional; newcomers without one
    fall back to the league mean (dc4.lambdas fallback). Returns a model dict, or None
    if the frame has no shot columns."""
    if not {"HS", "AS", "HST", "AST"}.issubset(df.columns):
        return None
    as_of = pd.Timestamp(as_of)
    att = _as_counts(df, "HS", "AS")
    sot = _as_counts(df, "HST", "AST")
    if len(att) < 30:
        return None
    ra = dc4.fit(att, as_of, xi, priors=priors, lam_prior=lam_prior, rho_bounds=(0.0, 0.0))
    rs = dc4.fit(sot, as_of, xi, priors=priors, lam_prior=lam_prior, rho_bounds=(0.0, 0.0))
    out = {"attempts": ra, "sot": rs, "as_of": str(as_of.date()), "xi": xi}
    for key, frame, r in (("k_attempts", att, ra), ("k_sot", sot, rs)):
        rec = frame[(frame.Date < as_of) & (frame.Date >= as_of - pd.Timedelta(days=DISPERSION_DAYS))]
        lh, la = dc4.lambdas(r, rec.HomeTeam.to_numpy(), rec.AwayTeam.to_numpy(), fallback=priors)
        y = np.concatenate([rec.FTHG.to_numpy(float), rec.FTAG.to_numpy(float)])
        out[key] = _nb_size(y, np.concatenate([lh, la]))
    return out


def predict(model: dict, home: str, away: str, fallback: dict | None = None, n_lines: int = 3):
    """The `shots` field for one fixture. Total-attempts lines are the half-integer line
    nearest the expected total plus neighbours 2 shots either side (n_lines=3)."""
    if not model:
        return None
    (h_att,), (a_att,) = dc4.lambdas(model["attempts"], [home], [away], fallback=fallback)
    (h_sot,), (a_sot,) = dc4.lambdas(model["sot"], [home], [away], fallback=fallback)
    ka, ks = model["k_attempts"], model["k_sot"]
    ph, pa = _pmf(h_att, ka), _pmf(a_att, ka)
    total = np.convolve(ph, pa)                       # exact sum of the two team NBs
    main = np.floor(h_att + a_att) + 0.5
    half = n_lines // 2
    totals = {}
    for L in [main + 2 * i for i in range(-half, half + 1)]:
        over = _over(total, L)
        totals[f"{L:.1f}"] = {"over": over, "under": 1.0 - over}
    team_sot = {}
    for side, m in (("home", h_sot), ("away", a_sot)):
        L = float(np.floor(m) + 0.5)
        over = _over(_pmf(m, ks), L)
        team_sot[side] = {"line": L, "over": over, "under": 1.0 - over}
    return {"home": float(h_att), "away": float(a_att),
            "home_sot": float(h_sot), "away_sot": float(a_sot),
            "totals": totals, "team_sot": team_sot,
            "model": MODEL, "k_attempts": round(ka, 2), "k_sot": round(ks, 2)}
