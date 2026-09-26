"""
Pulsematch v5 — adjustment model.

The ML no longer predicts matches from scratch. It learns small, regularised
corrections to the market price:

  1X2:     logit_k = log p_market_k + b_k . x + c_k   (k = home, away; draw is the reference)
  totals:  logit(p_over) = logit(p_market_over) + b . x + c

x are the gap features (features.py), standardised on the training set. Heavy L2
shrinkage on b means that with no real signal the model returns the market price
exactly; the penalty is picked on the most recent training season (time-ordered
validation, never random folds).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, log_softmax

LAMBDAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def _std(X, mu=None, sd=None):
    if mu is None:
        mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
    Z = np.nan_to_num((X - mu) / sd)          # missing feature -> 0 = "no information"
    return np.clip(Z, -4, 4), mu, sd


# ------------------------------------------------------------------ 1X2
def _nll_1x2(theta, Z, logp, y, lam):
    k = Z.shape[1]
    bH, cH, bA, cA = theta[:k], theta[k], theta[k + 1:2 * k + 1], theta[2 * k + 1]
    L = logp.copy()
    L[:, 0] += Z @ bH + cH
    L[:, 2] += Z @ bA + cA
    lp = log_softmax(L, axis=1)
    # intercepts are penalised too: an unpenalised intercept learned a small draw
    # correction that produced ~900 paper bets with -2% CLV in the first backtest
    f = -lp[np.arange(len(y)), y].sum() + lam * (bH @ bH + bA @ bA + cH * cH + cA * cA)
    P = np.exp(lp)
    R = P.copy(); R[np.arange(len(y)), y] -= 1.0           # dNLL/dlogit
    g = np.concatenate([Z.T @ R[:, 0] + 2 * lam * bH, [R[:, 0].sum() + 2 * lam * cH],
                        Z.T @ R[:, 2] + 2 * lam * bA, [R[:, 2].sum() + 2 * lam * cA]])
    return f, g


def predict_1x2(theta, Z, logp):
    k = Z.shape[1]
    L = logp.copy()
    L[:, 0] += Z @ theta[:k] + theta[k]
    L[:, 2] += Z @ theta[k + 1:2 * k + 1] + theta[2 * k + 1]
    return np.exp(log_softmax(L, axis=1))


# ------------------------------------------------------------------ totals
def _nll_ou(theta, Z, off, y, lam):
    b, c = theta[:-1], theta[-1]
    s = off + Z @ b + c
    p = expit(s)
    f = -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).sum() + lam * (b @ b + c * c)
    r = p - y
    return f, np.concatenate([Z.T @ r + 2 * lam * b, [r.sum() + 2 * lam * c]])


def predict_ou(theta, Z, off):
    return expit(off + Z @ theta[:-1] + theta[-1])


# ------------------------------------------------------------------ wrapper
@dataclass
class AdjustModel:
    kind: str                     # "1x2" or "ou"
    features: list
    lam: float = None
    theta: np.ndarray = None
    mu: np.ndarray = None
    sd: np.ndarray = None
    val_curve: dict = field(default_factory=dict)

    def _prep(self, df, fit_scaler=False):
        X = df[self.features].to_numpy(float)
        if fit_scaler:
            Z, self.mu, self.sd = _std(X)
        else:
            Z, _, _ = _std(X, self.mu, self.sd)
        if self.kind == "1x2":
            off = np.log(np.clip(df[["open_H", "open_D", "open_A"]].to_numpy(float), 1e-6, 1))
            y = df["y"].to_numpy(int) if "y" in df else None
        else:
            off = logit(np.clip(df["open_O25"].to_numpy(float), 1e-6, 1 - 1e-6))
            y = df["y_over"].to_numpy(float) if "y_over" in df else None
        return Z, off, y

    def _fit_once(self, Z, off, y, lam):
        k = Z.shape[1]
        if self.kind == "1x2":
            res = minimize(_nll_1x2, np.zeros(2 * k + 2), args=(Z, off, y, lam), jac=True, method="L-BFGS-B")
        else:
            res = minimize(_nll_ou, np.zeros(k + 1), args=(Z, off, y, lam), jac=True, method="L-BFGS-B")
        return res.x

    def _ll(self, theta, Z, off, y, per_row=False):
        if self.kind == "1x2":
            P = predict_1x2(theta, Z, off)
            l = -np.log(P[np.arange(len(y)), y])
        else:
            p = predict_ou(theta, Z, off)
            l = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        return l if per_row else float(l.mean())

    def fit(self, train, val_season=None, n_val: int = 2):
        """train: rows with outcomes, time-ordered seasons.

        Penalty selection (rolling origin, one-standard-error rule): for each of the last
        `n_val` seasons, fit on the seasons before it and score it. Keep the HEAVIEST
        penalty whose pooled validation loss is within one standard error of the best,
        so the model only departs from the market price when the evidence is clear.
        (A single validation season let the totals model pick almost no shrinkage from
        noise and lose 2% CLV in the first backtest.) Then refit on everything."""
        seasons = sorted(train.Season.unique())
        val_seasons = [s for s in seasons[1:]][-n_val:]
        if val_season is not None and val_seasons:
            per_lam = {lam: [] for lam in LAMBDAS}
            for vs in val_seasons:
                tr, va = train[train.Season < vs], train[train.Season == vs]
                Z, off, y = self._prep(tr, fit_scaler=True)
                Zv, offv, yv = self._prep(va)
                for lam in LAMBDAS:
                    per_lam[lam].append(self._ll(self._fit_once(Z, off, y, lam), Zv, offv, yv, per_row=True))
            losses = {lam: np.concatenate(v) for lam, v in per_lam.items()}
            self.val_curve = {lam: float(l.mean()) for lam, l in losses.items()}
            best = min(self.val_curve, key=self.val_curve.get)
            se = {lam: float((losses[lam] - losses[best]).std(ddof=1) / np.sqrt(len(losses[best])))
                  for lam in LAMBDAS}
            self.lam = max(lam for lam in LAMBDAS if self.val_curve[lam] <= self.val_curve[best] + se[lam])
        else:
            self.lam = self.lam or max(LAMBDAS)
        Z, off, y = self._prep(train, fit_scaler=True)
        self.theta = self._fit_once(Z, off, y, self.lam)
        return self

    def predict(self, df):
        Z, off, _ = self._prep(df)
        return predict_1x2(self.theta, Z, off) if self.kind == "1x2" else predict_ou(self.theta, Z, off)

    def coefficients(self):
        k = len(self.features)
        if self.kind == "1x2":
            return {"home": dict(zip(self.features, np.round(self.theta[:k], 4))), "home_c": round(self.theta[k], 4),
                    "away": dict(zip(self.features, np.round(self.theta[k + 1:2 * k + 1], 4))),
                    "away_c": round(self.theta[2 * k + 1], 4), "lambda": self.lam}
        return {"over": dict(zip(self.features, np.round(self.theta[:k], 4))), "c": round(self.theta[k], 4),
                "lambda": self.lam}
