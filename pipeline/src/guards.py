"""
Alpha Lines guards (2026-11-09, after the prediction-record audit and Codex's review).

Three things the audit found could reach a stored prediction or a live refit without anything stopping them:

  F3  A Dixon-Coles rho that makes a low-score cell negative. tau(0,1) = 1 + lam*rho goes below zero once
      lam > -1/rho; v4 fitted rho = -0.48, so four week-3 predictions had negative score probabilities.
  F4  An unrated team. v4 wrote Nott'm Forest v Coventry with mu = 0, i.e. P(away win) = 0; Coventry won 1-0.
  F12 A refit on incomplete results. On Nov 2 the Monday refit ran before week 9's results were in, so the week-10
      predictions were built on ratings that had not seen a single week-9 game.

Each guard either repairs something provably harmless (clamping rho into the range where every cell is >= 0;
no live prediction from weeks 6-10 changes, see test_guards.py) or raises GuardError so the job stops and reports
instead of writing a bad number. Nothing here changes a price: the blend weight is still 0.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

MIN_RATE = 0.05     # expected goals below this is a broken input, not a forecast (league minimum is ~0.3)
MAX_RATE = 8.0      # and above this
GRID_TOL = 1e-6


class GuardError(ValueError):
    """A prediction or refit input that must not be used."""


# ---------------------------------------------------------------- F4: rates and ratings
def check_rates(lam: float, mu: float, where: str = "") -> None:
    for name, v in (("lambda (home)", lam), ("mu (away)", mu)):
        if v is None or not math.isfinite(v) or not (MIN_RATE <= v <= MAX_RATE):
            raise GuardError(f"{where}{name} = {v!r} is outside {MIN_RATE}-{MAX_RATE}: "
                             "a team is probably unrated or its rating is broken")


def require_rated(attack: dict, defense: dict, home: str, away: str) -> None:
    missing = [t for t in (home, away) if t not in attack or t not in defense
               or attack[t] is None or defense[t] is None]
    if missing:
        raise GuardError(f"no rating for {', '.join(missing)}: refusing to write a prediction "
                         "(add the team's name mapping or wait for the next refit)")


def match_rates(attack: dict, defense: dict, home_adv: float, home: str, away: str) -> tuple[float, float]:
    """lam = exp(HA + att[home] - def[away]), mu = exp(att[away] - def[home]), refusing unrated teams.
    The strict counterpart of pipeline/v6/refit.lambdas, which falls back to the league mean on purpose
    (walk-forward backtests meet newly promoted teams) and must not be used to write live predictions."""
    require_rated(attack, defense, home, away)
    lam = math.exp(home_adv + attack[home] - defense[away])
    mu = math.exp(attack[away] - defense[home])
    check_rates(lam, mu, f"{home} v {away}: ")
    return lam, mu


# ---------------------------------------------------------------- F3: valid Dixon-Coles cells
def rho_bounds(lam: float, mu: float) -> tuple[float, float]:
    """The rho interval in which all four tau factors are >= 0:
    tau(0,0) = 1 - lam*mu*rho, tau(0,1) = 1 + lam*rho, tau(1,0) = 1 + mu*rho, tau(1,1) = 1 - rho."""
    lo = max(-1.0 / lam, -1.0 / mu)
    hi = min(1.0 / (lam * mu), 1.0)
    return lo, hi


def safe_rho(lam: float, mu: float, rho: float) -> float:
    lo, hi = rho_bounds(lam, mu)
    return float(min(max(rho, lo), hi))


def check_grid(G: np.ndarray, where: str = "") -> None:
    if not np.all(np.isfinite(G)):
        raise GuardError(f"{where}scoreline grid has NaN or infinite cells")
    if G.min() < -GRID_TOL:
        raise GuardError(f"{where}scoreline grid has a negative cell ({G.min():.3g})")
    s = float(G.sum())
    if abs(s - 1.0) > 1e-4:
        raise GuardError(f"{where}scoreline grid sums to {s:.6f}, not 1")


def check_probs(p: dict, keys=("home", "draw", "away"), where: str = "") -> None:
    vals = [p.get(k) for k in keys]
    if any(v is None or not math.isfinite(v) or v < -GRID_TOL or v > 1 + GRID_TOL for v in vals):
        raise GuardError(f"{where}probabilities out of range: {p}")
    if abs(sum(vals) - 1.0) > 1e-4:
        raise GuardError(f"{where}probabilities sum to {sum(vals):.6f}, not 1: {p}")


# ---------------------------------------------------------------- F12: complete results before a refit
def missing_results(db: Path, code: str, asof: str, lookback_days: int = 21, grace_hours: float = 3.0) -> list[str]:
    """League fixtures that kicked off between `lookback_days` before `asof` and `grace_hours` before the end of
    `asof` (UTC), are not marked postponed, and have no result with a score. db is an ArtifactData out_dir
    holding fixtures/ and results/. Returns "Home v Away (YYYY-MM-DD)" strings."""
    import pandas as pd
    db = Path(db)
    end = pd.Timestamp(asof).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(hours=grace_hours)
    now = pd.Timestamp.now("UTC") - pd.Timedelta(hours=grace_hours)
    end = min(end, now)
    start = end - pd.Timedelta(days=lookback_days)
    out = []
    for f in sorted((db / "fixtures").glob("*.json")):
        fx = json.loads(f.read_text())
        fx = fx.get("data", fx) if "id" not in fx else fx
        if fx.get("league") != code or fx.get("status") == "postponed":
            continue
        ko = pd.Timestamp(fx["kickoff"])
        ko = ko.tz_localize("UTC") if ko.tzinfo is None else ko
        if not (start <= ko <= end):
            continue
        rp = db / "results" / f.name
        r = json.loads(rp.read_text()) if rp.exists() else {}
        r = r.get("data", r) if "home_score" not in r else r
        if "home_score" not in r or r.get("home_score") is None:
            out.append(f"{fx.get('home')} v {fx.get('away')} ({str(fx['kickoff'])[:10]})")
    return out


def postponed_but_played(db: Path, code: str, fd_dir: Path) -> list[str]:
    """Fixtures marked postponed that football-data shows as played (same teams, within 7 days of the stored
    kickoff). The status stops every job from writing a result, so the match silently drops out of grading
    and the refit's results check. Found 2026-11-09: PSG v Lyon was moved one day and played, but stayed
    'postponed' for two weeks."""
    import pandas as pd
    db, fd_dir = Path(db), Path(fd_dir)
    out, cache = [], {}
    for f in sorted((db / "fixtures").glob("*.json")):
        fx = json.loads(f.read_text())
        fx = fx.get("data", fx) if "id" not in fx else fx
        if fx.get("league") != code or fx.get("status") != "postponed":
            continue
        ko = pd.Timestamp(fx["kickoff"])
        yy = ko.year % 100 if ko.month >= 7 else ko.year % 100 - 1
        p = fd_dir / f"{code}_{yy:02d}{yy + 1:02d}.csv"
        if p not in cache:
            cache[p] = pd.read_csv(p) if p.exists() else None
        d = cache[p]
        if d is None:
            continue
        m = d[(d.HomeTeam == fx.get("home")) & (d.AwayTeam == fx.get("away"))]
        dates = pd.to_datetime(m.Date, dayfirst=True)
        k = ko.tz_localize(None) if ko.tzinfo else ko
        hit = m[(abs((dates - k.normalize()).dt.days) <= 7).to_numpy() & m.FTHG.notna().to_numpy()]
        if len(hit):
            r = hit.iloc[0]
            out.append(f"{fx.get('home')} v {fx.get('away')} marked postponed but played {r.Date} "
                       f"({int(r.FTHG)}-{int(r.FTAG)})")
    return out
