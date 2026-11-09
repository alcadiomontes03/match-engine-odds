"""Tests for pipeline/src/guards.py and where it is wired in (2026-11-09).

Run from the repo root:  python3 -m pytest -q pipeline/test_guards.py

Covers the three audit findings the guards exist for:
  F3  negative Dixon-Coles cells (v4's rho = -0.48)
  F4  an unrated team written as mu = 0 (Nott'm Forest v Coventry, 2026-09-19)
  F12 a refit run before the weekend's results were in (Nov 2)
and checks that no live prediction from weeks 6-10 would have changed.
"""
import csv, json, math, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "v6"))
sys.path.insert(0, str(HERE / "v7"))
from src import dixon_coles, markets, guards  # noqa: E402
from src.guards import GuardError  # noqa: E402
import refit as R  # noqa: E402

STORED = HERE / "testdata" / "stored_predictions_2026-09-17_to_11-08.csv"


def x1x2(G):
    return {"home": float(np.tril(G, -1).sum()), "draw": float(np.trace(G)), "away": float(np.triu(G, 1).sum())}


def raw_grid(lam, mu, rho, n=10):
    """The pre-guard grid, for comparison."""
    from scipy.stats import poisson
    g = np.arange(n + 1)
    G = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    G[0, 0] *= 1 - lam * mu * rho; G[0, 1] *= 1 + lam * rho; G[1, 0] *= 1 + mu * rho; G[1, 1] *= 1 - rho
    return G / G.sum()


# ------------------------------------------------------------------ F3
V4_NEGATIVE = [  # (match, lam, mu, rho) as stored for week 3
    ("Brentford v Chelsea", 2.438, 2.525, -0.4844939349428375),
    ("Everton v Ipswich", 2.243, 1.174, -0.4844939349428375),
    ("Sevilla v Barcelona", 1.15, 3.403, -0.46280912956290593),
    ("Villarreal v Levante", 2.243, 1.669, -0.46280912956290593),
]


@pytest.mark.parametrize("name,lam,mu,rho", V4_NEGATIVE)
def test_v4_cases_had_negative_cells_and_now_do_not(name, lam, mu, rho):
    assert raw_grid(lam, mu, rho).min() < 0, f"{name}: expected the old grid to be broken"
    for G in (dixon_coles.scoreline_matrix(lam, mu, rho), R.grid(lam, mu, rho)):
        assert G.min() >= 0 and abs(G.sum() - 1) < 1e-9


def test_rho_bounds_are_the_exact_nonnegative_region():
    lam, mu = 2.4, 1.6
    lo, hi = guards.rho_bounds(lam, mu)
    assert raw_grid(lam, mu, lo).min() >= -1e-15
    assert raw_grid(lam, mu, lo - 0.01).min() < 0
    assert raw_grid(lam, mu, hi).min() >= -1e-15


def test_check_grid_rejects_bad_grids():
    G = dixon_coles.scoreline_matrix(1.4, 1.1, -0.08)
    bad = G.copy(); bad[0, 1] = -0.01; bad[0, 0] += 0.01
    with pytest.raises(GuardError):
        guards.check_grid(bad)
    with pytest.raises(GuardError):
        guards.check_grid(G * 1.1)
    nan = G.copy(); nan[3, 3] = np.nan
    with pytest.raises(GuardError):
        guards.check_grid(nan)


def test_build_prediction_refuses_a_broken_grid():
    G = raw_grid(2.438, 2.525, -0.4844939349428375)   # negative cell, as in week 3
    with pytest.raises(GuardError):
        markets.build_prediction(G, x1x2(G))
    good = dixon_coles.scoreline_matrix(1.4, 1.1, -0.08)
    pred = markets.build_prediction(good, x1x2(good), {"home": 2.1, "draw": 3.4, "away": 3.6})
    assert abs(sum(pred["market_devig"].values()) - 1) < 1e-9


# ------------------------------------------------------------------ F4
@pytest.mark.parametrize("lam,mu", [(1.342, 0.0), (0.0, 1.2), (float("nan"), 1.0), (1.2, 25.0)])
def test_broken_rates_are_refused(lam, mu):
    with pytest.raises(GuardError):
        dixon_coles.scoreline_matrix(lam, mu, -0.08)
    with pytest.raises(GuardError):
        R.grid(lam, mu, -0.08)


def test_unrated_team_is_refused_not_averaged():
    att = {"Forest": 0.10, "Arsenal": 0.40}
    dfn = {"Forest": 0.05, "Arsenal": 0.35}
    with pytest.raises(GuardError, match="Coventry"):
        guards.match_rates(att, dfn, 0.17, "Forest", "Coventry")
    lam, mu = guards.match_rates(att, dfn, 0.17, "Forest", "Arsenal")
    assert math.isclose(lam, math.exp(0.17 + 0.10 - 0.35)) and math.isclose(mu, math.exp(0.40 - 0.05))


# ------------------------------------------------------------------ no live prediction changes
def test_every_stored_live_prediction_is_unchanged_by_the_guards():
    rows = list(csv.DictReader(open(STORED)))
    live = [r for r in rows if r["model"] != "v4"]
    assert len(live) == 186
    moved = stored_gap = 0.0
    for r in live:
        lam, mu, rho = float(r["lambda"]), float(r["mu"]), float(r["rho"])
        p = x1x2(dixon_coles.scoreline_matrix(lam, mu, rho))
        old = x1x2(raw_grid(lam, mu, rho))
        for k, col in (("home", "p_home"), ("draw", "p_draw"), ("away", "p_away")):
            moved = max(moved, abs(p[k] - old[k]))
            stored_gap = max(stored_gap, abs(p[k] - float(r[col])))
    assert moved == 0.0, f"the guards moved a live prediction by {moved}"
    # and the stored numbers still reproduce (lambda/mu are stored to 3 decimals, hence the small gap)
    assert stored_gap < 1e-3, f"stored prediction not reproducible: {stored_gap}"


def test_only_week_3_predictions_are_affected():
    rows = list(csv.DictReader(open(STORED)))
    changed = []
    for r in rows:
        lam, mu, rho = float(r["lambda"]), float(r["mu"]), float(r["rho"])
        if not (guards.MIN_RATE <= lam <= guards.MAX_RATE and guards.MIN_RATE <= mu <= guards.MAX_RATE):
            changed.append((r["fixture_id"], "refused"))
        elif guards.safe_rho(lam, mu, rho) != rho:
            changed.append((r["fixture_id"], "rho clamped"))
    assert len(changed) == 5 and all(m == "v4" for m in
                                     (next(x["model"] for x in rows if x["fixture_id"] == f) for f, _ in changed))


# ------------------------------------------------------------------ F12
def _db(tmp, fixtures, results):
    (tmp / "fixtures").mkdir(); (tmp / "results").mkdir()
    for f in fixtures:
        (tmp / "fixtures" / f"{f['id']}.json").write_text(json.dumps(f))
    for fid, r in results.items():
        (tmp / "results" / f"{fid}.json").write_text(json.dumps(r))
    return tmp


def test_missing_results_finds_played_games_without_scores(tmp_path):
    fx = [
        {"id": "a", "league": "E0", "home": "Newcastle", "away": "Everton", "kickoff": "2026-11-02T20:00:00+00:00"},
        {"id": "b", "league": "E0", "home": "Leeds", "away": "Spurs", "kickoff": "2026-11-01T15:00:00+00:00"},
        {"id": "c", "league": "E0", "home": "Fulham", "away": "Hull", "kickoff": "2026-11-01T15:00:00+00:00",
         "status": "postponed"},
        {"id": "d", "league": "SP1", "home": "Betis", "away": "Getafe", "kickoff": "2026-11-01T15:00:00+00:00"},
        {"id": "e", "league": "E0", "home": "Brighton", "away": "Wolves", "kickoff": "2026-11-09T19:00:00+00:00"},
        {"id": "f", "league": "E0", "home": "Chelsea", "away": "Burnley", "kickoff": "2026-09-01T15:00:00+00:00"},
    ]
    db = _db(tmp_path, fx, {"b": {"home_score": 1, "away_score": 0, "actual": "H"}})
    miss = guards.missing_results(db, "E0", "2026-11-02", grace_hours=0)
    # a: played, no result -> missing. b: has result. c: postponed. d: other league.
    # e: not played by the as-of date. f: older than 21 days.
    assert miss == ["Newcastle v Everton (2026-11-02)"]


def test_refit_checks_fail_a_league_with_missing_results():
    import live as L
    r = {"converged": True, "home_adv": 0.17, "rho": -0.09, "att": {"A": 0.1, "B": -0.1}, "def": {"A": 0.0, "B": 0.0}}
    df = pd.DataFrame({"Date": pd.to_datetime(["2026-10-01"]), "HomeTeam": ["A"], "AwayTeam": ["B"],
                       "FTHG": [1], "FTAG": [1]})
    ok, ck = L.checks(df, r, 1.0, None, None, [("A", "B")], missing=["A v B (2026-11-02)"])
    assert not ok and any(c[0].startswith("results complete") and not c[1] for c in ck)
    ok, _ = L.checks(df, r, 1.0, None, None, [("A", "B")], missing=[])
    assert ok


def test_postponed_but_played_flags_psg_lyon(tmp_path):
    fx = [{"id": "x", "league": "F1", "home": "Paris SG", "away": "Lyon", "kickoff": "2026-10-25T19:45:00+00:00",
           "status": "postponed"},
          {"id": "y", "league": "F1", "home": "Lyon", "away": "Paris SG", "kickoff": "2026-10-25T19:45:00+00:00",
           "status": "postponed"}]
    db = _db(tmp_path, fx, {})
    hits = guards.postponed_but_played(db, "F1", HERE.parent / "data" / "fd")
    assert len(hits) == 1 and hits[0].startswith("Paris SG v Lyon") and "(2-0)" in hits[0]
