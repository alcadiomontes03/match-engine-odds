"""Alpha Lines v9 method: the folded stretch, the extras de-duplication and the shadow-file families."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline" / "v9"))
sys.path.insert(0, str(ROOT / "pipeline" / "v7"))
import method as V9  # noqa: E402
import live as L  # noqa: E402
from fixes import R, stretch  # noqa: E402


def _r():
    rng = np.random.default_rng(3)
    teams = [f"T{i}" for i in range(8)]
    return {"teams": teams, "att": dict(zip(teams, rng.normal(0, 0.3, 8))), "def": dict(zip(teams, rng.normal(-0.3, 0.2, 8))),
            "home_adv": 0.22, "rho": -0.05}


def test_fold_stretch_matches_the_backtest_stretch():
    r = _r(); rf = V9.fold_stretch(r, 1.13)
    for h, a in [("T0", "T1"), ("T5", "T2"), ("T7", "T3")]:
        lam, mu = R.lambdas(r, h, a)
        l2, m2 = stretch(np.array([lam]), np.array([mu]), r["home_adv"], 1.13)
        l3, m3 = R.lambdas(rf, h, a)
        assert abs(l2[0] - l3) < 1e-12 and abs(m2[0] - m3) < 1e-12
        assert abs(np.log(lam * mu) - np.log(l3 * m3)) < 1e-12      # total (in logs) unchanged


def test_fold_stretch_one_is_identity_for_every_match():
    r = _r(); rf = V9.fold_stretch(r, 1.0)
    assert np.allclose(R.lambdas(r, "T0", "T4"), R.lambdas(rf, "T0", "T4"))


def test_new_extras_drops_rows_football_data_already_has():
    d = pd.Timestamp("2026-11-08")
    df = pd.DataFrame({"Date": [d], "HomeTeam": ["A"], "AwayTeam": ["B"]})
    ex = pd.DataFrame({"Date": [d, d], "HomeTeam": ["A", "C"], "AwayTeam": ["B", "D"], "FTHG": [1, 2], "FTAG": [0, 2]})
    out = V9.new_extras(df, ex)
    assert list(out.HomeTeam) == ["C"]


def test_shadow_file_families():
    assert L.family_of("shadow_2026-11-02.csv") == "v8"          # written by the v7/v8 method
    assert L.family_of("shadow_2026-11-16.csv") == "v9"
    assert L.family_of("shadow_v8_2026-11-16.csv") == "v8"
    assert L.family_of("shadow_v61_2026-10-20.csv") == "v61"


def test_chosen_method_is_frozen():
    assert V9.CHOSEN == dict(wg=0.0, ws=0.0, wm=1.0, xi=0.010, P=5.0, use_nnls=True) and V9.S == 1.13
