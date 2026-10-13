"""Tests for pipeline/v6/benchmark.py. Run: python pipeline/test_benchmark.py"""
import sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent / "v6"))
import benchmark as B


def test_edge_and_kelly():
    e = B.ref_edge(2.20, 0.50)                      # fair price 2.00, offered 2.20
    assert abs(e["ev"] - 0.10) < 1e-12
    assert abs(e["prob_edge"] - (0.50 - 1 / 2.2)) < 1e-12
    assert abs(e["kelly_q"] - 0.25 * 0.10 / 1.20) < 1e-12
    n = B.ref_edge(1.80, 0.50)                      # negative edge -> no stake
    assert n["ev"] < 0 and n["kelly_q"] == 0
    assert np.isnan(B.ref_edge(np.nan, 0.5)["ev"]) and np.isnan(B.ref_edge(1.0, 0.5)["ev"])


def test_scores():
    assert abs(B.brier_1x2([.5, .3, .2], 0) - (.25 + .09 + .04)) < 1e-12
    assert abs(B.logloss([.5, .3, .2], 1) + np.log(.3)) < 1e-12


def test_join_and_clv():
    fd = B.load_fd()
    s = fd[(fd.lg == "E0") & (fd.season == "2425")].dropna(subset=["Pinnacle_close_H"]).head(30)
    alias = {"Man City": "Manchester City", "Man United": "Manchester United", "Nott'm Forest": "Nottingham Forest"}
    rows = []
    for r in s.itertuples():
        p = np.array([r.Pinnacle_close_H, r.Pinnacle_close_D, r.Pinnacle_close_A])
        o = 1 / (p * 1.05)                          # a book with a flat 5% margin
        rows.append(dict(fixture_id=f"f{r.Index}", league="E0", kickoff=r.Date.strftime("%Y-%m-%dT15:00:00+00:00"),
                         home=alias.get(r.HomeTeam, r.HomeTeam), away=alias.get(r.AwayTeam, r.AwayTeam),
                         ml_home=o[0], ml_draw=o[1], ml_away=o[2], home_goals=r.FTHG, away_goals=r.FTAG))
    with tempfile.TemporaryDirectory() as d:
        pth = Path(d) / "lines.csv"; pd.DataFrame(rows).to_csv(pth, index=False)
        t = B.clv_table(pth)
    assert t[["home", "kickoff"]].drop_duplicates().shape[0] == 30          # every fixture joined
    assert (t.ref_src == "pinnacle_close").all()
    assert np.allclose(t.ev, 1 / 1.05 - 1, atol=1e-9)                       # 5% margin -> -4.76% everywhere
    assert not t.flagged.any()
    assert abs(t.groupby(["home", "kickoff"]).ref_p.sum() - 1).max() < 1e-9


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
