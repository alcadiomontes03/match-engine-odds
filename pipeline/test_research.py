"""Tests for the research-5 fit columns in pipeline/v6/research.py.
Run: python3 -m pytest -q pipeline/test_research.py"""
import csv, json, sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "v6"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import research as R
from v5 import market as M

ML = {"home": 1.62, "draw": 4.2, "away": 5.25, "pulled_at": "2026-11-21T13:00:00+00:00"}
SIDE_TOTAL = {"pulled_at": "2026-11-21T13:00:00+00:00",
              "btts": {"yes": 1.8, "no": 1.95},
              "totals": {"1.5": {"over": 1.25, "under": 3.9}, "2.5": {"over": 1.8, "under": 2.0},
                         "3.0": {"over": 2.3, "under": 1.6}, "3.5": {"over": 2.9, "under": 1.4},
                         "2.75": {"over": 2.05, "under": 1.78}},
              "spreads": {"-1.5": {"home": 2.6, "away": 1.5}, "-0.5": {"home": 1.62, "away": 2.3},
                          "-1.0": {"home": 2.05, "away": 1.8}},
              "corners": {"9.5": {"over": 1.9, "under": 1.9}}}


def _db(tmp_path, side):
    fid = "f1"
    docs = {"fixtures": {"league": "E0", "kickoff": "2026-11-21T15:00:00+00:00", "home": "Arsenal",
                         "away": "Everton", "status": "final"},
            "results": {"home_score": 2, "away_score": 1, "stats": {"h": {"corners": 6}, "a": {"corners": 4}}},
            "odds": {"latest": ML, "side": side},
            "predictions": {"xg_home": 1.7, "xg_away": 0.9}}
    for coll, d in docs.items():
        (tmp_path / coll).mkdir()
        (tmp_path / coll / f"{fid}.json").write_text(json.dumps(d))
    return tmp_path


def _rebuild(row):
    """model_p_a from the four stored fit columns alone."""
    G = M.grid(float(row["fit_lam"]), float(row["fit_mu"]), float(row["fit_rho"]))
    return R.p_side_a(G, row["market"], row["line"])


@pytest.mark.parametrize("side,anchor", [
    (SIDE_TOTAL, "total"),
    ({k: v for k, v in SIDE_TOTAL.items() if k != "totals"}, "btts"),
    ({"pulled_at": SIDE_TOTAL["pulled_at"], "spreads": SIDE_TOTAL["spreads"]}, "1x2"),
])
def test_rows_rebuild_from_fit_columns(tmp_path, side, anchor):
    rows = R.build_lines(_db(tmp_path, side), "2026-11-23T00:00:00+00:00")
    assert rows and {r["fit_anchor"] for r in rows} == {anchor}
    checked = 0
    for r in rows:
        assert r["code_version"] == R.CODE_VERSION and R.CODE_VERSION.startswith("research-5")
        assert 0.05 <= r["fit_lam"] <= 6 and 0.05 <= r["fit_mu"] <= 6
        assert r["fit_rho"] == pytest.approx(M.RHO_MKT)
        assert 0 <= r["fit_resid"] < 0.02
        if r.get("model_p_a") is None:          # corners and quarter lines carry no grid probability
            assert r["market"] == "corners" or not R._is_half_or_whole(float(r["line"]))
            continue
        assert abs(_rebuild(r) - r["model_p_a"]) < 2e-5
        checked += 1
    assert checked >= 2


def test_resid_is_the_gap_to_the_fitted_prices(tmp_path):
    f = R.anchored_fit({"latest": ML, "side": SIDE_TOTAL}, None)
    pH, pD, pA = M.shin([ML["home"], ML["draw"], ML["away"]])
    p = M.grid_probs(f["G"])
    po = M.shin([1.8, 2.0])[0]                   # 2.5 is the line closest to 50%
    gaps = [p["H"] - pH, p["D"] - pD, p["A"] - pA, M.p_over_line(f["G"], 2.5) - po]
    assert f["line"] == 2.5 and f["resid"] == pytest.approx(max(abs(g) for g in gaps))


def test_anchored_grid_is_unchanged(tmp_path):
    od = {"latest": ML, "side": SIDE_TOTAL}
    G, anchor, line = R.anchored_grid(od, {"xg_home": 1.7, "xg_away": 0.9})
    lam, mu = M.implied_lambdas(*M.shin([ML["home"], ML["draw"], ML["away"]]), M.shin([1.8, 2.0])[0],
                                line=2.5, tot_prior=2.6)
    assert anchor == "total" and line == 2.5 and np.allclose(G, M.grid(lam, mu))
    assert R.anchored_grid({"latest": {"home": 1.5}}, None) == (None, None, None)


def test_old_file_is_widened_without_changing_values(tmp_path):
    old_cols = R.LINE_COLS[:-4]
    path = tmp_path / "lines.csv"
    old_rows = [{c: f"{c}{i}" for c in old_cols} | {"key": f"k{i}", "home": 'Brighton, "Hove"'} for i in range(3)]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=old_cols); w.writeheader(); w.writerows(old_rows)
    new = {c: "x" for c in R.LINE_COLS} | {"key": "k9", "kickoff": "2026-11-21"}
    assert len(R._append(path, R.LINE_COLS, [new, dict(new, key="k1")])) == 1      # k1 already recorded
    with path.open(newline="") as fh:
        got = list(csv.DictReader(fh))
    assert list(got[0]) == R.LINE_COLS and len(got) == 4
    for o, g in zip(old_rows, got):
        assert all(g[c] == o[c] for c in old_cols)
        assert all(g[c] == "" for c in R.LINE_COLS[-4:])
    assert got[3]["fit_lam"] == "x"
    before = path.read_bytes()
    assert R._append(path, R.LINE_COLS, [new]) == [] and path.read_bytes() == before   # re-run adds nothing


def test_widen_refuses_a_different_header(tmp_path):
    path = tmp_path / "lines.csv"
    path.write_bytes(b"key,something_else\r\na,b\r\n")
    with pytest.raises(ValueError):
        R._append(path, R.LINE_COLS, [])
    assert path.read_bytes() == b"key,something_else\r\na,b\r\n"
