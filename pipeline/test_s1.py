"""Tests for rule S1 (pipeline/picks/s1.py), the CLV re-pull (capture_odds.py) and the forward-test report.

Run from the repo root:  python3 -m pytest -q pipeline/test_s1.py
"""
import csv, importlib, json, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "picks")); sys.path.insert(0, str(ROOT))
import s1 as S1  # noqa: E402
import report as REP  # noqa: E402
from v5 import market as M  # noqa: E402

HOME, AWAY = "Home FC", "Away United"
KO = "2026-11-22T15:00:00Z"
PULL = "2026-11-22T10:00:00+00:00"


def fair_book(lam=1.6, mu=1.1, margin=1.06, tweak=None):
    """Snapshot rows priced fairly from one grid, with a proportional margin. tweak: {(market, outcome, point): price}."""
    G = M.grid(lam, mu, S1.RHO)
    I, J = np.indices(G.shape)
    rows = []
    def add(mk, oc, pt, p):
        price = round(1 / (p * margin), 3)
        if tweak and (mk, oc, pt) in tweak:
            price = tweak[(mk, oc, pt)]
        rows.append(dict(pulled_at=PULL, kind="late", comp="PL", event_id="ev1", commence=KO, home=HOME, away=AWAY,
                         bookmaker="draftkings", book_last_update="", market=mk, outcome=oc, point=pt, price=price))
    pH, pD, pA = float(G[I > J].sum()), float(np.trace(G)), float(G[I < J].sum())
    add("h2h", HOME, np.nan, pH); add("h2h", "Draw", np.nan, pD); add("h2h", AWAY, np.nan, pA)
    for L in (1.5, 2.5, 3.5, 4.5):
        po = float(G[(I + J) > L].sum())
        add("alternate_totals", "Over", L, po); add("alternate_totals", "Under", L, 1 - po)
    for h in (-1.5, -0.5, 0.5, 1.5, 0.25):
        ph = float(G[(I - J + h) > 0].sum()) if h != 0.25 else 0.5
        add("alternate_spreads", HOME, h, ph); add("alternate_spreads", AWAY, -h, 1 - ph)
    pb = float(G[1:, 1:].sum())
    add("btts", "Yes", np.nan, pb); add("btts", "No", np.nan, 1 - pb)
    return pd.DataFrame(rows)


def decide(df, done=()):
    return S1.build(df, set(done), now=datetime(2026, 11, 22, 10, 1, tzinfo=timezone.utc))


# ------------------------------------------------------------------ the rule
def test_fair_book_gives_no_priced_pick_with_a_reason():
    (r,) = decide(fair_book())
    assert r["decision"] == "no priced pick" and r["reason"].startswith("no line with EV >= 0.03")
    assert r["anchor"] == "total" and float(r["ev"]) < 0       # the best line is still recorded


def test_a_mispriced_line_is_picked_with_its_full_record():
    # DK prices BTTS Yes far too generously relative to its own 1X2 and main total
    (r,) = decide(fair_book(tweak={("btts", "Yes", np.nan): 2.40}))
    assert r["decision"] == "pick" and r["market"] == "btts" and r["side"] == "yes"
    assert r["dk_price"] == 2.40 and abs(r["ev"] - (r["p_grid"] * 2.40 - 1)) < 1e-5
    for k in ("side_pulled_at", "h2h_pulled_at", "lam", "mu", "anchor", "p_dk_shin", "code_commit", "rule", "threshold"):
        assert r[k] != "", k


def test_away_spread_uses_the_away_teams_own_handicap():
    # away +0.5 (= home -0.5 key) priced generously
    (r,) = decide(fair_book(tweak={("alternate_spreads", AWAY, 0.5): 2.9}))
    assert r["decision"] == "pick" and r["market"] == "sp" and r["side"] == "away"
    assert float(r["line"]) == -0.5 and r["label"] == f"{AWAY} +0.5"


def test_anchor_line_and_quarter_lines_are_never_picked():
    G = M.grid(1.6, 1.1, S1.RHO); I, J = np.indices(G.shape)
    # make the anchor (main total, 2.5 here) look like huge value; it must still not be picked
    df = fair_book(tweak={("alternate_totals", "Over", 2.5): 3.4, ("alternate_spreads", HOME, 0.25): 3.4})
    (r,) = decide(df)
    assert not (r["market"] == "tot" and float(r["line"] or 0) == float(r["anchor_line"]))
    assert not (r["market"] == "sp" and abs(float(r["line"]) - 0.25) < 1e-9)


def test_btts_only_pull_says_there_is_nothing_to_test():
    df = fair_book()
    df = df[df.market.isin(["h2h", "btts"])]
    (r,) = decide(df)
    assert r["decision"] == "no priced pick" and r["anchor"] == "btts" and "only the anchor" in r["reason"]


def test_missing_moneyline_and_post_kickoff_pulls_are_recorded_not_dropped():
    df = fair_book(); (r,) = decide(df[df.market != "h2h"])
    assert r["decision"] == "no priced pick" and "moneyline" in r["reason"]
    late = fair_book(); late["pulled_at"] = "2026-11-22T15:30:00+00:00"
    (r,) = decide(late)
    assert r["reason"] == "side pull was after kickoff"


def test_prices_outside_range_are_ineligible():
    (r,) = decide(fair_book(tweak={("alternate_totals", "Under", 1.5): 9.0}))   # big EV but priced above 3.50
    assert not (r["market"] == "tot" and r["side"] == "under" and float(r["line"]) == 1.5)


def test_append_only_one_row_per_match_and_sample_start(tmp_path, monkeypatch):
    out = tmp_path / "s1.csv"
    rows = decide(fair_book())
    S1.append(rows, out)
    assert decide(fair_book(), S1.decided_ids(out)) == []             # already decided -> nothing new
    # the registered file never takes rows from before the sample start
    monkeypatch.setattr(S1, "OUT", out)
    old = fair_book(); old["pulled_at"] = "2026-11-01T10:00:00+00:00"; old["commence"] = "2026-11-01T15:00:00Z"
    old["event_id"] = "ev_old"
    snap = tmp_path / "snap.csv"; old.to_csv(snap, index=False)
    monkeypatch.setattr(sys, "argv", ["s1.py", "--snap", str(snap), "--out", str(out)])
    S1.main()
    assert "ev_old" not in S1.decided_ids(out)


def test_week_10_replay_of_real_prices():
    snap = pd.read_csv(ROOT / "data" / "odds" / "snapshots.csv", dtype={"point": float})
    wk = snap[(snap.commence > "2026-11-06") & (snap.commence < "2026-11-10")]
    rows = decide(wk)
    ids = [r["event_id"] for r in rows]
    assert len(ids) == len(set(ids)) == 38                          # one decision per match, every match
    assert all(r["decision"] in ("pick", "no priced pick") and r["reason"] for r in rows)
    assert sum(r["decision"] == "pick" for r in rows) == 0          # DraftKings was internally consistent that week
    assert all(r["anchor"] == "total" for r in rows)                # full side pulls that week


# ------------------------------------------------------------------ CLV re-pull
def test_clv_repull_once_per_pick_with_weekly_cap(tmp_path, monkeypatch):
    import capture_odds as C
    picks = tmp_path / "s1.csv"
    now = datetime(2026, 11, 22, 14, 10, tzinfo=timezone.utc)
    with picks.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=S1.COLS); w.writeheader()
        for i in range(3):
            w.writerow({**{c: "" for c in S1.COLS}, "event_id": f"e{i}", "league": "E0", "kickoff": KO,
                        "decision": "pick", "dk_market": "btts"})
    monkeypatch.setattr(C, "S1_PICKS", picks)
    monkeypatch.setattr(C, "OUT", tmp_path)
    monkeypatch.setattr(C, "CLV_WEEKLY_CAP", 2)
    calls = []
    def fake_get(path, **params):
        calls.append((path, params["markets"]))
        eid = path.split("/")[4]
        body = {"id": eid, "commence_time": KO, "home_team": HOME, "away_team": AWAY,
                "bookmakers": [{"key": "draftkings", "last_update": "", "markets": [{"key": "btts", "outcomes": [
                    {"name": "Yes", "price": 1.8}, {"name": "No", "price": 1.95}]}]}]}
        return body, {"x-requests-remaining": "200"}
    monkeypatch.setattr(C, "get", fake_get)
    monkeypatch.setattr(C, "log_quota", lambda *a, **k: None)
    state = {}
    C.pull_clv("soccer_epl", "PL", state, now, 300)
    assert len(calls) == 2 and all(m == "btts" for _, m in calls)                    # cap respected
    assert sum(1 for k, v in state.items() if not k.startswith("_") and v.get("clv_skipped")) == 1
    C.pull_clv("soccer_epl", "PL", state, now, 300)
    assert len(calls) == 2                                                            # never twice
    rows = list(csv.reader((tmp_path / "clv.csv").open()))
    assert rows[0][0] == "pulled_at" and len(rows) == 1 + 2 * 2
    assert not (tmp_path / "snapshots.csv").exists()                                  # never mixed into snapshots


def test_clv_repull_only_inside_the_last_65_minutes(tmp_path, monkeypatch):
    import capture_odds as C
    picks = tmp_path / "s1.csv"
    with picks.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=S1.COLS); w.writeheader()
        w.writerow({**{c: "" for c in S1.COLS}, "event_id": "e0", "league": "E0", "kickoff": KO,
                    "decision": "pick", "dk_market": "btts"})
    monkeypatch.setattr(C, "S1_PICKS", picks)
    monkeypatch.setattr(C, "get", lambda *a, **k: pytest.fail("should not pull"))
    C.pull_clv("soccer_epl", "PL", {}, datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc), 300)   # 3 h before


# ------------------------------------------------------------------ report pieces
def test_grading_and_closing_probability():
    assert REP.grade("tot", "2.5", "over", 2, 1) and not REP.grade("tot", "2.5", "under", 2, 1)
    assert REP.grade("sp", "-0.5", "away", 1, 1)              # away +0.5 covers a draw
    assert not REP.grade("sp", "-0.5", "home", 1, 1)
    assert REP.grade("btts", "", "no", 2, 0)
    clv = fair_book(); clv = clv[clv.market == "btts"]
    pick = pd.Series(dict(event_id="ev1", dk_market="btts", home=HOME, away=AWAY, market="btts", line="", side="yes"))
    p = REP.close_prob(clv, pick)
    G = M.grid(1.6, 1.1, S1.RHO)
    assert abs(p - float(G[1:, 1:].sum())) < 0.01


def test_week_block_bootstrap_needs_two_weeks():
    assert REP.block_boot([0.1, 0.2], ["W1", "W1"]) == (None, None)
    lo, hi = REP.block_boot([0.1, 0.2, -0.1, 0.0], ["W1", "W1", "W2", "W2"])
    assert lo <= hi
