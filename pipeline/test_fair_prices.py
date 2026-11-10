"""Tests for scripts/fair_prices.py. Run: python3 -m pytest -q pipeline/test_fair_prices.py"""
import sys
from datetime import datetime, timezone
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fair_prices as F
from v5 import market as M

NOW = datetime(2026, 11, 9, tzinfo=timezone.utc)
FX = {"f1": {"kickoff": "2026-11-07T15:00:00+00:00"}, "f2": {"kickoff": "2026-11-08T15:00:00+00:00"},
      "f9": {"kickoff": "2026-11-21T15:00:00+00:00"}}
ODDS = {"f1": {"close": {"home": 2.0, "draw": 3.5, "away": 3.8},
               "side": {"btts": {"yes": 1.8, "no": 1.95}, "totals": {"2.5": {"over": 1.9, "under": 1.9}},
                        "spreads": {"-0.5": {"home": 2.0, "away": 1.8}}}},
        "f2": {"latest": {"home": 1.5, "draw": 4.2, "away": 6.5}}}
SHARP = {"f1": {"close": {"stale": False, "fair": {"home": 0.5, "draw": 0.27, "away": 0.23}}}}


def test_parse_leg_handles_both_spread_key_styles():
    assert F.parse_leg("ml:f1:away") == ("ml", "f1", None, "away")
    assert F.parse_leg("tot:f1:2.5:over") == ("tot", "f1", 2.5, "over")
    assert F.parse_leg("btts:f1::yes") == ("btts", "f1", None, "yes")
    assert F.parse_leg("sp:f1:-0.5:home") == ("sp", "f1", -0.5, "home")
    assert F.parse_leg("sp:f1:away:+0.5") == ("sp", "f1", -0.5, "away")      # away +0.5 is the home -0.5 line
    assert F.parse_leg("sp:f1:home:-0.5") == ("sp", "f1", -0.5, "home")


def test_leg_fair_sources():
    assert F.leg_fair("ml:f1:home", ODDS, SHARP) == (0.5, "pinnacle")
    p, src = F.leg_fair("ml:f2:home", ODDS, SHARP)                              # no Pinnacle: DraftKings' own close
    assert src == "draftkings" and p == pytest.approx(float(M.shin([1.5, 4.2, 6.5])[0]), abs=1e-5)
    p, src = F.leg_fair("tot:f1:2.5:under", ODDS, SHARP)
    assert src == "draftkings side" and p == pytest.approx(0.5)
    p, src = F.leg_fair("sp:f1:away:+0.5", ODDS, SHARP)
    assert p == pytest.approx(float(M.shin([2.0, 1.8])[1]))
    assert F.leg_fair("tot:f1:3.5:over", ODDS, SHARP)[0] is None
    assert F.leg_fair("cor:f1:9.5:over", ODDS, SHARP)[0] is None
    stale = {"f1": {"close": {"stale": True, "fair": SHARP["f1"]["close"]["fair"]}}}
    assert F.leg_fair("ml:f1:home", ODDS, stale)[1] == "draftkings"


def test_bet_docs():
    bets = {"single": {"price": 2.1, "legs": [{"key": "ml:f1:home"}], "status": "W"},
            "boost": {"price": 2.0, "legs": [{"key": "ml:f1:home"}], "promo": {"boosted_price": 2.25}, "status": "L"},
            "parlay": {"price": 3.0, "kind": "parlay", "legs": [{"key": "ml:f1:home"}, {"key": "ml:f2:home"}], "status": "L"},
            "sgp": {"price": 3.0, "legs": [{"key": "ml:f1:home"}, {"key": "tot:f1:2.5:over"}], "status": "L"},
            "noline": {"price": 1.9, "legs": [{"key": "tot:f1:3.5:over"}], "status": "W"},
            "open": {"price": 1.9, "legs": [{"key": "ml:f9:home"}], "status": "open"},
            "void": {"price": 1.9, "legs": [{"key": "ml:f1:home"}], "status": "V"},
            "paper": {"price": 1.9, "legs": [{"key": "ml:f1:home"}], "status": "W", "paper": True}}
    d = F.bet_docs(bets, ODDS, SHARP, FX, NOW)
    assert set(d) == {"single", "boost", "parlay", "sgp", "noline"}             # not started, void and paper are skipped
    assert d["single"]["clv"] == pytest.approx(2.1 * 0.5 - 1) and d["single"]["src"] == "pinnacle"
    assert d["boost"]["clv"] == pytest.approx(0.0) and d["boost"]["clv_boosted"] == pytest.approx(0.125)
    p2 = float(M.shin([1.5, 4.2, 6.5])[0])
    assert d["parlay"]["clv"] == pytest.approx(3.0 * 0.5 * p2 - 1, abs=1e-4) and d["parlay"]["src"] == "draftkings+pinnacle"
    assert d["sgp"]["clv"] is None and "same-game" in d["sgp"]["why"]
    assert d["noline"]["clv"] is None and d["noline"]["why"] == "no closing price on that line"


def test_sharp_docs_use_the_last_quote_before_kickoff(tmp_path):
    p = tmp_path / "sharp.csv"
    rows = ["pulled_at,kind,comp,event_id,commence,home,away,bookmaker,book_last_update,market,outcome,point,price"]
    for t, upd, h, d, a in (("2026-11-07T10:00:00+00:00", "2026-11-07T09:59:00Z", 2.0, 3.6, 3.9),
                            ("2026-11-07T14:00:00+00:00", "2026-11-07T13:59:30Z", 1.9, 3.7, 4.2),
                            ("2026-11-07T16:00:00+00:00", "2026-11-07T15:59:00Z", 1.2, 6.0, 15.0)):   # in play: ignored
        for o, px in (("Home FC", h), ("Draw", d), ("Away FC", a)):
            rows.append(f"{t},late,PL,f1,2026-11-07T15:00:00Z,Home FC,Away FC,pinnacle,{upd},h2h,{o},,{px}")
    p.write_text("\n".join(rows) + "\n")
    q = F.pinnacle_quotes(p)
    docs = F.sharp_docs(FX, q, NOW)
    assert set(docs) == {"f1"} and docs["f1"]["n_pulls"] == 2
    c = docs["f1"]["close"]
    assert c["home"] == 1.9 and c["mins_to_ko"] == 60.0 and not c["stale"]
    assert c["fair"]["home"] == pytest.approx(float(M.shin([1.9, 3.7, 4.2])[0]), abs=1e-5)
    assert sum(c["fair"].values()) == pytest.approx(1, abs=1e-4)
