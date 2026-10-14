"""
Tests for the v6.2 fixes (Codex review, 2026-10-13). One section per fix.

    python -m pytest test_v62.py -q            (or: python test_v62.py)

Self-contained: builds a tiny fake repo (four leagues, two teams each) and a fake run folder with
real manifests, so it needs no network and no checkout. One optional test runs the whole pipeline
against a real checkout when PULSEMATCH_REPO points at one.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import banker  # noqa: E402
import checker  # noqa: E402
import collect  # noqa: E402
import manifest as M  # noqa: E402
import orchestrate  # noqa: E402
import publish  # noqa: E402
import runner  # noqa: E402

NOW = datetime.now(timezone.utc)
LEAGUE_TEAMS = {"E0": ("Arsenal", "Chelsea"), "SP1": ("Betis", "Girona"),
                "D1": ("Bochum", "Mainz"), "F1": ("Lens", "Nice")}


# ------------------------------------------------------------------ fixtures

def make_root(tmp: Path) -> Path:
    root = tmp / "repo"
    fd = root / "data" / "fd"
    fd.mkdir(parents=True)
    for code, (a, b) in LEAGUE_TEAMS.items():
        done = pd.DataFrame({"Date": ["01/09/2025", "01/02/2026"], "HomeTeam": [a, b], "AwayTeam": [b, a],
                             "FTHG": [1, 2], "FTAG": [0, 2], "HST": [3, 4], "AST": [2, 5]})
        done.to_csv(fd / f"{code}_2526.csv", index=False)
        cur = pd.DataFrame({"Date": [(NOW - timedelta(days=2)).strftime("%d/%m/%Y")], "HomeTeam": [a],
                            "AwayTeam": [b], "FTHG": [1], "FTAG": [1], "HST": [3], "AST": [3]})
        cur.to_csv(fd / f"{code}_2627.csv", index=False)
    out = root / "pipeline" / "v5" / "out"
    out.mkdir(parents=True)
    (out / "backtest.json").write_text(json.dumps({"gate": {"open": False}, "generated": NOW.isoformat()}))
    return root


GOOD_ROW = {"event_id": "ev1", "comp": "PL", "commence": (NOW + timedelta(days=2)).isoformat(),
            "home": "Arsenal", "away": "Chelsea", "odds_home": 2.0, "odds_draw": 3.5, "odds_away": 4.0,
            "overround": 1.0357, "p_home": 0.48, "p_draw": 0.28, "p_away": 0.24, "lam": None, "mu": None,
            "goals_basis": "none", "flags": "no_total_line"}
SKIPPED_ROW = {"event_id": "ev2", "comp": "PL", "commence": (NOW + timedelta(days=2)).isoformat(),
               "home": "Betis", "away": "Girona", "flags": "missing_h2h"}


def make_run(tmp: Path, root: Path, rid: str = "R1", priced=None, picks=None, bank=None, plan=None,
             statuses=None, warnings=None) -> Path:
    """A run folder with every upstream agent's output and a real manifest for each."""
    run = tmp / "runs" / rid
    statuses, warnings = statuses or {}, warnings or {}
    counts = {f.stem: len(pd.read_csv(f)) for f in sorted((root / "data" / "fd").glob("*.csv"))}

    def finish(agent):
        M.write_manifest(run / agent, agent=agent, run_id=rid, status=statuses.get(agent, "OK"),
                         started=M.utc_now(), row_counts=counts if agent == "collector" else {},
                         warnings=warnings.get(agent, []))

    (run / "collector").mkdir(parents=True)
    (run / "collector" / "completeness.json").write_text("{}")
    finish("collector")

    (run / "pricer").mkdir()
    pd.DataFrame(priced if priced is not None else [GOOD_ROW]).to_csv(run / "pricer" / "priced.csv", index=False)
    finish("pricer")

    (run / "runner").mkdir()
    cols = ["event_id", "comp", "commence", "home", "away", "market", "selection", "point", "price", "ev",
            "close_price", "clv", "clv_basis"]
    pd.DataFrame(picks or [], columns=cols).to_csv(run / "runner" / "paper_picks.csv", index=False)
    (run / "runner" / "summary.json").write_text(json.dumps({
        "generated": NOW.isoformat(), "mode": "paper (DraftKings only)", "gate_open": False,
        "blend_weight": 0.0, "edge_threshold": 0.02}))
    (run / "runner" / "ratings.json").write_text(json.dumps({"leagues": {"E0": {
        "rho": -0.05, "attack": {"Arsenal": 0.3}, "defence": {"Arsenal": 0.2}}}}))
    finish("runner")

    (run / "banker").mkdir()
    (run / "banker" / "bankroll.json").write_text(json.dumps(bank or {"bankroll": 0.0, "unit": 0.0}))
    (run / "banker" / "plan.json").write_text(json.dumps(plan or {
        "gate_open": False, "unit": 0.0, "bets": [], "preview": [], "placed_today": 0.0}))
    finish("banker")
    return run


def check(root, run, baseline=None):
    return checker.Checker(root, run_dir=run, baseline=baseline).run()


def sev(report, cid):
    return [c["severity"] for c in report["checks"] if c["id"] == cid]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = make_root(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# ------------------------------------------------------------ fix 1: moneyline

class TestSkippedMoneyline(Base):
    def test_clean_run_passes(self):
        r = check(self.root, make_run(self.tmp, self.root))
        self.assertEqual(r["counts"]["BLOCK"], 0, r["blocking"])

    def test_skipped_fixture_warns_not_blocks(self):
        r = check(self.root, make_run(self.tmp, self.root, priced=[GOOD_ROW, SKIPPED_ROW]))
        self.assertEqual(sev(r, "market.skipped"), ["WARN"])
        self.assertEqual(sev(r, "market.probsum"), ["PASS"])
        self.assertEqual(r["counts"]["BLOCK"], 0, r["blocking"])

    def test_skipped_fixture_used_by_a_pick_blocks(self):
        pick = {"event_id": "ev2", "comp": "PL", "commence": SKIPPED_ROW["commence"], "home": "Betis",
                "away": "Girona", "market": "totals", "selection": "Over", "point": 2.5, "price": 2.1,
                "ev": 0.05, "close_price": None, "clv": None, "clv_basis": None}
        r = check(self.root, make_run(self.tmp, self.root, priced=[GOOD_ROW, SKIPPED_ROW], picks=[pick]))
        self.assertEqual(sev(r, "market.skipped"), ["BLOCK"])

    def test_only_skipped_fixtures(self):
        r = check(self.root, make_run(self.tmp, self.root, priced=[SKIPPED_ROW]))
        self.assertEqual(sev(r, "market.skipped"), ["WARN"])
        self.assertEqual(r["counts"]["BLOCK"], 0, r["blocking"])

    def test_unflagged_missing_probability_blocks(self):
        row = dict(GOOD_ROW, p_draw=None)
        r = check(self.root, make_run(self.tmp, self.root, priced=[row]))
        self.assertEqual(sev(r, "market.probsum"), ["BLOCK"])

    def test_missing_overround_blocks(self):
        row = dict(GOOD_ROW, overround=None)
        r = check(self.root, make_run(self.tmp, self.root, priced=[row]))
        self.assertEqual(sev(r, "market.overround"), ["BLOCK"])

    def test_claimed_goals_missing_blocks(self):
        row = dict(GOOD_ROW, goals_basis="1x2+total", lam=1.4, mu=None, flags="")
        r = check(self.root, make_run(self.tmp, self.root, priced=[row]))
        self.assertEqual(sev(r, "market.goals"), ["BLOCK"])

    def test_home_attempts_check_removed(self):
        r = check(self.root, make_run(self.tmp, self.root, priced=[dict(GOOD_ROW, home_attempts=99)]))
        self.assertEqual(sev(r, "market.attempts"), [])


# ------------------------------------------------------------ fix 2: baseline

class TestBaseline(Base):
    def test_missing_season_file_blocks(self):
        run = make_run(self.tmp, self.root)
        bl = self.tmp / "bl.json"
        counts = {f.stem: len(pd.read_csv(f)) for f in (self.root / "data" / "fd").glob("*.csv")}
        bl.write_text(json.dumps({**counts, "E0_2425": 380}))
        r = check(self.root, run, baseline=bl)
        self.assertEqual(sev(r, "hist.baseline"), ["BLOCK"])
        self.assertIn("E0_2425", r["blocking"][0]["detail"])

    def test_shrink_still_blocks_and_equal_passes(self):
        run = make_run(self.tmp, self.root)
        counts = {f.stem: len(pd.read_csv(f)) for f in (self.root / "data" / "fd").glob("*.csv")}
        bl = self.tmp / "bl.json"
        bl.write_text(json.dumps(counts))
        self.assertEqual(sev(check(self.root, run, baseline=bl), "hist.baseline"), ["PASS"])
        bl.write_text(json.dumps({**counts, "E0_2526": 99}))
        self.assertEqual(sev(check(self.root, run, baseline=bl), "hist.baseline"), ["BLOCK"])

    def _prepared(self):
        run = make_run(self.tmp, self.root)
        (run / "publisher").mkdir()
        (run / "publisher" / "state_v5.json").write_text("{}")
        M.write_manifest(run / "publisher", agent="publisher", run_id=run.name, status="OK", started=M.utc_now())
        return run

    def test_confirm_refuses_to_drop_a_season(self):
        run = self._prepared()
        bp = publish.baseline_path(run)
        bp.parent.mkdir(parents=True)
        counts = M.read_manifest(run / "collector")["row_counts"]
        bp.write_text(json.dumps({**counts, "E0_2425": 380}))
        m = publish.confirm(self.root, run, 3, "drivefolder")
        self.assertEqual(m["status"], "FAILED")
        self.assertIn("E0_2425", m["warnings"][0])
        self.assertIn("E0_2425", json.loads(bp.read_text()))          # baseline untouched

    def test_confirm_writes_baseline_atomically(self):
        run = self._prepared()
        m = publish.confirm(self.root, run, 3, "drivefolder")
        self.assertEqual(m["status"], "OK", m["warnings"])
        bp = publish.baseline_path(run)
        self.assertEqual(json.loads(bp.read_text()), M.read_manifest(run / "collector")["row_counts"])
        self.assertEqual([p.name for p in bp.parent.iterdir()], ["rowcounts.json"])   # no temp left behind

    def test_confirm_refuses_tampered_collector(self):
        run = self._prepared()
        (run / "collector" / "completeness.json").write_text('{"x": 1}')
        m = publish.confirm(self.root, run, 3, "drivefolder")
        self.assertEqual(m["status"], "FAILED")


# ------------------------------------------------- fix 3: Checker reads upstream

class TestUpstream(Base):
    def test_all_ok(self):
        r = check(self.root, make_run(self.tmp, self.root))
        for a in checker.UPSTREAM:
            self.assertEqual(sev(r, f"upstream.{a}"), ["PASS"], a)

    def test_partial_collector_warning_reaches_verdict(self):
        run = make_run(self.tmp, self.root, statuses={"collector": "PARTIAL"},
                       warnings={"collector": ["E0: newest result is 24 days old"]})
        r = check(self.root, run)
        self.assertEqual(sev(r, "upstream.collector"), ["WARN"])
        self.assertIn("24 days old", [c for c in r["checks"] if c["id"] == "upstream.collector"][0]["detail"])
        self.assertEqual(r["verdict"], "WARN")

    def test_failed_agent_blocks(self):
        r = check(self.root, make_run(self.tmp, self.root, statuses={"pricer": "FAILED"}))
        self.assertEqual(sev(r, "upstream.pricer"), ["BLOCK"])

    def test_missing_manifest_blocks(self):
        run = make_run(self.tmp, self.root)
        (run / "banker" / "manifest.json").unlink()
        self.assertEqual(sev(check(self.root, run), "upstream.banker"), ["BLOCK"])

    def test_manifest_from_another_run_blocks(self):
        run = make_run(self.tmp, self.root)
        m = json.loads((run / "runner" / "manifest.json").read_text())
        m["run_id"] = "OTHER"
        (run / "runner" / "manifest.json").write_text(json.dumps(m))
        self.assertEqual(sev(check(self.root, run), "upstream.runner"), ["BLOCK"])

    def test_unlisted_file_blocks(self):
        run = make_run(self.tmp, self.root)
        (run / "pricer" / "stale_from_last_week.csv").write_text("x\n")
        self.assertEqual(sev(check(self.root, run), "upstream.pricer"), ["BLOCK"])

    def test_required_inputs_no_longer_skip(self):
        run = make_run(self.tmp, self.root)
        (run / "pricer" / "priced.csv").unlink()
        (run / "runner" / "ratings.json").unlink()
        r = check(self.root, run)
        self.assertEqual(sev(r, "market.present"), ["BLOCK"])
        self.assertEqual(sev(r, "ratings.present"), ["WARN"])

    def test_repo_mode_still_skips(self):
        r = checker.Checker(self.root).run()
        self.assertEqual(sev(r, "upstream"), ["SKIP"])
        self.assertEqual(sev(r, "market.present"), ["SKIP"])


# ------------------------------------------------------ fix 4: Publisher strict

class TestPublisherStrict(Base):
    def _verdict(self, run):
        (run / "checker").mkdir()
        (run / "checker" / "verdict.json").write_text(json.dumps({"verdict": "PASS", "run_dir": str(run)}))
        M.write_manifest(run / "checker", agent="checker", run_id=run.name, status="OK", started=M.utc_now())

    def test_refuses_without_banker_manifest(self):
        run = make_run(self.tmp, self.root)
        self._verdict(run)
        (run / "banker" / "manifest.json").unlink()
        m = publish.prepare(self.root, run)
        self.assertEqual(m["status"], "FAILED")
        self.assertIn("banker", m["warnings"][0])

    def test_refuses_without_checker_manifest(self):
        run = make_run(self.tmp, self.root)
        self._verdict(run)
        (run / "checker" / "manifest.json").unlink()
        self.assertEqual(publish.prepare(self.root, run)["status"], "FAILED")

    def test_refuses_unlisted_file(self):
        run = make_run(self.tmp, self.root)
        self._verdict(run)
        (run / "runner" / "extra.csv").write_text("x\n")
        m = publish.prepare(self.root, run)
        self.assertEqual(m["status"], "FAILED")
        self.assertIn("not in the manifest", m["warnings"][0])

    def test_refuses_foreign_manifest(self):
        run = make_run(self.tmp, self.root)
        self._verdict(run)
        mp = run / "collector" / "manifest.json"
        m = json.loads(mp.read_text())
        m["run_id"] = "OTHER"
        mp.write_text(json.dumps(m))
        self.assertEqual(publish.prepare(self.root, run)["status"], "FAILED")

    def test_check_manifest_strict(self):
        run = make_run(self.tmp, self.root)
        self.assertEqual(M.check_manifest(run / "pricer", agent="pricer", run_id=run.name), [])
        (run / "pricer" / "x.txt").write_text("x")
        self.assertTrue(M.check_manifest(run / "pricer", agent="pricer", run_id=run.name))
        self.assertEqual(M.verify_outputs(run / "pricer"), [])           # non-strict keeps old behaviour


# ------------------------------------------------------------ fix 5: run ids

class TestRunIds(Base):
    def test_unique_and_second_resolution(self):
        a, b = M.new_run_id(NOW), M.new_run_id(NOW)
        self.assertNotEqual(a, b)
        self.assertRegex(a, r"^\d{4}-\d{2}-\d{2}T\d{6}Z-[0-9a-f]{4}$")

    def test_orchestrator_refuses_existing_folder(self):
        runs = self.tmp / "runs"
        (runs / "R9").mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            orchestrate.orchestrate(self.root, runs, "R9", agents_dir=self.tmp / "no_agents")
        rc = orchestrate.main(["--root", str(self.root), "--runs-dir", str(runs), "--run-id", "R9"])
        self.assertEqual(rc, 1)

    def test_collector_refuses_non_empty_folder(self):
        runs = self.tmp / "runs"
        (runs / "R8" / "collector").mkdir(parents=True)
        (runs / "R8" / "collector" / "old.json").write_text("{}")
        with self.assertRaises(FileExistsError):
            collect.collect(self.root, runs, "R8")

    def test_verdict_required_before_publisher(self):
        run = self.tmp / "runs" / "R7"
        (run / "checker").mkdir(parents=True)
        self.assertIn("no verdict", orchestrate._verdict_problem(run))
        (run / "checker" / "verdict.json").write_text(json.dumps({"verdict": "BLOCK", "run_dir": str(run)}))
        self.assertIn("BLOCK", orchestrate._verdict_problem(run))
        (run / "checker" / "verdict.json").write_text(json.dumps({"verdict": "WARN", "run_dir": "/elsewhere"}))
        self.assertIn("not this run", orchestrate._verdict_problem(run))
        (run / "checker" / "verdict.json").write_text(json.dumps({"verdict": "WARN", "run_dir": str(run)}))
        self.assertIsNone(orchestrate._verdict_problem(run))


# ------------------------------------------------------ fix 6: Banker screening

class TestBankerScreen(unittest.TestCase):
    def cand(self, **k):
        c = {"kind": "single", "market": "totals", "label": "Over 2.5", "price": 2.1, "ev": 0.05}
        c.update(k)
        if "p" not in k and c["price"] and c["ev"] is not None and c["price"] > 0:
            c["p"] = (c["ev"] + 1) / c["price"]
        return c

    def test_screen(self):
        ok, bad = banker.screen([self.cand(), self.cand(price=1.0), self.cand(price=float("nan")),
                                 self.cand(ev=0.0), self.cand(ev=-0.1), self.cand(ev=0.01),
                                 self.cand(p=1.2), self.cand(price=True)], min_edge=0.02)
        self.assertEqual(len(ok), 1)
        self.assertEqual(len(bad), 7)
        self.assertTrue(all("reason" in r for r in bad))

    def test_zero_edge_gets_zero_units(self):
        self.assertEqual(banker.tier(0.40, 2.5, 200, 4, 1.0), (0, 0.0))     # p*price = 1.0: no edge
        self.assertEqual(banker.tier(0.40, 2.0, 200, 4, 1.0), (0, 0.0))     # negative edge
        units, score = banker.tier(0.55, 2.1, 200, 4, 1.0)
        self.assertGreaterEqual(units, 1)

    def test_size_skips_zero_units_without_stopping(self):
        c0 = self.cand(price=2.0, ev=0.0, p=0.5)       # Kelly 0
        c1 = self.cand(price=2.1, ev=0.155, p=0.55)
        out = banker.size([c0, c1], 200.0, 0.0, {})
        self.assertEqual(len(out["bets"]), 1)
        self.assertEqual(out["not_sized"][0]["reason"], "no Kelly edge")
        self.assertIsNone(out["stopped"])

    def test_run_rejects_invalid_pick(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            run = tmp / "runs" / "R1"
            (run / "runner").mkdir(parents=True)
            ko = (NOW + timedelta(days=1)).isoformat()
            picks = pd.DataFrame([
                {"event_id": "a", "comp": "PL", "commence": ko, "home": "A", "away": "B", "market": "totals",
                 "selection": "Over", "point": 2.5, "price": 1.0, "ev": 0.05, "close_price": None, "clv": None},
                {"event_id": "b", "comp": "PL", "commence": ko, "home": "C", "away": "D", "market": "totals",
                 "selection": "Over", "point": 2.5, "price": 2.1, "ev": 0.06, "close_price": None, "clv": None}])
            picks.to_csv(run / "runner" / "paper_picks.csv", index=False)
            (run / "runner" / "summary.json").write_text(json.dumps({"gate_open": False, "edge_threshold": 0.02}))
            led = tmp / "runs" / "bankroll" / "ledger.json"
            led.parent.mkdir(parents=True)
            led.write_text(json.dumps({"transactions": [{"id": "t", "kind": "deposit", "amount": 125.0}],
                                       "bets": []}))
            m = banker.run(tmp, run)
            plan = json.loads((run / "banker" / "plan.json").read_text())
            self.assertEqual(m["row_counts"]["rejected"], 1)
            self.assertEqual(len(plan["preview"]), 1)
            self.assertTrue(any("invalid price" in r["reason"] for r in plan["not_sized"]))
            self.assertEqual(plan["bets"], [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------- Codex round 2 (v6.2 review)

class TestRound2(Base):
    def test_nested_folder_in_agent_output_blocks(self):
        run = make_run(self.tmp, self.root)
        (run / "runner" / "extra").mkdir()
        (run / "runner" / "extra" / "hidden.csv").write_text("x\n")
        self.assertEqual(sev(check(self.root, run), "upstream.runner"), ["BLOCK"])
        self.assertTrue(any("sub-folder" in p for p in
                            M.check_manifest(run / "runner", agent="runner", run_id=run.name)))

    def test_publisher_refuses_stray_top_level_entry(self):
        run = make_run(self.tmp, self.root)
        TestPublisherStrict._verdict(self, run)
        (run / "notes.txt").write_text("x")
        m = publish.prepare(self.root, run)
        self.assertEqual(m["status"], "FAILED")
        self.assertIn("unexpected entries", m["warnings"][0])

    def test_goals_columns_required_in_run_mode(self):
        row = {k: v for k, v in GOOD_ROW.items() if k not in ("lam", "mu")}
        r = check(self.root, make_run(self.tmp, self.root, priced=[row]))
        self.assertEqual(sev(r, "market.goals"), ["BLOCK"])

    def test_goals_basis_none_with_goals_blocks(self):
        r = check(self.root, make_run(self.tmp, self.root, priced=[dict(GOOD_ROW, lam=1.3, mu=1.1)]))
        self.assertEqual(sev(r, "market.goals"), ["BLOCK"])

    def test_good_goals_row_passes(self):
        row = dict(GOOD_ROW, goals_basis="1x2+total", lam=1.5, mu=1.1, flags="")
        r = check(self.root, make_run(self.tmp, self.root, priced=[row]))
        self.assertEqual(sev(r, "market.goals"), ["PASS"])

    def _banker_run(self, gate_open, edge):
        run = self.tmp / "runs" / "RB"
        (run / "runner").mkdir(parents=True)
        ko = (NOW + timedelta(days=1)).isoformat()
        pd.DataFrame([{"event_id": "b", "comp": "PL", "commence": ko, "home": "C", "away": "D",
                       "market": "totals", "selection": "Over", "point": 2.5, "price": 2.1, "ev": 0.06,
                       "close_price": None, "clv": None}]).to_csv(run / "runner" / "paper_picks.csv", index=False)
        summ = {"gate_open": gate_open}
        if edge != "absent":
            summ["edge_threshold"] = edge
        (run / "runner" / "summary.json").write_text(json.dumps(summ))
        led = self.tmp / "runs" / "bankroll" / "ledger.json"
        led.parent.mkdir(parents=True, exist_ok=True)
        led.write_text(json.dumps({"transactions": [{"id": "t", "kind": "deposit", "amount": 125.0}], "bets": []}))
        return banker.run(self.tmp, run), run

    def test_banker_gate_open_without_threshold_fails(self):
        for edge in ("absent", None, "x", float("nan"), 0, -0.01):
            shutil.rmtree(self.tmp / "runs", ignore_errors=True)
            m, _ = self._banker_run(True, edge)
            self.assertEqual(m["status"], "FAILED", edge)

    def test_banker_gate_closed_without_threshold_previews_nothing(self):
        m, run = self._banker_run(False, "absent")
        self.assertEqual(m["status"], "PARTIAL")
        plan = json.loads((run / "banker" / "plan.json").read_text())
        self.assertEqual(plan["preview"], [])
        self.assertTrue(all("edge threshold" in r["reason"] for r in plan["not_sized"]))

    def test_banker_gate_closed_with_threshold_previews(self):
        m, run = self._banker_run(False, 0.02)
        self.assertEqual(len(json.loads((run / "banker" / "plan.json").read_text())["preview"]), 1)


# --------------------------------------------------------------------- cleanups

class TestCleanups(Base):
    def test_atomic_write(self):
        p = self.tmp / "d" / "f.json"
        M.atomic_write_text(p, "abc")
        M.atomic_write_text(p, "xyz")
        self.assertEqual(p.read_text(), "xyz")
        self.assertEqual([x.name for x in p.parent.iterdir()], ["f.json"])

    def test_nested_blend_weight_blocks(self):
        run = make_run(self.tmp, self.root)
        (self.root / "pipeline" / "v5" / "out" / "weights.json").write_text(
            json.dumps({"1x2": {"E0": 0.0, "SP1": 0.3}}))
        self.assertEqual(sev(check(self.root, run), "gate.blend"), ["BLOCK"])

    def test_boolean_units_rejected(self):
        run = make_run(self.tmp, self.root, bank={"bankroll": 100.0, "unit": 2.0},
                       plan={"gate_open": False, "unit": 2.0, "placed_today": 0.0, "preview": [],
                             "bets": [{"label": "x", "units": True, "stake": 2.0}]})
        self.assertEqual(sev(check(self.root, run), "bank.tiers"), ["BLOCK"])

    def test_runtime_gate_strict(self):
        p = self.tmp / "runtime.py"
        for text, want in [("GATE_OPEN = False\n", False), ("GATE_OPEN = True  # note\n", True),
                           ("x = 1\nGATE_OPEN = False          # set only by backtest\n", False)]:
            p.write_text(text)
            self.assertEqual(runner._runtime_gate(p), want, text)
        for text in ["GATE_OPEN = bool(1)\n", "GATE_OPEN = False\nGATE_OPEN = True\n", "y = 2\n",
                     "GATE_OPEN = 'True'\n"]:
            p.write_text(text)
            with self.assertRaises(ValueError, msg=text):
                runner._runtime_gate(p)

    def test_malformed_history_filename(self):
        (self.root / "data" / "fd" / "notes.csv").write_text("a\n1\n")
        out = collect.inventory_history(self.root, NOW)
        self.assertTrue(any("notes.csv" in p for p in out["problems"]))
        r = check(self.root, make_run(self.tmp, self.root))
        self.assertEqual(sev(r, "hist.names"), ["BLOCK"])


# ------------------------------------------------------- optional end-to-end

@unittest.skipUnless(os.environ.get("PULSEMATCH_REPO"), "set PULSEMATCH_REPO to a checkout to run")
class TestEndToEnd(unittest.TestCase):
    def test_full_run_on_real_checkout(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            m = orchestrate.orchestrate(Path(os.environ["PULSEMATCH_REPO"]), tmp / "runs")
            self.assertEqual(m["outcome"], "COMPLETED", m["warnings"])
            self.assertTrue(m["publish_ready"])
            self.assertTrue(re.match(r"^\d{4}-\d{2}-\d{2}T\d{6}Z-[0-9a-f]{4}$", m["run_id"]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
