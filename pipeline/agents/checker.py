"""
Pulsematch — Checker agent (read-only).

The last step before anything is published. It re-reads the pipeline's own outputs,
runs every check below, and returns one verdict: PASS, WARN or BLOCK. It writes a
single verdict file and nothing else: it never edits, fixes or republishes the work
it reviews, and the Publisher is the only agent allowed to act on its verdict.

    python pipeline/agents/checker.py                     # repo root inferred
    python pipeline/agents/checker.py --root . --out verdict.json
    python pipeline/agents/checker.py --strict            # treat WARN as BLOCK
    python pipeline/agents/checker.py --break-until 2026-10-09

Exit code: 0 = publish allowed (PASS or WARN), 1 = BLOCK, 2 = the checker itself
failed to run (treat as BLOCK: an unexplained checker is not a pass).

Severities
  BLOCK  the output is wrong or unverifiable. Publishing is refused.
  WARN   the output is publishable but something should be looked at.
  PASS   checked and fine.
  SKIP   the input for that check is absent (recorded, never silently ignored).

v6.2 changes (Codex review, 2026-10-13)
  - upstream.<agent>: with --run-dir, the collector, pricer, runner and banker must each have a
    manifest for THIS run, unchanged outputs and no unlisted files (else BLOCK); FAILED blocks and
    PARTIAL carries its warnings into the verdict as WARN.
  - Fixtures the Pricer skipped (flag missing_h2h) are listed as WARN market.skipped instead of
    being summed as zero (which blocked the whole run); BLOCK only if a paper pick relies on one.
  - Missing or non-numeric probabilities, overround or claimed implied goals now BLOCK.
  - hist.baseline also blocks when a season file in the baseline has disappeared.
  - With --run-dir, priced.csv and paper_picks.csv absent = BLOCK, ratings absent = WARN (no SKIP).
  - Nested blend weights are inspected; boolean "units" are rejected; history files must be named
    CODE_SSSS.csv (hist.names); the home-shot-attempts check was removed (no such column).

Design rules this file follows (claude/agent-pipeline-design.md):
  - Code does the math; this agent only re-reads and compares.
  - Nothing it reviews is modified.
  - Every check reports its own id, severity and detail, so a BLOCK always says why.
  - Expected match counts are DERIVED from each season's team count, never hardcoded:
    Ligue 1 had 20 teams (380 matches) through 2022-23 and 18 (306) from 2023-24.
    Genuinely short seasons (COVID) are listed in ABANDONED, not treated as failures.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import manifest as M  # noqa: E402

# ---------------------------------------------------------------- configuration

LEAGUES = {"E0": "Premier League", "SP1": "La Liga", "D1": "Bundesliga", "F1": "Ligue 1"}

# Seasons that really did end early. Not failures.
ABANDONED = {
    ("F1", "1920"): (279, "Ligue 1 2019-20 abandoned in March 2020 (COVID)"),
}

RHO_MAX = 0.25          # the live fit bounds rho to +/- this; outside it the draw rate is wrong
RATING_MAX = 1.0        # attack/defence ratings live in roughly +/- 1; -13.88 was a degenerate fit
OVERROUND_MIN = 1.00
OVERROUND_MAX = 1.15
GOALS_MIN, GOALS_MAX = 0.2, 4.5
PROB_TOL = 1e-6
SHOTS_COVERAGE_MIN = 0.98
STALE_DAYS = 8

# The gate, restated here so the checker computes it independently of the model code.
GATE_MIN_BETS = 300
GATE_MIN_T = 2.0

# With --run-dir, these outputs are read from this run's agent folders instead of the repo,
# so the Checker judges what this run produced (agents never write into the public repo).
RUN_DIR_FILES = {
    "pipeline/v5/out/priced.csv": "pricer/priced.csv",
    "pipeline/v5/out/ratings.json": "runner/ratings.json",
    "data/v5/paper_picks.csv": "runner/paper_picks.csv",
    "data/v5/summary.json": "runner/summary.json",
    "banker/plan.json": "banker/plan.json",
    "banker/bankroll.json": "banker/bankroll.json",
}

# v6.2: with --run-dir, these agents must each have a manifest written for THIS run, with every
# output unchanged and no unlisted file in their folder. Their warnings are carried into the verdict.
UPSTREAM = ("collector", "pricer", "runner", "banker")
NAME_RE = re.compile(r"^([A-Za-z0-9]+)_(\d{4})$")   # history files: CODE_SSSS.csv

# The Banker's rules, restated so the checker verifies them independently of banker.py.
BANK_UNIT_PER_100 = 2.0
BANK_MAX_UNITS = 3
BANK_DAY_CAP = 0.10


def _numeric_leaves(x, prefix=""):
    """(path, value) for every number in a nested dict/list (booleans are not numbers here)."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _numeric_leaves(v, f"{prefix}{k}.")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from _numeric_leaves(v, f"{prefix}{i}.")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield prefix.rstrip("."), float(x)


class Result:
    """One check's outcome."""

    def __init__(self, cid: str, severity: str, detail: str):
        self.id, self.severity, self.detail = cid, severity, detail

    def as_dict(self) -> dict:
        return {"id": self.id, "severity": self.severity, "detail": self.detail}


class Checker:
    def __init__(self, root: Path, break_until: str | None = None, baseline: Path | None = None,
                 run_dir: Path | None = None):
        self.root = root
        self.run_dir = run_dir
        self.break_until = break_until
        self.baseline_path = baseline or root / "data" / "fd" / ".rowcounts.json"
        self.results: list[Result] = []
        self._cache: dict = {}          # each file is read (and, if broken, reported) once

    # -- recording -------------------------------------------------------------

    def block(self, cid, detail):
        self.results.append(Result(cid, "BLOCK", detail))

    def warn(self, cid, detail):
        self.results.append(Result(cid, "WARN", detail))

    def ok(self, cid, detail):
        self.results.append(Result(cid, "PASS", detail))

    def skip(self, cid, detail):
        self.results.append(Result(cid, "SKIP", detail))

    # -- helpers ---------------------------------------------------------------

    def _path(self, rel: str) -> Path:
        if self.run_dir is not None and rel in RUN_DIR_FILES:
            return self.run_dir / RUN_DIR_FILES[rel]
        return self.root / rel

    def _where(self, rel: str) -> str:
        if self.run_dir is not None and rel in RUN_DIR_FILES:
            return f"run {RUN_DIR_FILES[rel]}"
        return rel

    def _read_json(self, rel: str):
        key = ("json", rel)
        if key in self._cache:
            return self._cache[key]
        p = self._path(rel)
        out = None
        if p.exists():
            try:
                out = json.loads(p.read_text())
            except Exception as e:                              # noqa: BLE001
                self.block("file.json", f"{rel} is not readable JSON: {e}")
        self._cache[key] = out
        return out

    def _read_csv(self, rel: str):
        key = ("csv", rel)
        if key in self._cache:
            return None if self._cache[key] is None else self._cache[key].copy()
        p = self._path(rel)
        out = None
        if p.exists():
            try:
                out = pd.read_csv(p)
            except pd.errors.EmptyDataError:
                out = pd.DataFrame()
            except Exception as e:                              # noqa: BLE001
                self.block("file.csv", f"{rel} is not readable CSV: {e}")
        self._cache[key] = out
        return None if out is None else out.copy()

    @property
    def _run_mode(self) -> bool:
        """True when judging one pipeline run: its required inputs must exist, not just SKIP."""
        return self.run_dir is not None

    def _in_break(self) -> bool:
        if not self.break_until:
            return False
        try:
            until = datetime.fromisoformat(self.break_until).date()
        except ValueError:
            return False
        return datetime.now(timezone.utc).date() <= until

    # ======================================================== upstream manifests

    def check_upstream(self):
        """v6.2. Every upstream agent must have left a manifest for THIS run with unchanged outputs
        and no unlisted files; FAILED blocks, PARTIAL carries its warnings into the verdict as WARN.
        Before v6.2 the Checker never read these, so a collector or pricer warning could not reach
        the publish decision."""
        if not self._run_mode:
            self.skip("upstream", "no --run-dir: upstream manifests not checked")
            return
        rid = self.run_dir.name
        for agent in UPSTREAM:
            cid = f"upstream.{agent}"
            folder = self.run_dir / agent
            problems = M.check_manifest(folder, agent=agent, run_id=rid, strict=True)
            if problems:
                self.block(cid, "; ".join(problems))
                continue
            m = M.read_manifest(folder)
            if m["status"] == "FAILED":
                self.block(cid, f"{agent} reported FAILED: " + "; ".join(map(str, m.get("warnings", [])))[:400])
            elif m["status"] == "PARTIAL":
                ws = [str(w) for w in m.get("warnings", [])]
                self.warn(cid, f"{agent} reported PARTIAL ({len(ws)} warning(s)): " + "; ".join(ws)[:400])
            else:
                self.ok(cid, f"{agent} manifest OK for this run, outputs unchanged")

    # ========================================================== history checks

    def check_history(self):
        fd = self.root / "data" / "fd"
        if not fd.is_dir():
            self.block("hist.present", "data/fd is missing: there is no history to fit on")
            return

        files = sorted(fd.glob("*.csv"))
        if not files:
            self.block("hist.present", "data/fd holds no CSV files")
            return

        # H0 every file must be named CODE_SSSS.csv, or the season logic below cannot read it
        bad = [f.name for f in files if not NAME_RE.match(f.stem)]
        if bad:
            self.block("hist.names", f"history files not named CODE_SEASON.csv: {bad}")
        files = [f for f in files if NAME_RE.match(f.stem)]

        # H1 scope: nothing outside the four modelled leagues
        codes = {f.stem.split("_")[0] for f in files}
        extra = codes - set(LEAGUES)
        if extra:
            self.block("hist.scope", f"out-of-scope league files in data/fd: {sorted(extra)}")
        else:
            self.ok("hist.scope", f"only in-scope leagues present: {sorted(codes)}")

        missing = set(LEAGUES) - codes
        if missing:
            self.block("hist.leagues", f"no history for {sorted(missing)}")

        # the newest season per league is in progress and is expected to be partial
        latest = {}
        for f in files:
            code, season = f.stem.split("_")[0], f.stem.split("_")[1]
            if code in LEAGUES:
                latest[code] = max(latest.get(code, season), season)

        counts, short, dup, baddate, lowshots = {}, [], [], [], []
        for f in files:
            code, season = f.stem.split("_")[0], f.stem.split("_")[1]
            if code not in LEAGUES:
                continue
            try:
                d = pd.read_csv(f)
            except Exception as e:                              # noqa: BLE001
                self.block("hist.readable", f"{f.name} is not readable: {e}")
                continue

            counts[f.stem] = len(d)
            needed = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
            if not needed <= set(d.columns):
                self.block("hist.columns", f"{f.name} is missing columns {sorted(needed - set(d.columns))}")
                continue

            # H4 dates must all parse
            dates = pd.to_datetime(d.Date, format="mixed", dayfirst=True, errors="coerce")
            if dates.isna().any():
                baddate.append(f"{f.name} ({int(dates.isna().sum())} rows)")

            # H3 no fixture twice in one season
            d2 = d.dropna(subset=["HomeTeam", "AwayTeam"])
            n_dup = int(d2.duplicated(subset=["HomeTeam", "AwayTeam"]).sum())
            if n_dup:
                dup.append(f"{f.name} ({n_dup})")

            # H2 completed seasons must be complete: teams x (teams - 1)
            teams = len(set(d2.HomeTeam) | set(d2.AwayTeam))
            expected = teams * (teams - 1)
            is_current = season == latest.get(code)
            if (code, season) in ABANDONED:
                exp_rows, why = ABANDONED[(code, season)]
                if len(d) != exp_rows:
                    short.append(f"{f.name}: {len(d)} rows, known-short season expects {exp_rows} ({why})")
            elif not is_current and len(d) != expected:
                short.append(f"{f.name}: {len(d)} of {expected} rows ({teams} teams)")

            # H6 shots on target is the model's xG stand-in, so its coverage matters
            if {"HST", "AST"} <= set(d.columns) and len(d):
                cov = float(min(d.HST.notna().mean(), d.AST.notna().mean()))
                if cov < SHOTS_COVERAGE_MIN:
                    lowshots.append(f"{f.name} ({cov:.1%})")

        if short:
            self.block("hist.rowcount", "incomplete season files: " + "; ".join(short))
        else:
            self.ok("hist.rowcount", f"{len(counts)} season files have their full match count "
                                     "(expected count derived from each season's team count)")
        if dup:
            self.block("hist.duplicates", "duplicate fixtures: " + "; ".join(dup))
        else:
            self.ok("hist.duplicates", "no duplicated fixtures")
        if baddate:
            self.block("hist.dates", "unparseable dates: " + "; ".join(baddate))
        else:
            self.ok("hist.dates", "all dates parse")
        if lowshots:
            self.warn("hist.shots", f"shots on target below {SHOTS_COVERAGE_MIN:.0%} coverage: " + "; ".join(lowshots))
        else:
            self.ok("hist.shots", "shots on target fully covered")

        self._check_baseline(counts)

    def _check_baseline(self, counts: dict):
        """The current-season file must never shrink. This is the check that catches a
        truncated fetch: one season once loaded 63 of 380 rows and the fit ran anyway."""
        if not self.baseline_path.exists():
            self.warn("hist.baseline", f"no row-count baseline at {self.baseline_path.name}; "
                                       "cannot prove the history did not shrink this run")
            return
        try:
            prev = json.loads(self.baseline_path.read_text())
        except Exception as e:                                  # noqa: BLE001
            self.warn("hist.baseline", f"baseline unreadable ({e}); shrink check skipped")
            return
        shrunk = [f"{k}: {counts[k]} now vs {v} before"
                  for k, v in prev.items() if k in counts and counts[k] < v]
        # v6.2: a season file that was in the baseline and is now gone is the worst kind of shrink;
        # it used to pass because only files present in both were compared
        gone = [f"{k} ({v} rows before, file now missing)" for k, v in prev.items() if k not in counts]
        if shrunk or gone:
            self.block("hist.baseline", "history shrank since the last run: " + "; ".join(shrunk + gone)
                       + (" (to retire a season on purpose, remove it from the baseline file by hand)"
                          if gone else ""))
        else:
            self.ok("hist.baseline", f"no file shrank or disappeared (compared {len(prev)} baseline entries)")

    # =========================================================== ratings checks

    def check_ratings(self, rel="pipeline/v5/out/ratings.json"):
        r = self._read_json(rel)
        if r is None:
            # run mode: the runner should have produced ratings; their absence is worth a look
            (self.warn if self._run_mode else self.skip)(
                "ratings.present", f"{self._where(rel)} absent; rating bounds not checked this run")
            return

        leagues = r.get("leagues", r) if isinstance(r, dict) else {}
        bad_rho, bad_rating = [], []
        for lg, body in leagues.items():
            if not isinstance(body, dict):
                continue
            rho = body.get("rho")
            if rho is not None and abs(float(rho)) > RHO_MAX + 1e-9:
                bad_rho.append(f"{lg} rho={float(rho):+.3f}")
            for side in ("attack", "defence", "defense"):
                for team, v in (body.get(side) or {}).items():
                    try:
                        val = float(v)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(val) or abs(val) > RATING_MAX:
                        bad_rating.append(f"{lg} {team} {side}={val:+.2f}")

        if bad_rho:
            self.block("ratings.rho", f"rho outside +/-{RHO_MAX}: " + "; ".join(bad_rho) +
                                      " (an unbounded rho over-predicts draws)")
        else:
            self.ok("ratings.rho", f"rho within +/-{RHO_MAX} in every league")
        if bad_rating:
            shown = "; ".join(bad_rating[:6]) + (f" and {len(bad_rating) - 6} more" if len(bad_rating) > 6 else "")
            self.block("ratings.scale", f"ratings beyond +/-{RATING_MAX} (degenerate fit): {shown}")
        else:
            self.ok("ratings.scale", f"all team ratings within +/-{RATING_MAX}")

    # ============================================================ market checks

    def check_market(self, rel="pipeline/v5/out/priced.csv"):
        d = self._read_csv(rel)
        if d is None:
            if self._run_mode:   # the Pricer writes priced.csv even when it fails
                self.block("market.present", f"{self._where(rel)} absent: the Pricer output is missing")
            else:
                self.skip("market.present", f"{self._where(rel)} absent; priced-market checks not run")
            return
        if d.empty:
            if self._run_mode and not self._in_break():
                self.warn("market.present", f"{self._where(rel)} is empty outside a break: no fixtures priced")
            else:
                self.ok("market.present", f"{self._where(rel)} is empty (no fixtures priced)")
            return

        # v6.2: fixtures the Pricer skipped (incomplete moneyline, flag missing_h2h) carry no prices.
        # They used to be summed as zero and BLOCK the whole run. Now they are listed, and block only
        # if a paper pick relies on one.
        flags = (d["flags"].fillna("").astype(str).str.split(";").apply(set) if "flags" in d.columns
                 else pd.Series([set()] * len(d), index=d.index))
        skip_mask = flags.apply(lambda f: "missing_h2h" in f)
        skipped, d = d[skip_mask], d[~skip_mask]
        if len(skipped):
            names = [f"{r.get('home')} v {r.get('away')}" for _, r in skipped.iterrows()]
            ids = set(skipped["event_id"].astype(str)) if "event_id" in skipped.columns else set()
            picks = self._read_csv("data/v5/paper_picks.csv")
            relied = []
            if picks is not None and not picks.empty and "event_id" in picks.columns and ids:
                relied = sorted(set(picks.loc[picks.event_id.astype(str).isin(ids), "event_id"].astype(str)))
            if relied:
                self.block("market.skipped", f"{len(relied)} paper pick fixture(s) have no usable moneyline "
                                             f"(Pricer flag missing_h2h): {relied[:5]}")
            else:
                self.warn("market.skipped", f"{len(skipped)} fixture(s) skipped by the Pricer for an incomplete "
                                            "moneyline; no paper pick relies on them: " + "; ".join(names[:6]))
        if d.empty:
            self.ok("market.present", "every priced row was a skipped fixture; nothing else to check")
            return

        def num(cols):
            return d[cols].apply(pd.to_numeric, errors="coerce")

        trio = [c for c in ("p_home", "p_draw", "p_away") if c in d.columns]
        if len(trio) == 3:
            P = num(trio)
            bad = ~P.apply(lambda col: col.map(lambda v: math.isfinite(v))).all(axis=1)
            if bad.any():
                self.block("market.probsum", f"{int(bad.sum())} of {len(d)} priced fixtures have a missing or "
                                             "non-numeric 1X2 probability")
            else:
                s_ = P.sum(axis=1)
                off = (s_ - 1).abs() > PROB_TOL
                if off.any() or ((P < 0) | (P > 1)).any().any():
                    self.block("market.probsum", f"{int(off.sum())} of {len(d)} fixtures have 1X2 probabilities "
                                                 f"not summing to 1 or outside [0, 1] "
                                                 f"(worst off by {(s_ - 1).abs().max():.4f})")
                else:
                    self.ok("market.probsum", f"1X2 probabilities sum to 1 on all {len(d)} priced fixtures")
        else:
            self.block("market.probsum", "priced.csv has no p_home/p_draw/p_away columns")

        if "overround" in d.columns:
            o = pd.to_numeric(d.overround, errors="coerce")
            bad = ~o.map(lambda v: math.isfinite(v)) | (o < OVERROUND_MIN) | (o > OVERROUND_MAX)
            if bad.any():
                self.block("market.overround", f"{int(bad.sum())} fixtures with overround missing or outside "
                                               f"[{OVERROUND_MIN}, {OVERROUND_MAX}] "
                                               f"(min {o.min():.3f}, max {o.max():.3f})")
            else:
                self.ok("market.overround", f"overround within [{OVERROUND_MIN}, {OVERROUND_MAX}]")
        else:
            self.block("market.overround", "priced.csv has no overround column")

        if self._run_mode:
            # v6.2: the Pricer always writes these columns; lam/mu may be empty only when
            # goals_basis is "none" (no total line), and must be filled when it is "1x2+total"
            missing = [c for c in ("lam", "mu", "goals_basis") if c not in d.columns]
            if missing:
                self.block("market.goals", f"priced.csv is missing columns {missing}")
                return
            basis = d["goals_basis"].astype(str)
            unknown = sorted(set(basis) - {"none", "1x2+total"})
            if unknown:
                self.block("market.goals", f"unknown goals_basis value(s) {unknown}")
                return
            filled_but_none = d[(basis == "none") & d[["lam", "mu"]].apply(pd.to_numeric, errors="coerce")
                                .notna().any(axis=1)]
            if len(filled_but_none):
                self.block("market.goals", f"{len(filled_but_none)} fixture(s) say goals_basis none but carry "
                                           "implied goals")
                return
        gcols = [c for c in ("lam", "mu", "exp_goals_home", "exp_goals_away") if c in d.columns]
        if gcols:
            G = num(gcols)
            has_any = G.notna().any(axis=1)
            claims = (d["goals_basis"].astype(str) == "1x2+total") if "goals_basis" in d.columns else has_any
            rows = has_any | claims
            if not rows.any():
                self.skip("market.goals", "no fixture has implied goals this run (no total line), so bounds "
                                          "not checked")
            else:
                Gr = G[rows]
                bad = (~Gr.apply(lambda col: col.map(lambda v: math.isfinite(v))).all(axis=1)
                       | (Gr < GOALS_MIN).any(axis=1) | (Gr > GOALS_MAX).any(axis=1))
                if bad.any():
                    self.block("market.goals", f"{int(bad.sum())} fixtures with implied goals missing or outside "
                                               f"[{GOALS_MIN}, {GOALS_MAX}]")
                else:
                    self.ok("market.goals", f"implied goals within [{GOALS_MIN}, {GOALS_MAX}] on {int(rows.sum())} "
                                            "fixture(s)")
        else:
            self.skip("market.goals", "no implied-goals columns")
        # (v6.2: the home-shot-attempts check was removed; priced.csv never carries that column)

    # ====================================================== gate / stake checks

    def check_gate_and_stakes(self):
        bt = self._read_json("pipeline/v5/out/backtest.json")
        summary = self._read_json("data/v5/summary.json")
        picks = self._read_csv("data/v5/paper_picks.csv")

        # G1 recompute the gate from the picks themselves and compare
        mine = None
        if picks is not None and not picks.empty and "clv" in picks.columns:
            clv = pd.to_numeric(picks.clv, errors="coerce").dropna()
            n = len(clv)
            if n > 2 and clv.std(ddof=1) > 0:
                t = float(clv.mean() / (clv.std(ddof=1) / math.sqrt(n)))
                mine = bool(n >= GATE_MIN_BETS and clv.mean() > 0 and t >= GATE_MIN_T)
                detail = f"{n} picks with a closing price, mean CLV {clv.mean():+.4f}, t {t:+.2f}"
            else:
                mine, detail = False, f"{n} picks with a closing price: too few to open the gate"
        elif picks is not None:
            mine, detail = False, "no paper picks logged, so the gate cannot be open"
        else:
            detail = f"{self._where('data/v5/paper_picks.csv')} absent"
            if self._run_mode:   # the runner always writes the picks file, even when empty
                self.block("gate.present", f"{detail}: the runner output is missing")

        claimed = (bt or {}).get("gate", {}).get("open")
        if mine is None or claimed is None:
            self.skip("gate.agree", f"gate not independently checkable ({detail})")
        elif bool(claimed) != mine:
            self.block("gate.agree", f"the pipeline reports gate open={claimed} but the checker computes "
                                     f"{mine} from the picks: {detail}")
        else:
            self.ok("gate.agree", f"gate open={claimed} agrees with the checker: {detail}")

        gate_open = bool(claimed) if claimed is not None else bool(mine)

        # G2 blend weight must be 0 while the gate is shut
        weights = (self._read_json("pipeline/v5/out/weights.json")
                   or (bt or {}).get("blend_weights")
                   or ({"runner": (summary or {})["blend_weight"]} if "blend_weight" in (summary or {}) else {}))
        if isinstance(weights, dict) and weights:
            nz = dict(_numeric_leaves(weights))
            nz = {k: v for k, v in nz.items() if v != 0}
            if nz and not gate_open:
                self.block("gate.blend", f"gate is closed but blend weight is non-zero: {nz}")
            else:
                self.ok("gate.blend", "blend weights consistent with the gate"
                                      f"{' (all 0)' if not nz else ''}")
        else:
            self.skip("gate.blend", "no blend weights published this run")

        # G3 nothing may carry a stake while the gate is shut, and picks must say paper
        if picks is not None and not picks.empty:
            staked = [c for c in ("stake", "stake_units", "size", "bankroll_fraction") if c in picks.columns]
            offenders = 0
            for c in staked:
                offenders += int((pd.to_numeric(picks[c], errors="coerce").fillna(0) > 0).sum())
            if offenders and not gate_open:
                self.block("gate.stakes", f"{offenders} pick rows carry a stake while the gate is closed")
            else:
                self.ok("gate.stakes", "no stake on any pick" + ("" if staked else " (no stake column)"))

            mode = (summary or {}).get("mode", "")
            if "paper" not in str(mode).lower():
                self.block("gate.paper", f"summary mode is {mode!r}, not paper, while the gate is closed")
            else:
                self.ok("gate.paper", f"summary marked {mode!r}")

            # G4 picks only from the four modelled leagues
            if "comp" in picks.columns:
                names = set(LEAGUES) | set(LEAGUES.values()) | {"PL", "LaLiga", "Bundesliga", "Ligue1"}
                outside = sorted(set(picks.comp.dropna().astype(str)) - names)
                if outside:
                    self.block("gate.scope", f"picks from out-of-scope competitions: {outside}")
                else:
                    self.ok("gate.scope", "all picks are in-scope competitions")
        else:
            in_break = self._in_break()
            self.ok("gate.stakes", "no picks logged" +
                    (" (international break: expected)" if in_break else ""))
            if picks is None:
                self.skip("gate.paper", "no picks file to check the paper flag on")

        # S5 staleness
        for label, blob, key in (("backtest", bt, "generated"), ("paper summary", summary, "generated")):
            ts = (blob or {}).get(key)
            if not ts:
                continue
            try:
                when = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - when).days
                (self.warn if age > STALE_DAYS else self.ok)(
                    f"stale.{label.split()[0]}", f"{label} generated {age} day(s) ago")
            except ValueError:
                self.warn(f"stale.{label.split()[0]}", f"{label} timestamp unparseable: {ts!r}")

        # S4 how much CLV leaned on the fallback basis
        if picks is not None and not picks.empty and "clv_basis" in picks.columns:
            vc = picks.clv_basis.value_counts(dropna=True)
            grid = int(vc.get("grid", 0))
            total = int(vc.sum())
            if total and grid / total > 0.5:
                self.warn("picks.clv_basis", f"{grid} of {total} picks scored CLV from the closing goals grid "
                                             "rather than the same selection")
            elif total:
                self.ok("picks.clv_basis", f"{grid} of {total} picks used the grid fallback")

    # ================================================================ banker

    def check_bank(self):
        """Banker rules: no stakes while the gate is closed, unit = $2 per $100 bracket, 1-3 units a
        bet with stake = units x unit, today's new stakes within 10% of the bankroll, and the bankroll
        recomputed from the exported ledger. SKIP when this run has no Banker output."""
        if self.run_dir is None or not (self.run_dir / "banker").exists():
            for c in ("bank.gate", "bank.unit", "bank.tiers", "bank.daycap", "bank.bankroll"):
                self.skip(c, "no Banker output in this run")
            return
        plan = self._read_json("banker/plan.json")
        bank = self._read_json("banker/bankroll.json")
        if plan is None or bank is None:
            self.block("bank.files", "Banker ran but plan.json or bankroll.json is missing")
            return
        summary = self._read_json("data/v5/summary.json") or {}
        gate_open = bool(summary.get("gate_open", False))
        bets = plan.get("bets") or []

        if plan.get("gate_open") != gate_open:
            self.block("bank.gate", f"Banker thinks gate open={plan.get('gate_open')}, the runner says {gate_open}")
        elif not gate_open and (bets or any("stake" in r for r in plan.get("preview") or [])):
            self.block("bank.gate", f"gate is closed but the Banker plan carries {len(bets)} staked bet(s)")
        else:
            self.ok("bank.gate", "no stakes while the gate is closed" if not gate_open
                    else f"gate open: {len(bets)} bet(s) sized")

        b = float(bank.get("bankroll") or 0)
        want = 0.0 if b <= 0 else BANK_UNIT_PER_100 * math.ceil(round(b, 2) / 100)
        if abs(float(bank.get("unit") or 0) - want) > 1e-9 or abs(float(plan.get("unit") or 0) - want) > 1e-9:
            self.block("bank.unit", f"bankroll ${b:.2f} needs a ${want:.0f} unit; Banker used "
                                    f"${bank.get('unit')} / ${plan.get('unit')}")
        else:
            self.ok("bank.unit", f"bankroll ${b:.2f} -> ${want:.0f} unit")

        def units_ok(u):   # an int 1..3; True/False are ints in Python and are rejected explicitly
            return isinstance(u, int) and not isinstance(u, bool) and 1 <= u <= BANK_MAX_UNITS

        def stake_ok(r):
            try:
                return abs(float(r.get("stake")) - r["units"] * want) <= 0.005
            except (TypeError, ValueError):
                return False

        bad = [r for r in bets if not units_ok(r.get("units")) or not stake_ok(r)]
        if bad:
            self.block("bank.tiers", f"{len(bad)} bet(s) outside 1-{BANK_MAX_UNITS} units or with stake != "
                                     f"units x ${want:.0f}: " + "; ".join(str(r.get("label")) for r in bad[:3]))
        else:
            self.ok("bank.tiers", f"{len(bets)} bet(s), all 1-{BANK_MAX_UNITS} units at ${want:.0f}"
                    + (", parlays one slot each" if any(r.get("kind") == "parlay" for r in bets) else ""))

        cap = BANK_DAY_CAP * b
        used = float(plan.get("placed_today") or 0) + sum(float(r.get("stake") or 0) for r in bets)
        if used > cap + 0.005:
            self.block("bank.daycap", f"today's new stakes ${used:.2f} exceed 10% of ${b:.2f} (${cap:.2f})")
        else:
            self.ok("bank.daycap", f"today's new stakes ${used:.2f} of a ${cap:.2f} cap")

        lp = self.run_dir.parent / "bankroll" / "ledger.json"
        if not lp.exists():
            (self.block if gate_open else self.skip)(
                "bank.bankroll", "no exported bankroll ledger" + ("" if gate_open else " (gate closed)"))
            return
        try:
            d = json.loads(lp.read_text())
            dep = sum(float(t["amount"]) for t in d.get("transactions", []) if t["kind"] == "deposit")
            wd = sum(float(t["amount"]) for t in d.get("transactions", []) if t["kind"] == "withdrawal")
            pnl = sum(float(x.get("pnl") or 0) for x in d.get("bets", []) if x.get("status") in ("W", "L", "P", "V"))
            mine = round(dep - wd + pnl, 2)
        except Exception as e:                                   # noqa: BLE001
            self.block("bank.bankroll", f"bankroll ledger unreadable: {e}")
            return
        if abs(mine - b) > 0.005:
            self.block("bank.bankroll", f"ledger gives ${mine:.2f} but the Banker used ${b:.2f}")
        else:
            self.ok("bank.bankroll", f"${b:.2f} = deposits ${dep:.2f} - withdrawals ${wd:.2f} + settled P&L ${pnl:+.2f}")

    # =============================================================== run / report

    def run(self) -> dict:
        self.check_upstream()
        self.check_history()
        self.check_ratings()
        self.check_market()
        self.check_gate_and_stakes()
        self.check_bank()

        counts = {s: sum(1 for r in self.results if r.severity == s)
                  for s in ("BLOCK", "WARN", "PASS", "SKIP")}
        verdict = "BLOCK" if counts["BLOCK"] else ("WARN" if counts["WARN"] else "PASS")
        return {
            "agent": "checker",
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "root": str(self.root),
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "verdict": verdict,
            "counts": counts,
            "blocking": [r.as_dict() for r in self.results if r.severity == "BLOCK"],
            "warnings": [r.as_dict() for r in self.results if r.severity == "WARN"],
            "checks": [r.as_dict() for r in self.results],
            "break_until": self.break_until,
        }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pulsematch Checker agent (read-only).")
    ap.add_argument("--root", default=None, help="repo root (default: two levels above this file)")
    ap.add_argument("--out", default=None, help="write the verdict JSON here")
    ap.add_argument("--baseline", default=None, help="row-count baseline JSON")
    ap.add_argument("--break-until", default=None, help="ISO date; no fixtures expected up to it")
    ap.add_argument("--run-dir", default=None, help="this run's folder: check its pricer/runner outputs, not the repo's")
    ap.add_argument("--strict", action="store_true", help="treat WARN as BLOCK")
    ap.add_argument("--quiet", action="store_true", help="print the verdict only")
    a = ap.parse_args(argv)

    root = Path(a.root).resolve() if a.root else Path(__file__).resolve().parents[2]
    try:
        c = Checker(root, a.break_until, Path(a.baseline).resolve() if a.baseline else None,
                    Path(a.run_dir).resolve() if a.run_dir else None)
        report = c.run()
    except Exception as e:                                      # noqa: BLE001
        print(f"CHECKER FAILED TO RUN: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))

    if not a.quiet:
        for r in report["checks"]:
            if r["severity"] != "PASS":
                print(f"  {r['severity']:5} {r['id']}: {r['detail']}")
    c_ = report["counts"]
    print(f"{report['verdict']}  ({c_['BLOCK']} blocking, {c_['WARN']} warning, "
          f"{c_['PASS']} passed, {c_['SKIP']} skipped)")

    if report["verdict"] == "BLOCK":
        return 1
    if report["verdict"] == "WARN" and a.strict:
        print("strict mode: a warning blocks publishing")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
