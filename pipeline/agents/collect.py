"""
Pulsematch — Data collector agent.

What it does, and what it deliberately does not do
--------------------------------------------------
The raw fetching is already done by two GitHub Actions in the odds repo (football-data
history on Mondays, DraftKings odds capture through the week). Those are reliable and are
not truncated the way a WebFetch from a chat session is. So the collector does not
re-fetch. Its job is the one nothing else does:

  1. Take stock of everything the run is about to use (history files and the odds capture
     state), and prove it is complete and fresh enough to build on.
  2. Write that stock-take, with file hashes, into a per-run folder in Drive so that every
     later agent, and the Checker, reads the same facts.
  3. Say clearly when something is wrong, so the run stops before bad data is priced.

It is read-only against the repo. It writes only into the run folder.
v6.2: refuses to write into a collector folder that already holds files, and reports a history
file not named CODE_SSSS.csv as a problem instead of crashing on it.

    python collect.py --root <repo checkout> --runs-dir <runs folder>
    python collect.py --root . --runs-dir ./runs --break-until 2026-10-09

Exit code: 0 = usable (OK or PARTIAL), 1 = FAILED (downstream agents must not run),
2 = the collector itself crashed (treat as FAILED).

Status
  OK       everything present and complete
  PARTIAL  usable, but something should be looked at (reason in manifest warnings)
  FAILED   no usable history; stop the run

The Checker, not this agent, decides whether a run may be published. The collector
reports; it does not grade.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import manifest as M  # noqa: E402
from checker import ABANDONED, LEAGUES  # noqa: E402  (one source of truth for scope and short seasons)

RESULTS_STALE_DAYS = 10     # no match in the current season for this long, outside a break
CAPTURE_STALE_DAYS = 8      # the weekly odds job should have run inside this window
CAPTURE_GRACE_HOURS = 6     # a pull is written within this of its run being logged
QUOTA_FLOOR = 100           # BTTS capture pauses at or below this many credits
NAME_RE = re.compile(r"^([A-Za-z0-9]+)_(\d{4})$")   # e.g. E0_2627: league code, season


def parse_name(f: Path):
    """(code, season) from a history filename, or None if it does not follow CODE_SSSS.csv."""
    m = NAME_RE.match(f.stem)
    return (m.group(1), m.group(2)) if m else None


# ---------------------------------------------------------------- history

def inventory_history(root: Path, today: datetime):
    """Per-season-file facts, plus the in-progress season per league."""
    fd = root / "data" / "fd"
    out = {"files": [], "current": {}, "problems": [], "warnings": [], "in_scope_codes": []}
    if not fd.is_dir():
        out["problems"].append("data/fd is missing")
        return out
    files = sorted(fd.glob("*.csv"))
    if not files:
        out["problems"].append("data/fd holds no CSV files")
        return out

    bad_names = [f.name for f in files if parse_name(f) is None]
    for name in bad_names:
        out["problems"].append(f"{name} does not follow the CODE_SEASON.csv naming (e.g. E0_2627.csv); ignored")
    files = [f for f in files if parse_name(f) is not None]
    codes = {parse_name(f)[0] for f in files}
    out["in_scope_codes"] = sorted(codes & set(LEAGUES))
    for extra in sorted(codes - set(LEAGUES)):
        out["warnings"].append(f"out-of-scope league file(s) in data/fd: {extra}")
    for missing in sorted(set(LEAGUES) - codes):
        out["problems"].append(f"no history at all for {missing} ({LEAGUES[missing]})")

    latest = {}
    for f in files:
        code, season = parse_name(f)
        if code in LEAGUES:
            latest[code] = max(latest.get(code, season), season)

    for f in files:
        code, season = parse_name(f)
        if code not in LEAGUES:
            continue
        try:
            d = pd.read_csv(f)
        except Exception as e:  # noqa: BLE001
            out["problems"].append(f"{f.name} is unreadable: {e}")
            continue
        d2 = d.dropna(subset=[c for c in ("HomeTeam", "AwayTeam") if c in d.columns])
        teams = len(set(d2.get("HomeTeam", [])) | set(d2.get("AwayTeam", [])))
        expected = teams * (teams - 1)
        dates = pd.to_datetime(d.get("Date"), format="mixed", dayfirst=True, errors="coerce")
        is_current = season == latest[code]
        if (code, season) in ABANDONED and len(d) == ABANDONED[(code, season)][0]:
            status = "known-short"
        elif is_current:
            status = "complete" if len(d) == expected else "in-progress"
        else:
            status = "complete" if len(d) == expected else "INCOMPLETE"
        shots = None
        if {"HST", "AST"} <= set(d.columns) and len(d):
            shots = round(float(min(d.HST.notna().mean(), d.AST.notna().mean())), 4)
        rec = {"file": f.name, "league": code, "season": season, "rows": int(len(d)), "teams": int(teams),
               "expected_rows": int(expected), "status": status,
               "first_match": None if dates.isna().all() else dates.min().date().isoformat(),
               "last_match": None if dates.isna().all() else dates.max().date().isoformat(),
               "shots_coverage": shots, "sha256": M.sha256(f)}
        out["files"].append(rec)
        if status == "INCOMPLETE":
            out["problems"].append(f"{f.name}: {len(d)} of {expected} rows ({teams} teams) in a completed season")
        if is_current:
            last = dates.max()
            age = None if pd.isna(last) else (today.date() - last.date()).days
            out["current"][code] = {"file": f.name, "rows": int(len(d)), "expected_rows": int(expected),
                                    "last_match": rec["last_match"], "days_since_last_match": age}
    return out


# ------------------------------------------------------------------- odds

def inventory_odds(root: Path, today: datetime):
    o = {"present": False, "warnings": [], "notes": []}
    snap_p = root / "data" / "odds" / "snapshots.csv"
    run_p = root / "state" / "last_run.json"
    pulled_p = root / "state" / "pulled.json"
    quota_p = root / "data" / "odds" / "quota.csv"

    last_pull = None
    if snap_p.exists():
        o["present"] = True
        s = pd.read_csv(snap_p)
        o["snapshots_rows"] = int(len(s))
        if len(s) and "pulled_at" in s.columns:
            last_pull = pd.to_datetime(s.pulled_at, utc=True, errors="coerce").max()
            o["last_pulled_at"] = None if pd.isna(last_pull) else last_pull.isoformat(timespec="seconds")
            o["days_since_last_pull"] = None if pd.isna(last_pull) else (today - last_pull).days
        if "comp" in s.columns:
            counts = s.comp.value_counts().to_dict()
            o["rows_by_comp"] = {str(k): int(v) for k, v in counts.items()}
            known = set(LEAGUES.values()) | set(LEAGUES) | {"PL", "LaLiga", "Bundesliga", "Ligue1"}
            outside = sorted(str(c) for c in counts if str(c) not in known)
            if outside:
                o["notes"].append("earlier captures from competitions no longer modelled are still in "
                                  f"snapshots.csv: {outside}; they are history, not an error")
    else:
        o["warnings"].append("data/odds/snapshots.csv is missing")

    last_run = None
    if run_p.exists():
        try:
            lr = json.loads(run_p.read_text())
            when = pd.Timestamp(lr["utc"])
            if when.tzinfo is None:
                when = when.tz_localize("UTC")
            age = (today - when).days
            leagues = lr.get("leagues", {})
            listed = sum(int(v.get("events_from_api", 0)) for v in leagues.values())
            due_w = sum(int(v.get("due_weekly", 0)) for v in leagues.values())
            due_l = sum(int(v.get("due_late", 0)) for v in leagues.values())
            last_run = when
            o["last_run"] = {"utc": when.isoformat(timespec="seconds"), "age_days": age,
                             "fixtures_listed_by_api": listed, "due_weekly": due_w, "due_late": due_l,
                             "markets": lr.get("markets"), "bookmakers": lr.get("bookmakers")}
            if age > CAPTURE_STALE_DAYS:
                o["warnings"].append(f"the odds capture last ran {age} days ago "
                                     f"(expected inside {CAPTURE_STALE_DAYS})")
            if (due_w + due_l) > 0 and last_pull is not None:
                gap_h = (when - last_pull).total_seconds() / 3600
                if gap_h > CAPTURE_GRACE_HOURS:
                    o["warnings"].append(f"the last capture run listed {due_w + due_l} fixtures as due but the "
                                         f"newest snapshot is {gap_h:.0f}h older than that run: the pull "
                                         "may have failed silently")
            if (due_w + due_l) > 0 and last_pull is None:
                o["warnings"].append("fixtures were due but no snapshot exists")
        except Exception as e:  # noqa: BLE001
            o["warnings"].append(f"state/last_run.json is unreadable: {e}")
    else:
        o["warnings"].append("state/last_run.json is missing, so capture health cannot be confirmed")

    if pulled_p.exists():
        try:
            meta = json.loads(pulled_p.read_text()).get("_meta", {})
            o["weekly_marks"] = {k.split(":", 1)[1]: v for k, v in meta.items() if k.startswith("weekly:")}
        except Exception as e:  # noqa: BLE001
            o["warnings"].append(f"state/pulled.json is unreadable: {e}")

    if quota_p.exists():
        q = pd.read_csv(quota_p)
        if len(q):
            r = q.iloc[-1]
            o["quota"] = {"remaining": int(r.remaining), "as_of": str(r.utc)}
            if int(r.remaining) <= QUOTA_FLOOR:
                o["warnings"].append(f"only {int(r.remaining)} odds-API credits left; BTTS capture pauses at "
                                     f"{QUOTA_FLOOR} or below")
    return o


# ------------------------------------------------------------------- run

def collect(root: Path, runs_dir: Path, run_id: str | None = None,
            break_until: str | None = None, now: datetime | None = None) -> dict:
    today = now or datetime.now(timezone.utc)
    run_id = run_id or M.new_run_id(today)
    started = M.utc_now()
    folder = runs_dir / run_id / "collector"
    if folder.exists() and any(folder.iterdir()):
        # v6.2: never mix two runs' files in one folder (run ids used to collide within a minute)
        raise FileExistsError(f"{folder} already holds files from an earlier run; use a new run id")
    folder.mkdir(parents=True, exist_ok=True)

    in_break = False
    if break_until:
        try:
            in_break = today.date() <= datetime.fromisoformat(break_until).date()
        except ValueError:
            pass

    hist = inventory_history(root, today)
    odds = inventory_odds(root, today)

    warnings = list(hist["warnings"]) + list(odds["warnings"])
    notes = list(odds["notes"])
    if in_break:
        notes.append(f"international break until {break_until}: no new results or fixtures are expected")

    if not in_break:
        for code, c in hist["current"].items():
            age = c["days_since_last_match"]
            if age is not None and age > RESULTS_STALE_DAYS:
                warnings.append(f"{code}: newest result in {c['file']} is {age} days old outside a break")
            if c["rows"] == 0:
                warnings.append(f"{code}: current-season file {c['file']} has no rows")

    if hist["problems"]:
        # a missing league or unreadable folder means there is nothing safe to build on
        fatal = [p for p in hist["problems"] if "no history" in p or "missing" in p
                 or "no CSV" in p or "unreadable" in p]
        status = "FAILED" if fatal else "PARTIAL"
        warnings = hist["problems"] + warnings
    else:
        status = "PARTIAL" if warnings else "OK"

    completeness = {
        "generated": M.utc_now(), "run_id": run_id, "root": str(root),
        "in_break": in_break, "break_until": break_until,
        "history": {"files": hist["files"], "current_season": hist["current"],
                    "summary": {"files": len(hist["files"]),
                                "complete": sum(f["status"] == "complete" for f in hist["files"]),
                                "in_progress": sum(f["status"] == "in-progress" for f in hist["files"]),
                                "known_short": sum(f["status"] == "known-short" for f in hist["files"]),
                                "incomplete": sum(f["status"] == "INCOMPLETE" for f in hist["files"]),
                                "total_rows": sum(f["rows"] for f in hist["files"])}},
        "odds_capture": {k: v for k, v in odds.items() if k not in ("warnings", "notes")},
    }
    (folder / "completeness.json").write_text(json.dumps(completeness, indent=2))

    inputs = [{"path": f"data/fd/{f['file']}", "sha256": f["sha256"], "rows": f["rows"]} for f in hist["files"]]
    row_counts = {Path(f["file"]).stem: f["rows"] for f in hist["files"]}
    if "snapshots_rows" in odds:
        row_counts["odds_snapshots"] = odds["snapshots_rows"]

    return M.write_manifest(folder, agent="collector", run_id=run_id, status=status, started=started,
                            inputs=inputs, row_counts=row_counts, warnings=warnings, notes=notes)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pulsematch Data collector agent.")
    ap.add_argument("--root", required=True, help="checkout of the odds repo (data/fd, data/odds, state)")
    ap.add_argument("--runs-dir", required=True, help="folder that holds the per-run folders")
    ap.add_argument("--run-id", default=None, help="default: UTC start time, e.g. 2026-10-05T1600Z")
    ap.add_argument("--break-until", default=None, help="ISO date; no fixtures expected up to it")
    a = ap.parse_args(argv)
    try:
        m = collect(Path(a.root).resolve(), Path(a.runs_dir).resolve(), a.run_id, a.break_until)
    except Exception as e:  # noqa: BLE001
        print(f"COLLECTOR CRASHED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    for w in m["warnings"]:
        print(f"  WARN  {w}")
    for n in m["notes"]:
        print(f"  note  {n}")
    print(f"{m['status']}  run {m['run_id']}  ({len(m['inputs'])} history files, "
          f"{sum(v for k, v in m['row_counts'].items() if k != 'odds_snapshots')} rows)")
    return 1 if m["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
