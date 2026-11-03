"""Make sure today's football-data refresh (the fd-history workflow) has run before a job reads data/fd/.

Why (2026-11-03): fd-history is scheduled at 14:17 UTC on Mondays, but GitHub started it at 22:47 UTC on
Nov 2. The weekly refit (20:37 UTC) and the fixtures job's corners fit (22:17 UTC) both ran before it and
fitted on football-data that stopped at Oct 18-21, missing all of week 9.

  python3 scripts/ensure_fd.py [--since YYYY-MM-DDTHH:MMZ] [--wait-min 20]

1. Looks for an fd-history run created since --since (default: today 00:00 UTC) that finished successfully.
2. If one is queued or running, waits for it. If there is none, starts one (gh workflow run) and waits.
3. Then pulls the repo (git pull --rebase) and prints each league's latest match date in data/fd/.
Exit 0 = a successful run since --since exists and the clone is up to date; 3 = it could not confirm one
(gh missing or not allowed, the run failed, or it timed out). On 3 the caller carries on but must say in its
report that football-data may be stale, quoting the printed dates.
Needs the gh CLI with permission to run workflows on alcadiomontes03/match-engine-odds (the session's push
access from add_repo is enough).
"""
from __future__ import annotations

import argparse, csv, json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "alcadiomontes03/match-engine-odds"
WF = "fd-history.yml"


def gh(*args, timeout=60):
    return subprocess.run(["gh", *args, "-R", REPO], capture_output=True, text=True, timeout=timeout)


def runs():
    r = gh("run", "list", "--workflow", WF, "-L", "10", "--json", "databaseId,status,conclusion,createdAt,event")
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or r.stdout.strip())
    return json.loads(r.stdout or "[]")


def fd_dates():
    out = {}
    for code in ("E0", "SP1", "D1", "F1"):
        files = sorted((ROOT / "data" / "fd").glob(f"{code}_*.csv"))
        if not files:
            continue
        with files[-1].open(newline="") as fh:
            ds = [r["Date"] for r in csv.DictReader(fh) if r.get("Date")]
        if ds:
            out[code] = max(datetime.strptime(d, "%d/%m/%Y" if len(d) > 8 else "%d/%m/%y") for d in ds).date().isoformat()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None)
    ap.add_argument("--wait-min", type=float, default=20)
    a = ap.parse_args()
    now = datetime.now(timezone.utc)
    since = (datetime.fromisoformat(a.since.replace("Z", "+00:00")) if a.since
             else now.replace(hour=0, minute=0, second=0, microsecond=0))
    deadline = time.time() + a.wait_min * 60
    started = False
    ok = False
    try:
        while True:
            recent = [r for r in runs() if datetime.fromisoformat(r["createdAt"].replace("Z", "+00:00")) >= since]
            if any(r["status"] == "completed" and r["conclusion"] == "success" for r in recent):
                ok = True
                break
            active = [r for r in recent if r["status"] != "completed"]
            if not active and not started:
                d = gh("workflow", "run", WF, "--ref", "main")
                if d.returncode != 0:
                    raise RuntimeError("could not start fd-history: " + (d.stderr.strip() or d.stdout.strip()))
                print(f"started fd-history (no successful run since {since.isoformat(timespec='minutes')})", flush=True)
                started = True
            elif not active and started and recent and all(r["status"] == "completed" for r in recent):
                print("fd-history finished without success: " + ", ".join(f"{r['databaseId']} {r['conclusion']}" for r in recent))
                break
            if time.time() > deadline:
                print(f"timed out after {a.wait_min} min waiting for fd-history")
                break
            time.sleep(30)
    except Exception as e:  # noqa: BLE001
        print(f"ensure_fd: {e}")
    subprocess.run(["git", "-C", str(ROOT), "pull", "-q", "--rebase"], capture_output=True, text=True)
    print(json.dumps({"fd_history_ok": ok, "since": since.isoformat(timespec="minutes"), "fd_last": fd_dates()}))
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
