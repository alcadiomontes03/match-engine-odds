"""Results and match stats for dashboard fixtures, from the repo's football-data.co.uk files (data/fd/).

Why (2026-11-03): the scheduled grading jobs relied on the sportradar scores feed (fetch_sports_data), which
is not available to scheduled sessions. On Nov 2 the Monday grading job wrote no results at all, so the
refit, the research record and the Brier tile all missed week 9. football-data.co.uk is refreshed into
data/fd/ by the fd-history GitHub workflow, so every job can read it from a plain clone.

  python3 scripts/fd_results.py --db <ArtifactData export with fixtures/ and results/> --out <dir> [--now ISO]

Writes, for ArtifactData file_path writes:
  <out>/new/<fixture_id>.json    a new results doc {actual, home_score, away_score, settled_at, source, stats,
                                 stats_source} for a fixture that kicked off > 2.5 h ago and has no result
  <out>/stats/<fixture_id>.json  {stats, stats_source} for an existing result (kicked off in the last 10 days)
                                 that has no stats yet
  <out>/report.json              counts, plus "missing" (fixtures due a result that football-data doesn't have
                                 yet: use the fallback source for these), "stats_pending" (a result with no stats
                                 whose match football-data doesn't have yet, e.g. a Monday-night game) and
                                 "mismatch" (an existing result whose score differs from football-data: never
                                 overwritten, report it)
It never changes an existing result's score. Possession is not in football-data, so stats.*.poss is null.
Matching: same league, the fixture's football-data short names (fixtures.home/away use the same convention),
and the match date within 1 day of the kickoff's UTC date.
"""
from __future__ import annotations

import argparse, glob, json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FD = ROOT / "data" / "fd"
LEAGUES = ("E0", "SP1", "D1", "F1")


def load(db: Path, coll: str) -> dict:
    out = {}
    for f in glob.glob(str(db / coll / "*.json")):
        d = json.loads(Path(f).read_text())
        if isinstance(d, dict) and "data" in d and "id" not in d:
            d = d["data"]
        out[Path(f).stem] = d
    return out


def season_file(code: str, ko: datetime) -> Path:
    y = ko.year if ko.month >= 7 else ko.year - 1
    return FD / f"{code}_{y % 100:02d}{(y + 1) % 100:02d}.csv"


def num(v):
    try:
        if pd.isna(v):
            return None
        return int(float(v))
    except (TypeError, ValueError):
        return None


def stats_from(row) -> dict | None:
    g = lambda c: num(row[c]) if c in row.index else None
    if g("HS") is None or g("AS") is None:
        return None
    side = lambda p: {"shots": g(f"{p}S"), "sot": g(f"{p}ST"), "corners": g(f"{p}C"), "poss": None,
                      "yc": g(f"{p}Y"), "rc": g(f"{p}R")}
    return {"h": side("H"), "a": side("A")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--now", default=None)
    a = ap.parse_args()
    db, out = Path(a.db), Path(a.out)
    now = datetime.fromisoformat(a.now.replace("Z", "+00:00")) if a.now else datetime.now(timezone.utc)
    (out / "new").mkdir(parents=True, exist_ok=True)
    (out / "stats").mkdir(parents=True, exist_ok=True)
    fx, res = load(db, "fixtures"), load(db, "results")
    cache: dict[Path, pd.DataFrame | None] = {}
    rep = {"now": now.isoformat(timespec="seconds"), "fd_last": {}, "new": [], "stats": [], "missing": [],
           "mismatch": [], "no_stats_in_fd": [], "stats_pending": []}
    for code in LEAGUES:
        f = season_file(code, now)
        if f.exists():
            d = pd.read_csv(f, dayfirst=True)
            rep["fd_last"][code] = str(pd.to_datetime(d.Date, dayfirst=True).max().date())
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    for fid, f in sorted(fx.items()):
        if f.get("league") not in LEAGUES or f.get("status") == "postponed":
            continue
        ko = datetime.fromisoformat(str(f["kickoff"]).replace("Z", "+00:00"))
        if ko > now - timedelta(hours=2.5):
            continue
        r = res.get(fid)
        need_result = r is None or "home_score" not in r
        need_stats = (not need_result) and not r.get("stats") and ko >= now - timedelta(days=10)
        if not (need_result or need_stats):
            continue
        p = season_file(f["league"], ko)
        if p not in cache:
            cache[p] = pd.read_csv(p, dayfirst=True) if p.exists() else None
            if cache[p] is not None:
                cache[p]["_d"] = pd.to_datetime(cache[p].Date, dayfirst=True).dt.date
        df = cache[p]
        row = None
        if df is not None:
            m = df[(df.HomeTeam == f["home"]) & (df.AwayTeam == f["away"])]
            m = m[m._d.map(lambda d: abs((d - ko.date()).days) <= 1)]
            if len(m) == 1:
                row = m.iloc[0]
        label = f"{f['league']} {f['home']} v {f['away']} {ko.date()}"
        if row is None or num(row.FTHG) is None:
            rep["missing" if need_result else "stats_pending"].append({"fixture_id": fid, "match": label})
            continue
        hs, as_ = num(row.FTHG), num(row.FTAG)
        st = stats_from(row)
        if need_result:
            doc = {"actual": "H" if hs > as_ else "A" if as_ > hs else "D", "home_score": hs, "away_score": as_,
                   "settled_at": stamp, "source": "football-data.co.uk (data/fd)"}
            if st:
                doc.update(stats=st, stats_source="football-data.co.uk (data/fd)")
            (out / "new" / f"{fid}.json").write_text(json.dumps(doc))
            rep["new"].append({"fixture_id": fid, "match": label, "score": f"{hs}-{as_}", "stats": bool(st)})
        else:
            if (r.get("home_score"), r.get("away_score")) != (hs, as_):
                rep["mismatch"].append({"fixture_id": fid, "match": label,
                                        "dashboard": f"{r.get('home_score')}-{r.get('away_score')}",
                                        "football_data": f"{hs}-{as_}"})
                continue
            if st:
                (out / "stats" / f"{fid}.json").write_text(json.dumps({"stats": st,
                                                                        "stats_source": "football-data.co.uk (data/fd)"}))
                rep["stats"].append({"fixture_id": fid, "match": label})
            else:
                rep["no_stats_in_fd"].append({"fixture_id": fid, "match": label})
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: (v if k in ("now", "fd_last") else len(v)) for k, v in rep.items()}))
    for k in ("missing", "mismatch", "no_stats_in_fd", "stats_pending"):
        for x in rep[k]:
            print(f"{k.upper()}: {x['match']}" + (f" (dashboard {x['dashboard']}, football-data {x['football_data']})"
                                                  if k == "mismatch" else ""))


if __name__ == "__main__":
    main()
