"""Per-match expected goals (xG) from understat.com for the four leagues, 2017-18 onward.

Runs in GitHub Actions (open internet); Claude's sandbox can't reach understat. Writes one CSV per
league and season to data/xg/understat_<code>_<season>.csv with columns:
    id, datetime (UTC as understat gives it), home, away, home_goals, away_goals, home_xg, away_xg, is_result
Finished seasons are fetched once and skipped afterwards; the current season is refreshed every run.

Understat has served its data two ways: JSON embedded in the league page (`datesData = JSON.parse('...')`)
and, more recently, a JSON endpoint (`/getLeagueData/<league>/<season>`). Both are tried. Polite: one
request every 3 seconds, a normal browser User-Agent, at most ~40 requests a run.

  python scripts/fetch_xg.py [--seasons 2017-2026] [--force]
"""
from __future__ import annotations

import argparse, codecs, csv, json, re, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "xg"
LEAGUES = {"E0": "EPL", "SP1": "La_liga", "D1": "Bundesliga", "F1": "Ligue_1"}
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
COLS = ["id", "datetime", "home", "away", "home_goals", "away_goals", "home_xg", "away_xg", "is_result"]


def _get(url, ajax=False):
    h = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}
    if ajax:
        h.update({"X-Requested-With": "XMLHttpRequest", "Accept": "application/json, text/javascript, */*",
                  "Referer": url.replace("/getLeagueData/", "/league/")})
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read().decode("utf-8", "replace")


def _from_page(html):
    m = re.search(r"datesData\s*=\s*JSON\.parse\('(.*?)'\)", html, re.S)
    if not m:
        return None
    return json.loads(codecs.decode(m.group(1), "unicode_escape"))


def _from_api(txt):
    d = json.loads(txt)
    for k in ("dates", "datesData", "matches"):
        if isinstance(d, dict) and k in d:
            return d[k]
    return d if isinstance(d, list) else None


def fetch(league, season):
    errors = []
    try:
        rows = _from_page(_get(f"https://understat.com/league/{league}/{season}"))
        if rows:
            return rows, "page"
        errors.append("page: no datesData")
    except Exception as e:  # noqa: BLE001
        errors.append(f"page: {e}")
    time.sleep(3)
    try:
        rows = _from_api(_get(f"https://understat.com/getLeagueData/{league}/{season}", ajax=True))
        if rows:
            return rows, "api"
        errors.append("api: empty")
    except Exception as e:  # noqa: BLE001
        errors.append(f"api: {e}")
    raise RuntimeError("; ".join(errors))


def flat(m):
    g, x = m.get("goals") or {}, m.get("xG") or {}
    return {"id": m.get("id"), "datetime": m.get("datetime"),
            "home": (m.get("h") or {}).get("title"), "away": (m.get("a") or {}).get("title"),
            "home_goals": g.get("h"), "away_goals": g.get("a"), "home_xg": x.get("h"), "away_xg": x.get("a"),
            "is_result": m.get("isResult")}


def current_season(now):
    return now.year if now.month >= 7 else now.year - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2017-")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    now = datetime.now(timezone.utc)
    cur = current_season(now)
    lo, _, hi = a.seasons.partition("-")
    seasons = range(int(lo), int(hi or cur) + 1)
    OUT.mkdir(parents=True, exist_ok=True)
    log, bad = [], 0
    for code, lg in LEAGUES.items():
        for s in seasons:
            f = OUT / f"understat_{code}_{s}.csv"
            if f.exists() and s < cur and not a.force:
                continue
            try:
                rows, how = fetch(lg, s)
            except Exception as e:  # noqa: BLE001
                bad += 1
                log.append(f"FAIL {code} {s}: {e}")
                print(log[-1], flush=True)
                time.sleep(3)
                continue
            flat_rows = [flat(m) for m in rows]
            with f.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=COLS)
                w.writeheader(); w.writerows(flat_rows)
            n_res = sum(str(r["is_result"]).lower() == "true" for r in flat_rows)
            log.append(f"OK {code} {s}: {len(flat_rows)} matches ({n_res} played) via {how}")
            print(log[-1], flush=True)
            time.sleep(3)
    (OUT / "last_fetch.txt").write_text(now.isoformat(timespec="seconds") + "\n" + "\n".join(log) + "\n")
    sys.exit(1 if bad and not any(l.startswith("OK") for l in log) else 0)


if __name__ == "__main__":
    main()
