"""One-off probe (2026-11-01): what does understat (and alternatives) return to a GitHub runner?
Writes data/xg/probe.txt. Short timeouts; no retries."""
import json, time, urllib.request, urllib.error
from pathlib import Path
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
URLS = [
    ("understat league page", "https://understat.com/league/EPL/2025", False),
    ("understat getLeagueData", "https://understat.com/getLeagueData/EPL/2025", True),
    ("understat home", "https://understat.com/", False),
    ("fbref EPL schedule", "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures", False),
    ("fotmob api leagues", "https://www.fotmob.com/api/leagues?id=47", False),
    ("sofascore api", "https://api.sofascore.com/api/v1/unique-tournament/17/seasons", False),
    ("github raw (control)", "https://raw.githubusercontent.com/alcadiomontes03/match-engine-odds/main/README.md", False),
]
out = []
for name, url, ajax in URLS:
    h = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}
    if ajax:
        h.update({"X-Requested-With": "XMLHttpRequest", "Referer": "https://understat.com/league/EPL/2025"})
    t = time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=15) as r:
            body = r.read(400000).decode("utf-8", "replace")
            out.append(f"{name}: HTTP {r.status} in {time.time()-t:.1f}s, {len(body)} chars, "
                       f"datesData={'datesData' in body}, starts: {body[:160]!r}")
    except urllib.error.HTTPError as e:
        out.append(f"{name}: HTTP {e.code} in {time.time()-t:.1f}s, {e.read(300)!r}")
    except Exception as e:
        out.append(f"{name}: ERROR {type(e).__name__} in {time.time()-t:.1f}s: {e}")
Path("data/xg").mkdir(parents=True, exist_ok=True)
Path("data/xg/probe.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
