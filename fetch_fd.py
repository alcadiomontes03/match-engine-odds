"""
Pulsematch v5 — football-data.co.uk history fetcher (runs in GitHub Actions).

Downloads the main-league CSVs for every season from 2017-18 to the current one and
writes trimmed copies (results, shots, Pinnacle + market-average opening/closing odds
for 1X2, totals 2.5 and the Asian handicap) to data/fd/<code>_<season>.csv.

Why here: the Claude sandbox can't reach football-data.co.uk (egress policy), but
GitHub runners can. The v5 market layer reads these files.
"""
from __future__ import annotations
import csv, io, sys, time, urllib.request
from datetime import date
from pathlib import Path

LEAGUES = ["E0", "SP1", "D1", "F1"]          # PL, La Liga, Bundesliga, Ligue 1
FIRST_SEASON = 2017
KEEP = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST",
        "PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA",
        "AvgH", "AvgD", "AvgA", "AvgCH", "AvgCD", "AvgCA",
        "P>2.5", "P<2.5", "PC>2.5", "PC<2.5",
        "Avg>2.5", "Avg<2.5", "AvgC>2.5", "AvgC<2.5",
        "AHh", "AHCh", "PAHH", "PAHA", "PCAHH", "PCAHA"]
OUT = Path(__file__).resolve().parent / "data" / "fd"


def season_codes(today: date):
    last = today.year if today.month >= 7 else today.year - 1
    for y in range(FIRST_SEASON, last + 1):
        yield f"{y % 100:02d}{(y + 1) % 100:02d}", y == last


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "pulsematch-fetch/1.0"})
    for attempt in range(3):
        try:
            raw = urllib.request.urlopen(req, timeout=60).read()
            for enc in ("utf-8-sig", "latin-1"):
                try:
                    return raw.decode(enc)
                except UnicodeDecodeError:
                    continue
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt + 1}: {e}", file=sys.stderr)
            time.sleep(5)
    raise RuntimeError(f"failed: {url}")


def main(only_current: bool = False):
    OUT.mkdir(parents=True, exist_ok=True)
    wrote = 0
    for code in LEAGUES:
        for season, is_current in season_codes(date.today()):
            path = OUT / f"{code}_{season}.csv"
            if path.exists() and not is_current and not only_current:
                continue  # completed seasons never change
            if only_current and not is_current:
                continue
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
            print("GET", url)
            text = fetch(url)
            rows = [r for r in csv.DictReader(io.StringIO(text)) if r.get("HomeTeam")]
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=KEEP, extrasaction="ignore")
                w.writeheader()
                for r in rows:
                    w.writerow({k: (r.get(k) or "").strip() for k in KEEP})
            print(f"  {path.name}: {len(rows)} rows")
            wrote += 1
    print("files written:", wrote)


if __name__ == "__main__":
    main(only_current="current" in sys.argv[1:])
