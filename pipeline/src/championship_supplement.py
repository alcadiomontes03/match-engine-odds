"""
EFL Championship freshness supplement.

football-data.co.uk's E1 mirror runs roughly 11 days behind (it's a batch export,
not a live feed). This module documents and helps parse a supplementary results
scrape from worldfootball.net, which is current to within a day or two, to fill
that gap for the CURRENT ratings fit (it does not carry odds, so these matches
never enter the walk-forward market-comparison backtest — only the plain results
feed into dixon_coles.fit's expanding-window training set).

## Fetch procedure (done by the assistant via WebFetch each pipeline run —
## Python in this sandbox cannot reach arbitrary hosts; see loader.py's docstring)

1. Find the current-season URL if it's changed:
   WebSearch: "worldfootball.net EFL Championship <season> results"
   (season rolls from e.g. co20/england-championship each year; the path itself
   is stable, only the season shown on it changes automatically)

2. WebFetch this URL:
   https://www.worldfootball.net/competition/co20/england-championship/all-matches/
   with this prompt:
   "This should be a full fixture/results list for EFL Championship <season>
   season, matchday by matchday. First tell me what page/season it actually
   shows. Then extract ALL played matches (i.e. rows with a final score) as CSV:
   Date,HomeTeam,AwayTeam,HomeGoals,AwayGoals — do not skip any, do not
   summarize."

3. Keep only rows with a Date strictly AFTER the football-data.co.uk E1 file's
   latest date (check data/raw/E1_<season_code>.csv's max Date first) — the
   football-data rows are authoritative (they carry odds) wherever both sources
   overlap; the supplement only fills the tail the mirror hasn't reached yet.

4. Normalize team names to football-data.co.uk's short-name convention with
   NAME_MAP below (worldfootball uses full club names — "Wolverhampton
   Wanderers", "Queens Park Rangers" — where football-data uses "Wolves", "QPR").

5. Write to data/raw/E1_supplement.csv in the same 14-column shape as the other
   raw files, leaving the ten odds columns blank (loader.py's numeric coercion
   turns blanks into NaN, and every consumer downstream already tolerates
   missing odds: dixon_coles.fit only needs Date/HomeTeam/AwayTeam/FTHG/FTAG,
   and backtest.walk_forward's devig_1x2 call returns None for a match with no
   odds, which the summary already accounts for as "n_with_market_odds").

This is a manual-ish, once-a-run procedure rather than a fully automated scraper
because the sandbox's Python processes can't fetch arbitrary URLs themselves
(org egress policy allows only the assistant's WebFetch/WebSearch tools) — so
"scrape" here means the assistant fetches and reshapes the page each run, the
same way it does for the football-data.co.uk spine.
"""
from __future__ import annotations
import csv
import io
from datetime import datetime

NAME_MAP = {
    "Wolverhampton Wanderers": "Wolves",
    "Blackburn Rovers": "Blackburn",
    "Bolton Wanderers": "Bolton",
    "Preston North End": "Preston",
    "Bristol City": "Bristol City",
    "Millwall FC": "Millwall",
    "Charlton Athletic": "Charlton",
    "Derby County": "Derby",
    "Middlesbrough FC": "Middlesbrough",
    "Lincoln City": "Lincoln",
    "Norwich City": "Norwich",
    "West Bromwich Albion": "West Brom",
    "Portsmouth FC": "Portsmouth",
    "Queens Park Rangers": "QPR",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "Sheffield United": "Sheffield United",
    "Birmingham City": "Birmingham",
    "Watford FC": "Watford",
    "Southampton FC": "Southampton",
    "Burnley FC": "Burnley",
    "West Ham United": "West Ham",
    "Cardiff City": "Cardiff",
    "Wrexham AFC": "Wrexham",
    "Oxford United": "Oxford",
    "Leicester City": "Leicester",
    "Sheffield Wednesday": "Sheffield Weds",
    "Hull City": "Hull",
    "Coventry City": "Coventry",
}

RAW_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
               "AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5", "AHh", "AvgAHH", "AvgAHA"]


def normalize_team(name: str) -> str:
    return NAME_MAP.get(name.strip(), name.strip())


def parse_worldfootball_rows(rows, after_date: datetime | None = None) -> str:
    """
    rows: iterable of (date_str 'DD.MM.YYYY', home, away, home_goals, away_goals)
          as extracted from the worldfootball.net page by the assistant.
    after_date: if given, only keep rows strictly after this date (the
                football-data.co.uk mirror's current max date).
    Returns CSV text in the RAW_COLUMNS shape, ready to write to
    data/raw/E1_supplement.csv.
    """
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(RAW_COLUMNS)
    for date_str, home, away, hg, ag in rows:
        d = datetime.strptime(date_str.replace("-", "."), "%d.%m.%Y")
        if after_date is not None and d <= after_date:
            continue
        hg, ag = int(hg), int(ag)
        ftr = "H" if hg > ag else ("A" if ag > hg else "D")
        w.writerow([d.strftime("%d/%m/%Y"), normalize_team(home), normalize_team(away),
                    hg, ag, ftr, "", "", "", "", "", "", "", ""])
    return out.getvalue()
