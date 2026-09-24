"""
Loader: reads the raw football-data.co.uk-style CSVs cached under data/raw/
(fetched via WebFetch this run) and returns a clean, concatenated DataFrame per
league.

This module does NOT hit the network itself — network access to arbitrary
hosts is not available to Python processes in this sandbox (org egress policy);
only the assistant's WebFetch/WebSearch tools can reach the open internet. So
the fetch step is done by the assistant each pipeline run, saved here as CSV,
and this module just parses/cleans it.

v4.8 (2026-09-23): Premier League and La Liga now fit on 3 seasons. The two
completed seasons come from exact CSVs in the Drive folder "Match Engine —
Historical Data (2023-26)", downloaded by the Monday job into data/raw/hist/
(they carry per-match xG_H/xG_A, used by dixon_coles.fit(goals_weight=...)).
Those files use ISO dates (YYYY-MM-DD); football-data uses DD/MM/YYYY — _clean
accepts both. Only the current season is still fetched live from football-data.

v4.9 (2026-09-23): no season is ever dropped. Every file matching
data/raw/hist/<code>_*.csv is loaded (the Monday job downloads every archived
season from the Drive historical-data folder), and the live current-season file
name is derived from today's date (season starts 1 July), so nothing is
hardcoded to 2026-27. When a season ends, the Monday job archives it into Drive
as a new hist file; older seasons simply get less weight through the time decay.
Closing odds (AvgCH/AvgCD/AvgCA) are parsed when present.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

LEAGUES = ["E0", "SP1", "E1"]


def season_start(d=None) -> pd.Timestamp:
    """1 July of the season that contains date d (default: today)."""
    d = pd.Timestamp(d if d is not None else datetime.utcnow())
    if d.tzinfo is not None:
        d = d.tz_localize(None)
    year = d.year if d.month >= 7 else d.year - 1
    return pd.Timestamp(year=year, month=7, day=1)


def season_code(d=None, offset: int = 0) -> str:
    """football-data season code, e.g. '2627' for 2026-27. offset=-1 = last season."""
    y = season_start(d).year + offset
    return f"{y % 100:02d}{(y + 1) % 100:02d}"


def league_files(code: str, d=None) -> list:
    cur, prev = season_code(d), season_code(d, -1)
    # every archived season, oldest first (hist first so the exact Drive copy wins a dedupe)
    hist = sorted(p.relative_to(RAW_DIR).as_posix() for p in (RAW_DIR / "hist").glob(f"{code}_*.csv")
                  if p.stem.split("_")[-1] != cur)
    if code == "E1":
        # EFL Championship: football-data.co.uk mirror runs ~11 days behind, so it's
        # supplemented with a worldfootball.net scrape (results-only, no odds). See
        # championship_supplement.py. Out of scope for betting; kept as before.
        live = [f"E1_{prev}.csv", f"E1_{cur}.csv", "E1_supplement.csv"]
    else:
        live = [f"{code}_{cur}.csv"]
    return hist + live


# kept for backward compatibility with anything that imports it
LEAGUE_FILES = {c: league_files(c) for c in LEAGUES}

NUMERIC_COLS = ["FTHG", "FTAG", "AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5",
                 "AHh", "AvgAHH", "AvgAHA", "xG_H", "xG_A", "AvgCH", "AvgCD", "AvgCA"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    raw_date = df["Date"].astype(str).str.strip()
    df["Date"] = pd.to_datetime(raw_date, format="%d/%m/%Y", errors="coerce")
    iso = pd.to_datetime(raw_date, format="%Y-%m-%d", errors="coerce")
    df["Date"] = df["Date"].fillna(iso)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    dropped = before - len(df)
    df = df.sort_values("Date").reset_index(drop=True)
    df.attrs["rows_dropped_unparseable"] = dropped
    return df


def load_league(code: str, d=None) -> pd.DataFrame:
    files = league_files(code, d)
    frames = []
    total_dropped = 0
    for f in files:
        p = RAW_DIR / f
        if not p.exists():
            continue
        raw = pd.read_csv(p)
        cleaned = _clean(raw)
        total_dropped += cleaned.attrs.get("rows_dropped_unparseable", 0)
        frames.append(cleaned)
    if not frames:
        raise FileNotFoundError(f"no raw files found for {code} under {RAW_DIR}")
    out = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["Date", "HomeTeam", "AwayTeam"]
    ).sort_values("Date").reset_index(drop=True)
    out.attrs["rows_dropped_unparseable"] = total_dropped
    out.attrs["files"] = [f for f in files if (RAW_DIR / f).exists()]
    return out


def load_all(d=None):
    return {code: load_league(code, d) for code in LEAGUES}
