"""
Loader: reads the raw football-data.co.uk-style CSVs cached under data/raw/
(fetched via WebFetch this run) and returns a clean, concatenated DataFrame per
league.

This module does NOT hit the network itself — network access to arbitrary
hosts is not available to Python processes in this sandbox (org egress policy);
only the assistant's WebFetch/WebSearch tools can reach the open internet. So
the fetch step is done by the assistant each pipeline run, saved here as CSV,
and this module just parses/cleans it.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

LEAGUE_FILES = {
    "E0": ["E0_2526.csv", "E0_2627.csv"],   # Premier League
    "SP1": ["SP1_2526.csv", "SP1_2627.csv"],  # La Liga
    # EFL Championship: football-data.co.uk mirror runs ~11 days behind, so it's
    # supplemented with a worldfootball.net scrape (results-only, no odds) covering
    # the days the mirror hasn't caught up to yet. See championship_supplement.py
    # for the source URL, the fetch prompt, and the team-name normalization map.
    "E1": ["E1_2526.csv", "E1_2627.csv", "E1_supplement.csv"],
}

NUMERIC_COLS = ["FTHG", "FTAG", "AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5",
                 "AHh", "AvgAHH", "AvgAHA"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    dropped = before - len(df)
    df = df.sort_values("Date").reset_index(drop=True)
    df.attrs["rows_dropped_unparseable"] = dropped
    return df


def load_league(code: str) -> pd.DataFrame:
    files = LEAGUE_FILES[code]
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
    return out


def load_all():
    return {code: load_league(code) for code in LEAGUE_FILES}
