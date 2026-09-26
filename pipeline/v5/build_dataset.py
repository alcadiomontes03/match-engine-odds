"""
Pulsematch v5 — build the modelling dataset from data/fd/ (football-data history).

  python pipeline/v5/build_dataset.py   ->  pipeline/v5/out/dataset.csv
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import market, features  # noqa: E402

FD = HERE.parent.parent / "data" / "fd"
OUT = HERE / "out"
LEAGUES = ["E0", "SP1", "D1", "F1"]


def load(code: str) -> pd.DataFrame:
    frames = []
    for p in sorted(FD.glob(f"{code}_*.csv")):
        d = pd.read_csv(p)
        d["Season"] = p.stem.split("_")[1]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df.Date, format="mixed", dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    return df.sort_values("Date").reset_index(drop=True)


def build() -> pd.DataFrame:
    parts = []
    for code in LEAGUES:
        t = time.time()
        df = load(code)
        df = features.schedule_features(df)
        df = features.walk_forward_ratings(df)
        df = market.market_frame(df)
        parts.append(df)
        print(f"{code}: {len(df)} matches, {df.dc_lam.notna().sum()} with ratings ({time.time() - t:.0f}s)", flush=True)
    all_ = pd.concat(parts, ignore_index=True).sort_values(["Div", "Date"]).reset_index(drop=True)
    all_ = features.gap_features(all_)
    all_["y"] = (all_.FTHG > all_.FTAG).map({True: 0}).fillna((all_.FTHG == all_.FTAG).map({True: 1, False: 2})).astype(int)
    all_["y_over"] = ((all_.FTHG + all_.FTAG) > 2.5).astype(int)
    OUT.mkdir(exist_ok=True)
    all_.to_csv(OUT / "dataset.csv", index=False)
    return all_


if __name__ == "__main__":
    d = build()
    print(d[["Div", "Season"]].value_counts().sort_index().to_string())
