"""
Pulsematch v5 — build the dashboard's `state/v5` document.

Combines pipeline/v5/out/backtest.json, pipeline/v5/out/checks.json and
data/v5/summary.json + paper_picks.csv into one JSON object that the dashboard's
v5 page reads. The Monday "v5 weekly" scheduled task writes it with ArtifactData.

  python pipeline/v5/dashboard_doc.py  ->  pipeline/v5/out/state_v5.json
"""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SEASON_LABEL = lambda s: f"20{s[:2]}-{s[2:]}"


def _commit():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


def build() -> dict:
    bt = json.loads((HERE / "out" / "backtest.json").read_text())
    ck = json.loads((HERE / "out" / "checks.json").read_text())
    summ_p = ROOT / "data" / "v5" / "summary.json"
    picks_p = ROOT / "data" / "v5" / "paper_picks.csv"
    ps = json.loads(summ_p.read_text()) if summ_p.exists() else {"picks": 0}
    picks = []
    if picks_p.exists() and picks_p.stat().st_size > 0:
        df = pd.read_csv(picks_p).sort_values("pulled_at", ascending=False).head(40)
        for r in df.itertuples():
            picks.append({"pulled_at": r.pulled_at, "comp": r.comp, "commence": r.commence, "home": r.home,
                          "away": r.away, "market": r.market, "selection": r.selection,
                          "point": None if pd.isna(r.point) else float(r.point), "price": float(r.price),
                          "ev": float(r.ev), "exp_goals": r.exp_goals,
                          "clv": None if pd.isna(r.clv) else float(r.clv),
                          "clv_basis": None if pd.isna(getattr(r, "clv_basis", None)) else r.clv_basis})
    seasons = [{"season": SEASON_LABEL(s["season"]), "n": s["n"],
                "close_source": "/".join(k for k in s["close_source"]),
                "ll_open": s["ll_1x2"]["open"], "ll_v5": s["ll_1x2"]["v5"], "ll_close": s["ll_1x2"]["close"],
                "ll_ratings": s["ll_1x2"]["ratings_only"],
                "ou_open": s["ll_ou"]["open"], "ou_v5": s["ll_ou"]["v5"], "ou_close": s["ll_ou"]["close"]}
               for s in bt["seasons"]]
    dp = ck["derivative_pricing_asian_handicap"]
    return {
        "generated_at": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
        "code_commit": _commit(),
        "mode": "paper",
        "gate": {"open": bt["gate"]["open"], "rule": bt["gate"]["rule"], "recent": bt["gate"]["recent"]},
        "backtest": {"seasons": seasons, "v5_bets": bt["pooled_2021_2526"]["v5"],
                     "ratings_only_bets": bt["pooled_2021_2526"]["ratings_only"]},
        "engine_check": {"n": dp["n"], "mean_abs_diff": dp["mean_abs_diff_vs_pinnacle"],
                         "p90_abs_diff": dp["p90_abs_diff"], "brier_engine": dp["brier_engine"],
                         "brier_pinnacle": dp["brier_pinnacle"]},
        "dk_paper": {"summary": ps, "picks": picks},
    }


if __name__ == "__main__":
    doc = build()
    (HERE / "out" / "state_v5.json").write_text(json.dumps(doc, indent=1, default=str))
    print(json.dumps({k: doc[k] for k in ("generated_at", "gate", "engine_check")}, indent=1, default=str))
    print("picks:", len(doc["dk_paper"]["picks"]), "seasons:", len(doc["backtest"]["seasons"]))
