"""
Walk-forward validation gate.

For each match in the "current" (out-of-sample) season, refit ratings using only
data strictly BEFORE that match's date (prior season history + already-played
current-season matches), generate a model prediction, and compare its Brier
score against the de-vigged closing-line market and against blends of the two.

This mirrors the v3 finding described in the project record: the gate opens only
if some blend weight w>0 beats the pure-market Brier score out of sample.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import dixon_coles as dc
from . import markets as mk


def walk_forward(history_df: pd.DataFrame, oos_df: pd.DataFrame, xi: float,
                  league: str, min_train_matches: int = 30):
    """
    history_df: matches strictly before the OOS season (e.g. prior season) — always
                available as training data.
    oos_df: the out-of-sample season's matches, in date order, each used once as a
            held-out test point; the training set expands to include it afterwards.
    Returns a list of per-match records plus summary Brier scores.

    Implementation note: this uses positional slicing into one pre-sorted,
    pre-concatenated frame rather than repeated pd.concat of single rows —
    concatenating a transposed single-row Series repeatedly silently upcasts the
    Date column to object dtype after the first append, which then breaks the
    `.dt` accessor inside dixon_coles.fit on every subsequent iteration (caught,
    but silently truncates the backtest to n=1). Slicing avoids that entirely.
    """
    full = pd.concat([history_df, oos_df], ignore_index=True).sort_values("Date").reset_index(drop=True)
    n_history = len(history_df)
    records = []

    for i in range(n_history, len(full)):
        train = full.iloc[:i]
        row = full.iloc[i]
        if len(train) < min_train_matches:
            continue

        try:
            ratings = dc.fit(train, as_of=row["Date"], xi=xi, league=league)
        except Exception:
            continue

        pred = dc.market_probs(ratings, row["HomeTeam"], row["AwayTeam"])
        market = mk.devig_1x2(row.get("AvgH"), row.get("AvgD"), row.get("AvgA"))

        rec = {
            "date": row["Date"].isoformat(),
            "home": row["HomeTeam"], "away": row["AwayTeam"],
            "result": row["FTR"],
            "model_p_home": pred["p_home"], "model_p_draw": pred["p_draw"],
            "model_p_away": pred["p_away"],
        }
        rec["model_brier_1x2"] = mk.brier_1x2(pred["p_home"], pred["p_draw"], pred["p_away"], row["FTR"])

        if market is not None:
            rec["market_p_home"] = market["p_home"]
            rec["market_p_draw"] = market["p_draw"]
            rec["market_p_away"] = market["p_away"]
            rec["market_overround"] = market["overround"]
            rec["market_brier_1x2"] = mk.brier_1x2(
                market["p_home"], market["p_draw"], market["p_away"], row["FTR"])
        else:
            rec["market_brier_1x2"] = None

        records.append(rec)

    return records


def summarize(records):
    df = pd.DataFrame(records)
    if df.empty:
        return {"n": 0, "note": "no out-of-sample matches evaluated (insufficient history)"}
    valid = df.dropna(subset=["market_brier_1x2"])
    out = {
        "n_evaluated": int(len(df)),
        "n_with_market_odds": int(len(valid)),
        "model_brier_mean": float(df["model_brier_1x2"].mean()),
    }
    if len(valid):
        out["market_brier_mean"] = float(valid["market_brier_1x2"].mean())
        # grid search blend weight w in [0,1], w=1 pure model, w=0 pure market
        best_w, best_brier = 0.0, out["market_brier_mean"]
        for w in np.linspace(0, 1, 21):
            briers = []
            for _, r in valid.iterrows():
                pm = {"p_home": r.model_p_home, "p_draw": r.model_p_draw, "p_away": r.model_p_away}
                pk = {"p_home": r.market_p_home, "p_draw": r.market_p_draw, "p_away": r.market_p_away}
                blended = mk.blend(pm, pk, w)
                briers.append(mk.brier_1x2(blended["p_home"], blended["p_draw"],
                                            blended["p_away"], r.result))
            mean_b = float(np.mean(briers))
            if mean_b < best_brier:
                best_brier = mean_b
                best_w = float(w)
        out["best_blend_weight"] = best_w
        out["best_blend_brier_mean"] = best_brier
        out["gate_open"] = bool(best_w > 0.0 and best_brier < out["market_brier_mean"] - 1e-9)
    else:
        out["gate_open"] = False
        out["note"] = "no matches had usable market odds"
    return out
