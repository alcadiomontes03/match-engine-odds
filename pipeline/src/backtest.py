"""
Walk-forward validation gate.

For each match in the "current" (out-of-sample) season, refit ratings using only
data from before that match (all archived seasons + already-played current-season
matches), generate a model prediction, and compare its Brier score against the
de-vigged market and against blends of the two.

v4.9 (2026-09-23) fixes:
  * Weekly refit. The OOS matches are grouped by the Monday before kickoff and
    the model is refit once per group on everything before that Monday — the
    same schedule the live job uses (and much faster than one refit per match
    now that all seasons are kept).
  * Honest gate. v4.8 and earlier picked the best blend weight on the same
    matches it then scored, so `gate_open` was in-sample and optimistic. Now each
    match's blend weight is chosen only from matches BEFORE it (expanding
    window, needs MIN_PRIOR earlier matches), and the gate opens only if that
    out-of-sample blend beats the market on the same matches.
    `best_blend_weight` (full-sample) is still reported as the weight to use
    going forward.
  * Closing odds. The market benchmark now uses football-data's closing
    average (AvgCH/AvgCD/AvgCA) when present, falling back to the pre-closing
    average (AvgH/AvgD/AvgA). Each record says which it used.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import dixon_coles as dc
from . import markets as mk

MIN_PRIOR = 20          # earlier matches needed before an OOS blend weight is chosen
W_GRID = np.linspace(0, 1, 21)


def _market(row):
    m = mk.devig_1x2(row.get("AvgCH"), row.get("AvgCD"), row.get("AvgCA"))
    if m is not None:
        return m, "closing"
    m = mk.devig_1x2(row.get("AvgH"), row.get("AvgD"), row.get("AvgA"))
    return (m, "pre-closing") if m is not None else (None, None)


def walk_forward(history_df: pd.DataFrame, oos_df: pd.DataFrame, xi: float,
                  league: str, min_train_matches: int = 30,
                  goals_weight: float | None = None):
    """
    history_df: matches strictly before the OOS season (all archived seasons).
    oos_df: the current season's matches; each is predicted once from a fit on
            everything before the Monday of its week.
    Returns a list of per-match records.
    """
    full = pd.concat([history_df, oos_df], ignore_index=True).sort_values("Date").reset_index(drop=True)
    oos = full.iloc[len(history_df):]
    records = []
    week = oos.Date.dt.normalize() - pd.to_timedelta(oos.Date.dt.weekday, unit="D")

    for cut, grp in oos.groupby(week):
        train = full[full.Date < cut]
        if len(train) < min_train_matches:
            continue
        try:
            ratings = dc.fit(train, as_of=cut, xi=xi, league=league,
                             goals_weight=goals_weight)  # same fit as live
        except Exception:
            continue

        for _, row in grp.iterrows():
            pred = dc.market_probs(ratings, row["HomeTeam"], row["AwayTeam"])
            market, source = _market(row)
            rec = {
                "date": row["Date"].isoformat(),
                "home": row["HomeTeam"], "away": row["AwayTeam"],
                "result": row["FTR"],
                "refit_as_of": cut.isoformat(),
                "model_p_home": pred["p_home"], "model_p_draw": pred["p_draw"],
                "model_p_away": pred["p_away"],
            }
            rec["model_brier_1x2"] = mk.brier_1x2(pred["p_home"], pred["p_draw"], pred["p_away"], row["FTR"])
            if market is not None:
                rec["market_p_home"] = market["p_home"]
                rec["market_p_draw"] = market["p_draw"]
                rec["market_p_away"] = market["p_away"]
                rec["market_overround"] = market["overround"]
                rec["market_source"] = source
                rec["market_brier_1x2"] = mk.brier_1x2(
                    market["p_home"], market["p_draw"], market["p_away"], row["FTR"])
            else:
                rec["market_brier_1x2"] = None
            records.append(rec)

    return records


def _blend_briers(valid: pd.DataFrame) -> np.ndarray:
    """(n_matches, n_weights) Brier of the model/market blend at each grid weight."""
    out = np.zeros((len(valid), len(W_GRID)))
    for i, (_, r) in enumerate(valid.iterrows()):
        pm = {"p_home": r.model_p_home, "p_draw": r.model_p_draw, "p_away": r.model_p_away}
        pk = {"p_home": r.market_p_home, "p_draw": r.market_p_draw, "p_away": r.market_p_away}
        for j, w in enumerate(W_GRID):
            b = mk.blend(pm, pk, w)
            out[i, j] = mk.brier_1x2(b["p_home"], b["p_draw"], b["p_away"], r.result)
    return out


def summarize(records):
    df = pd.DataFrame(records)
    if df.empty:
        return {"n": 0, "note": "no out-of-sample matches evaluated (insufficient history)",
                "gate_open": False}
    valid = df.dropna(subset=["market_brier_1x2"]).sort_values("date").reset_index(drop=True)
    out = {
        "n_evaluated": int(len(df)),
        "n_with_market_odds": int(len(valid)),
        "model_brier_mean": float(df["model_brier_1x2"].mean()),
        "gate_method": "expanding-window (weight chosen only from earlier matches)",
    }
    if not len(valid):
        out["gate_open"] = False
        out["note"] = "no matches had usable market odds"
        return out

    out["market_source"] = valid["market_source"].value_counts().to_dict() if "market_source" in valid else {}
    out["market_brier_mean"] = float(valid["market_brier_1x2"].mean())
    B = _blend_briers(valid)

    # full-sample weight: the best estimate to use going forward (NOT a gate test)
    j = int(np.argmin(B.mean(axis=0)))
    out["best_blend_weight"] = float(W_GRID[j])
    out["best_blend_brier_mean_in_sample"] = float(B[:, j].mean())

    # honest out-of-sample gate: match i uses the weight that was best on matches < i
    oos_b, mkt_b = [], []
    for i in range(MIN_PRIOR, len(valid)):
        w_i = int(np.argmin(B[:i].mean(axis=0)))
        oos_b.append(B[i, w_i])
        mkt_b.append(valid["market_brier_1x2"].iloc[i])
    out["n_gate_matches"] = len(oos_b)
    if len(oos_b):
        out["blend_brier_oos"] = float(np.mean(oos_b))
        out["market_brier_on_gate_matches"] = float(np.mean(mkt_b))
        out["gate_open"] = bool(out["best_blend_weight"] > 0.0 and
                                out["blend_brier_oos"] < out["market_brier_on_gate_matches"] - 1e-9)
    else:
        out["gate_open"] = False
        out["note"] = f"fewer than {MIN_PRIOR + 1} matches with odds — gate stays closed"
    return out
