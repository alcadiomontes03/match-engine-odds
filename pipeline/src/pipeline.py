"""
Weekly runtime: load data -> fit current ratings -> walk-forward validation gate
check -> generate this-week predictions -> write a single state.json.

state.json is the artifact this run produces in the (ephemeral) container. It is
NOT the persistence layer — per the rebuild rule, the assistant pushes the
relevant parts of it into Google Drive and the dashboard's Artifact database
immediately after this script runs. Nothing here should be assumed to survive a
container reset.
"""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import loader, dixon_coles as dc, backtest, markets as mk, xg as xg_mod

STATE_PATH = Path(__file__).resolve().parent.parent / "state" / "state.json"

# v4.9 blend rule (replaces the Monday job's old step-4b "floor at 0.7", which was
# inverted: in markets.blend w=1 is pure MODEL, so flooring at 0.7 forced at
# least 70% model exactly when the model was worse than the market).
# Now: gate closed -> 0 (pure market); gate open but model alone worse than the
# market -> at most MAX_MODEL_WEIGHT_WHEN_WORSE model weight.
MAX_MODEL_WEIGHT_WHEN_WORSE = 0.0
# v4.9.1: a gate opened on a small sample can't hand the model a big weight.
# v4.9.2 (2026-09-24, Alcadio's decision): both caps are now 0.0 (pure market).
# Full market test vs closing odds, PL 2023-26 + La Liga 2023-26 (2,280 matches,
# see claude/model-v4.9-handoff.md): the model was worse than the close in every
# league and every season (PL 2023-24 0.950 vs 0.901, La Liga 0.976 vs 0.954),
# blending at any weight never beat the close, and paper value bets at Bet365
# opening odds lost 19-26%. So the model only gets weight once a gate backed by
# at least MIN_GATE_MATCHES out-of-sample matches shows it beating the market.
MAX_MODEL_WEIGHT_SMALL_SAMPLE = 0.0
MIN_GATE_MATCHES = 100


def choose_blend_weight(v: dict) -> tuple[float, str]:
    if not v or not v.get("gate_open"):
        return 0.0, "gate closed -> pure market"
    w = float(v.get("best_blend_weight", 0.0))
    mb, kb = v.get("model_brier_mean"), v.get("market_brier_mean")
    if mb is not None and kb is not None and mb > kb and w > MAX_MODEL_WEIGHT_WHEN_WORSE:
        return MAX_MODEL_WEIGHT_WHEN_WORSE, f"capped from {w:.2f}: model Brier worse than market"
    n = int(v.get("n_gate_matches", 0) or 0)
    if n < MIN_GATE_MATCHES and w > MAX_MODEL_WEIGHT_SMALL_SAMPLE:
        return MAX_MODEL_WEIGHT_SMALL_SAMPLE, f"capped from {w:.2f}: only {n} gate matches (< {MIN_GATE_MATCHES})"
    return w, "best_blend_weight"


# goals_weight (v4.8): Poisson target = w*goals + (1-w)*xG on rows that have xG
# (archived seasons that carry per-match xG; rows without xG, including the
# current season, use goals).
# None = goals only. Backtest: claude/backtest-3season-xg-2026-09-23.md.
LEAGUES = {
    "E0": {"name": "Premier League", "betting_universe": True, "xi": 0.0020, "goals_weight": 0.3},
    "SP1": {"name": "La Liga", "betting_universe": True, "xi": 0.0020, "goals_weight": 0.3},
    # v4.10 (2026-09-24, Alcadio's decision): Bundesliga and Ligue 1 added, same
    # settings. Walk-forward 2023-24..2025-26 (918 matches each): the xG blend
    # beats goals-only on 1X2 (D1 0.9939 vs 0.9985, F1 1.0063 vs 1.0070), but the
    # model still trails the closing market (D1 0.994 vs 0.961, F1 1.006 vs 0.982),
    # so their blend weight is 0 like the others until a gate says otherwise.
    "D1": {"name": "Bundesliga", "betting_universe": True, "xi": 0.0020, "goals_weight": 0.3},
    "F1": {"name": "Ligue 1", "betting_universe": True, "xi": 0.0020, "goals_weight": 0.3},
}
# v4.9.3 (2026-09-24, Alcadio's decision): the EFL Championship (E1) is out of
# scope and no longer fitted. Removing it cuts the Monday job's slowest fetches
# (two seasons of monthly E1 chunks + the worldfootball supplement).


def run():
    as_of = datetime.now(timezone.utc)
    all_data = {code: loader.load_league(code) for code in LEAGUES}  # v4.10: E0/SP1/D1/F1
    state = {
        "generated_at": as_of.isoformat(),
        "leagues": {},
        "known_gaps": [
            "xG layer: partially resolved 2026-09-15. xgstat.com verified as a "
            "real, current-season, free/no-login source for Premier League team "
            "xG (see src/xg.py) — surfaced as an over/underperformance check, not "
            "blended into the Dixon-Coles fit (it's season-aggregate, not "
            "per-match shot data). La Liga and Championship still have no xG "
            "source: understat.com blocks the fetch outright (robots.txt), and "
            "both fbref.com and footystats.org returned tables for a different, "
            "older season on first try and were discarded rather than trusted — "
            "worth re-checking periodically in case that was a transient glitch "
            "rather than a hard block.",
            "No upcoming-fixture list wired up yet (only completed matches with "
            "closing odds) — dashboard shows fitted ratings and the walk-forward "
            "backtest, not next-matchweek prices, until a fixtures source is added.",
            "Backtest sample sizes are small (early in the 2026/27 season) — "
            "gate readings should be treated as low-confidence until more "
            "matchweeks accumulate.",
            "v4.8 (2026-09-23): PL and La Liga fit on 3 seasons (2024-25 and "
            "2025-26 from the Drive historical-data folder + the live current "
            "season), with the goal target blended 30% goals / 70% xG on the two "
            "completed seasons. Current-season rows are still goals-only because "
            "no per-match xG feed is wired up for 2026-27 yet.",
            "v4.9 (2026-09-23): every archived season in Drive is kept (never "
            "dropped; older seasons just get less weight via the time decay); "
            "the current season is detected from the date; the gate now picks "
            "its blend weight out-of-sample and benchmarks against closing odds "
            "when available; blend_weight is computed in code (gate closed = 0, "
            "model worse than market = at most 0.3 model).",
            "v4.9.2 (2026-09-24): after the full market test (PL + La Liga "
            "2023-26, 2,280 matches, model worse than the closing odds every "
            "season), both blend caps are 0.0: the model gets no weight unless a "
            "gate backed by 100+ out-of-sample matches shows it beating the market.",
            "v4.9.3 (2026-09-24): the EFL Championship is out of scope and no "
            "longer fitted; only the Premier League and La Liga are refit. The "
            "Champions League is not modelled yet (results history for 2023-26 "
            "is in the Drive historical folder).",
            "v4.10 (2026-09-24): Bundesliga (D1) and Ligue 1 (F1) added with "
            "2022-23..2025-26 history in Drive (xG from FBref/Opta, xglab for "
            "2025-26). Market test 2023-26: the model trails the closing odds in "
            "both, so blend weight is 0 (pure market) like PL and La Liga.",
        ],
    }

    for code, cfg in LEAGUES.items():
        df = all_data[code]
        current_season_cutoff = loader.season_start(as_of)  # v4.9: from the date, not hardcoded
        history = df[df.Date < current_season_cutoff]
        oos = df[df.Date >= current_season_cutoff]

        league_state = {
            "name": cfg["name"],
            "betting_universe": cfg["betting_universe"],
            "n_history_matches": int(len(history)),
            "n_oos_matches": int(len(oos)),
            "rows_dropped_unparseable": int(df.attrs.get("rows_dropped_unparseable", 0)),
            "goals_weight": cfg.get("goals_weight"),
            "n_rows_with_xg": int(df["xG_H"].notna().sum()) if "xG_H" in df.columns else 0,
            "fit_from": str(df.Date.min().date()) if len(df) else None,
            "season": loader.season_code(as_of),
            "files_loaded": list(df.attrs.get("files", [])),
        }

        # current ratings fit on everything available, decayed to "as_of"
        try:
            ratings = dc.fit(df, as_of=as_of.replace(tzinfo=None), xi=cfg["xi"], league=code,
                             goals_weight=cfg.get("goals_weight"))
            league_state["ratings"] = ratings.to_dict()
            top_attack = sorted(ratings.attack.items(), key=lambda kv: -kv[1])[:5]
            league_state["top_attack_teams"] = top_attack
        except Exception as e:
            league_state["ratings_error"] = str(e)

        # xG overlay (currently Premier League only — see src/xg.py)
        xg_data = xg_mod.load_xg(code)
        if xg_data:
            league_state["xg"] = xg_data

        # walk-forward validation gate
        try:
            records = backtest.walk_forward(history, oos, xi=cfg["xi"], league=code,
                                             min_train_matches=30,
                                             goals_weight=cfg.get("goals_weight"))
            summary = backtest.summarize(records)
            league_state["validation"] = summary
            league_state["validation_detail"] = records
            w, why = choose_blend_weight(summary)
            league_state["blend_weight"] = w
            league_state["blend_weight_reason"] = why
        except Exception as e:
            league_state["validation_error"] = str(e)

        state["leagues"][code] = league_state

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)

    return state


if __name__ == "__main__":
    s = run()
    print(json.dumps({k: v for k, v in s.items() if k != "leagues"}, indent=2))
    for code, ls in s["leagues"].items():
        print(f"\n=== {code} ({ls['name']}) ===")
        print(f"history matches: {ls.get('n_history_matches')}, oos matches: {ls.get('n_oos_matches')}, dropped: {ls.get('rows_dropped_unparseable')}")
        print(f"fit from: {ls.get('fit_from')}, rows with xG: {ls.get('n_rows_with_xg')}, goals_weight: {ls.get('goals_weight')}")
        print(f"files: {ls.get('files_loaded')}")
        print(f"blend_weight: {ls.get('blend_weight')} ({ls.get('blend_weight_reason')})")
        if "validation" in ls:
            print("validation:", json.dumps(ls["validation"], indent=2))
        if "ratings_error" in ls:
            print("ratings error:", ls["ratings_error"])
