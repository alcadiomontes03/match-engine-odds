"""De-vig market odds and score model markets against them."""
from __future__ import annotations
import numpy as np


def devig_1x2(oh, od, oa):
    """Multiplicative de-vig (normalize implied probabilities to sum to 1)."""
    if any(o is None or o <= 1.0 or np.isnan(o) for o in (oh, od, oa)):
        return None
    raw = np.array([1 / oh, 1 / od, 1 / oa])
    overround = raw.sum()
    return {"p_home": raw[0] / overround, "p_draw": raw[1] / overround,
            "p_away": raw[2] / overround, "overround": float(overround)}


def devig_2way(o_over, o_under):
    if any(o is None or o <= 1.0 or np.isnan(o) for o in (o_over, o_under)):
        return None
    raw = np.array([1 / o_over, 1 / o_under])
    overround = raw.sum()
    return {"p_over": raw[0] / overround, "p_under": raw[1] / overround,
            "overround": float(overround)}


def brier_1x2(p_home, p_draw, p_away, result):
    """Multi-class Brier score (0=best, 2=worst) for a single match."""
    outcome = np.array([1.0 if result == "H" else 0.0,
                         1.0 if result == "D" else 0.0,
                         1.0 if result == "A" else 0.0])
    pred = np.array([p_home, p_draw, p_away])
    return float(np.sum((pred - outcome) ** 2))


def brier_2way(p_yes, outcome_yes: bool):
    return float((p_yes - (1.0 if outcome_yes else 0.0)) ** 2)


def blend(p_model, p_market, w):
    """w=0 -> pure market, w=1 -> pure model."""
    return {k: w * p_model[k] + (1 - w) * p_market[k] for k in p_model if k in p_market}


# --- Derived markets from a scoreline grid --------------------------------
# Added because pipeline.py only ever fits ratings and never builds a
# prediction: every dashboard field below (totals/spread/btts/grid4/top_score)
# was previously reverse-engineered ad hoc from the dashboard's client-side JS
# on every Sunday job run instead of coming from a real, saved function here.
# All of these take a pre-computed scoreline grid (rows=home goals,
# cols=away goals, summing to 1) — the same grid dixon_coles.py's own
# market_probs() already builds internally with np.tril/np.trace/np.triu for
# x1x2/p_over25 — so this module stays decoupled from dixon_coles.py and just
# consumes whatever grid the caller already has.

def totals(grid, lines=(1.5, 2.5, 3.5)):
    """Over/under fair probabilities for each goal line, from a scoreline grid."""
    i, j = np.indices(grid.shape)
    total_goals = i + j
    out = {}
    for line in lines:
        over = float(grid[total_goals > line].sum())
        out[f"{line:.1f}"] = {"over": over, "under": 1.0 - over}
    return out


def spread(grid, lines=(-1.5, -0.5, 0.5, 1.5)):
    """Home-team spread cover probabilities. Key is the home line
    ("-1.5" = home favoured by 1.5, "+1.5" = home getting 1.5); the away
    side of the same key is the mirror line's cover probability."""
    i, j = np.indices(grid.shape)
    margin = i - j
    out = {}
    for line in lines:
        home_cover = float(grid[margin > -line].sum())
        out[f"{line:+.1f}"] = {"home": home_cover, "away": 1.0 - home_cover}
    return out


def btts(grid):
    """Both-teams-to-score probabilities."""
    i, j = np.indices(grid.shape)
    yes = float(grid[(i > 0) & (j > 0)].sum())
    return {"yes": yes, "no": 1.0 - yes}


def grid4(grid, size=4):
    """Top-left size x size corner of the scoreline grid (raw probabilities,
    not renormalized) — what the dashboard's scoreline heatmap displays."""
    return [[float(grid[i, j]) for j in range(size)] for i in range(min(size, grid.shape[0]))]


def top_score(grid):
    """The single most likely final scoreline."""
    idx = np.unravel_index(int(np.argmax(grid)), grid.shape)
    return {"home": int(idx[0]), "away": int(idx[1]), "p": float(grid[idx])}


def build_prediction(grid, x1x2, price=None, blend_weight=0.0, shots=None):
    """Assemble every derived-market field the dashboard/predictions schema
    expects from one scoreline grid, the model's own x1x2 {home,draw,away},
    and (optionally) a DraftKings moneyline price {home,draw,away} to blend
    against. If no price is given, bet_prob falls back to the pure model and
    market_devig/ev_at_open are left null — the same convention already used
    for a gated-closed league.

    shots (v4.7): the dict from shots.predict(); written as the prediction's `shots`
    field, which the dashboard's Goal attempts card reads. Left out entirely (not
    null) when there is no shot model, so the card keeps its placeholder."""
    result = {
        "totals": totals(grid),
        "spread": spread(grid),
        "btts": btts(grid),
        "grid4": grid4(grid),
        "top_score": top_score(grid),
    }
    mk = devig_1x2(price["home"], price["draw"], price["away"]) if price else None
    if mk:
        market_devig = {"home": mk["p_home"], "draw": mk["p_draw"], "away": mk["p_away"]}
        bet_prob = blend(x1x2, market_devig, blend_weight)
        ev_at_open = {s: bet_prob[s] * price[s] - 1.0 for s in ("home", "draw", "away")}
    else:
        market_devig, bet_prob, ev_at_open = None, dict(x1x2), None
    result["market_devig"] = market_devig
    result["bet_prob"] = bet_prob
    result["ev_at_open"] = ev_at_open
    if shots:
        result["shots"] = shots
    return result
