"""
Pulsematch v5 — live pricing and staking (paper mode).

For one fixture: the sharpest available line gives fair probabilities (Shin), the
implied expected goals, and fair prices for every derivative market. A bookmaker
price (e.g. DraftKings) is a candidate bet only where it beats the fair price by
>= EDGE, and stakes are quarter Kelly with per-bet and per-match caps.

    from v5.runtime import price_fixture, stake_plan
    fair = price_fixture(sharp_1x2=(2.10, 3.40, 3.60), sharp_ou25=(1.95, 1.90))
    plan = stake_plan(fair, book={"H": 2.20, "O2.5": 2.05, "spread:-0.5": 2.20}, bankroll=1000)

PAPER MODE: the v5 gate (backtest.py) is closed, so stake_plan output is for the
paper ledger only and must not be used to size real bets.
"""
from __future__ import annotations
import numpy as np
from . import market

EDGE = 0.02
KELLY_FRAC, BET_CAP, MATCH_CAP = 0.25, 0.02, 0.03
GATE_OPEN = False          # set only by a passing backtest.py gate; never by hand


def price_fixture(sharp_1x2, sharp_ou25=None, adjust=None) -> dict:
    """sharp_1x2: decimal odds (home, draw, away) from the sharpest source available.
    sharp_ou25: (over, under) 2.5 odds, optional. adjust: optional callable taking and
    returning (pH, pD, pA, pO) — the fitted AdjustModel hook (currently the identity,
    because the backtest shrinks every adjustment to zero)."""
    pH, pD, pA = market.shin(sharp_1x2)
    pO = market.shin(sharp_ou25)[0] if sharp_ou25 else None
    if adjust:
        pH, pD, pA, pO = adjust(pH, pD, pA, pO)
    lam, mu = market.implied_lambdas(pH, pD, pA, pO)
    fair = market.price_derivatives(lam, mu)
    fair.update({"H": pH, "D": pD, "A": pA})
    if pO is not None:
        fair["O25"] = pO
    return fair


def _fair_prob(fair: dict, sel: str) -> float:
    if sel in ("H", "D", "A"):
        return fair[sel]
    if sel.startswith("O") or sel.startswith("U"):
        line = sel[1:]
        t = fair["totals"].get(line)
        return None if t is None else (t["over"] if sel[0] == "O" else t["under"])
    if sel.startswith("spread:"):
        return fair["spreads"].get(sel.split(":", 1)[1])
    if sel == "BTTS:yes":
        return fair["btts"]
    if sel == "BTTS:no":
        return 1 - fair["btts"]
    return None


def stake_plan(fair: dict, book: dict, bankroll: float = 1.0) -> list:
    """book: {selection: decimal odds}. Selections: H/D/A, O2.5/U2.5 (any listed line),
    spread:<home line>, BTTS:yes/no. Returns candidate paper bets with EV and stake."""
    out = []
    for sel, o in book.items():
        p = _fair_prob(fair, sel)
        if p is None or not o or o <= 1:
            continue
        ev = p * o - 1
        if ev >= EDGE:
            f = min(KELLY_FRAC * ev / (o - 1), BET_CAP)
            out.append({"selection": sel, "odds": o, "fair_prob": round(p, 4), "fair_odds": round(1 / p, 3),
                        "ev": round(ev, 4), "stake_frac": f})
    tot = sum(b["stake_frac"] for b in out)
    for b in out:          # same-match bets are correlated: cap them as one position
        if tot > MATCH_CAP:
            b["stake_frac"] *= MATCH_CAP / tot
        b["stake"] = round(b["stake_frac"] * bankroll, 2)
        b["stake_frac"] = round(b["stake_frac"], 4)
        b["paper_only"] = not GATE_OPEN
    return sorted(out, key=lambda b: -b["ev"])
