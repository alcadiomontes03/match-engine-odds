"""
Pulsematch — Banker agent.

Sizes bets in units once the CLV gate is open. Runs after the Model runner and before the
Checker. While the gate is closed it stakes nothing: it still works out the bankroll and the
unit so the dashboard can show them, and writes a clearly labelled preview of what it would
size, with no stake column.

The rules (agreed with Alcadio, 2026-10-02/03):

  Unit ...... $2 for every $100 bracket the bankroll is in, recalculated on every run and
              never sticky:  $0-100 -> $2, $101-200 -> $4, $201-300 -> $6, $301-400 -> $8,
              +$2 per further $100. If the bankroll drops into a lower bracket, the unit
              drops with it at once ($120 -> $99 takes the unit from $4 to $2).
  Tier ...... 1, 2 or 3 units, never more. Quarter-Kelly and confidence choose it together:
                  kelly units = 0.25 * (p*o - 1)/(o - 1) * bankroll / unit
                  score       = kelly units * confidence
                  tier        = score rounded, clamped to 1..3
              Confidence is per market, from the paper CLV record (placeholder until real
              bets exist): 1.0 if mean CLV > 0 with t >= 2; 0.9 if mean CLV > 0;
              0.75 if fewer than 50 picks have a closing price; 0.5 if mean CLV < 0.
  Parlay .... one tier slot for the whole parlay, never one per leg.
  Day cap ... stakes newly placed on one day (Pacific) total at most 10% of the bankroll.
              Bets placed on earlier days, open or settled, do not count. Bets go in by score,
              best first; at the first one that does not fit, the Banker stops sizing and says
              why. It never shrinks a bet to squeeze it in.

  Screen .... (v6.2) before sizing, every candidate is re-checked here: decimal price a finite
              number > 1, probability strictly between 0 and 1, edge positive and at least the
              runner's edge_threshold. Rejects go to `not_sized` with the reason. A candidate
              whose Kelly score is 0 gets 0 units (it used to get the 1-unit minimum).

It never opens the gate, never places a bet, and never moves the bankroll. The bankroll only
changes through Alcadio's own entries on the dashboard (deposits and withdrawals) and settled
bets; the weekly session exports those to <runs>/bankroll/ledger.json before this runs:

    {"exported_at": "...",
     "transactions": [{"id": "...", "ts": "...", "kind": "deposit"|"withdrawal",
                       "amount": 50.0, "note": "..."}],
     "bets": [{"id": "...", "placed_at": "...", "stake": 4.0,
               "status": "open"|"W"|"L"|"P"|"V", "pnl": 3.6}]}

    bankroll = deposits - withdrawals + P&L of settled bets

Output (run folder):
    banker/bankroll.json   bankroll, unit, bracket, day cap, today's placed stakes, cap left
    banker/plan.json       gate state; `bets` (empty while the gate is closed); `preview`
                           (gate closed only: units the Banker would use, no dollars staked);
                           `not_sized` with the reason for each pick left out
    banker/manifest.json

Status: FAILED only if the gate is open and there is no usable bankroll (missing ledger or
bankroll <= 0); PARTIAL if the ledger is unreadable or malformed; OK otherwise.

    python banker.py --root <repo checkout> --run-dir <runs/run_id> [--bankroll ledger.json]
Exit codes: 0 usable, 1 FAILED, 2 crashed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import manifest as M  # noqa: E402

KELLY_FRAC = 0.25
MAX_UNITS = 3
DAY_CAP = 0.10
UNIT_PER_BRACKET = 2.0
BRACKET = 100.0
TZ = ZoneInfo("America/Los_Angeles")
CONF_MIN_CLOSES = 50
SETTLED = ("W", "L", "P", "V")


# --------------------------------------------------------------------------- rules

def unit_value(bankroll: float) -> float:
    """$2 per $100 bracket: 0-100 -> 2, 100.01-200 -> 4, ... Zero when there is no bankroll."""
    if bankroll is None or bankroll <= 0:
        return 0.0
    return UNIT_PER_BRACKET * math.ceil(round(bankroll, 2) / BRACKET)


def bracket(bankroll: float) -> list[float] | None:
    if bankroll is None or bankroll <= 0:
        return None
    k = math.ceil(round(bankroll, 2) / BRACKET)
    return [round((k - 1) * BRACKET, 2), round(k * BRACKET, 2)]


def kelly_fraction(p: float, price: float) -> float:
    """Quarter-Kelly fraction of bankroll for decimal odds `price`; 0 when there is no edge."""
    if price is None or price <= 1 or p is None or not (0 < p < 1):
        return 0.0
    return max(0.0, KELLY_FRAC * (p * price - 1) / (price - 1))


def confidence(stats: dict | None) -> tuple[float, str]:
    """Per-market confidence from the paper CLV record: (multiplier, reason)."""
    s = stats or {}
    n, mean, t = s.get("n", 0), s.get("mean"), s.get("t")
    if n < CONF_MIN_CLOSES or mean is None:
        return 0.75, f"only {n} picks with a closing price"
    if mean < 0:
        return 0.5, f"paper CLV {mean:+.4f} over {n}"
    if t is not None and t >= 2:
        return 1.0, f"paper CLV {mean:+.4f}, t {t:+.2f} over {n}"
    return 0.9, f"paper CLV {mean:+.4f} over {n}, not yet significant"


def tier(p: float, price: float, bankroll: float, unit: float, conf: float) -> tuple[int, float]:
    """(units 1..3, score). A positive score gets at least 1 unit; a score of 0 (no Kelly edge,
    or an invalid price/probability) gets 0 units and is never sized (v6.2: it used to get 1)."""
    if unit <= 0:
        return 0, 0.0
    score = kelly_fraction(p, price) * bankroll / unit * conf
    if not (score > 0) or not math.isfinite(score):
        return 0, 0.0
    return max(1, min(MAX_UNITS, int(math.floor(score + 0.5)))), round(score, 3)


def screen(candidates: list[dict], min_edge: float | None) -> tuple[list[dict], list[dict]]:
    """min_edge None means no valid threshold is known: every candidate is rejected."""
    """v6.2: the Banker re-checks every candidate itself instead of trusting upstream.
    Rejects: decimal price not a finite number > 1; probability not strictly between 0 and 1;
    edge (ev) not finite, not positive, or below the runner's published edge threshold."""
    ok, rejected = [], []
    for c in candidates:
        price, p, ev = c.get("price"), c.get("p"), c.get("ev")
        why = None
        if not _finite(price) or price <= 1:
            why = f"invalid price {price!r}"
        elif not _finite(p) or not (0 < p < 1):
            why = f"invalid probability {p!r}"
        elif not _finite(ev) or ev <= 0:
            why = f"no edge (ev {ev!r})"
        elif min_edge is None:
            why = "no valid edge threshold from the runner"
        elif ev < min_edge - 1e-12:
            why = f"edge {ev:+.4f} below the {min_edge:.4f} threshold"
        if why:
            rejected.append({**c, "reason": why})
        else:
            ok.append(c)
    return ok, rejected


def _finite(x) -> bool:
    if isinstance(x, bool):
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def parlay(legs: list[dict]) -> dict:
    """Combine legs into one candidate: one tier slot, probabilities and prices multiplied."""
    p = math.prod(l["p"] for l in legs)
    price = math.prod(l["price"] for l in legs)
    return {"kind": "parlay", "p": p, "price": price, "market": "parlay",
            "label": " + ".join(l["label"] for l in legs), "legs": legs}


def size(candidates: list[dict], bankroll: float, placed_today: float, conf_by_market: dict) -> dict:
    """Tier each candidate, then fill today's 10% cap best-first; stop at the first that won't fit."""
    unit = unit_value(bankroll)
    cap = round(DAY_CAP * bankroll, 2) if bankroll and bankroll > 0 else 0.0
    rows = []
    for c in candidates:
        mult, why = conf_by_market.get(c.get("market"), (0.75, "no paper record for this market"))
        if c.get("kind") == "parlay":
            mults = [conf_by_market.get(l.get("market"), (0.75, ""))[0] for l in c["legs"]]
            mult, why = min(mults), "parlay: weakest leg's confidence"
        units, score = tier(c["p"], c["price"], bankroll, unit, mult)
        rows.append({**c, "units": units, "score": score, "confidence": mult, "confidence_why": why,
                     "kelly_fraction": round(kelly_fraction(c["p"], c["price"]), 5)})
    rows.sort(key=lambda r: (-r["score"], -(r["p"] * r["price"] - 1)))
    bets, not_sized, used, stopped = [], [], round(placed_today, 2), None
    for r in rows:
        if r["units"] < 1:                      # no Kelly edge: never sized, never stops the cap walk
            not_sized.append({**r, "reason": "no Kelly edge"})
            continue
        stake = round(r["units"] * unit, 2)
        if stopped is None and used + stake > cap + 1e-9:
            stopped = (f"daily cap reached: ${used:.2f} of ${cap:.2f} already used today and the next bet "
                       f"needs ${stake:.2f} ({r['units']}u x ${unit:.0f}); stopped sizing rather than "
                       f"shrinking it")
        if stopped is not None:
            not_sized.append({**r, "reason": "daily cap"})
            continue
        bets.append({**r, "stake": stake, "unit_value": unit})
        used = round(used + stake, 2)
    return {"unit": unit, "day_cap": cap, "placed_today": round(placed_today, 2), "used_after": used,
            "cap_left": round(cap - used, 2), "bets": bets, "not_sized": not_sized, "stopped": stopped}


# --------------------------------------------------------------------------- inputs

def read_bankroll(path: Path, today) -> dict:
    """Bankroll from the exported dashboard ledger. Raises ValueError if malformed."""
    d = json.loads(path.read_text())
    tx, bets = d.get("transactions", []), d.get("bets", [])
    if not isinstance(tx, list) or not isinstance(bets, list):
        raise ValueError("transactions and bets must be lists")
    dep = wd = 0.0
    for t in tx:
        amt = float(t["amount"])
        if amt < 0:
            raise ValueError(f"transaction {t.get('id')} has a negative amount; use kind instead")
        if t["kind"] == "deposit":
            dep += amt
        elif t["kind"] == "withdrawal":
            wd += amt
        else:
            raise ValueError(f"transaction {t.get('id')} has unknown kind {t['kind']!r}")
    pnl = open_stakes = placed_today = 0.0
    for b in bets:
        st = str(b.get("status", "open"))
        if st in SETTLED:
            pnl += float(b.get("pnl") or 0)
        else:
            open_stakes += float(b.get("stake") or 0)
        when = b.get("placed_at")
        if when and _pacific_date(when) == today:
            placed_today += float(b.get("stake") or 0)
    bank = round(dep - wd + pnl, 2)
    return {"bankroll": bank, "deposits": round(dep, 2), "withdrawals": round(wd, 2),
            "settled_pnl": round(pnl, 2), "open_stakes": round(open_stakes, 2),
            "placed_today": round(placed_today, 2), "transactions": len(tx), "bets_logged": len(bets),
            "exported_at": d.get("exported_at")}


def _pacific_date(ts: str):
    t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(TZ).date()


def market_confidence(picks: pd.DataFrame) -> dict:
    out = {}
    if picks is None or picks.empty or "clv" not in picks.columns:
        return out
    for m, g in picks.groupby("market"):
        c = pd.to_numeric(g.clv, errors="coerce").dropna()
        n = len(c)
        mean = float(c.mean()) if n else None
        t = float(mean / (c.std(ddof=1) / math.sqrt(n))) if n > 2 and c.std(ddof=1) > 0 else None
        out[m] = confidence({"n": n, "mean": mean, "t": t})
    return out


def candidates_from(picks: pd.DataFrame, now: pd.Timestamp) -> list[dict]:
    if picks is None or picks.empty:
        return []
    up = picks[pd.to_datetime(picks.commence, utc=True, errors="coerce") > now]
    up = up[up.close_price.isna()] if "close_price" in up.columns else up
    out = []
    for _, r in up.iterrows():
        price = pd.to_numeric(r.price, errors="coerce")
        ev = pd.to_numeric(r.ev, errors="coerce")
        price = None if pd.isna(price) else float(price)
        ev = None if pd.isna(ev) else float(ev)
        p = (ev + 1) / price if (price is not None and ev is not None and price > 0) else None
        pt = "" if pd.isna(r.point) else f" {r.point:+g}" if r.market == "spreads" else f" {r.point:g}"
        out.append({"kind": "single", "event_id": str(r.event_id), "comp": r.comp, "commence": str(r.commence),
                    "match": f"{r.home} v {r.away}", "market": r.market, "selection": str(r.selection),
                    "point": None if pd.isna(r.point) else float(r.point), "price": price,
                    "ev": None if ev is None else round(ev, 4), "p": p, "label": f"{r.selection}{pt}"})
    return out


# --------------------------------------------------------------------------- run

def run(root: Path, run_dir: Path, bankroll_path: Path | None = None, now: pd.Timestamp | None = None,
        break_until: str | None = None) -> dict:
    started = M.utc_now()
    now = now or pd.Timestamp.now("UTC")
    today = now.tz_convert(TZ).date()
    out = run_dir / "banker"
    out.mkdir(parents=True, exist_ok=True)
    rid = run_dir.name
    warnings, notes, inputs = [], [], []
    bankroll_path = bankroll_path or run_dir.parent / "bankroll" / "ledger.json"

    summary_p, picks_p = run_dir / "runner" / "summary.json", run_dir / "runner" / "paper_picks.csv"
    if not summary_p.exists():
        return M.write_manifest(out, agent="banker", run_id=rid, status="FAILED", started=started,
                                warnings=["no runner/summary.json in this run: the gate state is unknown"])
    summary = json.loads(summary_p.read_text())
    gate_open = bool(summary.get("gate_open", False))
    min_edge = summary.get("edge_threshold")
    min_edge = float(min_edge) if _finite(min_edge) and float(min_edge) > 0 else None
    if min_edge is None:
        # v6.2: with real stakes the threshold is mandatory; with the gate closed, preview only
        msg = (f"runner summary has no valid edge_threshold ({summary.get('edge_threshold')!r}); "
               "every candidate is rejected")
        if gate_open:
            return M.write_manifest(out, agent="banker", run_id=rid, status="FAILED", started=started,
                                    inputs=inputs, warnings=[msg + ": the gate is open, so nothing can be sized"])
        warnings.append(msg + " (gate closed: preview only)")
    inputs.append({**M.describe(summary_p), "path": "runner/summary.json"})
    picks = pd.read_csv(picks_p) if picks_p.exists() else pd.DataFrame()
    if picks_p.exists():
        inputs.append({**M.describe(picks_p, len(picks)), "path": "runner/paper_picks.csv"})

    bank = None
    if bankroll_path.exists():
        try:
            bank = read_bankroll(bankroll_path, today)
            inputs.append({**M.describe(bankroll_path), "path": "bankroll/ledger.json"})
        except Exception as e:  # noqa: BLE001
            warnings.append(f"bankroll ledger unreadable ({type(e).__name__}: {e}); bankroll unknown")
    else:
        (warnings if gate_open else notes).append(
            "no bankroll ledger exported yet (bankroll/ledger.json): log a deposit on the dashboard")

    b = bank["bankroll"] if bank else 0.0
    if gate_open and b <= 0:
        return M.write_manifest(out, agent="banker", run_id=rid, status="FAILED", started=started,
                                inputs=inputs, warnings=warnings + [
                                    f"gate is open but the bankroll is ${b:.2f}: nothing can be sized"])

    conf = market_confidence(picks)
    raw = candidates_from(picks, now)
    cands, rejected = screen(raw, min_edge)
    if rejected:
        warnings.append(f"{len(rejected)} candidate(s) rejected before sizing (invalid price/probability or "
                        f"edge below threshold): " + "; ".join(f"{r['label']}: {r['reason']}" for r in rejected[:3]))
    sized = size(cands, b, bank["placed_today"] if bank else 0.0, conf) if b > 0 else \
        {"unit": 0.0, "day_cap": 0.0, "placed_today": 0.0, "used_after": 0.0, "cap_left": 0.0,
         "bets": [], "not_sized": [{**c, "reason": "no bankroll"} for c in cands], "stopped": None}
    sized["not_sized"] = sized["not_sized"] + rejected

    bankroll_doc = {
        "as_of": now.isoformat(timespec="seconds"), "date_pacific": str(today),
        "bankroll": b, "unit": unit_value(b), "bracket": bracket(b),
        "day_cap": round(DAY_CAP * b, 2) if b > 0 else 0.0,
        "placed_today": bank["placed_today"] if bank else 0.0,
        "cap_left": round(DAY_CAP * b - (bank["placed_today"] if bank else 0.0), 2) if b > 0 else 0.0,
        **({k: bank[k] for k in ("deposits", "withdrawals", "settled_pnl", "open_stakes",
                                  "transactions", "bets_logged", "exported_at")} if bank else {}),
        "rule": "unit = $2 per $100 bracket of the current bankroll; 1-3 units per bet; "
                "10% of bankroll per day in new stakes",
    }
    (out / "bankroll.json").write_text(json.dumps(bankroll_doc, indent=2))

    def strip(r, keep_stake):
        r = {k: v for k, v in r.items() if k not in ("p",)}
        r["fair_p"] = (round((r["ev"] + 1) / r["price"], 4)
                       if r.get("kind") == "single" and _finite(r.get("ev")) and _finite(r.get("price"))
                       and r["price"] > 0 else None)
        if not keep_stake:
            r.pop("stake", None)
            r.pop("unit_value", None)
        return r

    plan = {"generated": now.isoformat(timespec="seconds"), "date_pacific": str(today),
            "gate_open": gate_open, "unit": sized["unit"], "day_cap": sized["day_cap"],
            "placed_today": sized["placed_today"], "bets": [], "preview": [],
            "not_sized": [strip(r, False) | {"reason": r["reason"]} for r in sized["not_sized"]],
            "stopped": sized["stopped"]}
    if gate_open:
        plan["bets"] = [strip(r, True) for r in sized["bets"]]
        plan["total_staked"] = round(sum(r["stake"] for r in sized["bets"]), 2)
        notes.append(f"sized {len(plan['bets'])} bet(s), ${plan['total_staked']:.2f} of the "
                     f"${sized['day_cap']:.2f} daily cap; the Banker places nothing itself")
    else:
        plan["preview"] = [strip(r, False) | {"preview_units": r["units"]} for r in sized["bets"]]
        for r in plan["preview"]:
            r.pop("units", None)
        plan["preview_note"] = "PAPER PREVIEW: gate closed, nothing is staked; units shown for illustration only"
        notes.append("gate closed: no stakes; preview of "
                     f"{len(plan['preview'])} pick(s) at the current ${sized['unit']:.0f} unit")
    if sized["stopped"]:
        notes.append(sized["stopped"])
    (out / "plan.json").write_text(json.dumps(plan, indent=2, default=str))

    notes.append(f"bankroll ${b:.2f} -> unit ${unit_value(b):.0f}, day cap ${bankroll_doc['day_cap']:.2f}")
    if b > 0:
        lo = bracket(b)[0]
        if lo > 0 and MAX_UNITS * unit_value(b) > DAY_CAP * b:
            notes.append(f"near the bottom of the ${lo:.0f}-${lo + BRACKET:.0f} bracket a single "
                         f"{MAX_UNITS}u bet (${MAX_UNITS * unit_value(b):.0f}) is more than the day cap")
    status = "PARTIAL" if warnings else "OK"
    return M.write_manifest(out, agent="banker", run_id=rid, status=status, started=started, inputs=inputs,
                            row_counts={"candidates": len(raw), "rejected": len(rejected),
                                        "sized": len(sized["bets"]),
                                        "not_sized": len(sized["not_sized"])},
                            warnings=warnings, notes=notes)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pulsematch Banker agent.")
    ap.add_argument("--root", required=True, help="checkout of the odds repo")
    ap.add_argument("--run-dir", required=True, help="this run's folder")
    ap.add_argument("--bankroll", default=None, help="exported ledger (default <runs>/bankroll/ledger.json)")
    ap.add_argument("--break-until", default=None)
    a = ap.parse_args(argv)
    try:
        m = run(Path(a.root).resolve(), Path(a.run_dir).resolve(),
                Path(a.bankroll).resolve() if a.bankroll else None, break_until=a.break_until)
    except Exception as e:  # noqa: BLE001
        print(f"BANKER CRASHED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    for w in m["warnings"]:
        print(f"  WARN  {w}")
    for n in m["notes"]:
        print(f"  note  {n}")
    print(f"{m['status']}  banker  ({m['row_counts'].get('sized', 0)} sized, "
          f"{m['row_counts'].get('not_sized', 0)} not sized)")
    return 1 if m["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
