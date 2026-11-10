"""
Fair-price reference for the dashboard (2026-11-09). Paper/diagnostic only: nothing here is a pick rule or part of
a registered test (docs/prereg-sharp-2026-11-01.md and docs/prereg-forward-2026-11.md are untouched).

Why: DraftKings is the only book bet at, so the useful question for any bet is "how far is this DraftKings price
from the fair price?". The sharpest fair price collected is Pinnacle's moneyline (data/odds/sharp.csv), Shin
de-vigged. Side markets have no Pinnacle price here; their reference is DraftKings' own two-way price, de-vigged.

  python scripts/fair_prices.py --db <ArtifactData out_dir with fixtures/, odds/, bank_bets/> --out <dir>

Writes two folders of JSON documents for the dashboard database (new collections, nothing existing is edited):
  <out>/sharp/<fixture_id>.json     Pinnacle 1X2: latest quote and, once kicked off, the last quote before kickoff,
                                    each with raw prices, Shin fair chances and when it was pulled.
  <out>/bet_fair/all.json           {bets: {<bet_id>: ...}}. For each real bet: the fair chance of every leg at the close and the bet's
                                    closing-line value, clv = price taken x fair chance at close - 1.
                                    Moneyline legs use Pinnacle's close when it exists, else DraftKings' own
                                    closing moneyline. Totals, spreads and BTTS use DraftKings' side price.
                                    Same-game parlays get no number (their legs move together); nor do bets with
                                    a leg that has no closing price.
"""
from __future__ import annotations
import argparse, csv, glob, json, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from v5 import market as M  # noqa: E402

SHARP = ROOT / "data" / "odds" / "sharp.csv"
STALE_MIN = 60          # same staleness limit as pipeline/sharp/compare.py: the book's own update vs our pull


def _ts(x):
    return datetime.fromisoformat(str(x).replace("Z", "+00:00")) if x else None


def _load(db: Path, coll: str) -> dict:
    out = {}
    for f in glob.glob(str(db / coll / "*.json")):
        d = json.loads(Path(f).read_text())
        if isinstance(d, dict) and isinstance(d.get("data"), dict) and "id" not in d:
            d = d["data"]
        out[Path(f).stem] = d
    return out


def _fair3(h, d, a):
    p = M.shin([h, d, a])
    return None if any(x != x for x in p) else {"home": round(float(p[0]), 5), "draw": round(float(p[1]), 5),
                                                 "away": round(float(p[2]), 5)}


def pinnacle_quotes(path: Path = SHARP) -> dict:
    """{event_id: [quote, ...] oldest first}; a quote is one complete Pinnacle 1X2 pull."""
    pulls = {}
    if not path.exists():
        return {}
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("bookmaker") != "pinnacle" or r.get("market") != "h2h":
                continue
            side = "draw" if r["outcome"] == "Draw" else "home" if r["outcome"] == r["home"] else \
                   "away" if r["outcome"] == r["away"] else None
            try:
                price = float(r["price"])
            except (TypeError, ValueError):
                continue
            if side is None or not price > 1:
                continue
            q = pulls.setdefault((r["event_id"], r["pulled_at"]), {"pulled_at": r["pulled_at"],
                                                                  "commence": r["commence"],
                                                                  "book_last_update": r.get("book_last_update")})
            q[side] = price
    out = {}
    for (eid, _), q in sorted(pulls.items(), key=lambda kv: kv[0][1]):
        if not all(k in q for k in ("home", "draw", "away")):
            continue
        lag = None
        if q.get("book_last_update"):
            lag = (_ts(q["pulled_at"]) - _ts(q["book_last_update"])).total_seconds() / 60.0
        q["stale"] = bool(lag is None or lag > STALE_MIN)
        q["fair"] = _fair3(q["home"], q["draw"], q["away"])
        if q["fair"]:
            out.setdefault(eid, []).append(q)
    return out


def _pub(q):
    return None if q is None else {k: q[k] for k in ("home", "draw", "away", "pulled_at", "fair", "stale")}


def sharp_docs(fixtures: dict, quotes: dict, now: datetime) -> dict:
    docs = {}
    for fid, f in fixtures.items():
        qs = quotes.get(f.get("odds_event_id") or fid) or quotes.get(fid)
        if not qs:
            continue
        ko = _ts(f.get("kickoff"))
        pre = [q for q in qs if ko is None or _ts(q["pulled_at"]) <= ko]
        if not pre:
            continue
        doc = {"id": fid, "fixture_id": fid, "book": "pinnacle", "method": "Shin de-vig of the 1X2",
               "latest": _pub(pre[-1]), "n_pulls": len(pre), "updated_at": now.isoformat(timespec="seconds")}
        if ko is not None and ko <= now:
            doc["close"] = dict(_pub(pre[-1]), mins_to_ko=round((ko - _ts(pre[-1]["pulled_at"])).total_seconds() / 60.0, 1))
        docs[fid] = doc
    return docs


def parse_leg(key: str):
    """(market, fixture_id, home_line, side) from a pick key. Spread lines come back as the HOME handicap.
    Hand-logged spread keys are 'sp:<fid>:<home|away>:<that team's own line>'; page keys are
    'sp:<fid>:<home handicap>:<home|away>'."""
    parts = str(key).split(":")
    m = parts[0]
    if m == "ml" and len(parts) == 3:
        return m, parts[1], None, parts[2]
    if m in ("tot", "btts", "sp") and len(parts) == 4:
        fid, a, b = parts[1], parts[2], parts[3]
        if m == "sp" and a in ("home", "away"):
            own = float(b)
            return m, fid, (own if a == "home" else -own), a
        return m, fid, (None if a == "" else float(a)), b
    return m, (parts[1] if len(parts) > 1 else None), None, None


def _two_way(pa, pb):
    if not (pa and pb and float(pa) > 1 and float(pb) > 1):
        return None
    p = M.shin([float(pa), float(pb)])
    return None if p[0] != p[0] else (float(p[0]), float(p[1]))


def leg_fair(key: str, odds: dict, sharp: dict):
    """Fair chance of one leg at the close: (p, source) or (None, reason)."""
    m, fid, line, side = parse_leg(key)
    od = odds.get(fid) or {}
    if m == "ml":
        sc = (sharp.get(fid) or {}).get("close")
        if sc and not sc.get("stale") and side in sc["fair"]:
            return sc["fair"][side], "pinnacle"
        px = od.get("close") or od.get("latest") or od.get("open") or {}
        if all((px.get(k) or 0) > 1 for k in ("home", "draw", "away")) and side in ("home", "draw", "away"):
            return _fair3(px["home"], px["draw"], px["away"])[side], "draftkings"
        return None, "no closing moneyline"
    sd = od.get("side") or {}
    if m == "btts":
        t = _two_way((sd.get("btts") or {}).get("yes"), (sd.get("btts") or {}).get("no"))
        return (t[0 if side == "yes" else 1], "draftkings side") if t and side in ("yes", "no") else (None, "no closing BTTS price")
    if m in ("tot", "sp") and line is not None:
        book = sd.get("totals" if m == "tot" else "spreads") or {}
        hit = next((v for k, v in book.items() if abs(float(k) - line) < 1e-9), None)
        a, b = ("over", "under") if m == "tot" else ("home", "away")
        t = _two_way((hit or {}).get(a), (hit or {}).get(b))
        return (t[0 if side == a else 1], "draftkings side") if t and side in (a, b) else (None, "no closing price on that line")
    return None, "market has no fair price here"


def bet_docs(bets: dict, odds: dict, sharp: dict, fixtures: dict, now: datetime) -> dict:
    docs = {}
    for bid, b in bets.items():
        if b.get("paper") or str(b.get("status") or "open") == "V":
            continue
        legs = b.get("legs") or ([{"key": b["pick_key"]}] if b.get("pick_key") else [])
        price = float(b.get("price") or 0)
        if not legs or not price > 1:
            continue
        fids = [parse_leg(g.get("key"))[1] for g in legs]
        started = all((fixtures.get(f) or {}).get("kickoff") and _ts(fixtures[f]["kickoff"]) <= now for f in fids)
        if not started:
            continue                      # no close yet; the page shows the live comparison for open bets
        out_legs, p, why = [], 1.0, None
        for g in legs:
            lp, src = leg_fair(g.get("key"), odds, sharp)
            out_legs.append({"key": g.get("key"), "fair_p": None if lp is None else round(lp, 5),
                             "src": src if lp is not None else None, "why": None if lp is not None else src})
            if lp is None:
                why = why or src
            else:
                p *= lp
        if len(set(fids)) < len(fids):
            why = "same-game parlay: the legs move together, so the leg chances can't be multiplied"
        boosted = ((b.get("promo") or {}).get("boosted_price") or 0)
        doc = {"id": bid, "price": price, "legs": out_legs, "computed_at": now.isoformat(timespec="seconds"),
               "fair_p": None, "clv": None, "clv_boosted": None, "why": why,
               "src": "+".join(sorted({g["src"] for g in out_legs if g["src"]})) or None}
        if why is None:
            doc["fair_p"] = round(p, 5)
            doc["clv"] = round(price * p - 1, 4)
            if boosted and float(boosted) > 1:
                doc["clv_boosted"] = round(float(boosted) * p - 1, 4)
        docs[bid] = doc
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--now", help="ISO time to treat as now (tests)")
    a = ap.parse_args()
    db, out = Path(a.db), Path(a.out)
    now = _ts(a.now) if a.now else datetime.now(timezone.utc)
    fixtures, odds, bets = _load(db, "fixtures"), _load(db, "odds"), _load(db, "bank_bets")
    sharp = sharp_docs(fixtures, pinnacle_quotes(), now)
    bf = bet_docs(bets, odds, sharp, fixtures, now)
    (out / "sharp").mkdir(parents=True, exist_ok=True)
    for k, d in sharp.items():
        (out / "sharp" / f"{k}.json").write_text(json.dumps(d))
    (out / "bet_fair").mkdir(parents=True, exist_ok=True)       # one document, so a weekly run is one write
    (out / "bet_fair" / "all.json").write_text(json.dumps(
        {"id": "all", "computed_at": now.isoformat(timespec="seconds"), "bets": bf}))
    got = [d for d in bf.values() if d["clv"] is not None]
    print(json.dumps({"sharp_docs": len(sharp), "bet_docs": len(bf), "bets_with_clv": len(got),
                      "mean_clv": round(sum(d["clv"] for d in got) / len(got), 4) if got else None}))


if __name__ == "__main__":
    main()
