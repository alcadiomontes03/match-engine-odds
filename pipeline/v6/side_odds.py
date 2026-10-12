"""
Pulsematch v6.1 - DraftKings side-market prices for the dashboard.

Reads the odds bot's data/odds/snapshots.csv and, for each dashboard fixture, builds the
DraftKings BTTS, totals and spreads prices from its newest pre-kickoff pull that carries
them. Writes one JSON file per odds doc that needs a change, holding {"side": {...}}, for the
odds-sync job to merge into odds/<id> with ArtifactData "update".

  python pipeline/v6/side_odds.py --db /tmp/me/db --out /tmp/me/side

--db is an ArtifactData out_dir holding fixtures/<id>.json and odds/<id>.json.

side = {pulled_at,
        btts:    {yes, no},
        totals:  {"2.5": {over, under}, ...},            # every line DraftKings posted
        spreads: {"-0.5": {home, away}, ...}}             # keyed by the HOME line, like predictions
Decimal prices, exactly as captured. Alternate lines count as ordinary lines. A market
DraftKings didn't post is left out. Never invents a price.
"""
from __future__ import annotations
import argparse, csv, json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "data" / "odds" / "snapshots.csv"
SIDE_MARKETS = {"btts": "btts", "totals": "totals", "alternate_totals": "totals",
                "spreads": "spreads", "alternate_spreads": "spreads"}


def _ts(x: str) -> datetime:
    return datetime.fromisoformat(x.replace("Z", "+00:00"))


def _line(x: float) -> str:
    """Spread key in the predictions' format: '-1.5', '-0.5', '+0.5', '+1.5'."""
    return f"{x:+.1f}"


def build_side(rows: list[dict], home: str, away: str) -> dict | None:
    out: dict = {}
    for r in rows:
        m = SIDE_MARKETS.get(r["market"])
        try:
            price = float(r["price"])
        except (TypeError, ValueError):
            continue
        if not m or price <= 1:
            continue
        if m == "btts" and r["outcome"] in ("Yes", "No"):
            out.setdefault("btts", {})[r["outcome"].lower()] = price
        elif m == "totals" and r["outcome"] in ("Over", "Under") and r["point"] not in ("", None):
            k = f"{float(r['point']):.1f}"
            out.setdefault("totals", {}).setdefault(k, {})[r["outcome"].lower()] = price
        elif m == "spreads" and r["point"] not in ("", None):
            pt = float(r["point"])
            if r["outcome"] == home:
                out.setdefault("spreads", {}).setdefault(_line(pt), {})["home"] = price
            elif r["outcome"] == away:
                out.setdefault("spreads", {}).setdefault(_line(-pt), {})["away"] = price
    # keep only complete two-way markets
    if "btts" in out and len(out["btts"]) < 2:
        del out["btts"]
    for k in ("totals", "spreads"):
        if k in out:
            out[k] = {L: v for L, v in out[k].items() if len(v) == 2}
            if not out[k]:
                del out[k]
    return out or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--days-back", type=int, default=4)
    a = ap.parse_args()
    db, out = Path(a.db), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    by_event: dict[str, list[dict]] = {}
    with SNAP.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["bookmaker"] == "draftkings" and r["market"] in SIDE_MARKETS:
                by_event.setdefault(r["event_id"], []).append(r)
    changed = []
    for f in sorted((db / "odds").glob("*.json")):
        od = json.loads(f.read_text())
        fid = od.get("fixture_id") or f.stem
        fx_path = db / "fixtures" / f"{fid}.json"
        if not fx_path.exists():
            continue
        fx = json.loads(fx_path.read_text())
        ko = _ts(fx["kickoff"])
        if ko < now - timedelta(days=a.days_back):
            continue
        eid = fx.get("odds_event_id") or fid
        rows = [r for r in by_event.get(eid, []) if _ts(r["pulled_at"]) < ko]
        if not rows:
            continue
        last = max(r["pulled_at"] for r in rows)
        side = build_side([r for r in rows if r["pulled_at"] == last],
                          fx.get("odds_home_name") or rows[0]["home"], fx.get("odds_away_name") or rows[0]["away"])
        if not side:
            continue
        side = {"pulled_at": last, **side}
        if (od.get("side") or {}) == side:
            continue
        (out / f"{f.stem}.json").write_text(json.dumps({"side": side}, indent=1))
        changed.append(f.stem)
    print(json.dumps({"changed": changed, "n": len(changed)}))


if __name__ == "__main__":
    main()
