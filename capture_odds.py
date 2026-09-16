"""
Match Engine — Odds API capture (two-pull cadence).

Runs on a schedule (GitHub Actions). Every run:
  1. lists upcoming fixtures via the FREE /events endpoint
  2. SLATE pull: once per match, when kickoff is <= 24h15m away
  3. LATE pull:  once per match, when kickoff is 10-70 min away (after team news)
  4. saves raw JSON + flat rows to data/odds/, records what was pulled in
     state/pulled.json, and logs remaining credits to data/odds/quota.csv

Credits are spent only in steps 2-3: one call per competition per pull type,
covering every due match at once. Cost per call = markets x regions.

The API key is read from the ODDS_API_KEY environment variable (a GitHub
secret). It is never written to any file or log.

Modes:
  python capture_odds.py            normal scheduled run
  python capture_odds.py probe      ONE-TIME test: which regions return soccer
                                    totals/spreads (costs up to 9 credits)
  python capture_odds.py check      free: validate key + sport keys, show quota
"""
from __future__ import annotations
import csv, json, os, sys, time, urllib.parse, urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = "https://api.the-odds-api.com/v4"
ROOT = Path(__file__).resolve().parent
STATE = ROOT / "state" / "pulled.json"
OUT = ROOT / "data" / "odds"

# ---- configuration -------------------------------------------------------
SPORTS = {                      # betting universe
    "soccer_epl": "PL",
    "soccer_spain_la_liga": "LaLiga",
    "soccer_efl_champ": "Championship",
}
CUPS = {                        # predictions-only; OFF until decided
    "soccer_fa_cup": "FACup",
    "soccer_uefa_europa_conference_league": "UECL",
}
INCLUDE_CUPS = os.environ.get("INCLUDE_CUPS", "false").lower() == "true"
REGIONS = os.environ.get("ODDS_REGIONS", "eu")          # set after the probe
MARKETS = os.environ.get("ODDS_MARKETS", "h2h,totals,spreads")
CREDIT_RESERVE = int(os.environ.get("CREDIT_RESERVE", "40"))  # never go below
SLATE_MAX_MIN = 24 * 60 + 15
SLATE_MIN_MIN = 90
LATE_MAX_MIN = 70
LATE_MIN_MIN = 10
# --------------------------------------------------------------------------


def _key():
    k = os.environ.get("ODDS_API_KEY", "").strip()
    if not k:
        sys.exit("ODDS_API_KEY is not set (add it as a repository secret).")
    return k


def get(path, **params):
    params["apiKey"] = _key()
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "match-engine-capture"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read().decode())
            hdr = {k: r.headers.get(k) for k in
                   ("x-requests-remaining", "x-requests-used", "x-requests-last")}
            return body, hdr
    except urllib.error.HTTPError as e:
        # never echo the URL: it contains the key
        msg = e.read().decode(errors="replace")[:300]
        raise RuntimeError(f"HTTP {e.code} on {path}: {msg}") from None


def log_quota(hdr, note):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "quota.csv"; new = not f.exists()
    with f.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["utc", "note", "remaining", "used", "last_cost"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), note,
                    hdr.get("x-requests-remaining"), hdr.get("x-requests-used"),
                    hdr.get("x-requests-last")])


def last_logged_remaining():
    f = OUT / "quota.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(f.open()))
    for r in reversed(rows):
        if r.get("remaining") not in (None, "", "None"):
            return int(float(r["remaining"]))
    return None


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(s):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    # drop entries older than 14 days so the file stays small
    cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
    s = {k: v for k, v in s.items() if v.get("commence", "9") >= cutoff}
    STATE.write_text(json.dumps(s, indent=1, sort_keys=True))


def flatten(events, kind, pulled_at, sport_label):
    rows = []
    for ev in events:
        for bm in ev.get("bookmakers", []):
            for mk in bm.get("markets", []):
                for oc in mk.get("outcomes", []):
                    rows.append([pulled_at, kind, sport_label, ev["id"], ev["commence_time"],
                                 ev["home_team"], ev["away_team"], bm["key"],
                                 bm.get("last_update", ""), mk["key"], oc["name"],
                                 oc.get("point", ""), oc["price"]])
    return rows


def append_rows(rows):
    f = OUT / "snapshots.csv"; new = not f.exists()
    with f.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["pulled_at", "kind", "comp", "event_id", "commence", "home", "away",
                        "bookmaker", "book_last_update", "market", "outcome", "point", "price"])
        w.writerows(rows)


def minutes_to(ts, now):
    return (datetime.fromisoformat(ts.replace("Z", "+00:00")) - now).total_seconds() / 60


def due(events, state, now):
    slate, late = [], []
    for ev in events:
        m = minutes_to(ev["commence_time"], now)
        k = state.get(ev["id"], {})
        if not k.get("slate") and SLATE_MIN_MIN < m <= SLATE_MAX_MIN:
            slate.append(ev)
        if not k.get("late") and LATE_MIN_MIN < m <= LATE_MAX_MIN:
            late.append(ev)
    return slate, late


def pull(sport, label, evs, kind, state, now, remaining):
    cost = len(MARKETS.split(",")) * len(REGIONS.split(","))
    # unknown balance (first ever pull) -> allow; the response header sets it
    if remaining is not None and remaining - cost < CREDIT_RESERVE:
        print(f"SKIP {label} {kind}: {remaining} credits left, reserve {CREDIT_RESERVE}")
        return remaining
    ids = ",".join(e["id"] for e in evs)
    body, hdr = get(f"/sports/{sport}/odds", regions=REGIONS, markets=MARKETS,
                    oddsFormat="decimal", dateFormat="iso", eventIds=ids)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    d = OUT / "raw" / now.strftime("%Y/%m/%d"); d.mkdir(parents=True, exist_ok=True)
    (d / f"{label}_{kind}_{stamp}.json").write_text(json.dumps(body))
    append_rows(flatten(body, kind, now.isoformat(timespec="seconds"), label))
    for e in evs:
        s = state.setdefault(e["id"], {"commence": e["commence_time"], "comp": label,
                                       "home": e["home_team"], "away": e["away_team"]})
        s[kind] = now.isoformat(timespec="seconds")
    log_quota(hdr, f"{label} {kind} x{len(evs)}")
    print(f"PULLED {label} {kind}: {len(evs)} matches, {len(body)} returned, "
          f"cost {hdr.get('x-requests-last')}, left {hdr.get('x-requests-remaining')}")
    rem = hdr.get("x-requests-remaining")
    return int(float(rem)) if rem is not None else remaining


def run():
    now = datetime.now(timezone.utc)
    comps = dict(SPORTS, **(CUPS if INCLUDE_CUPS else {}))
    state = load_state()
    # Balance: last value logged after a paid call. Scheduled runs never call
    # /sports, so polling every 15 min cannot consume credits even if that
    # endpoint were ever billed.
    remaining = last_logged_remaining()
    for sport, label in comps.items():
        try:
            events, hdr = get(f"/sports/{sport}/events", dateFormat="iso")   # free
        except RuntimeError as e:
            print(f"WARN {label}: {e}"); continue
        if hdr.get("x-requests-remaining") is not None:
            remaining = int(float(hdr["x-requests-remaining"]))
        slate, late = due(events, state, now)
        if slate:
            remaining = pull(sport, label, slate, "slate", state, now, remaining)
        if late:
            remaining = pull(sport, label, late, "late", state, now, remaining)
        if not slate and not late:
            print(f"{label}: nothing due ({len(events)} upcoming)")
    save_state(state)


def check():
    sports, hdr = get("/sports", all="true")        # free per docs; manual use only
    keys = {s["key"]: s for s in sports}
    for k, label in dict(SPORTS, **CUPS).items():
        s = keys.get(k)
        print(f"{label:13s} {k:40s} " + ("MISSING" if not s else
              f"found, active={s['active']}"))
    print("credits remaining:", hdr.get("x-requests-remaining"),
          "| used:", hdr.get("x-requests-used"), "| this call cost:", hdr.get("x-requests-last"))
    log_quota(hdr, "check")


def probe():
    """One-time: which regions carry soccer totals/spreads. Up to 9 credits."""
    events, _ = get("/sports/soccer_epl/events", dateFormat="iso")
    if not events:
        sys.exit("No upcoming PL events to probe right now.")
    ev = events[0]["id"]
    report = {}
    for region in ("us", "uk", "eu"):
        body, hdr = get("/sports/soccer_epl/odds", regions=region, markets="h2h,totals,spreads",
                        oddsFormat="decimal", eventIds=ev)
        log_quota(hdr, f"probe {region}")
        books = body[0]["bookmakers"] if body else []
        report[region] = {m: sorted({b["key"] for b in books for mk in b["markets"] if mk["key"] == m})
                          for m in ("h2h", "totals", "spreads")}
        report[region]["cost"] = hdr.get("x-requests-last")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe_regions.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"run": run, "check": check, "probe": probe}[mode]()
