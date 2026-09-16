"""
Match Engine — Odds API capture (two-pull cadence).

Runs on a schedule (GitHub Actions). Every run:
  1. lists upcoming fixtures via the FREE /events endpoint
  2. WEEKLY pull: once per week per league, on the first run after Monday
     15:00 UTC (8am Pacific in summer, 7am in winter); covers every match in
     the next 7 days in a single call (DraftKings' early lines)
  3. LATE pull:  once per match, when kickoff is 10-70 min away (after team
     news) - DraftKings' near-closing line, used for CLV
  4. saves raw JSON + flat rows to data/odds/, records what was pulled in
     state/pulled.json, and logs remaining credits to data/odds/quota.csv

Credits are spent only in steps 2-3: one call per competition per pull type,
covering every due match at once. Only DraftKings is requested (ODDS_BOOKMAKERS);
cost per call = markets returned x 1 (up to 10 bookmakers bill as one region).

The API key is read from the ODDS_API_KEY environment variable (a GitHub
secret). It is never written to any file or log.

Modes:
  python capture_odds.py            normal scheduled run
  python capture_odds.py weekly     force this week's weekly pull now (manual)
  python capture_odds.py probe      ONE-TIME test: which markets the bookmaker
                                    offers per league (at most 3 credits each)
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
# Only these bookmakers are pulled. Up to 10 bookmakers bill as one region;
# if BOOKMAKERS is empty, REGIONS is used instead.
BOOKMAKERS = os.environ.get("ODDS_BOOKMAKERS", "draftkings").strip()
REGIONS = os.environ.get("ODDS_REGIONS", "us")
MARKETS = os.environ.get("ODDS_MARKETS", "h2h,totals,spreads")
CREDIT_RESERVE = int(os.environ.get("CREDIT_RESERVE", "40"))  # never go below
WEEKLY_WEEKDAY = 0               # Monday
WEEKLY_HOUR_UTC = 15             # 15:00 UTC = 8am PDT / 7am PST
WEEKLY_WINDOW_MIN = 7 * 24 * 60  # the weekly pull covers the next 7 days
LATE_MAX_MIN = 70
LATE_MIN_MIN = 10
# Optional per-league team filter, JSON: {"LaLiga": ["Real Madrid", "Sevilla"]}.
# A match is kept if EITHER side matches a listed name (accent/case-insensitive
# substring). Leagues not listed are unfiltered.
TEAM_FILTER = json.loads(os.environ.get("ODDS_TEAM_FILTER", "") or "{}")
# --------------------------------------------------------------------------


def _fold(x):
    import unicodedata
    return unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode().lower()


def keep_event(ev, label):
    teams = TEAM_FILTER.get(label)
    if not teams:
        return True
    names = [_fold(t) for t in teams]
    sides = (_fold(ev["home_team"]), _fold(ev["away_team"]))
    return any(n in side for n in names for side in sides)


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


def week_key(now):
    """Weeks start Monday 15:00 UTC; returns an ISO-week label for that week."""
    shifted = now - timedelta(days=WEEKLY_WEEKDAY, hours=WEEKLY_HOUR_UTC)
    y, w, _ = shifted.isocalendar()
    return f"{y}-W{w:02d}"


def weekly_done(state, label, now):
    return state.get("_meta", {}).get(f"weekly:{label}") == week_key(now)


def mark_weekly(state, label, now):
    state.setdefault("_meta", {})[f"weekly:{label}"] = week_key(now)


def due_weekly(events, now):
    return [ev for ev in events if LATE_MIN_MIN < minutes_to(ev["commence_time"], now) <= WEEKLY_WINDOW_MIN]


def due_late(events, state, now):
    """One late pull per kickoff slot - it has to come after team news."""
    return [ev for ev in events
            if not state.get(ev["id"], {}).get("late")
            and LATE_MIN_MIN < minutes_to(ev["commence_time"], now) <= LATE_MAX_MIN]


def book_params():
    """Bookmakers take priority over regions in the API; send only one of them."""
    return {"bookmakers": BOOKMAKERS} if BOOKMAKERS else {"regions": REGIONS}


def max_cost():
    n_markets = len(MARKETS.split(","))
    if BOOKMAKERS:
        return n_markets * -(-len(BOOKMAKERS.split(",")) // 10)
    return n_markets * len(REGIONS.split(","))


def pull(sport, label, evs, kind, state, now, remaining):
    """Returns (remaining, pulled?)."""
    cost = max_cost()   # upper bound: markets a book doesn't offer are not billed
    # unknown balance (first ever pull) -> allow; the response header sets it
    if remaining is not None and remaining - cost < CREDIT_RESERVE:
        print(f"SKIP {label} {kind}: {remaining} credits left, reserve {CREDIT_RESERVE}")
        return remaining, False
    ids = ",".join(e["id"] for e in evs)
    body, hdr = get(f"/sports/{sport}/odds", markets=MARKETS, oddsFormat="decimal",
                    dateFormat="iso", eventIds=ids, **book_params())
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
    return (int(float(rem)) if rem is not None else remaining), True


def run(force_weekly=False):
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
        events = [e for e in events if keep_event(e, label)]
        did = False
        if force_weekly or not weekly_done(state, label, now):
            wk = due_weekly(events, now)
            if wk:
                remaining, ok = pull(sport, label, wk, "weekly", state, now, remaining)
                if ok:
                    mark_weekly(state, label, now)
                did = True
            else:
                mark_weekly(state, label, now)   # nothing scheduled this week
        late = due_late(events, state, now)
        if late:
            remaining, _ = pull(sport, label, late, "late", state, now, remaining)
            did = True
        if not did:
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
    """One-time: which markets the configured bookmaker(s) offer for one upcoming
    match in each league. At most 3 credits per league; markets not offered
    are not billed."""
    report = {}
    for sport, label in SPORTS.items():
        events, _ = get(f"/sports/{sport}/events", dateFormat="iso")
        if not events:
            report[label] = "no upcoming events"; continue
        events = [e for e in events if keep_event(e, label)] or events
        ev = events[0]
        body, hdr = get(f"/sports/{sport}/odds", markets=MARKETS, oddsFormat="decimal",
                        eventIds=ev["id"], **book_params())
        log_quota(hdr, f"probe {label}")
        books = body[0]["bookmakers"] if body else []
        report[label] = {
            "match": f'{ev["home_team"]} v {ev["away_team"]} {ev["commence_time"]}',
            "books": {b["key"]: {mk["key"]: [(o["name"], o.get("point"), o["price"]) for o in mk["outcomes"]]
                                 for mk in b["markets"]} for b in books},
            "cost": hdr.get("x-requests-last"), "remaining": hdr.get("x-requests-remaining")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe_bookmaker.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"run": run, "weekly": lambda: run(force_weekly=True),
     "check": check, "probe": probe}[mode]()
