"""
Match Engine — Odds API capture (two-pull cadence).

Runs on a schedule (GitHub Actions). Every run:
  1. lists upcoming fixtures via the FREE /events endpoint
  2. WEEKLY pull: once per week per league, on the first run after Monday
     15:00 UTC (8am Pacific in summer, 7am in winter); covers every match in
     the next 7 days in a single call (DraftKings' early lines)
  3. LATE pull:  within 3h of kickoff, plus one refresh inside the last 65 min
     (after team news) if the first pull came earlier - DraftKings'
     near-closing line, used for CLV. The wide window exists because GitHub
     runs "every 15 min" schedules only every ~2h in practice.
  4. saves raw JSON + flat rows to data/odds/, records what was pulled in
     state/pulled.json, and logs remaining credits to data/odds/quota.csv

Credits are spent only in steps 2-3: one call per competition per pull type,
covering every due match at once. Only DraftKings is requested (ODDS_BOOKMAKERS);
cost per call = markets requested x 1 (up to 10 bookmakers bill as one region).

v5 (2026-09-25): markets are moneyline (h2h), spreads and totals in the bulk call
(3 credits per call). BTTS is an "additional market" that The Odds API only serves
per event, so it is pulled once per match at its first late pull (1 credit per match),
and only while more than BTTS_RESERVE credits remain, so the main markets never run
dry. If DraftKings returns no BTTS for three matches in a row in a league, that
league's BTTS is skipped for the rest of the week.

The API key is read from the ODDS_API_KEY environment variable (a GitHub
secret). It is never written to any file or log.

Modes:
  python capture_odds.py            normal scheduled run
  python capture_odds.py weekly     force this week's weekly pull now (manual)
  python capture_odds.py probe      ONE-TIME test: which markets the bookmaker
                                    offers per league (1 credit per market each)
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
    "soccer_germany_bundesliga": "Bundesliga",   # added 2026-09-24 (model v4.10)
    "soccer_france_ligue_one": "Ligue1",         # added 2026-09-24 (model v4.10)
}
# The EFL Championship was removed 2026-09-24 (out of scope; model v4.9.3).
CUPS = {                        # predictions-only; OFF until decided
    "soccer_fa_cup": "FACup",
    "soccer_uefa_europa_conference_league": "UECL",
}
INCLUDE_CUPS = os.environ.get("INCLUDE_CUPS", "false").lower() == "true"
# Only these bookmakers are pulled. Up to 10 bookmakers bill as one region;
# if BOOKMAKERS is empty, REGIONS is used instead.
BOOKMAKERS = os.environ.get("ODDS_BOOKMAKERS", "draftkings").strip()
REGIONS = os.environ.get("ODDS_REGIONS", "us")
MARKETS = os.environ.get("ODDS_MARKETS", "h2h,spreads,totals")
BTTS = os.environ.get("ODDS_BTTS", "true").lower() == "true"
BTTS_RESERVE = int(os.environ.get("BTTS_RESERVE", "100"))    # stop BTTS below this
CREDIT_RESERVE = int(os.environ.get("CREDIT_RESERVE", "40"))  # never go below
WEEKLY_WEEKDAY = 0               # Monday
WEEKLY_HOUR_UTC = 15             # 15:00 UTC = 8am PDT / 7am PST
WEEKLY_WINDOW_MIN = 7 * 24 * 60  # the weekly pull covers the next 7 days
LATE_MAX_MIN = 180               # first late pull: any run within 3h of kickoff
TEAM_NEWS_MIN = 65               # one refresh once inside this window
LATE_MIN_MIN = 5
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
    """At most two late pulls per match: the first run inside 3h of kickoff,
    and one refresh inside the team-news window if the first came earlier."""
    out = []
    for ev in events:
        m = minutes_to(ev["commence_time"], now)
        if not (LATE_MIN_MIN < m <= LATE_MAX_MIN):
            continue
        k = state.get(ev["id"], {})
        if k.get("late_final"):
            continue
        if not k.get("late") or m <= TEAM_NEWS_MIN:
            out.append(ev)
    return out


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
    cost = max_cost()   # billed per market requested, even if the book offers none
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
        if kind == "late" and minutes_to(e["commence_time"], now) <= TEAM_NEWS_MIN:
            s["late_final"] = True
    log_quota(hdr, f"{label} {kind} x{len(evs)}")
    print(f"PULLED {label} {kind}: {len(evs)} matches, {len(body)} returned, "
          f"cost {hdr.get('x-requests-last')}, left {hdr.get('x-requests-remaining')}")
    rem = hdr.get("x-requests-remaining")
    return (int(float(rem)) if rem is not None else remaining), True


def pull_btts(sport, label, evs, state, now, remaining):
    """BTTS per event (additional market). Once per match; guarded by BTTS_RESERVE."""
    meta = state.setdefault("_meta", {})
    if not BTTS or meta.get(f"btts_none:{label}") == week_key(now):
        return remaining
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    d = OUT / "raw" / now.strftime("%Y/%m/%d"); d.mkdir(parents=True, exist_ok=True)
    for e in evs:
        s = state.get(e["id"], {})
        if s.get("btts"):
            continue
        if remaining is not None and remaining - 1 < max(CREDIT_RESERVE, BTTS_RESERVE):
            print(f"SKIP {label} btts: {remaining} credits left, BTTS reserve {BTTS_RESERVE}")
            return remaining
        try:
            body, hdr = get(f"/sports/{sport}/events/{e['id']}/odds", markets="btts",
                            oddsFormat="decimal", dateFormat="iso", **book_params())
        except RuntimeError as err:
            print(f"WARN {label} btts: {err}")
            return remaining
        (d / f"{label}_btts_{e['id']}_{stamp}.json").write_text(json.dumps(body))
        rows = flatten([body], "late", now.isoformat(timespec="seconds"), label)
        append_rows(rows)
        s["btts"] = now.isoformat(timespec="seconds")
        log_quota(hdr, f"{label} btts x1")
        rem = hdr.get("x-requests-remaining")
        remaining = int(float(rem)) if rem is not None else remaining
        # one empty reply can just mean DraftKings hasn't posted BTTS for that match;
        # three in a row for a league means it doesn't offer it -> skip for the week
        key = f"btts_empty:{label}"
        meta[key] = 0 if rows else meta.get(key, 0) + 1
        if meta[key] >= 3:
            meta[f"btts_none:{label}"] = week_key(now)
            meta[key] = 0
            print(f"{label}: DraftKings returned no BTTS 3 times; skipping BTTS for this league this week")
            return remaining
    return remaining


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
            remaining, ok = pull(sport, label, late, "late", state, now, remaining)
            if ok:
                remaining = pull_btts(sport, label, late, state, now, remaining)
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
    match in each league. Costs 1 credit per market requested,
    per league."""
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


def diag():
    """FREE: what the scheduled run sees per league (fixtures only; no odds calls)."""
    now = datetime.now(timezone.utc)
    state = load_state()
    rep = {"utc": now.isoformat(timespec="seconds"), "week_key": week_key(now),
           "team_filter_set": bool(TEAM_FILTER), "team_filter_leagues": sorted(TEAM_FILTER),
           "markets": MARKETS, "bookmakers": BOOKMAKERS, "leagues": {}}
    for sport, label in dict(SPORTS, **(CUPS if INCLUDE_CUPS else {})).items():
        try:
            events, hdr = get(f"/sports/{sport}/events", dateFormat="iso")
        except RuntimeError as e:
            rep["leagues"][label] = {"error": str(e)}; continue
        kept = [e for e in events if keep_event(e, label)]
        rep["leagues"][label] = {
            "events_returned": len(events), "after_team_filter": len(kept),
            "due_weekly_now": len(due_weekly(kept, now)), "due_late_now": len(due_late(kept, state, now)),
            "weekly_done_mark": state.get("_meta", {}).get(f"weekly:{label}"),
            "next_kickoffs": sorted(e["commence_time"] for e in events)[:5],
            "credits_remaining_header": hdr.get("x-requests-remaining")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diag.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"run": run, "weekly": lambda: run(force_weekly=True),
     "check": check, "probe": probe, "diag": diag}[mode]()
