"""
Match Engine — Odds API capture (two-pull cadence).

Runs on a schedule (GitHub Actions). Every run:
  1. lists upcoming fixtures via the FREE /events endpoint
  2. WEEKLY pull: once per week per league, on the first run after Monday
     15:00 UTC (8am Pacific in summer, 7am in winter); covers every match in
     the next 7 days in a single call (DraftKings' early lines)
  3. LATE pull:  within 8h of kickoff, plus one refresh inside the last 65 min
     (after team news) if the first pull came earlier - DraftKings'
     near-closing line, used for CLV. The wide window exists because GitHub
     runs "every 15 min" schedules only every ~2h in practice.
  4. saves raw JSON + flat rows to data/odds/, records what was pulled in
     state/pulled.json, and logs remaining credits to data/odds/quota.csv

Credits are spent only in steps 2-3: one call per competition per pull type,
covering every due match at once. Only DraftKings is requested (ODDS_BOOKMAKERS);
cost per call = markets requested x 1 (up to 10 bookmakers bill as one region).

v6.1 (2026-10-12): the bulk call is moneyline (h2h) only, 1 credit per call. DraftKings
never returned spreads or totals there for these leagues. The side markets (BTTS, totals,
spreads, and alternate goal lines while credits allow) are pulled per event at the match's
first late pull; the event endpoint bills one credit per market it returns. A market
DraftKings returns nothing for three matches in a row in a league is skipped for that
league for the rest of the week. Late window: first pull within 8h of kickoff, because
GitHub runs the schedule only every few hours.

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
# Sharp reference book(s), moneyline only (2026-11-01). Added to the bulk h2h calls, which bill
# per group of up to 10 bookmakers, so this costs no extra credits there. The per-match side
# calls stay DraftKings-only: there each market a book returns is billed, and Pinnacle returns
# main totals/spreads that DraftKings doesn't. Sharp rows go to data/odds/sharp.csv, never to
# snapshots.csv, so every existing DraftKings consumer (fixtures, odds sync, grading, pricer)
# keeps reading DraftKings only.
SHARP_BOOKMAKERS = os.environ.get("ODDS_SHARP_BOOKMAKERS", "pinnacle").strip()
REGIONS = os.environ.get("ODDS_REGIONS", "us")
MARKETS = os.environ.get("ODDS_MARKETS", "h2h")
BTTS = os.environ.get("ODDS_BTTS", "true").lower() == "true"
BTTS_RESERVE = int(os.environ.get("BTTS_RESERVE", "100"))    # stop BTTS below this
# v6.1 (2026-10-12): DraftKings never returned spreads or totals in the bulk call for these
# leagues (every pull since 2026-09-25 came back h2h only, billed 3 credits). The side markets
# are requested per match instead, at its first late pull, in tiers by credits left. The event
# endpoint bills only the markets it returns, and a market DraftKings returns nothing for three
# times in a row in a league is dropped for that league for the rest of the week.
EVENT_MARKETS_FULL = os.environ.get("EVENT_MARKETS_FULL", "btts,totals,spreads,alternate_totals,alternate_spreads")
EVENT_MARKETS_CORE = os.environ.get("EVENT_MARKETS_CORE", "btts,alternate_totals,alternate_spreads")
FULL_RESERVE = int(os.environ.get("FULL_RESERVE", "250"))     # plain totals/spreads only above this
CORE_RESERVE = int(os.environ.get("CORE_RESERVE", "130"))     # alternate totals/spreads only above this
CREDIT_RESERVE = int(os.environ.get("CREDIT_RESERVE", "40"))  # never go below
# 2026-11-09 (docs/prereg-forward-2026-11.md, rule S1): for each S1 pick only, re-pull that one market inside
# the last TEAM_NEWS_MIN minutes before kickoff, for CLV against DraftKings' own near-close price. Billed one
# credit per market returned. Hard cap per odds-bot week; rows go to data/odds/clv.csv, never snapshots.csv, so
# no existing consumer (fixtures, odds sync, research record) ever sees them.
CLV_WEEKLY_CAP = int(os.environ.get("CLV_WEEKLY_CAP", "15"))
S1_PICKS = ROOT / "data" / "picks" / "s1.csv"
WEEKLY_WEEKDAY = 0               # Monday
WEEKLY_HOUR_UTC = 15             # 15:00 UTC = 8am PDT / 7am PST
WEEKLY_WINDOW_MIN = 7 * 24 * 60  # the weekly pull covers the next 7 days
# v6.1: GitHub runs the "every 15 min" schedule only every 3-7 hours in practice (Oct 4-12:
# 30 runs in 8 days), so a 3h window missed most Sunday matches entirely. The first late pull
# now happens in any run within 8h of kickoff; the team-news refresh is unchanged.
LATE_MAX_MIN = int(os.environ.get("LATE_MAX_MIN", "480"))
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


SNAP_COLS = ["pulled_at", "kind", "comp", "event_id", "commence", "home", "away",
             "bookmaker", "book_last_update", "market", "outcome", "point", "price"]


def _primary_books():
    return {b.strip() for b in BOOKMAKERS.split(",") if b.strip()}


def _write_rows(f, rows):
    if not rows:
        return
    new = not f.exists()
    with f.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(SNAP_COLS)
        w.writerows(rows)


def append_rows(rows):
    """DraftKings (the primary books) -> snapshots.csv; any other book -> sharp.csv.
    With BOOKMAKERS empty (regions mode) everything goes to snapshots.csv, as before."""
    prim = _primary_books()
    if not prim:
        _write_rows(OUT / "snapshots.csv", rows)
        return
    _write_rows(OUT / "snapshots.csv", [r for r in rows if r[7] in prim])
    _write_rows(OUT / "sharp.csv", [r for r in rows if r[7] not in prim])


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


def book_params(sharp=False):
    """Bookmakers take priority over regions in the API; send only one of them.
    sharp=True adds the sharp reference book(s) (bulk moneyline calls only)."""
    if not BOOKMAKERS:
        return {"regions": REGIONS}
    books = [b.strip() for b in BOOKMAKERS.split(",") if b.strip()]
    if sharp:
        books += [b.strip() for b in SHARP_BOOKMAKERS.split(",") if b.strip() and b.strip() not in books]
    return {"bookmakers": ",".join(books)}


def max_cost(sharp=False):
    n_markets = len(MARKETS.split(","))
    if BOOKMAKERS:
        return n_markets * -(-len(book_params(sharp)["bookmakers"].split(",")) // 10)
    return n_markets * len(REGIONS.split(","))


def pull(sport, label, evs, kind, state, now, remaining):
    """Returns (remaining, pulled?)."""
    cost = max_cost(sharp=True)   # billed per market requested, even if the book offers none
    # unknown balance (first ever pull) -> allow; the response header sets it
    if remaining is not None and remaining - cost < CREDIT_RESERVE:
        print(f"SKIP {label} {kind}: {remaining} credits left, reserve {CREDIT_RESERVE}")
        return remaining, False
    ids = ",".join(e["id"] for e in evs)
    body, hdr = get(f"/sports/{sport}/odds", markets=MARKETS, oddsFormat="decimal",
                    dateFormat="iso", eventIds=ids, **book_params(sharp=True))
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


def event_markets(label, meta, now, remaining):
    """Which per-match markets to ask for, by credits left; minus any market DraftKings
    has returned nothing for three times in a row in this league this week."""
    if remaining is None or remaining > FULL_RESERVE:
        want = EVENT_MARKETS_FULL
    elif remaining > CORE_RESERVE:
        want = EVENT_MARKETS_CORE
    else:
        want = "btts"
    wk = week_key(now)
    out = [m for m in want.split(",") if m and meta.get(f"none:{m}:{label}") != wk]
    if not BTTS or meta.get(f"btts_none:{label}") == wk:
        out = [m for m in out if m != "btts"]
    return out


def pull_btts(sport, label, evs, state, now, remaining):
    """Side markets per event (BTTS, totals, spreads and alternate goal lines; corners
    are disabled). Once per match, at its first late pull. The event endpoint bills
    one credit per market it returns; guarded by BTTS_RESERVE overall."""
    meta = state.setdefault("_meta", {})
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    d = OUT / "raw" / now.strftime("%Y/%m/%d"); d.mkdir(parents=True, exist_ok=True)
    for e in evs:
        s = state.get(e["id"], {})
        if s.get("btts") or s.get("side"):
            continue
        mks = event_markets(label, meta, now, remaining)
        if not mks:
            return remaining
        if remaining is not None and remaining - len(mks) < max(CREDIT_RESERVE, BTTS_RESERVE):
            print(f"SKIP {label} side markets: {remaining} credits left, reserve {BTTS_RESERVE}")
            return remaining
        try:
            body, hdr = get(f"/sports/{sport}/events/{e['id']}/odds", markets=",".join(mks),
                            oddsFormat="decimal", dateFormat="iso", **book_params())
        except RuntimeError as err:
            print(f"WARN {label} side markets: {err}")
            return remaining
        (d / f"{label}_side_{e['id']}_{stamp}.json").write_text(json.dumps(body))
        evs_body = body if isinstance(body, list) else [body] if isinstance(body, dict) else []
        evs_body = [b for b in evs_body if isinstance(b, dict) and "id" in b]
        rows = flatten(evs_body, "late", now.isoformat(timespec="seconds"), label)
        append_rows(rows)
        s["side"] = now.isoformat(timespec="seconds")
        got = {r[9] for r in rows}
        log_quota(hdr, f"{label} side x1 {'+'.join(sorted(got)) or 'none'}")
        rem = hdr.get("x-requests-remaining")
        remaining = int(float(rem)) if rem is not None else remaining
        # one empty reply can just mean DraftKings hasn't posted a market for that match yet;
        # three in a row in a league means it doesn't offer it -> skip it for the week
        for m in mks:
            key = f"empty:{m}:{label}"
            meta[key] = 0 if m in got else meta.get(key, 0) + 1
            if meta[key] >= 3:
                meta[f"none:{m}:{label}"] = week_key(now)
                meta[key] = 0
                print(f"{label}: DraftKings returned no {m} 3 times; skipping it for this league this week")
    return remaining


def s1_picks(label=None):
    """S1 picks from data/picks/s1.csv: {event_id: (dk_market, kickoff)}. Standard library only."""
    if not S1_PICKS.exists():
        return {}
    league = {"PL": "E0", "LaLiga": "SP1", "Bundesliga": "D1", "Ligue1": "F1"}.get(label)
    out = {}
    with S1_PICKS.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("decision") == "pick" and r.get("dk_market") and (league is None or r.get("league") == league):
                out[r["event_id"]] = (r["dk_market"], r["kickoff"])
    return out


def pull_clv(sport, label, state, now, remaining):
    """Near-kickoff re-pull of each S1 pick's market (CLV). Once per pick, inside the last TEAM_NEWS_MIN
    minutes, at most CLV_WEEKLY_CAP per odds-bot week. Over the cap the pick is marked and has no CLV."""
    meta = state.setdefault("_meta", {})
    wk = f"clv:{week_key(now)}"
    for eid, (mk, ko) in s1_picks(label).items():
        s = state.setdefault(eid, {})
        if s.get("clv") or s.get("clv_skipped"):
            continue
        m = minutes_to(ko, now)
        if not (LATE_MIN_MIN < m <= TEAM_NEWS_MIN):
            continue
        if meta.get(wk, 0) >= CLV_WEEKLY_CAP:
            s["clv_skipped"] = f"weekly cap {CLV_WEEKLY_CAP} reached"
            print(f"SKIP {label} CLV re-pull {eid}: weekly cap {CLV_WEEKLY_CAP} reached")
            continue
        if remaining is not None and remaining - 1 < CREDIT_RESERVE:
            print(f"SKIP {label} CLV re-pull {eid}: {remaining} credits left, reserve {CREDIT_RESERVE}")
            return remaining
        try:
            body, hdr = get(f"/sports/{sport}/events/{eid}/odds", markets=mk,
                            oddsFormat="decimal", dateFormat="iso", **book_params())
        except RuntimeError as err:
            print(f"WARN {label} CLV re-pull {eid}: {err}")
            continue
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        d = OUT / "raw" / now.strftime("%Y/%m/%d"); d.mkdir(parents=True, exist_ok=True)
        (d / f"{label}_clv_{eid}_{stamp}.json").write_text(json.dumps(body))
        evs_body = body if isinstance(body, list) else [body] if isinstance(body, dict) else []
        evs_body = [b for b in evs_body if isinstance(b, dict) and "id" in b]
        rows = flatten(evs_body, "clv", now.isoformat(timespec="seconds"), label)
        _write_rows(OUT / "clv.csv", [r for r in rows if r[7] in _primary_books()] if _primary_books() else rows)
        s["clv"] = now.isoformat(timespec="seconds")
        meta[wk] = meta.get(wk, 0) + 1
        log_quota(hdr, f"{label} clv x1 {mk}")
        print(f"PULLED {label} CLV re-pull {eid} {mk}: {len(rows)} rows, {int(m)} min before kickoff")
        rem = hdr.get("x-requests-remaining")
        remaining = int(float(rem)) if rem is not None else remaining
    return remaining


def run(force_weekly=False):
    now = datetime.now(timezone.utc)
    comps = dict(SPORTS, **(CUPS if INCLUDE_CUPS else {}))
    state = load_state()
    # Balance: last value logged after a paid call. Scheduled runs never call
    # /sports, so polling every 15 min cannot consume credits even if that
    # endpoint were ever billed.
    remaining = last_logged_remaining()
    diag = {}
    for sport, label in comps.items():
        try:
            events, hdr = get(f"/sports/{sport}/events", dateFormat="iso")   # free
        except RuntimeError as e:
            print(f"WARN {label}: {e}"); diag[label] = {"error": str(e)[:200]}; continue
        if hdr.get("x-requests-remaining") is not None:
            remaining = int(float(hdr["x-requests-remaining"]))
        n_api = len(events)
        events = [e for e in events if keep_event(e, label)]
        wk_due = due_weekly(events, now)
        d = diag[label] = {"events_from_api": n_api, "after_team_filter": len(events),
                           "team_filter": TEAM_FILTER.get(label, []),
                           "due_weekly": len(wk_due), "weekly_done": weekly_done(state, label, now)}
        did = False
        if force_weekly or not weekly_done(state, label, now):
            if wk_due:
                remaining, ok = pull(sport, label, wk_due, "weekly", state, now, remaining)
                if ok:
                    mark_weekly(state, label, now)
                did = True
            elif n_api:
                # the API listed fixtures but none fall in the next 7 days -> genuinely
                # nothing this week. If the API listed NOTHING, don't mark: that is more
                # likely a hiccup, and the next (free) run retries.
                mark_weekly(state, label, now)
                print(f"{label}: weekly marked done, nothing in the next 7 days "
                      f"({n_api} listed, {len(events)} after team filter)")
        late = due_late(events, state, now)
        d["due_late"] = len(late)
        if late:
            remaining, ok = pull(sport, label, late, "late", state, now, remaining)
            if ok:
                remaining = pull_btts(sport, label, late, state, now, remaining)
            did = True
        remaining = pull_clv(sport, label, state, now, remaining)
        if not did:
            print(f"{label}: nothing due ({n_api} listed by the API, {len(events)} after team filter)")
    save_state(state)
    # Diagnostics committed with the odds so a quiet week can be explained later.
    # Written only when the counts change, so it doesn't create a commit every run.
    f = ROOT / "state" / "last_run.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    if old.get("leagues") != diag:
        f.write_text(json.dumps({"utc": now.isoformat(timespec="seconds"), "markets": MARKETS,
                                 "bookmakers": BOOKMAKERS, "leagues": diag}, indent=1))


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


def soon():
    """Exit 0 if a tracked match still needs a late pull within the next 10 minutes or more
    (kicks off within LATE_MAX_MIN + 15 min and its team-news refresh hasn't happened), else 1.
    The workflow keeps looping while this is true, so near-kickoff pulls no longer depend on
    GitHub's unreliable cron (2026-11-01). Reads only the state file; free."""
    now = datetime.now(timezone.utc)
    for k, v in load_state().items():
        if k.startswith("_") or not isinstance(v, dict) or not v.get("commence") or v.get("late_final"):
            continue
        if LATE_MIN_MIN < minutes_to(v["commence"], now) <= LATE_MAX_MIN + 15:
            print(f"soon: {v.get('comp')} {v.get('home')} v {v.get('away')} at {v['commence']}")
            sys.exit(0)
    st = load_state()
    for eid, (_mk, ko) in s1_picks().items():          # an S1 pick still waiting for its CLV re-pull
        v = st.get(eid, {})
        if not v.get("clv") and not v.get("clv_skipped") and LATE_MIN_MIN < minutes_to(ko, now) <= TEAM_NEWS_MIN + 15:
            print(f"soon: S1 pick {eid} needs its CLV re-pull before {ko}")
            sys.exit(0)
    print("soon: nothing within the late window")
    sys.exit(1)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"run": run, "weekly": lambda: run(force_weekly=True),
     "check": check, "probe": probe, "diag": diag, "soon": soon}[mode]()
