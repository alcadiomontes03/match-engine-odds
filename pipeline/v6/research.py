"""
Alpha Lines v6.1 - research record, every-line test, sanity checks (2026-10-12).

Why: an outside review (Codex, 2026-10-12) pointed out that judging the model only on the pick it
chose per market is a selected sample (the biggest disagreement with DraftKings), and that nothing
stored what the model knew at the time in a form that later code changes can't rewrite. This file:

  record   Append every DraftKings side line of every settled fixture to data/research/lines.csv,
           and every auto-pick to data/research/picks.csv. Append-only by key: a row is never
           changed once written, and git history makes the record permanent. Re-running is safe.
  report   The pre-registered tests (docs/v6.md, "Research record"): model vs DraftKings' own
           de-vigged price on every line, scored by log loss, with 95% intervals from a bootstrap
           that resamples whole matches (lines and picks on one match move together).
  sanity   Data checks that would have caught past bugs (goal level, missing/duplicate fixtures,
           impossible probabilities or prices, flags, how close to kickoff "closing" prices were).
  devig    One-off: which de-vig method (proportional, additive, power, Shin) scores best on
           football-data's historical closing prices.
  weekly   record + sanity + report, writing data/research/sanity/<date>.json and
           data/research/reports/<date>.md.

  python pipeline/v6/research.py weekly --db /tmp/me/db [--snap data/odds/snapshots.csv]

--db is an ArtifactData out_dir holding fixtures/, results/, odds/, predictions/, corners/,
auto_picks/ (and optionally state/summary.json). Nothing here writes to the dashboard.
"""
from __future__ import annotations
import argparse, csv, glob, json, math, sys
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
from v5 import market as M  # noqa: E402

OUT = ROOT / "data" / "research"
CODE_VERSION = "research-1 (2026-10-12)"
V61_START = "2026-10-12T04:00:00Z"   # auto-picks logged after this use the v6.1 best-pick rule
LINE_COLS = ["key", "fixture_id", "league", "kickoff", "home", "away", "market", "line", "sel_a", "sel_b",
             "dk_a", "dk_b", "dk_overround", "dk_shin_a", "side_pulled_at", "side_mins_to_ko",
             "ml_home", "ml_draw", "ml_away", "ml_pulled_at", "fit_anchor", "fit_line", "used_in_fit",
             "model_p_a", "dc_p_a", "corners_p_a", "home_goals", "away_goals", "corners_total",
             "outcome_a", "recorded_at", "code_version"]
PICK_COLS = ["key", "fixture_id", "league", "kickoff", "market", "line", "side", "label", "rule", "logged_at",
             "mins_to_ko", "model_prob", "dk_price", "ev", "result", "flat_pnl", "recorded_at", "code_version"]


# ------------------------------------------------------------------ loading
def _load(db: Path, coll: str) -> dict:
    out = {}
    for f in glob.glob(str(db / coll / "*.json")):
        d = json.loads(Path(f).read_text())
        if isinstance(d, dict) and "data" in d and isinstance(d["data"], dict) and "id" not in d:
            d = d["data"]
        out[Path(f).stem] = d
    return out


def _ts(x):
    if not x:
        return None
    return datetime.fromisoformat(str(x).replace("Z", "+00:00"))


def _mins(a, b):
    a, b = _ts(a), _ts(b)
    return None if a is None or b is None else round((b - a).total_seconds() / 60.0, 1)


def _is_half_or_whole(x: float) -> bool:
    return abs(x * 2 - round(x * 2)) < 1e-9


# ------------------------------------------------------------------ the market-anchored grid (same as the dashboard)
def anchored_grid(od: dict, pred: dict | None):
    """Mirror of the dashboard's mktPred(): Shin 1X2 from DraftKings' latest pre-kickoff moneyline,
    plus its main total (the half/whole line whose Shin over-probability is closest to 0.5); with
    no total, its BTTS price; with neither, a weak prior on total goals. Returns (G, anchor, line)."""
    px = od.get("latest") or od.get("open") or {}
    if not all((px.get(k) or 0) > 1 for k in ("home", "draw", "away")):
        return None, None, None
    pH, pD, pA = M.shin([px["home"], px["draw"], px["away"]])
    side = od.get("side") or {}
    main = None
    for L, t in (side.get("totals") or {}).items():
        l = float(L)
        if not _is_half_or_whole(l) or not ((t.get("over") or 0) > 1 and (t.get("under") or 0) > 1):
            continue
        po = M.shin([t["over"], t["under"]])[0]
        if main is None or abs(po - 0.5) < abs(main[1] - 0.5):
            main = (l, po)
    prior = ((pred or {}).get("xg_home") or 1.35) + ((pred or {}).get("xg_away") or 1.35)
    if main:
        lam, mu = M.implied_lambdas(pH, pD, pA, main[1], line=main[0], tot_prior=prior)
        return M.grid(lam, mu), "total", main[0]
    b = side.get("btts") or {}
    if (b.get("yes") or 0) > 1 and (b.get("no") or 0) > 1:
        pB = M.shin([b["yes"], b["no"]])[0]
        lam, mu = _lambdas_btts(pH, pD, pA, pB, prior)
        return M.grid(lam, mu), "btts", None
    lam, mu = M.implied_lambdas(pH, pD, pA, None, tot_prior=prior)
    return M.grid(lam, mu), "1x2", None


def _lambdas_btts(pH, pD, pA, pB, prior):
    from scipy.optimize import least_squares

    def resid(x):
        lam, mu = np.exp(x)
        G = M.grid(lam, mu)
        p = M.grid_probs(G)
        return np.array([p["H"] - pH, p["D"] - pD, p["A"] - pA, 2.0 * (float(G[1:, 1:].sum()) - pB)])
    sup = np.log(max(pH, 1e-3) / max(pA, 1e-3)) * 0.6
    x0 = np.array([np.log(1.35) + sup / 2, np.log(1.35) - sup / 2])
    r = least_squares(resid, x0, bounds=([np.log(0.05)] * 2, [np.log(6.0)] * 2))
    return tuple(float(v) for v in np.exp(r.x))


_I, _J = np.indices((M.MAXG + 1, M.MAXG + 1))


def p_side_a(G, market: str, line):
    """Model probability of selection A (over / home covers / yes), pushes excluded. None for quarter lines."""
    if market == "btts":
        return float(G[1:, 1:].sum())
    L = float(line)
    if not _is_half_or_whole(L):
        return None
    m = (_I + _J - L) if market == "total" else (_I - _J + L)
    w, l = G[m > 1e-9].sum(), G[m < -1e-9].sum()
    return float(w / (w + l))


def outcome_a(market, line, hg, ag, corners=None):
    """1 = A won, 0 = A lost, 'P' = push, 'Q' = quarter line (half results, not scored), '' = unknown."""
    if market == "btts":
        return int(hg > 0 and ag > 0)
    L = float(line)
    if market == "corners":
        if corners is None:
            return ""
        v = corners - L
    elif market == "total":
        v = hg + ag - L
    else:
        v = hg - ag + L
    if not _is_half_or_whole(L):
        return "Q"
    return "P" if abs(v) < 1e-9 else int(v > 0)


# ------------------------------------------------------------------ record
def _read_csv(path: Path, cols):
    if not path.exists():
        return pd.DataFrame(columns=cols)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _append(path: Path, cols, rows):
    old = _read_csv(path, cols)
    have = set(old["key"]) if len(old) else set()
    new = [r for r in rows if r["key"] not in have]
    if not new:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in sorted(new, key=lambda r: (r.get("kickoff", ""), r["key"])):
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})
    return len(new)


def build_lines(db: Path, now_iso: str):
    fx, res, odds = _load(db, "fixtures"), _load(db, "results"), _load(db, "odds")
    preds, cor = _load(db, "predictions"), _load(db, "corners")
    rows = []
    for fid, od in odds.items():
        f, r = fx.get(fid), res.get(fid)
        side = (od or {}).get("side")
        if not f or not r or not side or "home_score" not in r:
            continue
        if f.get("status") == "postponed":
            continue
        hg, ag = int(r["home_score"]), int(r["away_score"])
        st = r.get("stats") or {}
        ctot = None
        if (st.get("h") or {}).get("corners") is not None and (st.get("a") or {}).get("corners") is not None:
            ctot = int(st["h"]["corners"]) + int(st["a"]["corners"])
        pred = preds.get(fid) or {}
        G, anchor, fit_line = anchored_grid(od, pred)
        ml = od.get("latest") or od.get("open") or {}
        base = dict(fixture_id=fid, league=f.get("league"), kickoff=f.get("kickoff"), home=f.get("home"),
                    away=f.get("away"), side_pulled_at=side.get("pulled_at"),
                    side_mins_to_ko=_mins(side.get("pulled_at"), f.get("kickoff")),
                    ml_home=ml.get("home"), ml_draw=ml.get("draw"), ml_away=ml.get("away"),
                    ml_pulled_at=ml.get("pulled_at"), fit_anchor=anchor, fit_line=fit_line,
                    home_goals=hg, away_goals=ag, corners_total=ctot, recorded_at=now_iso,
                    code_version=CODE_VERSION)
        items = []
        if (side.get("btts") or {}).get("yes"):
            items.append(("btts", "", "yes", "no", side["btts"]["yes"], side["btts"]["no"]))
        for L, t in (side.get("totals") or {}).items():
            items.append(("total", L, "over", "under", t.get("over"), t.get("under")))
        for L, t in (side.get("spreads") or {}).items():
            items.append(("spread", L, "home", "away", t.get("home"), t.get("away")))
        for L, t in (side.get("corners") or {}).items():
            items.append(("corners", L, "over", "under", t.get("over"), t.get("under")))
        for mk, L, a, b, pa, pb in items:
            if not (pa and pb and float(pa) > 1 and float(pb) > 1):
                continue
            pa, pb = float(pa), float(pb)
            row = dict(base, key=f"{fid}|{mk}|{L}", market=mk, line=L, sel_a=a, sel_b=b, dk_a=pa, dk_b=pb,
                       dk_overround=round(1 / pa + 1 / pb, 5), dk_shin_a=round(float(M.shin([pa, pb])[0]), 5))
            row["used_in_fit"] = int((mk == "total" and anchor == "total" and fit_line is not None
                                      and abs(float(L) - fit_line) < 1e-9) or (mk == "btts" and anchor == "btts"))
            if G is not None and mk != "corners":
                p = p_side_a(G, mk, L)
                row["model_p_a"] = None if p is None else round(p, 5)
            if mk == "btts" and pred.get("btts"):
                row["dc_p_a"] = pred["btts"].get("yes")
            elif mk == "total" and (pred.get("totals") or {}).get(f"{float(L):.1f}"):
                t = pred["totals"][f"{float(L):.1f}"]
                row["dc_p_a"] = round(t["over"] / (t["over"] + t["under"]), 5)
            elif mk == "spread":
                k = f"{float(L):+.1f}"
                if (pred.get("spread") or {}).get(k):
                    row["dc_p_a"] = pred["spread"][k].get("home")
            if mk == "corners":
                ct = ((cor.get(fid) or {}).get("totals") or {}).get(f"{float(L):.1f}")
                if ct:
                    row["corners_p_a"] = round(ct["over"] / (ct["over"] + ct["under"]), 5)
            row["outcome_a"] = outcome_a(mk, L, hg, ag, ctot)
            rows.append(row)
    return rows


def _grade_pick(p, r):
    hg, ag = int(r["home_score"]), int(r["away_score"])
    st = r.get("stats") or {}
    m, side = p.get("market"), p.get("side")
    L = p.get("line")

    def ou(v, line):
        if v is None:
            return None
        line = float(line)
        return "P" if abs(v - line) < 1e-9 else ("W" if (side == "over") == (v > line) else "L")
    if m == "ml":
        return "W" if {"home": hg > ag, "away": ag > hg, "draw": hg == ag}[side] else "L"
    if m == "tot":
        return ou(hg + ag, L)
    if m == "btts":
        y = hg > 0 and ag > 0
        return "W" if (side == "yes") == y else "L"
    if m == "sp":
        h = float(L)
        v = (hg - ag + h) if side == "home" else (ag - hg - h)
        return "P" if abs(v) < 1e-9 else ("W" if v > 0 else "L")
    if m in ("sh", "cor"):
        k = "shots" if m == "sh" else "corners"
        a, b = (st.get("h") or {}).get(k), (st.get("a") or {}).get(k)
        return None if a is None or b is None else ou(int(a) + int(b), L)
    if m == "sot":
        t, n = str(L).split(":")
        v = (st.get("h" if t == "home" else "a") or {}).get("sot")
        return None if v is None else ou(int(v), n)
    return None


def build_picks(db: Path, now_iso: str):
    fx, res, aps = _load(db, "fixtures"), _load(db, "results"), _load(db, "auto_picks")
    rows = []
    for pid, p in aps.items():
        fid = p.get("fixture_id")
        f, r = fx.get(fid), res.get(fid)
        if not f or not r or "home_score" not in r:
            continue
        g = "V" if f.get("status") == "postponed" else _grade_pick(p, r)
        if g is None:
            continue
        price = p.get("dk_price")
        pnl = None
        if price and g in ("W", "L", "P", "V"):
            pnl = round(float(price) - 1, 4) if g == "W" else (-1.0 if g == "L" else 0.0)
        rows.append(dict(key=pid, fixture_id=fid, league=p.get("league"), kickoff=p.get("kickoff"),
                         market=p.get("market"), line=p.get("line"), side=p.get("side"), label=p.get("label"),
                         rule="v6.1" if (p.get("logged_at") or "") >= V61_START else "safest_line",
                         logged_at=p.get("logged_at"), mins_to_ko=_mins(p.get("logged_at"), p.get("kickoff")),
                         model_prob=p.get("model_prob"), dk_price=price, ev=p.get("ev"), result=g, flat_pnl=pnl,
                         recorded_at=now_iso, code_version=CODE_VERSION))
    return rows


def record(db: Path):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n1 = _append(OUT / "lines.csv", LINE_COLS, build_lines(db, now))
    n2 = _append(OUT / "picks.csv", PICK_COLS, build_picks(db, now))
    return {"lines_added": n1, "picks_added": n2}


# ------------------------------------------------------------------ report (pre-registered tests)
def _ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def cluster_boot(values: np.ndarray, clusters: np.ndarray, n=2000, seed=7, stat=np.mean, level=0.95):
    """Interval for stat(values) (95% by default), resampling whole clusters (matches) with replacement."""
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    if len(values) == 0:
        return (math.nan, math.nan, math.nan)
    ids, inv = np.unique(clusters, return_inverse=True)
    groups = [values[inv == k] for k in range(len(ids))]
    rng = np.random.default_rng(seed)
    est = float(stat(values))
    if len(ids) < 2:
        return (est, math.nan, math.nan)
    sims = []
    for _ in range(n):
        pick = rng.integers(0, len(ids), len(ids))
        sims.append(stat(np.concatenate([groups[k] for k in pick])))
    a = (1 - level) / 2 * 100
    lo, hi = np.percentile(sims, [a, 100 - a])
    return (est, float(lo), float(hi))


def _ci(lo, hi, fmt="+.4f"):
    return "n/a (one match)" if not (np.isfinite(lo) and np.isfinite(hi)) else f"{lo:{fmt}} to {hi:{fmt}}"


def report(today: str) -> str:
    L = _read_csv(OUT / "lines.csv", LINE_COLS)
    P = _read_csv(OUT / "picks.csv", PICK_COLS)
    out = [f"# Research report {today}", "",
           f"Code: {CODE_VERSION}. Every number has a 95% interval from a bootstrap that resamples whole matches.",
           "Primary tests are fixed in docs/v6.md (\"Research record\"); everything else here is exploratory.", ""]
    if len(L):
        d = L[L.outcome_a.isin(["0", "1"])].copy()
        d["y"] = d.outcome_a.astype(int)
        out += ["## Test 1 (primary): model vs DraftKings on every line", "",
                "Mean log-loss difference, model minus DraftKings' own de-vigged (Shin) price, on every scored line. "
                "Negative means the model priced the line better. Lines used to fit the model are excluded.", "",
                "| Market | League | Lines | Matches | Model − DK (log loss) | 95% interval |", "|---|---|---|---|---|---|"]
        g = d[(d.used_in_fit != "1") & (d.model_p_a != "") & (d.market != "corners")]
        for (mk, lg), s in [(("ALL", "ALL"), g)] + list(g.groupby(["market", "league"])) + \
                [((m, "ALL"), s) for m, s in g.groupby("market")]:
            if not len(s):
                continue
            diff = _ll(s.model_p_a.astype(float), s.y) - _ll(s.dk_shin_a.astype(float), s.y)
            e, lo, hi = cluster_boot(diff, s.fixture_id.values)
            out.append(f"| {mk} | {lg} | {len(s)} | {s.fixture_id.nunique()} | {e:+.4f} | {_ci(lo, hi)} |")
            if mk == "ALL":
                _, lo99, hi99 = cluster_boot(diff, s.fixture_id.values, level=0.99)
                primary = (e, lo99, hi99, len(s), s.fixture_id.nunique())
        if g.empty:
            primary = None
        if primary:
            e, lo99, hi99, n, m = primary
            verdict = ("model better" if hi99 < 0 else "DraftKings better" if lo99 > 0 else "no difference shown yet")
            out += ["", f"**Primary read (Test 1, all goal markets pooled):** {e:+.4f} on {n} lines / {m} matches; "
                        f"99% interval {_ci(lo99, hi99)} -> {verdict}. Checkpoint looks use the 99% interval; "
                        "only the end-of-season look uses 95%."]
        r = d[(d.used_in_fit != "1") & (d.dc_p_a != "") & (d.market != "corners")]
        if len(r):
            diff = _ll(r.dc_p_a.astype(float), r.y) - _ll(r.dk_shin_a.astype(float), r.y)
            e, lo, hi = cluster_boot(diff, r.fixture_id.values)
            out += ["", f"Exploratory: the Dixon-Coles ratings (not market-anchored) vs DraftKings on the same kind of lines: "
                        f"{e:+.4f} on {len(r)} lines (95% {_ci(lo, hi)})."]
        c = d[(d.market == "corners") & (d.corners_p_a != "")]
        out += ["", "## Test 2 (primary): corners model vs DraftKings' corners price", ""]
        if len(c):
            diff = _ll(c.corners_p_a.astype(float), c.y) - _ll(c.dk_shin_a.astype(float), c.y)
            e, lo, hi = cluster_boot(diff, c.fixture_id.values)
            out.append(f"{len(c)} lines on {c.fixture_id.nunique()} matches: model − DK log loss {e:+.4f} "
                       f"(95% {_ci(lo, hi)}).")
        else:
            out.append("No DraftKings corners prices yet (capture starts week 7).")
        out += ["", "## Exploratory: DraftKings margin by market", "", "| Market | Lines | Mean overround |", "|---|---|---|"]
        for mk, s in L.groupby("market"):
            out.append(f"| {mk} | {len(s)} | {s.dk_overround.astype(float).mean():.4f} |")
        if len(d):
            out += ["", "## Exploratory: which de-vig fits DraftKings' side lines best", "",
                    "| Method | Log loss on all scored side lines | Minus Shin | 95% interval |", "|---|---|---|---|"]
            pa, pb = d.dk_a.astype(float).to_numpy(), d.dk_b.astype(float).to_numpy()
            base = _ll(d.dk_shin_a.astype(float), d.y)
            for m in ("proportional", "additive", "power", "shin"):
                pr = np.array([_devig_all([x, y])[m][0] for x, y in zip(pa, pb)])
                ll = _ll(pr, d.y)
                ok = np.isfinite(ll)
                e, lo, hi = cluster_boot((ll - base)[ok], d.fixture_id.values[ok])
                out.append(f"| {m} | {np.nanmean(ll):.4f} | {e:+.5f} | {_ci(lo, hi, '+.5f')} |")
        t = L[L.side_mins_to_ko != ""]
        if len(t):
            m = t.drop_duplicates("fixture_id").side_mins_to_ko.astype(float)
            out += ["", f"Side prices were pulled a median {m.median():.0f} minutes before kickoff "
                        f"(range {m.min():.0f} to {m.max():.0f}, {len(m)} matches)."]
    else:
        out.append("No settled lines with DraftKings side prices yet.")
    out += ["", "## Exploratory: auto-picks (flat 1-unit stakes, DraftKings-priced only)", "",
            "| Rule | Market | Priced picks | Matches | ROI per bet | 95% interval |", "|---|---|---|---|---|---|"]
    if len(P):
        q = P[(P.flat_pnl != "") & (P.result.isin(["W", "L", "P"]))]
        for (rule, mk), s in q.groupby(["rule", "market"]):
            e, lo, hi = cluster_boot(s.flat_pnl.astype(float), s.fixture_id.values)
            out.append(f"| {rule} | {mk} | {len(s)} | {s.fixture_id.nunique()} | {e:+.3f} | {_ci(lo, hi, '+.3f')} |")
        h = P[P.result.isin(["W", "L"])]
        out += ["", f"All graded auto-picks (priced or not): {len(h)}, hit rate {(h.result == 'W').mean():.1%}. "
                    "Hit rate without a price says nothing about value."]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ sanity
def sanity(db: Path, snap: Path | None):
    fx, res, odds = _load(db, "fixtures"), _load(db, "results"), _load(db, "odds")
    preds, aps = _load(db, "predictions"), _load(db, "auto_picks")
    now = datetime.now(timezone.utc)
    checks = []

    def add(name, level, detail, items=None):
        checks.append({"check": name, "level": level, "detail": detail, "items": (items or [])[:20]})

    # 1. goal level: predicted vs actual mean goals per league, settled matches
    rows = []
    for fid, r in res.items():
        f, p = fx.get(fid), preds.get(fid)
        if f and p and "home_score" in r and p.get("xg_home") is not None:
            G, _, _ = anchored_grid(odds.get(fid) or {}, p) if odds.get(fid) else (None, None, None)
            mkt = None if G is None else float(((_I + _J) * G).sum())
            rows.append((f["league"], float(p["xg_home"]) + float(p["xg_away"]), mkt,
                         int(r["home_score"]) + int(r["away_score"])))
    if rows:
        df = pd.DataFrame(rows, columns=["lg", "dc", "mkt", "act"])
        for lg, s in df.groupby("lg"):
            for col, name in (("dc", "ratings"), ("mkt", "market grid")):
                v = s[col].dropna()
                if len(v) < 8:
                    continue
                diff = s.loc[v.index, "act"] - v
                se = diff.std(ddof=1) / math.sqrt(len(diff))
                z = diff.mean() / se if se > 0 else 0
                lvl = "fail" if abs(z) > 3 else "warn" if abs(z) > 2 else "ok"
                add(f"goal level {lg} ({name})", lvl,
                    f"{len(diff)} matches: predicted {v.mean():.2f}, actual {s.loc[v.index,'act'].mean():.2f} goals/game "
                    f"(diff {diff.mean():+.2f}, z {z:+.1f})")
    # 2. missing results
    miss = [f"{f.get('league')} {f.get('home')} v {f.get('away')} {f.get('kickoff','')[:10]}"
            for fid, f in fx.items() if f.get("status") != "postponed" and fid not in res
            and _ts(f.get("kickoff")) and (now - _ts(f["kickoff"])).total_seconds() > 6 * 3600]
    add("missing results (kicked off 6h+ ago)", "warn" if miss else "ok", f"{len(miss)} fixtures", miss)
    # 3. missing match stats (needed to grade shots and corners)
    nost = [fid for fid, r in res.items() if "home_score" in r and not r.get("stats")
            and fx.get(fid) and _ts(fx[fid].get("kickoff")) and (now - _ts(fx[fid]["kickoff"])).days < 10]
    add("results without match stats (last 10 days)", "warn" if nost else "ok", f"{len(nost)} results", nost)
    # 4. duplicate fixtures
    seen, dup = {}, []
    for fid, f in fx.items():
        k = (f.get("league"), f.get("home"), f.get("away"))
        d = _ts(f.get("kickoff"))
        for fid2, d2 in seen.get(k, []):
            if d and d2 and abs((d - d2).days) <= 3:
                dup.append(f"{k} {fid} / {fid2}")
        seen.setdefault(k, []).append((fid, d))
    add("duplicate fixtures", "fail" if dup else "ok", f"{len(dup)} pairs", dup)
    # 5. results consistent with scores
    bad = [fid for fid, r in res.items() if "home_score" in r and r.get("actual") !=
           ("H" if r["home_score"] > r["away_score"] else "A" if r["away_score"] > r["home_score"] else "D")]
    add("result letter matches the score", "fail" if bad else "ok", f"{len(bad)} mismatches", bad)
    # 6. probabilities
    badp, gate = [], []
    for fid, p in preds.items():
        x = p.get("x1x2") or {}
        if x and abs(sum(float(x.get(k, 0)) for k in ("home", "draw", "away")) - 1) > 0.01:
            badp.append(f"{fid} x1x2 sums to {sum(float(v) for v in x.values()):.3f}")
        for L, t in (p.get("totals") or {}).items():
            if abs(t["over"] + t["under"] - 1) > 0.01 or not (0 <= t["over"] <= 1):
                badp.append(f"{fid} total {L}")
        if p.get("gate_open"):
            gate.append(fid)
    add("prediction probabilities", "fail" if badp else "ok", f"{len(badp)} problems", badp)
    add("predictions carrying gate_open: true", "warn" if gate else "ok",
        f"{len(gate)} predictions copy the league's old blend-validation flag from state/summary; blend weight is 0, "
        "so no price is affected, but the flag reads as if the CLV gate were open", gate)
    # 7. prices
    badq = []
    for fid, od in odds.items():
        for tag in ("open", "latest", "close"):
            px = od.get(tag) or {}
            if px and all(px.get(k) for k in ("home", "draw", "away")):
                ovr = sum(1 / float(px[k]) for k in ("home", "draw", "away"))
                if not (1.0 <= ovr <= 1.15):
                    badq.append(f"{fid} {tag} 1X2 overround {ovr:.3f}")
        for mk in ("totals", "spreads", "corners"):
            for L, t in ((od.get("side") or {}).get(mk) or {}).items():
                v = [x for x in t.values() if isinstance(x, (int, float))]
                if len(v) == 2 and not (1.0 <= sum(1 / x for x in v) <= 1.30):
                    badq.append(f"{fid} {mk} {L} overround {sum(1/x for x in v):.3f}")
    add("odds overrounds in a sane range", "warn" if badq else "ok", f"{len(badq)} prices", badq)
    # 8. team-name mapping between the odds feed and the fixtures
    badn = []
    for fid, f in fx.items():
        for a, b in (("home", "odds_home_name"), ("away", "odds_away_name")):
            if f.get(b) and f.get(a):
                r = SequenceMatcher(None, f[a].lower(), f[b].lower()).ratio()
                if r < 0.35:
                    badn.append(f"{f[a]} <- {f[b]} ({r:.2f})")
    add("odds-feed names close to fixture names", "warn" if badn else "ok",
        f"{len(badn)} weak matches (check by eye; abbreviations can be fine)", sorted(set(badn)))
    # 9. auto-picks point at real fixtures
    orphan = [pid for pid, p in aps.items() if p.get("fixture_id") not in fx]
    add("auto-picks with no fixture", "warn" if orphan else "ok", f"{len(orphan)}", orphan)
    # 10. how close to kickoff the 'closing' and side prices were
    gaps = {"close": [], "side": []}
    for fid, od in odds.items():
        ko = (fx.get(fid) or {}).get("kickoff")
        if (od.get("close") or {}).get("pulled_at"):
            gaps["close"].append(_mins(od["close"]["pulled_at"], ko))
        if (od.get("side") or {}).get("pulled_at"):
            gaps["side"].append(_mins(od["side"]["pulled_at"], ko))
    for k, v in gaps.items():
        v = [x for x in v if x is not None]
        if v:
            add(f"{k} price timing", "warn" if sum(x <= 65 for x in v) < len(v) / 2 else "ok",
                f"{len(v)} matches, median {np.median(v):.0f} min before kickoff, "
                f"{sum(x <= 65 for x in v)} within 65 min, max {max(v):.0f}")
        else:
            add(f"{k} price timing", "ok", "no prices yet")
    # 11. odds bot credits
    q = ROOT / "data" / "odds" / "quota.csv"
    if q.exists():
        qq = pd.read_csv(q)
        col = [c for c in qq.columns if "remain" in c.lower()]
        if col and len(qq):
            left = float(qq[col[0]].iloc[-1])
            add("odds API credits", "fail" if left < 40 else "warn" if left < 130 else "ok",
                f"{left:.0f} left (side markets drop to BTTS only below 130)")
    worst = "fail" if any(c["level"] == "fail" for c in checks) else "warn" if any(c["level"] == "warn" for c in checks) else "ok"
    return {"run_at": now.isoformat(timespec="seconds"), "code_version": CODE_VERSION, "status": worst, "checks": checks}


# ------------------------------------------------------------------ de-vig comparison (historical)
def _devig_all(o):
    o = np.asarray(o, float)
    pi = 1 / o
    S = pi.sum()
    out = {"proportional": pi / S}
    a = pi - (S - 1) / len(o)
    out["additive"] = a / a.sum() if np.all(a > 0) else np.full(len(o), np.nan)
    from scipy.optimize import brentq
    try:
        k = brentq(lambda k: (pi ** k).sum() - 1, 0.5, 3.0)
        out["power"] = pi ** k
    except ValueError:
        out["power"] = np.full(len(o), np.nan)
    out["shin"] = M.shin(o)
    return out


def devig_test():
    fd = ROOT / "data" / "fd"
    frames = []
    for f in sorted(fd.glob("*.csv")):
        d = pd.read_csv(f)
        d["lg"], d["season"] = f.stem.split("_")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True).dropna(subset=["FTHG", "FTAG"])
    sets = {"Pinnacle close 1X2": ["PSCH", "PSCD", "PSCA"], "Market-average close 1X2": ["AvgCH", "AvgCD", "AvgCA"],
            "Pinnacle close O/U 2.5": ["PC>2.5", "PC<2.5"], "Market-average close O/U 2.5": ["AvgC>2.5", "AvgC<2.5"]}
    lines = ["# De-vig comparison on football-data closing prices", "",
             "Mean log loss of the actual outcome under each de-vig method (lower is better). "
             "Intervals for the difference against Shin resample whole seasons-leagues.", ""]
    for name, cols in sets.items():
        if not all(c in df.columns for c in cols):
            continue
        s = df.dropna(subset=cols)
        s = s[(s[cols] > 1).all(axis=1)]
        if not len(s):
            continue
        if len(cols) == 3:
            y = np.select([s.FTHG > s.FTAG, s.FTHG == s.FTAG], [0, 1], 2)
        else:
            y = np.where(s.FTHG + s.FTAG > 2, 0, 1)
        ll = {m: [] for m in ("proportional", "additive", "power", "shin")}
        for o, yy in zip(s[cols].to_numpy(float), y):
            for m, p in _devig_all(o).items():
                ll[m].append(-math.log(max(p[yy], 1e-9)) if np.isfinite(p[yy]) else np.nan)
        cl = (s.lg + "_" + s.season).to_numpy()
        lines += [f"## {name} ({len(s)} matches, seasons {s.season.min()}-{s.season.max()})", "",
                  "| Method | Log loss | Minus Shin | 95% interval |", "|---|---|---|---|"]
        base = np.array(ll["shin"])
        for m, v in ll.items():
            v = np.array(v)
            ok = np.isfinite(v) & np.isfinite(base)
            e, lo, hi = cluster_boot(v[ok] - base[ok], cl[ok])
            lines.append(f"| {m} | {np.nanmean(v):.5f} | {e:+.5f} | {lo:+.5f} to {hi:+.5f} |")
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------ cli
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["record", "report", "sanity", "devig", "weekly"])
    ap.add_argument("--db")
    ap.add_argument("--snap", default=str(ROOT / "data" / "odds" / "snapshots.csv"))
    a = ap.parse_args()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if a.cmd == "devig":
        txt = devig_test()
        (OUT / "reports").mkdir(parents=True, exist_ok=True)
        (OUT / "reports" / f"devig-{today}.md").write_text(txt)
        print(txt)
        return
    if a.cmd == "report":
        print(report(today))
        return
    db = Path(a.db)
    if a.cmd == "record":
        print(json.dumps(record(db)))
        return
    s = sanity(db, Path(a.snap))
    if a.cmd == "sanity":
        print(json.dumps(s, indent=1))
        return
    r = record(db)
    (OUT / "sanity").mkdir(parents=True, exist_ok=True)
    (OUT / "sanity" / f"{today}.json").write_text(json.dumps(s, indent=1))
    rep = report(today)
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    (OUT / "reports" / f"{today}.md").write_text(rep)
    flagged = [f"{c['level'].upper()} {c['check']}: {c['detail']}" for c in s["checks"] if c["level"] != "ok"]
    try:   # market benchmark (free, football-data only); a failure here must never block the weekly job
        import benchmark
        benchmark.OUT.mkdir(parents=True, exist_ok=True)
        _t = benchmark.clv_table()
        _t.to_csv(benchmark.OUT / "dk_vs_reference.csv", index=False)
        (benchmark.OUT / f"clv-{today}.md").write_text(benchmark.clv_report(_t))
    except Exception as e:  # noqa: BLE001
        flagged.append(f"WARN benchmark: {e}")
    print(json.dumps({"record": r, "sanity_status": s["status"], "flags": flagged,
                      "report": f"data/research/reports/{today}.md"}, indent=1))


if __name__ == "__main__":
    main()
