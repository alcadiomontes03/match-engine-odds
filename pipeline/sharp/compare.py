"""DraftKings vs Pinnacle moneyline: the pre-registered test in docs/prereg-sharp-2026-11-01.md.

Rebuilt from scratch on every run from append-only inputs, so it never edits history:
  data/odds/snapshots.csv   DraftKings quotes (the odds bot)
  data/odds/sharp.csv       Pinnacle quotes from the same API calls (since 2026-11-02)
  --db                      dashboard export (ArtifactData list out_dir) with fixtures/, results/, predictions/

Writes data/research/sharp/report_<date>.md and data/research/sharp/matches.csv (one row per eligible match).
Every rule here is fixed by the registration; changing one means a new registration, not an edit.

  python pipeline/sharp/compare.py --db /tmp/me/rdb [--start 2026-11-02T15:00Z]
"""
from __future__ import annotations

import argparse, glob, json, math, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
from v5 import market as M  # noqa: E402

REG = "docs/prereg-sharp-2026-11-01.md"
START = "2026-11-02T15:00:00Z"        # sample start (kickoff after this)
COMPS = {"PL": "E0", "LaLiga": "SP1", "Bundesliga": "D1", "Ligue1": "F1"}
P1 = 0.02                             # the one primary threshold
SECONDARY = (0.00, 0.05)
STALE_MIN = 60                        # Pinnacle quote older than this at pull time = stale
W_V7 = 0.10                           # fixed pool weight on v7 (forecast test F1)
LOOKS = (100, 200, 400)
OUT = ROOT / "data" / "research" / "sharp"
B = 4000


def _ts(x):
    try:
        t = pd.Timestamp(x)
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    except (TypeError, ValueError):
        return None


def week_key(t: pd.Timestamp) -> str:
    """The odds bot's week: Monday 15:00 UTC to Monday 15:00 UTC."""
    s = t - timedelta(hours=15)
    y, w, _ = s.isocalendar()
    return f"{y}-W{w:02d}"


def quotes(path: Path, book: str) -> pd.DataFrame:
    """One row per (event, pull): home/draw/away prices and the book's last update. Incomplete quotes dropped."""
    if not path.exists():
        return pd.DataFrame(columns=["event_id", "pulled_at", "kind", "comp", "commence", "home", "away",
                                     "H", "D", "A", "book_last_update"])
    s = pd.read_csv(path, dtype={"point": str})
    s = s[(s.bookmaker == book) & (s.market == "h2h") & s.comp.isin(list(COMPS))].copy()
    if s.empty:
        return pd.DataFrame(columns=["event_id", "pulled_at", "kind", "comp", "commence", "home", "away",
                                     "H", "D", "A", "book_last_update"])
    s["side"] = np.where(s.outcome == s.home, "H", np.where(s.outcome == s.away, "A",
                         np.where(s.outcome == "Draw", "D", "")))
    s = s[s.side != ""]
    key = ["event_id", "pulled_at", "kind", "comp", "commence", "home", "away"]
    q = s.pivot_table(index=key, columns="side", values="price", aggfunc="last").reset_index()
    lu = s.groupby(key)["book_last_update"].max().reset_index()
    q = q.merge(lu, on=key, how="left")
    for c in "HDA":
        if c not in q:
            q[c] = np.nan
    q = q.dropna(subset=["H", "D", "A"])
    q = q[(q[["H", "D", "A"]] > 1).all(axis=1)]
    q["pulled_at"] = q.pulled_at.map(_ts)
    q["commence"] = q.commence.map(_ts)
    q["book_last_update"] = q.book_last_update.map(_ts)
    return q


def _shin(r):
    return M.shin([r["H"], r["D"], r["A"]])


def _prop(r):
    p = 1 / np.array([r["H"], r["D"], r["A"]], float)
    return p / p.sum()


def _load(db: Path, coll: str) -> dict:
    out = {}
    for f in glob.glob(str(db / coll / "*.json")):
        d = json.loads(Path(f).read_text())
        if isinstance(d, dict) and "data" in d and isinstance(d["data"], dict) and "id" not in d:
            d = d["data"]
        out[Path(f).stem] = d
    return out


def build(db: Path | None, start: str = START, snap: Path | None = None, sharp: Path | None = None):
    dk = quotes(snap or ROOT / "data" / "odds" / "snapshots.csv", "draftkings")
    pn = quotes(sharp or ROOT / "data" / "odds" / "sharp.csv", "pinnacle")
    t0 = _ts(start)
    counts = {"events_dk": 0, "events_both": 0, "no_bet_time": 0, "stale_bet_time": 0, "no_close": 0,
              "stale_close": 0, "no_result": 0, "postponed": 0, "eligible": 0}
    fx = _load(db, "fixtures") if db else {}
    res = _load(db, "results") if db else {}
    preds = _load(db, "predictions") if db else {}
    by_event = {f.get("odds_event_id"): fid for fid, f in fx.items() if f.get("odds_event_id")}
    rows = []
    dk = dk[dk.commence > t0]
    pn = pn[pn.commence > t0]
    counts["events_dk"] = int(dk.event_id.nunique())
    for ev, d in dk.groupby("event_id"):
        p = pn[pn.event_id == ev]
        if p.empty:
            continue
        counts["events_both"] += 1
        ko = d.commence.iloc[0]
        dl = d[(d.kind == "late") & (d.pulled_at < ko)]
        both = sorted(set(dl.pulled_at) & set(p[p.pulled_at < ko].pulled_at))
        if not both:
            counts["no_bet_time"] += 1
            continue
        tb = both[0]                                     # first late pull with both quotes
        qd = dl[dl.pulled_at == tb].iloc[0]
        qp = p[p.pulled_at == tb].iloc[0]
        lag = (tb - qp.book_last_update).total_seconds() / 60 if qp.book_last_update is not None else None
        if lag is None or lag > STALE_MIN:
            counts["stale_bet_time"] += 1
            continue
        row = {"event_id": ev, "league": COMPS[qd.comp], "home": qd.home, "away": qd.away,
               "kickoff": ko.isoformat(), "week": week_key(ko), "t_bet": tb.isoformat(),
               "mins_to_ko_bet": round((ko - tb).total_seconds() / 60), "pin_lag_bet": round(lag, 1),
               "dk_H": qd.H, "dk_D": qd.D, "dk_A": qd.A, "pin_H": qp.H, "pin_D": qp.D, "pin_A": qp.A}
        ps, pp, ds = _shin(qp), _prop(qp), _shin(qd)
        for i, c in enumerate("HDA"):
            row[f"pin_p{c}"] = ps[i]; row[f"pin_prop_p{c}"] = pp[i]; row[f"dk_p{c}"] = ds[i]
            row[f"ev_{c}"] = qd[c] * ps[i] - 1
        best = max("HDA", key=lambda c: row[f"ev_{c}"])
        row["best"] = best; row["best_ev"] = row[f"ev_{best}"]; row["best_price"] = qd[best]
        row["best_ev_prop"] = qd[best] * row[f"pin_prop_p{best}"] - 1
        # close: last Pinnacle quote after T_bet and before kickoff
        pc = p[(p.pulled_at > tb) & (p.pulled_at < ko)].sort_values("pulled_at")
        row["clv"] = np.nan
        if pc.empty:
            counts["no_close"] += 1
        else:
            qc = pc.iloc[-1]
            lagc = (qc.pulled_at - qc.book_last_update).total_seconds() / 60 if qc.book_last_update is not None else None
            if lagc is None or lagc > STALE_MIN:
                counts["stale_close"] += 1
            else:
                pcs = _shin(qc)
                row["clv"] = qd[best] * pcs["HDA".index(best)] - 1
                row["t_close"] = qc.pulled_at.isoformat()
                row["mins_to_ko_close"] = round((ko - qc.pulled_at).total_seconds() / 60)
        # early (Monday weekly pull) EV, exploratory
        dw, pw = d[d.kind == "weekly"], p[p.kind == "weekly"]
        ew = sorted(set(dw.pulled_at) & set(pw.pulled_at))
        if ew:
            a, b = dw[dw.pulled_at == ew[0]].iloc[0], pw[pw.pulled_at == ew[0]].iloc[0]
            pse = _shin(b)
            row["early_best_ev"] = max(a[c] * pse[i] - 1 for i, c in enumerate("HDA"))
        # result and v7
        fid = by_event.get(ev)
        r = res.get(fid) if fid else None
        if fid and (fx.get(fid) or {}).get("status") == "postponed":
            counts["postponed"] += 1
            continue
        if not r or "home_score" not in r:
            counts["no_result"] += 1
            row["y"] = None
        else:
            hs, as_ = int(r["home_score"]), int(r["away_score"])
            row["y"] = "H" if hs > as_ else "A" if as_ > hs else "D"
        x = (preds.get(fid) or {}).get("x1x2") if fid else None
        if x:
            row["v7_pH"], row["v7_pD"], row["v7_pA"] = float(x["home"]), float(x["draw"]), float(x["away"])
        counts["eligible"] += 1
        rows.append(row)
    return pd.DataFrame(rows), counts


def wboot(values: np.ndarray, weeks: np.ndarray, level: float, seed=7):
    """Mean with a bootstrap interval that resamples whole weeks."""
    v = np.asarray(values, float); w = np.asarray(weeks)
    ok = np.isfinite(v); v, w = v[ok], w[ok]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    uw = np.unique(w)
    groups = [v[w == u] for u in uw]
    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        s = np.concatenate([groups[i] for i in pick])
        sims.append(s.mean())
    a = (1 - level) / 2
    lo, hi = np.percentile(sims, [100 * a, 100 * (1 - a)])
    return float(v.mean()), float(lo), float(hi), len(v)


def _f(x, f="+.3f"):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, f)


def report(df: pd.DataFrame, counts: dict, today: str, start: str = START) -> str:
    out = [f"# DraftKings vs Pinnacle report {today}", "",
           f"Registration: `{REG}` (commit 51aa20a, before any Pinnacle data). Sample: kickoffs after {start}.",
           "Paper only. Decisions use only the primary endpoint at the registered looks.", "",
           "## Data", "",
           f"DraftKings events in sample: {counts['events_dk']}; with Pinnacle quotes too: {counts['events_both']}.",
           f"Excluded: no late pull with both books {counts['no_bet_time']}; stale Pinnacle at bet time "
           f"{counts['stale_bet_time']}; postponed {counts['postponed']}. Eligible matches: {counts['eligible']} "
           f"(of which {counts['no_result']} have no result yet).",
           f"No CLV: no Pinnacle pull after bet time {counts['no_close']}; stale Pinnacle close {counts['stale_close']}.", ""]
    if df.empty:
        out += ["No eligible matches yet.", ""]
        return "\n".join(out) + "\n"
    lag = df.pin_lag_bet.astype(float)
    out += [f"Bet time: median {df.mins_to_ko_bet.median():.0f} min before kickoff. Pinnacle quote age at bet time: "
            f"median {lag.median():.0f} min, 90th percentile {lag.quantile(.9):.0f} min.", ""]
    if "mins_to_ko_close" in df and df.mins_to_ko_close.notna().any():
        out += [f"Close: median {df.mins_to_ko_close.median():.0f} min before kickoff "
                f"({int(df.mins_to_ko_close.notna().sum())} matches).", ""]
    # P1
    out += ["## Rule P1 (primary): DraftKings price beats Pinnacle Shin fair by at least 2%", "",
            "| Threshold | Candidates | With CLV | Mean EV at bet | Mean CLV | 95% | 99% | Settled | Hit rate | ROI | 95% |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    status = None
    for thr in (P1,) + SECONDARY:
        c = df[df.best_ev >= thr].copy()
        m95 = wboot(c.clv.values, c.week.values, .95)
        m99 = wboot(c.clv.values, c.week.values, .99)
        s = c[c.y.notna()]
        pnl = np.where(s.y == s.best, s.best_price - 1, -1.0) if len(s) else np.array([])
        roi = wboot(pnl, s.week.values, .95) if len(s) else (float("nan"),) * 3 + (0,)
        hit = float((s.y == s.best).mean()) if len(s) else float("nan")
        tag = "**2% (primary)**" if thr == P1 else f"{thr:.0%} (secondary)"
        out.append(f"| {tag} | {len(c)} | {m95[3]} | {_f(c.best_ev.mean() if len(c) else float('nan'))} | "
                   f"{_f(m95[0])} | {_f(m95[1])} to {_f(m95[2])} | {_f(m99[1])} to {_f(m99[2])} | {len(s)} | "
                   f"{_f(hit, '.0%')} | {_f(roi[0])} | {_f(roi[1])} to {_f(roi[2])} |")
        if thr == P1:
            n = m95[3]
            reached = [L for L in LOOKS if n >= L]
            if not reached:
                status = f"No decision look yet: {n} P1 candidates with CLV; the first look is at {LOOKS[0]}."
            else:
                L = reached[-1]
                if m99[1] > 0:
                    status = f"Look at {L}: PROMISING (99% interval for mean CLV entirely above 0)."
                elif n >= 200 and m95[2] < 0:
                    status = f"Look at {L}: STOP (95% interval entirely below 0). P1 is retired; no re-tuning."
                else:
                    status = f"Look at {L}: CONTINUE (interval includes 0)."
    out += ["", f"**Decision status:** {status}", "",
            "Proportional de-vig (sensitivity only): P1 candidates would be "
            f"{int((df.best_ev_prop >= P1).sum())} instead of {int((df.best_ev >= P1).sum())}.", ""]
    # F1
    f = df[df.y.notna() & df.v7_pH.notna()].copy() if "v7_pH" in df else pd.DataFrame()
    out += ["## Forecast test F1: does v7 add anything at bet time?", ""]
    if f.empty:
        out += ["No settled matches with a v7 prediction yet.", ""]
    else:
        yi = f.y.map({"H": 0, "D": 1, "A": 2}).values; k = np.arange(len(f))
        P = {"Pinnacle": f[["pin_pH", "pin_pD", "pin_pA"]].values,
             "DraftKings": f[["dk_pH", "dk_pD", "dk_pA"]].values,
             "v7": f[["v7_pH", "v7_pD", "v7_pA"]].values}
        x = P["Pinnacle"] ** (1 - W_V7) * P["v7"] ** W_V7
        P[f"pool (v7 weight {W_V7})"] = x / x.sum(1, keepdims=True)
        ll = {n: -np.log(np.clip(p[k, yi], 1e-12, 1)) for n, p in P.items()}
        out += [f"{len(f)} settled matches.", "", "| Forecast | Log loss | Minus Pinnacle | 95% (week blocks) |",
                "|---|---|---|---|"]
        for n, v in ll.items():
            if n == "Pinnacle":
                out.append(f"| Pinnacle (Shin, bet time) | {v.mean():.4f} | | |")
            else:
                m = wboot(v - ll["Pinnacle"], f.week.values, .95)
                tag = " **(primary F1)**" if n.startswith("pool") else ""
                out.append(f"| {n}{tag} | {v.mean():.4f} | {_f(m[0], '+.4f')} | {_f(m[1], '+.4f')} to {_f(m[2], '+.4f')} |")
        out.append("")
    # exploratory
    out += ["## Exploratory (no decisions)", ""]
    if "early_best_ev" in df and df.early_best_ev.notna().any():
        e = df.early_best_ev.dropna()
        out.append(f"- Monday weekly pull: {int((e >= P1).sum())} of {len(e)} matches had a DraftKings price 2%+ over "
                   f"Pinnacle fair (bet time: {int((df.best_ev >= P1).sum())} of {len(df)}).")
    parts = []
    for lg, s_ in df.groupby("league"):
        parts.append(f"{lg} {int((s_.best_ev >= P1).sum())}/{len(s_)} candidates (mean best EV {s_.best_ev.mean():+.3f})")
    out.append("- By league: " + "; ".join(parts))
    ovr_dk = (1 / df[["dk_H", "dk_D", "dk_A"]]).sum(axis=1).mean(); ovr_pn = (1 / df[["pin_H", "pin_D", "pin_A"]]).sum(axis=1).mean()
    out.append(f"- Mean overround: DraftKings {ovr_dk:.3f}, Pinnacle {ovr_pn:.3f}.")
    out.append("")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", help="dashboard export dir with fixtures/, results/, predictions/")
    ap.add_argument("--start", default=START, help="sample start; only change for testing")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--snap", help="DraftKings snapshots (testing)")
    ap.add_argument("--sharp", help="Pinnacle snapshots (testing)")
    a = ap.parse_args()
    df, counts = build(Path(a.db) if a.db else None, a.start,
                       Path(a.snap) if a.snap else None, Path(a.sharp) if a.sharp else None)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = report(df, counts, today, a.start)
    if a.start != START or a.sharp or a.snap:
        rep = rep.replace("# DraftKings vs Pinnacle report", "# TEST RUN (non-registered start) DraftKings vs Pinnacle report")
    (out / f"report_{today}.md").write_text(rep)
    df.to_csv(out / "matches.csv", index=False)
    print(json.dumps({"eligible": counts["eligible"], "p1_candidates": int((df.best_ev >= P1).sum()) if len(df) else 0,
                      "report": str(out / f"report_{today}.md")}))


if __name__ == "__main__":
    main()
