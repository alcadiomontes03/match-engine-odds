"""
Weekly report for docs/prereg-forward-2026-11.md (Test A and Test B). Descriptive only: a weekly look never
triggers a decision. Decision looks are defined in the registration and are flagged here when reached.

  python pipeline/picks/report.py [--date YYYY-MM-DD] [--out data/picks/reports]

Everything comes from this repo, so anyone can re-run it:
  Test A  model = v7/v8 prediction files pipeline/v7/out/shadow_<date>.csv (not shadow_v61_*), timed by the git
          commit that added the file; DraftKings = the latest complete h2h pull in data/odds/snapshots.csv at or
          before that commit, Shin de-vigged; results = data/fd (football-data.co.uk).
  Test B  picks = data/picks/s1.csv; closing price = data/odds/clv.csv; results = data/fd.
Matches before the registration sample start are reported separately, as exploration, and never mixed in.
"""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "pipeline" / "picks"))
from v5 import market as M  # noqa: E402
import s1 as S1  # noqa: E402

SAMPLE_START = pd.Timestamp(S1.SAMPLE_START)
SHADOW = ROOT / "pipeline" / "v7" / "out"
SNAP = ROOT / "data" / "odds" / "snapshots.csv"
CLV = ROOT / "data" / "odds" / "clv.csv"
PICKS = ROOT / "data" / "picks" / "s1.csv"
FD = ROOT / "data" / "fd"
LOOKS = (100, 200, 400)
OUT_ = ["H", "D", "A"]


# ---------------------------------------------------------------- shared
def week_key(ts: pd.Timestamp) -> str:
    """Odds-bot week: Monday 15:00 UTC to Monday."""
    t = ts - pd.Timedelta(hours=15)
    y, w, _ = t.isocalendar()
    return f"{y}-W{w:02d}"


def block_boot(values, weeks, n=4000, seed=11, level=0.95):
    v, wk = np.asarray(values, float), np.asarray(weeks)
    if len(v) == 0:
        return None, None
    groups = [v[wk == u] for u in np.unique(wk)]
    rng = np.random.default_rng(seed)
    if len(groups) < 2:
        return None, None
    stats = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(n)]
    a = (1 - level) / 2
    return float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))


def fd_results() -> pd.DataFrame:
    frames = []
    for p in sorted(FD.glob("*_2627.csv")):
        d = pd.read_csv(p)
        d["league"] = p.name.split("_")[0]
        frames.append(d[["league", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]])
    d = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(d):
        d["date"] = pd.to_datetime(d.Date, dayfirst=True).dt.date
    return d.dropna(subset=["FTHG", "FTAG"]) if len(d) else d


def find_result(res, league, home, away, kickoff):
    if not len(res):
        return None
    k = pd.Timestamp(kickoff).date()
    m = res[(res.league == league) & (res.HomeTeam == home) & (res.AwayTeam == away)]
    m = m[m.date.map(lambda d: abs((d - k).days) <= 1)]
    return (int(m.iloc[0].FTHG), int(m.iloc[0].FTAG)) if len(m) == 1 else None


def git_added(path: Path):
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "log", "--diff-filter=A", "--format=%cI", "--", str(path)],
                             capture_output=True, text=True, timeout=30).stdout.split()
        return pd.Timestamp(out[-1]).tz_convert("UTC") if out else None
    except Exception:
        return None


def ci_txt(lo, hi, d=4):
    return "–" if lo is None else f"{lo:+.{d}f} to {hi:+.{d}f}"


# ---------------------------------------------------------------- Test A
def test_a(snap: pd.DataFrame, res: pd.DataFrame):
    dk = snap[(snap.bookmaker == "draftkings") & (snap.market == "h2h")].copy()
    dk["t"] = pd.to_datetime(dk.pulled_at, utc=True)
    rows, excluded, name_map = [], {"no git time": 0, "no matched price": 0, "no result": 0, "unknown team": 0}, {}
    for f in sorted(SHADOW.glob("shadow_2*.csv")):
        t_commit = git_added(f)
        sh = pd.read_csv(f)
        for _, r in sh.iterrows():
            api = dk[dk.event_id == r.fixture_id]
            if len(api):
                name_map[api.iloc[0].home] = r.home
                name_map[api.iloc[0].away] = r.away
            if t_commit is None:
                excluded["no git time"] += 1; continue
            if isinstance(r.unknown_teams, str) and r.unknown_teams.strip():
                excluded["unknown team"] += 1; continue
            before = api[api.t <= t_commit]
            quote = None
            for t, g in before.sort_values("t", ascending=False).groupby("t", sort=False):
                pr = {}
                for _, x in g.iterrows():
                    pr["home" if x.outcome == x.home else "away" if x.outcome == x.away else "draw"] = x.price
                if all(pr.get(k, 0) > 1 for k in ("home", "draw", "away")):
                    quote = (t, pr); break
            if quote is None:
                excluded["no matched price"] += 1; continue
            sc = find_result(res, r.league, r.home, r.away, r.kickoff)
            if sc is None:
                excluded["no result"] += 1; continue
            y = 0 if sc[0] > sc[1] else 1 if sc[0] == sc[1] else 2
            pm = np.array([r.p_home, r.p_draw, r.p_away], float); pm = pm / pm.sum()
            pd_ = M.shin([quote[1]["home"], quote[1]["draw"], quote[1]["away"]])
            pool = pd_ ** 0.9 * pm ** 0.1; pool = pool / pool.sum()
            rows.append(dict(fixture_id=r.fixture_id, league=r.league, kickoff=r.kickoff, week=week_key(pd.Timestamp(r.kickoff)),
                             in_sample=pd.Timestamp(r.kickoff) > SAMPLE_START and t_commit > SAMPLE_START,
                             ll_model=-math.log(max(pm[y], 1e-12)), ll_dk=-math.log(pd_[y]), ll_pool=-math.log(pool[y]),
                             br_model=float(((pm - np.eye(3)[y]) ** 2).sum()), br_dk=float(((pd_ - np.eye(3)[y]) ** 2).sum()),
                             mins_gap=(pd.Timestamp(t_commit) - quote[0]).total_seconds() / 60))
    return pd.DataFrame(rows), excluded, name_map


def summarise_a(df: pd.DataFrame) -> list[str]:
    if not len(df):
        return ["No graded matches yet."]
    diff = df.ll_model - df.ll_dk; pdiff = df.ll_pool - df.ll_dk
    lo, hi = block_boot(diff, df.week); plo, phi = block_boot(pdiff, df.week)
    lo99, hi99 = block_boot(diff, df.week, level=0.99)
    return [f"| Matches | Weeks | Model LL | DK LL (matched) | Model − DK | 95% (week blocks) | 99% | Pool − DK | 95% |",
            "|---|---|---|---|---|---|---|---|---|",
            f"| {len(df)} | {df.week.nunique()} | {df.ll_model.mean():.4f} | {df.ll_dk.mean():.4f} | {diff.mean():+.4f} | "
            f"{ci_txt(lo, hi)} | {ci_txt(lo99, hi99)} | {pdiff.mean():+.4f} | {ci_txt(plo, phi)} |",
            "", f"Brier: model {df.br_model.mean():.4f}, DK {df.br_dk.mean():.4f}. "
                f"Median gap between the DK pull used and the model's commit: {df.mins_gap.median():.0f} min."]


# ---------------------------------------------------------------- Test B
def grade(kind, line, side, hg, ag):
    if kind == "tot":
        over = hg + ag > float(line)
        return over if side == "over" else not over
    if kind == "sp":
        home = hg - ag + float(line) > 0
        return home if side == "home" else not home
    yes = hg > 0 and ag > 0
    return yes if side == "yes" else not yes


def close_prob(clv: pd.DataFrame, pick) -> float | None:
    g = clv[(clv.event_id == pick.event_id) & (clv.market == pick.dk_market)]
    if not len(g):
        return None
    g = g[g.pulled_at == g.pulled_at.max()]
    lines = S1.side_lines(g, pick.home, pick.away)
    key = (pick.market, None if pick.market == "btts" else float(pick.line))
    d = lines.get(key)
    if not d:
        return None
    sides = list(d)
    p = M.shin([d[sides[0]][0], d[sides[1]][0]])
    return float(p[sides.index(pick.side)])


def test_b(name_map, res):
    if not PICKS.exists():
        return pd.DataFrame(), pd.DataFrame()
    allrows = pd.read_csv(PICKS, dtype={"line": str})
    picks = allrows[allrows.decision == "pick"].copy()
    clv = pd.read_csv(CLV, dtype={"point": float}) if CLV.exists() else pd.DataFrame(columns=S1.COLS)
    out = []
    for _, p in picks.iterrows():
        pc = close_prob(clv, p) if len(clv) else None
        h, a = name_map.get(p.home), name_map.get(p.away)
        sc = find_result(res, p.league, h, a, p.kickoff) if h and a else None
        won = None if sc is None else grade(p.market, p.line, p.side, *sc)
        out.append(dict(event_id=p.event_id, league=p.league, market=p.market, week=week_key(pd.Timestamp(p.side_pulled_at)),
                        price=float(p.dk_price), ev=float(p.ev), clv=None if pc is None else float(p.dk_price) * pc - 1,
                        pnl=None if won is None else (float(p.dk_price) - 1 if won else -1.0)))
    return allrows, pd.DataFrame(out)


def summarise_b(allrows, b) -> list[str]:
    if not len(allrows):
        return ["No S1 decisions yet."]
    lines = [f"Matches decided: {len(allrows)}; picks: {int((allrows.decision == 'pick').sum())}.", "",
             "| Why no priced pick | Matches |", "|---|---|"]
    reasons = allrows[allrows.decision != "pick"].reason.str.replace(r" \(best.*", "", regex=True).value_counts()
    lines += [f"| {k} | {v} |" for k, v in reasons.items()] or ["| – | 0 |"]
    if len(allrows) and "ev" in allrows:
        e = pd.to_numeric(allrows.ev, errors="coerce").dropna()
        lines += ["", f"Best-line EV over all priced matches: mean {e.mean():+.4f}; "
                      f"matches at ≥ 0.00: {(e >= 0).sum()}, ≥ 0.03: {(e >= 0.03).sum()}, ≥ 0.05: {(e >= 0.05).sum()} "
                      "(secondary thresholds, no decisions)."]
    if not len(b):
        return lines
    c = b.dropna(subset=["clv"])
    lo, hi = block_boot(c.clv, c.week) if len(c) else (None, None)
    lo99, hi99 = block_boot(c.clv, c.week, level=0.99) if len(c) else (None, None)
    g = b.dropna(subset=["pnl"])
    rlo, rhi = block_boot(g.pnl, g.week) if len(g) else (None, None)
    lines += ["", "| Picks | With CLV | Mean CLV | 95% | 99% | Graded | Hit rate | ROI / bet | 95% |",
              "|---|---|---|---|---|---|---|---|---|",
              f"| {len(b)} | {len(c)} | {c.clv.mean() if len(c) else float('nan'):+.4f} | {ci_txt(lo, hi)} | {ci_txt(lo99, hi99)} | "
              f"{len(g)} | {(g.pnl > 0).mean() * 100 if len(g) else float('nan'):.0f}% | "
              f"{g.pnl.mean() * 100 if len(g) else float('nan'):+.1f}% | {ci_txt(rlo, rhi, 3)} |"]
    reached = [n for n in LOOKS if len(c) >= n]
    if reached:
        lines += ["", f"**Decision look reached: {max(reached)} candidates with CLV.** Read it with the rules in "
                      "docs/prereg-forward-2026-11.md section 'Looks and decisions'."]
    return lines


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--out", default=str(ROOT / "data" / "picks" / "reports"))
    a = ap.parse_args()
    snap = pd.read_csv(SNAP, dtype={"point": float})
    res = fd_results()
    A, excl, name_map = test_a(snap, res)
    allrows, B = test_b(name_map, res)
    ins = A[A.in_sample] if len(A) else A
    pre = A[~A.in_sample] if len(A) else A
    doc = [f"# Forward test report {a.date} (docs/prereg-forward-2026-11.md)", "",
           "Descriptive only. Weekly looks never trigger a decision.", "",
           "## Test A: model vs DraftKings at the same moment (registered sample)", ""] + summarise_a(ins) + \
          ["", f"Excluded and counted: " + ", ".join(f"{k} {v}" for k, v in excl.items()), "",
           "## Test B: rule S1 on DraftKings side markets", ""] + summarise_b(allrows, B) + \
          ["", "## Exploration only: Test A before the sample start (never used for a decision)", ""] + summarise_a(pre)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.date}.md").write_text("\n".join(doc) + "\n")
    print("\n".join(doc))


if __name__ == "__main__":
    main()
