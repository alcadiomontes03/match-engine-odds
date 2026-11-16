"""
Alpha Lines v9 rating method (2026-11-15). Chosen from the walk-forward in pipeline/v8/gap.py (rounds 1 and 2,
out/gap_summary.md and out/gap_summary_round2.md): variant mkt_x10_prom5+stretch.

What changed from v8 (70% goals + 30% shot proxy, 347-day half-life):
  target   Each past match is represented by the goals the closing market expected from it, not by the goals
           scored: the (lam, mu) whose Dixon-Coles grid reproduces the Shin-de-vigged closing 1X2 and the fair
           over-2.5 price (football-data: Pinnacle close, else the market-average close). Far less noisy than
           one match's goals or shots. A match with no closing price in football-data uses DraftKings' last
           pre-kickoff moneyline from data/odds/snapshots.csv; with no price at all, its shot proxy, else goals.
  decay    xi = 0.010 a day (69-day half-life). Because the target is low-noise, recent matches can carry most
           of the weight. (On goals the same decay makes the ratings worse: gap.py "goals_x6".)
  promoted A newly promoted team is pulled toward the level of last season's relegated teams, fading as its own
           matches arrive (P = 5, k = 10; pipeline/v8/fixes.py).
  stretch  The strength difference between two sides is multiplied by S = 1.13 (fitted on the last 730 days of
           out-of-sample backtest rows as of 2026-11-16; frozen until the next registration).
  rho      Fixed at the market's -0.05 (with no goals term it is not identified by the fit).
Unchanged: centred ridge 1, the 1,100-day window, the recent goal-level factor, the guards.

Backtest (4,506 matches, Aug 2023 - Nov 2026): gap to the closing market +0.0221 -> +0.0127 (change -0.0093,
95% -0.0136 to -0.0052). Still behind the market, and the ratings still add nothing on top of the price: this
makes the displayed numbers more accurate, it is not a pick source. Blend weight stays 0.

The stretch and the level factor are folded into the stored attack/defence, so the dashboard's own formula
lam = exp(home_adv + att[home] - def[away]), mu = exp(att[away] - def[home]) reproduces the method exactly.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "pipeline" / "v8"))
import gap as GAP              # noqa: E402  (fit with the three targets; load with market-implied goals)
from fixes import C, R, M      # noqa: E402

MODEL = "alpha-lines-v9-market-goals-xi010-prom5-stretch113"
CHOSEN = dict(wg=0.0, ws=0.0, wm=1.0, xi=0.010, P=5.0, use_nnls=True)
S = 1.13
SNAP = ROOT / "data" / "odds" / "snapshots.csv"


def fold_stretch(r: dict, s: float = S) -> dict:
    """Multiply every pairwise strength difference by s, keeping log(lam) + log(mu) of every match unchanged.
    With strength = att + def and style = att - def:  att' = (s*(strength - mean) + style)/2,
    def' = (s*(strength - mean) - style)/2."""
    teams = list(r["att"])
    st = np.array([r["att"][t] + r["def"][t] for t in teams]); ty = np.array([r["att"][t] - r["def"][t] for t in teams])
    st = s * (st - st.mean())
    out = dict(r)
    out["att"] = dict(zip(teams, ((st + ty) / 2).tolist()))
    out["def"] = dict(zip(teams, ((st - ty) / 2).tolist()))
    out["stretch"] = s
    return out


def dk_last_h2h(snap_path: Path = SNAP) -> dict:
    """event_id -> (commence, Shin fair [home, draw, away]) from DraftKings' last complete pull before kickoff."""
    if not Path(snap_path).exists():
        return {}
    s = pd.read_csv(snap_path)
    s = s[(s.bookmaker == "draftkings") & (s.market == "h2h")].copy()
    s["pulled_at"] = pd.to_datetime(s.pulled_at, utc=True); s["commence"] = pd.to_datetime(s.commence, utc=True)
    s = s[s.pulled_at < s.commence]
    out = {}
    for eid, g in s.groupby("event_id"):
        for _, p in sorted(g.groupby("pulled_at"), key=lambda kv: kv[0], reverse=True):
            px = {o: float(v) for o, v in zip(p.outcome, p.price)}
            h, a = p.home.iloc[0], p.away.iloc[0]
            if all(px.get(k, 0) > 1 for k in (h, "Draw", a)):
                out[eid] = M.shin([px[h], px["Draw"], px[a]]); break
    return out


def new_extras(df: pd.DataFrame, ex: pd.DataFrame) -> pd.DataFrame:
    """Dashboard results that football-data does not have yet. candidate.extras_from_db compares kickoff times with
    the midnight of football-data's last day, so matches played on that day came back as "extra" although their
    rows were already in the CSV, and were counted twice in the fit (found 2026-11-15). Drop those."""
    if not len(ex):
        return ex
    have = set(zip(df.Date, df.HomeTeam, df.AwayTeam))
    keep = [k not in have for k in zip(ex.Date, ex.HomeTeam, ex.AwayTeam)]
    return ex[keep].reset_index(drop=True)


def add_extras(df: pd.DataFrame, ex: pd.DataFrame, db: Path, code: str) -> pd.DataFrame:
    """Append dashboard results newer than football-data, with market-implied goals from DraftKings' last
    pre-kickoff moneyline where the bot captured one (total-goals prior = the league's recent market mean)."""
    if not len(ex):
        return df
    ex = ex.copy(); ex["mlam"] = np.nan; ex["mmu"] = np.nan; ex["extra"] = True
    import json
    dk = dk_last_h2h()
    rec = df.dropna(subset=["mlam"]).tail(60)
    tot = float((rec.mlam + rec.mmu).mean()) if len(rec) else 2.7
    ev = {}
    for f in (Path(db) / "fixtures").glob("*.json"):
        fx = json.loads(f.read_text()); fx = fx.get("data", fx) if "id" not in fx else fx
        if fx.get("league") == code and fx.get("odds_event_id"):
            ev[(pd.Timestamp(fx["kickoff"]).tz_localize(None).normalize(), fx["home"], fx["away"])] = fx["odds_event_id"]
    for i, m in ex.iterrows():
        q = dk.get(ev.get((m.Date, m.HomeTeam, m.AwayTeam)))
        if q is not None:
            ex.loc[i, ["mlam", "mmu"]] = M.implied_lambdas(q[0], q[1], q[2], None, tot_prior=tot)
    return pd.concat([df, ex], ignore_index=True).sort_values("Date").reset_index(drop=True)


def fit_league(db: Path, code: str, asof: str):
    """(df, r, k, meta): r has the stretch folded in; k is the goal-level factor (computed before the stretch,
    as in the backtest)."""
    df = GAP.load(code)
    fd_last = df.Date.max()
    ex = new_extras(df, C.extras_from_db(Path(db), code, fd_last))
    df = add_extras(df, ex, Path(db), code)
    a = pd.Timestamp(asof) + pd.Timedelta(days=1)
    r = GAP.fit(df, a, code=code, **CHOSEN)
    k = R.level_factor(df, r)
    win = df[(df.Date < a) & (df.Date >= a - pd.Timedelta(days=R.WINDOW))]
    meta = {"n_matches": int(len(win)), "n_extra": int(len(ex)), "fd_last": str(fd_last.date()),
            "n_market_target": int(win.mlam.notna().sum()),
            "n_extra_with_dk_price": int(df.loc[df.get("extra", False) == True, "mlam"].notna().sum()) if len(ex) else 0,  # noqa: E712
            "promoted": (r.get("prom") or {}).get("promoted", [])}
    return df, fold_stretch(r), k, meta
