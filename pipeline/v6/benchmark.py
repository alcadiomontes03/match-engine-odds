"""
Alpha Lines v6 — market benchmark (free: football-data columns only, zero API credits).

Treats the sharp market price as the yardstick:

  python pipeline/v6/benchmark.py score   -> how good is each reference line? (Pinnacle / market average,
                                             open vs close, per league), Brier + log loss on 1X2
  python pipeline/v6/benchmark.py clv     -> DraftKings moneyline vs the de-vigged reference CLOSE for every
                                             settled match in data/research/lines.csv (edge, CLV, flat P&L)
  python pipeline/v6/benchmark.py weekly  -> both, written to data/research/benchmark/

Definitions (all on de-vigged Shin probabilities, 1X2):
  ref_p        reference fair probability (Pinnacle close where published, else market-average close)
  dk_implied   1 / DraftKings decimal price (includes DraftKings' margin)
  prob_edge    ref_p - dk_implied            "edge" in the video's sense
  ev           ref_p * dk_price - 1          expected return per unit at the DraftKings price, if ref_p is truth
  kelly_q      0.25 * ev / (dk_price - 1)    quarter-Kelly stake as a share of bankroll
A bet is "flagged" when ev >= MIN_EV (default 2%, same as the existing pick rule).

Limits, stated plainly:
  * football-data has no pre-match odds for fixtures that haven't been played, so this measures DraftKings'
    late price against the sharp CLOSE after the fact. It is a benchmark and a CLV tracker, not a live
    pre-bet signal. A live reference needs a second book from an odds API (costs credits, not used here).
  * From the second half of 2025-26 the Pinnacle close is missing; the market-average close is used and each
    row records which source it came from (ref_src).
  * 1X2 only. football-data has no BTTS, corners or DraftKings columns.
"""
from __future__ import annotations
import argparse, sys, unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
from v5 import market as M  # noqa: E402

FD = ROOT / "data" / "fd"
OUT = ROOT / "data" / "research" / "benchmark"
LEAGUES = ["E0", "SP1", "D1", "F1"]
MIN_EV = 0.02
KELLY_FRACTION = 0.25
SOURCES = {   # label -> (columns, tag)
    "Pinnacle open": ["PSH", "PSD", "PSA"],
    "Market-avg open": ["AvgH", "AvgD", "AvgA"],
    "Pinnacle close": ["PSCH", "PSCD", "PSCA"],
    "Market-avg close": ["AvgCH", "AvgCD", "AvgCA"],
}


# ------------------------------------------------------------------ pure maths
def ref_edge(dk_price: float, ref_p: float) -> dict:
    """Edge of one selection: reference fair probability against the DraftKings price."""
    if not (np.isfinite(dk_price) and np.isfinite(ref_p)) or dk_price <= 1:
        return {"dk_implied": np.nan, "prob_edge": np.nan, "ev": np.nan, "kelly_q": np.nan}
    ev = ref_p * dk_price - 1
    return {"dk_implied": 1 / dk_price, "prob_edge": ref_p - 1 / dk_price, "ev": ev,
            "kelly_q": max(0.0, KELLY_FRACTION * ev / (dk_price - 1))}


def brier_1x2(p, y) -> float:
    p = np.asarray(p, float)
    t = np.zeros(3); t[int(y)] = 1
    return float(((p - t) ** 2).sum())


def logloss(p, y) -> float:
    return float(-np.log(max(float(p[int(y)]), 1e-9)))


# ------------------------------------------------------------------ football-data loading
def load_fd() -> pd.DataFrame:
    frames = []
    for lg in LEAGUES:
        for f in sorted(FD.glob(f"{lg}_*.csv")):
            d = pd.read_csv(f, encoding="latin1", on_bad_lines="skip")
            d["lg"], d["season"] = lg, f.stem.split("_")[1]
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df.Date, format="mixed", dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).reset_index(drop=True)
    df["y"] = np.select([df.FTHG > df.FTAG, df.FTHG == df.FTAG], [0, 1], 2)
    for label, cols in SOURCES.items():
        key = label.replace(" ", "_").replace("-", "")
        P = np.full((len(df), 3), np.nan)
        if all(c in df.columns for c in cols):
            O = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            for i in range(len(df)):
                if np.all(O[i] > 1):
                    P[i] = M.shin(O[i])
        df[[f"{key}_H", f"{key}_D", f"{key}_A"]] = P
    return df


def _src_cols(df, label):
    key = label.replace(" ", "_").replace("-", "")
    return df[[f"{key}_H", f"{key}_D", f"{key}_A"]].to_numpy(float)


# ------------------------------------------------------------------ score: how good is each reference?
def score(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    labels = list(SOURCES)
    P = {l: _src_cols(df, l) for l in labels}
    have = np.all([np.isfinite(P[l]).all(axis=1) for l in labels], axis=0)
    s = df[have].reset_index(drop=True)
    rows = []
    for scope, sub in [("All four leagues", s)] + [(lg, s[s.lg == lg]) for lg in LEAGUES]:
        for l in labels:
            p = _src_cols(sub, l)
            br = np.array([brier_1x2(p[i], sub.y.iloc[i]) for i in range(len(sub))])
            ll = np.array([logloss(p[i], sub.y.iloc[i]) for i in range(len(sub))])
            rows.append({"scope": scope, "source": l, "n": len(sub), "brier": br.mean(), "logloss": ll.mean(),
                         "ll_se": ll.std(ddof=1) / np.sqrt(len(ll))})
    tab = pd.DataFrame(rows)
    # paired difference against the Pinnacle close, with a standard error
    pc = _src_cols(s, "Pinnacle close")
    base = np.array([logloss(pc[i], s.y.iloc[i]) for i in range(len(s))])
    diffs = []
    for l in labels:
        p = _src_cols(s, l)
        ll = np.array([logloss(p[i], s.y.iloc[i]) for i in range(len(s))])
        d = ll - base
        diffs.append((l, d.mean(), d.std(ddof=1) / np.sqrt(len(d)) if l != "Pinnacle close" else 0.0))
    # coverage by season
    cov = df.groupby(["season"]).apply(lambda g: pd.Series({
        "matches": len(g),
        "pinnacle_close": int(np.isfinite(_src_cols(g, "Pinnacle close")).all(axis=1).sum()),
        "avg_close": int(np.isfinite(_src_cols(g, "Market-avg close")).all(axis=1).sum())}), include_groups=False)

    md = ["# Market benchmark — how sharp is each reference line? (1X2)", "",
          f"Matches priced by **all four** sources at once: {len(s)} "
          f"(seasons {s.season.min()}–{s.season.max()}, E0 SP1 D1 F1). Lower is better.", "",
          "| Scope | Source | n | Brier | Log loss | ± SE |", "|---|---|---|---|---|---|"]
    for r in tab.itertuples():
        md.append(f"| {r.scope} | {r.source} | {r.n} | {r.brier:.4f} | {r.logloss:.4f} | {r.ll_se:.4f} |")
    md += ["", "## Paired log-loss difference vs the Pinnacle close (all four leagues)", "",
           "| Source | Mean difference | SE | Reading |", "|---|---|---|---|"]
    for l, m, se in diffs:
        read = "baseline" if l == "Pinnacle close" else ("worse (clear)" if m > 2 * se else "no clear difference")
        md.append(f"| {l} | {m:+.5f} | {se:.5f} | {read} |")
    md += ["", "## Closing-price coverage by season (all matches)", "", "| Season | Matches | Pinnacle close | Market-avg close |",
           "|---|---|---|---|"]
    for sea, r in cov.iterrows():
        md.append(f"| {sea} | {int(r.matches)} | {int(r.pinnacle_close)} | {int(r.avg_close)} |")
    md += ["", "Use: score any model against the Pinnacle close where it exists, and the market-average close "
           "after that. If the model's log loss is above these numbers it is worse than the market."]
    return "\n".join(md), tab


# ------------------------------------------------------------------ CLV: DraftKings vs reference close
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    for t in (" fc", "fc ", " cf", " afc", "1. ", " sc", " ud", " ca ", "rc ", "real ", "sv ", "borussia ", "olympique "):
        s = s.replace(t, " ")
    return " ".join(s.split())


def _sim(a, b):
    a, b = _norm(a), _norm(b)
    if a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def match_lines_to_fd(lines: pd.DataFrame, fd: pd.DataFrame) -> pd.DataFrame:
    """Join each settled DraftKings fixture to its football-data row: same league, kickoff within a day,
    SAME SCORE (a hard check), best home+away name similarity."""
    out, used = [], set()
    for r in lines.itertuples():
        ko = pd.to_datetime(r.kickoff, utc=True).tz_localize(None).normalize()
        c = fd[(fd.lg == r.league) & ((fd.Date - ko).abs() <= pd.Timedelta(days=1))
               & (fd.FTHG == r.home_goals) & (fd.FTAG == r.away_goals) & (~fd.index.isin(used))]
        if c.empty:
            continue
        sc = c.apply(lambda x: _sim(x.HomeTeam, r.home) + _sim(x.AwayTeam, r.away), axis=1)
        if sc.max() < 0.9:                      # names too different to trust even with the score agreeing
            continue
        i = sc.idxmax(); used.add(i)
        out.append((r.Index, i))
    return pd.DataFrame(out, columns=["line_idx", "fd_idx"])


def clv_table(lines_path: Path | None = None) -> pd.DataFrame:
    lines_path = lines_path or ROOT / "data" / "research" / "lines.csv"
    L = pd.read_csv(lines_path).drop_duplicates("fixture_id")
    L = L.dropna(subset=["ml_home", "ml_draw", "ml_away", "home_goals", "away_goals"]).reset_index(drop=True)
    fd = load_fd()
    link = match_lines_to_fd(L, fd)
    pin, avg = _src_cols(fd, "Pinnacle close"), _src_cols(fd, "Market-avg close")
    rows = []
    for lr in link.itertuples():
        l, f = L.loc[lr.line_idx], fd.loc[lr.fd_idx]
        use_pin = np.isfinite(pin[lr.fd_idx]).all()
        ref = pin[lr.fd_idx] if use_pin else avg[lr.fd_idx]
        if not np.isfinite(ref).all():
            continue
        src = "pinnacle_close" if use_pin else "market_avg_close"
        dk = np.array([l.ml_home, l.ml_draw, l.ml_away], float)
        dk_fair = M.shin(dk)
        for k, name in enumerate(["home", "draw", "away"]):
            e = ref_edge(dk[k], ref[k])
            won = int(f.y == k)
            rows.append({"league": l.league, "kickoff": l.kickoff, "home": l.home, "away": l.away, "fd_home": f.HomeTeam,
                         "fd_away": f.AwayTeam, "selection": name, "dk_price": dk[k], "dk_shin_p": dk_fair[k],
                         "ref_p": ref[k], "ref_src": src, **e, "flagged": e["ev"] >= MIN_EV, "won": won,
                         "flat_pnl": (dk[k] - 1) if won else -1.0})
    return pd.DataFrame(rows)


def clv_report(t: pd.DataFrame) -> str:
    if t.empty:
        return "# DraftKings vs reference close\n\nNo settled fixtures matched yet."
    n_fix = t[["league", "kickoff", "home"]].drop_duplicates().shape[0]
    md = ["# DraftKings moneyline vs the sharp reference close", "",
          f"{n_fix} settled fixtures matched to football-data ({len(t)} selections). "
          f"`ev` = ref_p × DraftKings price − 1; a positive mean means DraftKings' late price beat the sharp close.", ""]

    def block(title, g):
        out = [f"## {title}", "", "| Group | Selections | Mean ev | Mean prob edge | Flagged (ev ≥ 2%) | Flagged flat P&L (units) |",
               "|---|---|---|---|---|---|"]
        for k, s in g:
            fl = s[s.flagged]
            out.append(f"| {k} | {len(s)} | {s.ev.mean():+.2%} | {s.prob_edge.mean():+.2%} | {len(fl)} | "
                       f"{fl.flat_pnl.sum():+.2f} |" if len(fl) else
                       f"| {k} | {len(s)} | {s.ev.mean():+.2%} | {s.prob_edge.mean():+.2%} | 0 | — |")
        return out + [""]

    md += block("All selections", [("All", t)])
    md += block("By league", t.groupby("league"))
    md += block("By outcome", t.groupby("selection"))
    md += block("By reference source", t.groupby("ref_src"))
    fl = t[t.flagged].sort_values("ev", ascending=False)
    md += ["## Flagged selections (DraftKings price beat the sharp close by ≥ 2%)", ""]
    if fl.empty:
        md.append("None this period.")
    else:
        md += ["| League | Match | Pick | DK price | Ref p | ev | ¼-Kelly | Result |", "|---|---|---|---|---|---|---|---|"]
        for r in fl.itertuples():
            md.append(f"| {r.league} | {r.home} v {r.away} | {r.selection} | {r.dk_price:.2f} | {r.ref_p:.3f} | "
                      f"{r.ev:+.1%} | {r.kelly_q:.2%} | {'W' if r.won else 'L'} |")
    md += ["", "Read it as a tracker, not a verdict: tens of fixtures say nothing about edge. The mean ev is the number "
           "to watch once several hundred selections have accumulated, and it will normally sit below zero because "
           "DraftKings charges a margin the sharp close does not."]
    return "\n".join(md)


# ------------------------------------------------------------------ cli
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "clv", "weekly"])
    a = ap.parse_args()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd in ("score", "weekly"):
        txt, tab = score(load_fd())
        (OUT / f"score-{today}.md").write_text(txt)
        tab.to_csv(OUT / "score.csv", index=False)
        print(txt)
    if a.cmd in ("clv", "weekly"):
        t = clv_table()
        t.to_csv(OUT / "dk_vs_reference.csv", index=False)
        txt = clv_report(t)
        (OUT / f"clv-{today}.md").write_text(txt)
        print(txt)


if __name__ == "__main__":
    main()
