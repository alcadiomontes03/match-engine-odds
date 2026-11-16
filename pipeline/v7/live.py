"""
Alpha Lines live ratings. v7 went live 2026-10-20 and was renamed v8 on 2026-11-01 (same method). On 2026-11-15
the live method became **v9** (pipeline/v9/method.py: market-implied goals as the fit target, 69-day half-life,
promoted-team prior, stretch 1.13), chosen from the walk-forward in pipeline/v8/gap.py and registered in
docs/prereg-forward-2026-11b.md. This file keeps its name and commands so the scheduled jobs need no change:
`refit` writes v9 ratings, `shadow` writes v9 predictions (shadow_<date>.csv) and v8 shadow predictions
(shadow_v8_<date>.csv), `score` compares them. Everything below this paragraph describes v7/v8, kept for history.

Alpha Lines v7 - live ratings (went live 2026-10-20, Alcadio's decision).
Renamed "Alpha Lines v8" from week 9 (Alcadio, 2026-11-01): same rating method, written as model_version "Alpha Lines v8".

v7 replaced the v6.1 Dixon-Coles ratings in the dashboard state (state/summary, state/detail_<code>). Prices are
unchanged: blend weight is 0 in every league, so every bet price is still DraftKings' own; the ratings drive the
dashboard's per-match detail (expected goals, 1X2, totals, spread, BTTS) and the exploratory Dixon-Coles comparison.

Model (candidate.CHOSEN): Dixon-Coles fitted on 70% goals + 30% a shot-based expected-goals proxy, centred ridge 1,
time decay xi 0.002 over 1,100 days, and the recent goal-level factor. The level factor is folded into attack
(attack = fitted attack + ln(level_factor)), so the Monday fixtures job's own formula
lam = exp(home_adv + att[home] - def[away]), mu = exp(att[away] - def[home]) reproduces v7 exactly.

  refit   python pipeline/v7/live.py refit --db <ArtifactData out_dir with state/, fixtures/, results/> --asof YYYY-MM-DD
          Fits all four leagues, runs the guard checks, and writes out/refit_<date>/ : detail_<code>.json
          ({attack, defense, ratings_refit}) and summary.json ({leagues, model_version, generated_at, refit_source})
          ready for ArtifactData "update", plus checks.json. Exit code 0 = every league passed, 2 = at least one
          league failed (its files are not written; the others are).
  shadow  python pipeline/v7/live.py shadow --db <out_dir> --asof YYYY-MM-DD
          Writes out/shadow_<date>.csv (v7) and out/shadow_v61_<date>.csv (v6.1 method refit the same way) for
          unplayed fixtures not already in an earlier file of the same model.
  score   python pipeline/v7/live.py score --db <out_dir>
          out/scorecard.md: 1X2, over 2.5 and BTTS log loss per model on the same settled matches, with DraftKings.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "pipeline" / "v6"))
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "pipeline" / "v9"))
import candidate as C  # noqa: E402
import method as V9  # noqa: E402
import refit as R  # noqa: E402
from src import guards as G  # noqa: E402

OUT = HERE / "out"
MODEL = V9.MODEL
MODEL_VERSION = "Alpha Lines v9"
V9_SINCE = "2026-11-15"     # shadow_<date>.csv files dated before this are the v7/v8 method
CONFIGS = {"v9": "v9", "v8": dict(C.CHOSEN), "v61": dict(ridge=5.0, shots=0.0, level=False)}
SHADOW_FAMILIES = ["v9", "v8"]     # v6.1 shadow files stopped 2026-11-15; the old ones are still scored
PREFIX = {"v9": "shadow_", "v8": "shadow_v8_", "v61": "shadow_v61_"}
LABEL = {"v9": "v9 (market-implied goals, 69-day half-life, promoted prior, stretch 1.13)",
         "v8": "v8 method (shots30, ridge 1, goal level)", "v61": "v6.1 method (ridge 5, goals only)"}


def family_of(name: str) -> str:
    """Which method wrote a shadow file."""
    if name.startswith(PREFIX["v61"]):
        return "v61"
    if name.startswith(PREFIX["v8"]):
        return "v8"
    return "v9" if name[len("shadow_"):len("shadow_") + 10] >= V9_SINCE else "v8"


def fit_league(db: Path, code: str, asof: str, cfg):
    if cfg == "v9":
        return V9.fit_league(db, code, asof)
    df = C.load(code)
    fd_last = df.Date.max()
    ex = V9.new_extras(df, C.extras_from_db(db, code, fd_last))   # 2026-11-15: no double-counting of fd's last day
    if len(ex):
        df = pd.concat([df, ex], ignore_index=True).sort_values("Date").reset_index(drop=True)
    kw = dict(cfg); lvl = kw.pop("level")
    a = pd.Timestamp(asof) + pd.Timedelta(days=1)
    r = C.fit(df, a, **kw)
    k = R.level_factor(df, r) if lvl else 1.0
    n = int(((df.Date < a) & (df.Date >= a - pd.Timedelta(days=R.WINDOW))).sum())
    return df, r, k, {"n_matches": n, "n_extra": int(len(ex)), "fd_last": str(fd_last.date())}


LIVE_SINCE = "2026-10-09"   # first week the dashboard tracked all four leagues


def live_validation(db, code):
    """Live 1X2 Brier (sum over home/draw/away) on settled dashboard predictions since LIVE_SINCE: the
    locked prediction's x1x2 vs its market_devig (DraftKings, de-vigged). Replaces the frozen early-season
    validation block the dashboard tile read (2026-11-02). None if predictions weren't exported."""
    import glob as _g, json as _j
    from pathlib import Path as _P
    db = _P(db)
    if not (db / "predictions").exists():
        return None
    def load(c):
        out = {}
        for f in _g.glob(str(db / c / "*.json")):
            d = _j.loads(_P(f).read_text())
            out[_P(f).stem] = d["data"] if isinstance(d, dict) and "data" in d and "id" not in d else d
        return out
    fx, res, pr = load("fixtures"), load("results"), load("predictions")
    n = bm = bk = 0
    for fid, f in fx.items():
        r, p = res.get(fid), pr.get(fid)
        if f.get("league") != code or not r or "home_score" not in r or not p or f.get("status") == "postponed":
            continue
        if str(f.get("kickoff", ""))[:10] < LIVE_SINCE:
            continue
        x, m = p.get("x1x2") or {}, p.get("market_devig") or {}
        if not all(k in x for k in ("home", "draw", "away")) or not all(m.get(k) for k in ("home", "draw", "away")):
            continue
        hs, as_ = int(r["home_score"]), int(r["away_score"])
        y = {"home": float(hs > as_), "draw": float(hs == as_), "away": float(as_ > hs)}
        bm += sum((float(x[k]) - y[k]) ** 2 for k in y); bk += sum((float(m[k]) - y[k]) ** 2 for k in y); n += 1
    if not n:
        return None
    return {"n_evaluated": n, "model_brier_mean": bm / n, "market_brier_mean": bk / n,
            "brier_source": f"live: locked dashboard predictions vs DraftKings de-vigged, settled since {LIVE_SINCE}"}


def checks(df, r, k, prev_att, prev_def, upcoming, missing=None, method_changed=False):
    """Guard rails before anything is written to the dashboard. Returns (ok, list of (check, ok, detail)).
    missing (2026-11-09): league fixtures from the last 21 days that have kicked off but have no result in the
    dashboard. A refit on incomplete results is what forced the Nov 3 hand rebuild of week 10, so it now fails
    the league (it keeps last week's ratings) until the results are in."""
    out = []
    if missing is not None:
        out.append(("results complete for played matches (last 21 days)", not missing,
                    f"{len(missing)} missing: " + "; ".join(missing[:6]) + (" ..." if len(missing) > 6 else "")
                    if missing else "all played matches have results"))
    out.append(("converged", bool(r["converged"]), str(r["converged"])))
    out.append(("home advantage 0 to 0.5", 0 <= r["home_adv"] <= 0.5, f"{r['home_adv']:.3f}"))
    out.append(("rho inside bounds", -0.25 < r["rho"] < 0.25, f"{r['rho']:.3f}"))
    out.append(("goal-level factor 0.85 to 1.20", 0.85 <= k <= 1.20, f"{k:.3f}"))
    mx = max(max(abs(v) for v in r["att"].values()), max(abs(v) for v in r["def"].values()))
    out.append(("ratings magnitude below 1.5", mx < 1.5, f"{mx:.3f}"))
    missing = sorted({t for h, a in upcoming for t in (h, a)} - set(r["att"]))
    out.append(("every upcoming team rated", not missing, ", ".join(missing) or "all rated"))
    if prev_att and method_changed:
        out.append(("largest weekly change in team strength below 0.35", True,
                    "skipped once: the previous ratings were fitted by a different method"))
    elif prev_att:
        common = [t for t in r["att"] if t in prev_att and t in prev_def]
        lk = math.log(k)
        moves = [abs((r["att"][t] + lk + r["def"][t]) - (prev_att[t] + prev_def[t])) for t in common]
        big = max(moves) if moves else 0.0
        out.append(("largest weekly change in team strength below 0.35", big < 0.35, f"{big:.3f} over {len(common)} teams"))
    recent = df[df.Date >= df.Date.max() - pd.Timedelta(days=60)]
    if len(recent) >= 20:
        pt = np.mean([sum(R.lambdas(r, h, a)) * k for h, a in zip(recent.HomeTeam, recent.AwayTeam)])
        act = float((recent.FTHG + recent.FTAG).mean())
        out.append(("recent goals: predicted within 0.6 of actual", abs(pt - act) < 0.6,
                    f"predicted {pt:.2f}, actual {act:.2f} over {len(recent)} matches (in-sample)"))
    return all(c[1] for c in out), out


def _upcoming(db: Path, code: str, asof: str):
    t = pd.Timestamp(asof).tz_localize("UTC")
    ups = []
    for f in (db / "fixtures").glob("*.json"):
        fx = json.loads(f.read_text())
        if fx.get("league") == code and pd.Timestamp(fx["kickoff"]) > t and not (db / "results" / f.name).exists():
            ups.append((fx["home"], fx["away"]))
    return ups


def refit(db: Path, asof: str) -> int:
    summ = json.loads((db / "state" / "summary.json").read_text())
    summ = summ.get("data", summ)
    leagues = summ["leagues"]
    od = OUT / f"refit_{asof}"
    od.mkdir(parents=True, exist_ok=True)
    report, status = {}, 0
    for code in C.LEAGUES:
        det = json.loads((db / "state" / f"detail_{code}.json").read_text()); det = det.get("data", det)
        df, r, k, meta = fit_league(db, code, asof, CONFIGS["v9"])
        prev_model = (det.get("ratings_refit") or {}).get("model")
        ok, ck = checks(df, r, k, det.get("attack"), det.get("defense"), _upcoming(db, code, asof),
                        missing=G.missing_results(db, code, asof), method_changed=prev_model != MODEL)
        pbp = G.postponed_but_played(db, code, ROOT / "data" / "fd")   # warning only, does not fail the league
        report[code] = {"ok": bool(ok), "checks": [{"check": c, "ok": bool(o), "detail": d} for c, o, d in ck],
                        "postponed_but_played": pbp,
                        "level_factor": float(k), "home_adv": r["home_adv"], "rho": r["rho"], **meta}
        if not ok:
            status = 2
            continue
        lk = math.log(k)
        rr = {"as_of": asof, "model": MODEL, "ridge": 1.0, "xi": V9.CHOSEN["xi"], "half_life_days": 69,
              "window_days": R.WINDOW,
              "target": "goals the closing market implied for each past match (Shin 1X2 + fair over 2.5), Poisson",
              "n_market_target": meta["n_market_target"], "stretch": V9.S,
              "stretch_note": "strength differences multiplied by this; folded into attack and defense",
              "promoted_prior": {"P": V9.CHOSEN["P"], "fade_k": 10, "teams": meta["promoted"]},
              "rho_note": "fixed at the market value", "centre_defence": True,
              "level_factor": round(k, 4), "level_factor_note": "folded into attack: attack = fitted attack + ln(level_factor)",
              "n_matches": meta["n_matches"],
              "extra_results": f"{meta['n_extra']} dashboard results newer than the football-data CSV (last {meta['fd_last']}); "
                               f"{meta['n_extra_with_dk_price']} of them priced from DraftKings' last moneyline",
              "previous": "v8: 70% goals + 30% shot proxy, xi 0.002 (until the 2026-11-03 refit)",
              "code": "pipeline/v7/live.py refit (method: pipeline/v9/method.py)",
              "note": "matches[] below is still the v6.1 walk-forward"}
        (od / f"detail_{code}.json").write_text(json.dumps({
            "attack": {t: round(a + lk, 6) for t, a in r["att"].items()},
            "defense": {t: round(v, 6) for t, v in r["def"].items()}, "ratings_refit": rr}))
        lg = dict(leagues[code])
        lg.update(home_adv=round(r["home_adv"], 6), rho=round(r["rho"], 6), ratings_as_of=asof, ratings_model=MODEL,
                  n_matches_fit=meta["n_matches"], blend_weight=0, xi=V9.CHOSEN["xi"])
        lv = live_validation(db, code)
        if lv:
            lg["validation"] = dict(lg.get("validation") or {}, **lv)
        leagues[code] = lg
    now = pd.Timestamp.now("UTC").isoformat(timespec="seconds")
    passed = [c for c in C.LEAGUES if report[c]["ok"]]
    if passed:
        (od / "summary.json").write_text(json.dumps({
            "leagues": leagues, "model_version": MODEL_VERSION, "generated_at": now,
            "refit_source": f"weekly v9 refit (pipeline/v7/live.py, method pipeline/v9/method.py) as of {asof}; leagues refit: {', '.join(passed)}; "
                            f"prices unchanged (blend weight 0)"}))
    (od / "checks.json").write_text(json.dumps(report, indent=1))
    for c in C.LEAGUES:
        bad = [x for x in report[c]["checks"] if not x["ok"]]
        print(f"{c}: {'PASS' if report[c]['ok'] else 'FAIL'}  level {report[c]['level_factor']:.3f}  "
              f"home_adv {report[c]['home_adv']:.3f}" + ("" if not bad else "  failed: " +
              "; ".join(f"{x['check']} ({x['detail']})" for x in bad)))
        for w in report[c].get("postponed_but_played") or []:
            print(f"{c}: WARNING {w} - clear the postponed status so its result can be recorded")
    return status


def shadow(db: Path, asof: str):
    n = {}
    for fam in SHADOW_FAMILIES:
        cfg = CONFIGS[fam]
        done = set()   # a fixture is predicted once per method: the first file of that method that saw it
        for prev in OUT.glob("shadow_*.csv"):
            if family_of(prev.name) == fam and prev.name != f"{PREFIX[fam]}{asof}.csv":
                done |= set(pd.read_csv(prev).fixture_id)
        recs = []
        t = pd.Timestamp(asof).tz_localize("UTC")
        for code in C.LEAGUES:
            _, r, k, _ = fit_league(db, code, asof, cfg)
            for f in (db / "fixtures").glob("*.json"):
                fx = json.loads(f.read_text())
                if fx.get("league") != code or pd.Timestamp(fx["kickoff"]) <= t or fx["id"] in done:
                    continue
                if (db / "results" / f.name).exists():
                    continue
                lam, mu = R.lambdas(r, fx["home"], fx["away"]); lam, mu = lam * k, mu * k
                G = R.grid(lam, mu, r["rho"]); i, j = np.indices(G.shape)
                unknown = [x for x in (fx["home"], fx["away"]) if x not in r["att"]]
                recs.append({"fixture_id": fx["id"], "league": code, "kickoff": fx["kickoff"], "home": fx["home"],
                             "away": fx["away"], "xg_home": round(lam, 4), "xg_away": round(mu, 4),
                             "p_home": round(float(G[i > j].sum()), 5), "p_draw": round(float(G[i == j].sum()), 5),
                             "p_away": round(float(G[i < j].sum()), 5), "p_o15": round(float(G[(i + j) > 1].sum()), 5),
                             "p_o25": round(float(G[(i + j) > 2].sum()), 5), "p_o35": round(float(G[(i + j) > 3].sum()), 5),
                             "p_btts": round(float(G[(i > 0) & (j > 0)].sum()), 5), "unknown_teams": ";".join(unknown),
                             "model": LABEL[fam], "made_at": pd.Timestamp.now("UTC").isoformat(timespec="seconds")})
        if recs:
            pd.DataFrame(recs).sort_values(["kickoff", "league"]).to_csv(OUT / f"{PREFIX[fam]}{asof}.csv", index=False)
        n[fam] = len(recs)
    return n


def score(db: Path) -> str:
    from v5 import market as M
    rows = []
    for f in sorted(OUT.glob("shadow_*.csv")):
        fam = family_of(f.name)
        for r in pd.read_csv(f).to_dict("records"):
            rp, op = db / "results" / f"{r['fixture_id']}.json", db / "odds" / f"{r['fixture_id']}.json"
            if not rp.exists():
                continue
            res = json.loads(rp.read_text()); res = res.get("data", res)
            if "home_score" not in res:
                continue
            hg, ag = int(res["home_score"]), int(res["away_score"])
            y = "home" if hg > ag else ("draw" if hg == ag else "away")
            od = json.loads(op.read_text()) if op.exists() else {}
            od = od.get("data", od)
            px = od.get("latest") or od.get("open") or {}
            dk = None
            if all((px.get(k) or 0) > 1 for k in ("home", "draw", "away")):
                dk = dict(zip(("home", "draw", "away"), M.shin([px["home"], px["draw"], px["away"]])))
            o25, bt = int(hg + ag > 2), int(hg > 0 and ag > 0)
            rows.append({"fam": fam, "week": f.name.split("_")[-1][:-4], "fixture_id": r["fixture_id"],
                         "ll": -math.log(max(r[f"p_{y}"], 1e-9)), "ll_dk": None if dk is None else -math.log(dk[y]),
                         "ll_o25": float(R._ll(np.array([r["p_o25"]]), np.array([o25]))[0]),
                         "ll_btts": float(R._ll(np.array([r["p_btts"]]), np.array([bt]))[0])})
    d = pd.DataFrame(rows)
    lines = ["# Ratings scorecard", "",
             "Log loss on the same settled matches; lower is better. Every prediction file was written before kickoff. "
             "DraftKings = Shin de-vig of the latest pre-kickoff moneyline.", ""]

    nm = {"v9": "v9", "v8": "v8", "v61": "v6.1"}

    def section(title, new, old, note):
        out = [f"## {title}", "", note, "",
               f"| Week file | Matches | 1X2 {nm[new]} | 1X2 {nm[old]} | 1X2 DraftKings | Over 2.5 {nm[new]} | Over 2.5 {nm[old]} | "
               f"BTTS {nm[new]} | BTTS {nm[old]} |", "|---|---|---|---|---|---|---|---|---|"]
        if not len(d):
            return out + ["| (nothing settled yet) | | | | | | | | |", ""]
        a = d[d.fam == new].drop_duplicates("fixture_id").set_index("fixture_id")
        b = d[d.fam == old].drop_duplicates("fixture_id").set_index("fixture_id")
        both = a.join(b, lsuffix="_n", rsuffix="_o", how="inner").dropna(subset=["ll_dk_n"])
        if not len(both):
            return out + ["| (nothing settled yet) | | | | | | | | |", ""]
        for wk, g in list(both.groupby("week_n")) + [("all", both)]:
            out.append(f"| {wk} | {len(g)} | {g.ll_n.mean():.4f} | {g.ll_o.mean():.4f} | {g.ll_dk_n.mean():.4f} | "
                       f"{g.ll_o25_n.mean():.4f} | {g.ll_o25_o.mean():.4f} | {g.ll_btts_n.mean():.4f} | {g.ll_btts_o.mean():.4f} |")
        if len(both) >= 20:
            dd = (both.ll_n - both.ll_o).values
            rng = np.random.default_rng(7)
            bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
            out += ["", f"{nm[new]} minus {nm[old]}, 1X2: {dd.mean():+.4f} (95% {np.percentile(bs, 2.5):+.4f} to "
                        f"{np.percentile(bs, 97.5):+.4f}) on {len(dd)} matches. Negative = {nm[new]} better."]
        return out + [""]

    lines += section(f"v9 (live from {V9_SINCE}) against the v8 method", "v9", "v8",
                     "v9 = the live dashboard ratings (market-implied goals, 69-day half-life, promoted prior, stretch); "
                     "v8 = the previous method refit the same way, kept as a shadow.")
    lines += section("v8 (live 2026-10-20 to 2026-11-15) against v6.1 - history", "v8", "v61",
                     "v8 = the v7/v8 method (70% goals + 30% shot proxy); v6.1 = ridge 5, goals only. Closed: no new "
                     "v6.1 files are written.")
    txt = "\n".join(lines) + "\n"
    (OUT / "scorecard.md").write_text(txt)
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["refit", "shadow", "score"])
    ap.add_argument("--db", required=True)
    ap.add_argument("--asof")
    a = ap.parse_args()
    db = Path(a.db)
    if a.cmd == "refit":
        sys.exit(refit(db, a.asof))
    if a.cmd == "shadow":
        print(json.dumps(shadow(db, a.asof)))
        return
    print(score(db))


if __name__ == "__main__":
    main()
