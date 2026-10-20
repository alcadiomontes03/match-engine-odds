"""
Alpha Lines v7 - live ratings (went live 2026-10-20, Alcadio's decision).

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
import candidate as C  # noqa: E402
import refit as R  # noqa: E402

OUT = HERE / "out"
MODEL = "alpha-lines-v7-shots30-ridge1-level"
CONFIGS = {"v7": dict(C.CHOSEN), "v61": dict(ridge=5.0, shots=0.0, level=False)}
PREFIX = {"v7": "shadow_", "v61": "shadow_v61_"}
LABEL = {"v7": "v7 (shots30, ridge 1, goal level)", "v61": "v6.1 method (ridge 5, goals only)"}


def fit_league(db: Path, code: str, asof: str, cfg: dict):
    df = C.load(code)
    fd_last = df.Date.max()
    ex = C.extras_from_db(db, code, fd_last)
    if len(ex):
        df = pd.concat([df, ex], ignore_index=True).sort_values("Date").reset_index(drop=True)
    kw = dict(cfg); lvl = kw.pop("level")
    a = pd.Timestamp(asof) + pd.Timedelta(days=1)
    r = C.fit(df, a, **kw)
    k = R.level_factor(df, r) if lvl else 1.0
    n = int(((df.Date < a) & (df.Date >= a - pd.Timedelta(days=R.WINDOW))).sum())
    return df, r, k, {"n_matches": n, "n_extra": int(len(ex)), "fd_last": str(fd_last.date())}


def checks(df, r, k, prev_att, prev_def, upcoming):
    """Guard rails before anything is written to the dashboard. Returns (ok, list of (check, ok, detail))."""
    out = []
    out.append(("converged", bool(r["converged"]), str(r["converged"])))
    out.append(("home advantage 0 to 0.5", 0 <= r["home_adv"] <= 0.5, f"{r['home_adv']:.3f}"))
    out.append(("rho inside bounds", -0.25 < r["rho"] < 0.25, f"{r['rho']:.3f}"))
    out.append(("goal-level factor 0.85 to 1.20", 0.85 <= k <= 1.20, f"{k:.3f}"))
    mx = max(max(abs(v) for v in r["att"].values()), max(abs(v) for v in r["def"].values()))
    out.append(("ratings magnitude below 1.5", mx < 1.5, f"{mx:.3f}"))
    missing = sorted({t for h, a in upcoming for t in (h, a)} - set(r["att"]))
    out.append(("every upcoming team rated", not missing, ", ".join(missing) or "all rated"))
    if prev_att:
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
        df, r, k, meta = fit_league(db, code, asof, CONFIGS["v7"])
        ok, ck = checks(df, r, k, det.get("attack"), det.get("defense"), _upcoming(db, code, asof))
        report[code] = {"ok": bool(ok), "checks": [{"check": c, "ok": bool(o), "detail": d} for c, o, d in ck],
                        "level_factor": float(k), "home_adv": r["home_adv"], "rho": r["rho"], **meta}
        if not ok:
            status = 2
            continue
        lk = math.log(k)
        rr = {"as_of": asof, "model": MODEL, "ridge": CONFIGS["v7"]["ridge"], "shots_weight": CONFIGS["v7"]["shots"],
              "shot_proxy_coef": [round(x, 4) for x in r["shot_coef"]], "xi": R.XI, "window_days": R.WINDOW,
              "target": "70% goals (Dixon-Coles) + 30% shot proxy (Poisson)", "centre_defence": True,
              "level_factor": round(k, 4), "level_factor_note": "folded into attack: attack = fitted attack + ln(level_factor)",
              "n_matches": meta["n_matches"],
              "extra_results": f"{meta['n_extra']} dashboard results newer than the football-data CSV (last {meta['fd_last']})",
              "code": "pipeline/v7/live.py refit", "note": "matches[] below is still the v6.1 walk-forward"}
        (od / f"detail_{code}.json").write_text(json.dumps({
            "attack": {t: round(a + lk, 6) for t, a in r["att"].items()},
            "defense": {t: round(v, 6) for t, v in r["def"].items()}, "ratings_refit": rr}))
        lg = dict(leagues[code])
        lg.update(home_adv=round(r["home_adv"], 6), rho=round(r["rho"], 6), ratings_as_of=asof, ratings_model=MODEL,
                  n_matches_fit=meta["n_matches"], blend_weight=0)
        leagues[code] = lg
    now = pd.Timestamp.now("UTC").isoformat(timespec="seconds")
    passed = [c for c in C.LEAGUES if report[c]["ok"]]
    if passed:
        (od / "summary.json").write_text(json.dumps({
            "leagues": leagues, "model_version": "Alpha Lines v7", "generated_at": now,
            "refit_source": f"weekly v7 refit (pipeline/v7/live.py) as of {asof}; leagues refit: {', '.join(passed)}; "
                            f"prices unchanged (blend weight 0)"}))
    (od / "checks.json").write_text(json.dumps(report, indent=1))
    for c in C.LEAGUES:
        bad = [x for x in report[c]["checks"] if not x["ok"]]
        print(f"{c}: {'PASS' if report[c]['ok'] else 'FAIL'}  level {report[c]['level_factor']:.3f}  "
              f"home_adv {report[c]['home_adv']:.3f}" + ("" if not bad else "  failed: " +
              "; ".join(f"{x['check']} ({x['detail']})" for x in bad)))
    return status


def shadow(db: Path, asof: str):
    n = {}
    for fam, cfg in CONFIGS.items():
        done = set()
        for prev in OUT.glob(PREFIX[fam] + "*.csv"):
            if fam == "v7" and prev.name.startswith(PREFIX["v61"]):
                continue
            if prev.name != f"{PREFIX[fam]}{asof}.csv":
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
        fam = "v61" if f.name.startswith(PREFIX["v61"]) else "v7"
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
    lines = ["# v7 scorecard (v7 live from 2026-10-20)", "",
             "Log loss on the same settled matches; lower is better. v7 = the live dashboard ratings; v6.1 = the old "
             "method refit the same way, kept as a shadow; DraftKings = Shin de-vig of the latest pre-kickoff moneyline.", "",
             "| Week file | Matches | 1X2 v7 | 1X2 v6.1 | 1X2 DraftKings | Over 2.5 v7 | Over 2.5 v6.1 | BTTS v7 | BTTS v6.1 |",
             "|---|---|---|---|---|---|---|---|---|"]
    d = pd.DataFrame(rows)
    if len(d):
        v7, v6 = d[d.fam == "v7"].set_index("fixture_id"), d[d.fam == "v61"].set_index("fixture_id")
        both = v7.join(v6, lsuffix="_7", rsuffix="_6", how="inner").dropna(subset=["ll_dk_7"])
        for wk, g in list(both.groupby("week_7")) + [("all", both)]:
            lines.append(f"| {wk} | {len(g)} | {g.ll_7.mean():.4f} | {g.ll_6.mean():.4f} | {g.ll_dk_7.mean():.4f} | "
                         f"{g.ll_o25_7.mean():.4f} | {g.ll_o25_6.mean():.4f} | {g.ll_btts_7.mean():.4f} | {g.ll_btts_6.mean():.4f} |")
        if len(both) >= 20:
            dd = (both.ll_7 - both.ll_6).values
            rng = np.random.default_rng(7)
            bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
            lines += ["", f"v7 minus v6.1, 1X2: {dd.mean():+.4f} (95% {np.percentile(bs, 2.5):+.4f} to "
                          f"{np.percentile(bs, 97.5):+.4f}) on {len(dd)} matches. Negative = v7 better."]
    else:
        lines.append("| (nothing settled yet) | | | | | | | | |")
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
