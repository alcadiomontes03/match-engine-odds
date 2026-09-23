"""
Match Engine v4 — FA Cup, EFL Cup and UEFA Conference League walk-forward.

No free source has historical FA Cup, EFL Cup or UECL odds, so the market gate
cannot be run for these competitions. This script answers the question that
CAN be answered now: does the joint model forecast cup matches better than
naive base rates, and is it calibrated? Market comparison has to come from
the Odds API pulls going forward.

Data:
  domestic  data/ext/dom_2020plus.csv  (xgabora mirror of football-data, 27 divisions;
            github.com/xgabora/Club-Football-Match-Data-2000-2025 data/Matches.csv,
            filtered to MatchDate >= 2020-07-01, columns Division, MatchDate,
            HomeTeam, AwayTeam, FTHome, FTAway)
  cups      data/ext/{facup,eflcup,cl,el,conf}_<season>.txt  (openfootball)

EFL Cup fields every English professional club (Premier League through League
Two) and no non-league or foreign entrants, unlike FA Cup — so it never needs
the "eng_nonleague" group prior; every EFL Cup club already has a domestic
rating from the same 27-division dataset the FA Cup fit uses.
"""
from __future__ import annotations
import json, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import dc4, cups, names as N

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
EPS = 1e-12
TEST = ["2022-23", "2023-24", "2024-25"]
GROUPS = ["eng_nonleague"]
ENGLISH_ONLY_COMPS = ("FAC", "EFC")  # cups that only ever field English clubs


def team_group(t):
    """English non-league clubs share one group. Clubs from countries with no
    domestic data get a per-country group ("unc:CRO"), and those country means
    are shrunk toward a shared hyper-mean. A single pooled group was biased
    upward: the only such clubs in the data are ones that reached UEFA group
    stages, so a pooled prior would overrate a Gibraltar or Andorra entrant."""
    if t.startswith("ENG:"):
        return "eng_nonleague"
    if ":" in t:
        return "unc:" + t.split(":")[0]
    return None


def PARENTS(g):
    return "uefa_uncovered" if g.startswith("unc:") else None


team_group.parent = PARENTS  # lets prediction route a never-seen country to the hyper-mean


def assemble(english_only=False):
    d = pd.read_csv(ROOT / "data/ext/dom_2020plus.csv")
    d[["HomeTeam", "AwayTeam"]] = d[["HomeTeam", "AwayTeam"]].replace(N.DOM_FIX)
    d["Country"] = d.Division.map(N.DIV_COUNTRY)
    d = d.dropna(subset=["Country", "FTHome", "FTAway"])
    if english_only:
        d = d[d.Country == "ENG"]
    dom = pd.DataFrame({"Date": pd.to_datetime(d.MatchDate), "HomeTeam": d.HomeTeam, "AwayTeam": d.AwayTeam,
                        "FTHG": d.FTHome.astype(int), "FTAG": d.FTAway.astype(int),
                        "Comp": d.Division, "Neutral": False, "Season": None})
    pools = {k: set(g.HomeTeam) | set(g.AwayTeam) for k, g in d.groupby("Country")}

    c = cups.load_cups()
    if english_only:
        # FA Cup and EFL Cup both field only English clubs, so both are
        # available from the same english-only domestic pool.
        c = c[c.Comp.isin(ENGLISH_ONLY_COMPS)]
    names = {}
    for side in ["Home", "Away"]:
        for ct, n in zip(c[side + "Country"], c[side + "Team"]):
            names.setdefault(ct, set()).add(n)
    mp, unmatched = N.build_map(names, pools)
    c = c.copy()
    c["HomeTeam"] = [mp[(ct, n)] for ct, n in zip(c.HomeCountry, c.HomeTeam)]
    c["AwayTeam"] = [mp[(ct, n)] for ct, n in zip(c.AwayCountry, c.AwayTeam)]
    allm = pd.concat([dom, c[dom.columns]], ignore_index=True).sort_values("Date").reset_index(drop=True)
    comps = sorted(allm.Comp.unique())
    return allm, c, comps, unmatched


def ll_multi(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], EPS, 1))


def ll_bin(p, y):
    p = np.clip(p, EPS, 1 - EPS); return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# Forward-chosen shrink toward competition base rates (logit pool weight on the
# model). Chosen from the walk-forward numbers this script produces on 2022-23
# through 2024-25 (data/cupgate_results.json), not guessed:
#   FAC:  n=416  model_minus_base_logloss = -0.062 [-0.122, +0.004]  -> model beats base -> shrink 0.6
#   UECL: n=435  model_minus_base_logloss = -0.057 [-0.101, -0.009]  -> model beats base -> shrink 0.75
#   EFC:  n=279  model_minus_base_logloss = +0.163 [+0.077, +0.257]  -> model LOSES to base -> shrink 0.3
# EFL Cup is the one competition where the raw joint model is confidently WORSE
# than just using the round's historical result frequency (CI entirely positive,
# not just noise). by_stage_model_vs_base shows why: Round 1/2/3 are clearly
# worse than base (heaviest live squad rotation), while Quarterfinals onward
# are roughly at or better than base once the biggest clubs have mostly stopped
# rotating. A single scalar shrink can't capture that stage split, so 0.3 is a
# deliberately conservative compromise (mostly base rate, some model signal)
# rather than a value tuned to look good — see data/cupgate_results.json for
# the full by-stage breakdown before trusting EFL Cup predictions for anything
# more than "informational only". Re-run this script each time a new EFL Cup
# season of results lands to check whether this still holds.
CUP_SHRINK = {"FAC": 0.6, "UECL": 0.75, "EFC": 0.3}

MIN_NEFF = 10.0  # decay-weighted matches below which a club counts as UNRATED


def walk(allm, cupdf, comps, target, xi=0.002, lam_grp=3.0, lam_hyp=2.0):
    test = cupdf[(cupdf.Comp == target) & cupdf.Season.isin(TEST)].copy()
    test["Week"] = test.Date - pd.to_timedelta(test.Date.dt.weekday, unit="D")
    warm, out = None, []
    for wk, block in test.groupby("Week"):
        warm = dc4.fit_joint(allm, wk, xi, comps, GROUPS, team_group, lam_grp, warm, parents=PARENTS, lam_hyp=lam_hyp)
        lam, mu = dc4.lambdas_joint(warm, block.HomeTeam, block.AwayTeam, block.Comp, block.Neutral, team_group)
        G = dc4.grids(lam, mu, warm["rho"])
        b = block.copy()
        b[["m_H", "m_D", "m_A"]] = dc4.probs_1x2(G)
        b["m_O25"] = dc4.p_over(G, 2.5)
        b["home_seen"] = [warm["n_eff"].get(t, 0) >= MIN_NEFF for t in b.HomeTeam]
        b["away_seen"] = [warm["n_eff"].get(t, 0) >= MIN_NEFF for t in b.AwayTeam]
        out.append(b)
    return pd.concat(out, ignore_index=True), warm


def evaluate(p, cupdf, target):
    p["y"] = p.FTR.map({"H": 0, "D": 1, "A": 2})
    p["y_O25"] = (p.FTHG + p.FTAG > 2.5).astype(float)
    res = {"n": int(len(p))}
    m_ll = ll_multi(p[["m_H", "m_D", "m_A"]].to_numpy(), p.y.to_numpy())
    # baseline: result / over-2.5 frequencies from earlier seasons of the same competition
    base_ll, base_o = [], []
    hist = cupdf[cupdf.Comp == target]
    for s, g in p.groupby("Season"):
        prior = hist[hist.Season < s]
        freq = prior.FTR.value_counts(normalize=True).reindex(["H", "D", "A"]).to_numpy()
        o = float((prior.FTHG + prior.FTAG > 2.5).mean())
        base_ll.append(ll_multi(np.tile(freq, (len(g), 1)), g.y.to_numpy()))
        base_o.append(ll_bin(np.full(len(g), o), g.y_O25.to_numpy()))
    b_ll = np.concatenate(base_ll); b_o = np.concatenate(base_o)
    m_o = ll_bin(p.m_O25.to_numpy(), p.y_O25.to_numpy())
    rng = np.random.default_rng(1)
    def ci(x):
        bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        return [round(float(x.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
    res["logloss_1x2"] = {"model": round(float(m_ll.mean()), 4), "base_rate": round(float(b_ll.mean()), 4),
                          "model_minus_base": ci(m_ll - b_ll)}
    res["logloss_o25"] = {"model": round(float(m_o.mean()), 4), "base_rate": round(float(b_o.mean()), 4),
                          "model_minus_base": ci(m_o - b_o)}
    # calibration of home-win probability
    bins = pd.cut(p.m_H, [0, .2, .35, .5, .65, .8, 1])
    cal = p.groupby(bins, observed=True).agg(n=("y", "size"), pred=("m_H", "mean"), actual=("y", lambda s: (s == 0).mean()))
    res["calibration_home_win"] = [[str(k), int(r.n), round(r.pred, 3), round(r.actual, 3)] for k, r in cal.iterrows()]
    res["pred_mean_goals_vs_actual"] = [round(float((p.m_O25).mean()), 3), round(float(p.y_O25.mean()), 3)]
    rated = (p.home_seen & p.away_seen).to_numpy()
    res["share_with_unrated_team"] = round(float((~rated).mean()), 3)
    if rated.any() and (~rated).any():
        res["model_minus_base_1x2_rated_only"] = ci((m_ll - b_ll)[rated])
        res["model_minus_base_1x2_with_unrated"] = ci((m_ll - b_ll)[~rated])
    by_stage = p.assign(mll=m_ll, bll=b_ll).groupby("Stage")[["mll", "bll"]].mean().round(3)
    res["by_stage_model_vs_base"] = by_stage.reset_index().values.tolist()
    return res


if __name__ == "__main__":
    results = {}
    t = time.time()
    allm, c, comps, un = assemble(english_only=True)
    p, warm = walk(allm, c, comps, "FAC")
    results["FA_Cup"] = evaluate(p, c, "FAC")
    results["FA_Cup"]["home_adv_FAC_vs_E0"] = [round(warm["home"]["FAC"], 3), round(warm["home"]["E0"], 3)]
    results["FA_Cup"]["nonleague_group_mean_att_def"] = [round(warm["group_att"]["eng_nonleague"], 3),
                                                          round(warm["group_def"]["eng_nonleague"], 3)]
    p.to_csv(ROOT / "data/wf_facup.csv", index=False)
    print("FA Cup done", round(time.time() - t), "s", flush=True)

    t = time.time()
    allm, c, comps, un = assemble(english_only=True)
    p, warm = walk(allm, c, comps, "EFC")
    results["EFL_Cup"] = evaluate(p, c, "EFC")
    results["EFL_Cup"]["home_adv_EFC_vs_E0"] = [round(warm["home"]["EFC"], 3), round(warm["home"]["E0"], 3)]
    p.to_csv(ROOT / "data/wf_eflcup.csv", index=False)
    print("EFL Cup done", round(time.time() - t), "s", flush=True)

    t = time.time()
    allm, c, comps, un = assemble(english_only=False)
    p, warm = walk(allm, c, comps, "UECL")
    results["Conference_League"] = evaluate(p, c, "UECL")
    results["Conference_League"]["home_adv_UECL"] = round(warm["home"]["UECL"], 3)
    results["Conference_League"]["uncovered_hyper_att_def"] = [round(warm["hyper_att"]["uefa_uncovered"], 3),
                                                                 round(warm["hyper_def"]["uefa_uncovered"], 3)]
    results["Conference_League"]["uncovered_country_means"] = {
        g: round(warm["group_att"][g] + warm["group_def"][g], 3) for g in warm["group_att"] if g.startswith("unc:")}
    results["Conference_League"]["fit_size"] = [warm["n_matches"], warm["n_teams"]]
    p.to_csv(ROOT / "data/wf_uecl.csv", index=False)
    print("UECL done", round(time.time() - t), "s", flush=True)
    print(json.dumps(results, indent=1))
    (ROOT / "data/cupgate_results.json").write_text(json.dumps(results, indent=1))
