"""Synthetic-data check for shots.py: recovers NB dispersion and team rates, and
build_prediction() attaches the shots field only when given one."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from src import dc4, shots, markets
rng = np.random.default_rng(7)
teams = [f"T{i:02d}" for i in range(20)]
att = dict(zip(teams, rng.normal(0, .2, 20))); dfn = dict(zip(teams, rng.normal(0, .15, 20)))
att = {t: v - np.mean(list(att.values())) for t, v in att.items()}
rows = []; d0 = pd.Timestamp("2024-08-10"); k_true = 12.0
for day in range(0, 700, 7):
    perm = rng.permutation(teams)
    for h, a in zip(perm[::2], perm[1::2]):
        mh = np.exp(np.log(12) + 0.1 + att[h] - dfn[a]); ma = np.exp(np.log(12) + att[a] - dfn[h])
        hs = rng.negative_binomial(k_true, k_true/(k_true+mh)); as_ = rng.negative_binomial(k_true, k_true/(k_true+ma))
        rows.append(dict(Date=d0+pd.Timedelta(days=day), HomeTeam=h, AwayTeam=a,
                         HS=hs, AS=as_, HST=rng.binomial(hs, .35), AST=rng.binomial(as_, .35),
                         FTHG=rng.poisson(mh*.11), FTAG=rng.poisson(ma*.11)))
df = pd.DataFrame(rows)
m = shots.fit(df, "2026-07-20", xi=0.002)
print("k_attempts", round(m["k_attempts"],1), "(true 12) k_sot", round(m["k_sot"],1))
est = m["attempts"]["att"]; print("att corr vs truth", round(np.corrcoef([att[t] for t in teams],[est[t] for t in teams])[0,1],3))
p = shots.predict(m, "T00", "T01"); print(json.dumps(p, indent=1)[:900])
# unseen team -> league mean, not (0,0)
u = shots.predict(m, "Newbie", "T01"); print("unseen home attempts", round(u["home"],2))
# probabilities sane
for L,v in p["totals"].items(): assert abs(v["over"]+v["under"]-1)<1e-9 and 0<v["over"]<1
# markets integration
r = dc4.fit(df, "2026-07-20", 0.002); lh, la = dc4.lambdas(r, ["T00"], ["T01"])
G = dc4.grids(lh, la, r["rho"])[0]; H,D,A = dc4.probs_1x2(G[None])[0]
pred = markets.build_prediction(G, {"home":H,"draw":D,"away":A}, {"home":2.1,"draw":3.4,"away":3.6}, 0.3, shots=p)
print("prediction keys", sorted(pred)); assert "shots" in pred
assert "shots" not in markets.build_prediction(G, {"home":H,"draw":D,"away":A})
# old call signatures still work
dc4.fit(df, "2026-07-20", 0.002, rho_bounds=(0.0,0.0)); dc4.lambdas(r, ["X"], ["T01"], fallback={"X":(0.1,0.0)})
print("ALL OK")
