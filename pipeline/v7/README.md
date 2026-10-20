# Alpha Lines v7 candidate (shadow, 2026-10-20)

Research only. Nothing here prices a bet, changes the dashboard or touches the v6.1 tracking frozen by the
week 7-10 edge plan. `candidate.py` is the code; results are in `out/`.

**Why:** week 7 showed the Dixon-Coles ratings under-rating favourites (market favourites over 55%: market 65.5%,
model 55.9%). The walk-forward below confirms it is long-running, not one week.

**Walk-forward, 2023-08 to 2026-10, 4,353 matches, 28-day refits** (`out/backtest.json`, `out/backtest_round2.json`).
Lower log loss is better; fav_gap = model minus market probability on market favourites above 55%.

| Variant | 1X2 | 1X2 minus market | Over 2.5 | BTTS | Goal bias | fav_gap |
|---|---|---|---|---|---|---|
| v6.1 (ridge 5, goals only) | 0.9884 | +0.0241 | 0.6706 | 0.6828 | -0.061 | -0.060 |
| ridge 1 + goal level | 0.9873 | +0.0230 | 0.6726 | 0.6841 | -0.033 | -0.033 |
| **30% shots, ridge 1, goal level (chosen)** | **0.9860** | **+0.0217** | **0.6696** | **0.6820** | **-0.031** | **-0.046** |
| 50% shots, ridge 1, goal level | 0.9865 | +0.0223 | 0.6687 | 0.6813 | -0.030 | -0.055 |

Chosen vs v6.1, paired bootstrap over matches: 1X2 -0.0024 (95% -0.0039 to -0.0009), Over 2.5 -0.0009
(-0.0026 to +0.0008), BTTS -0.0008 (-0.0021 to +0.0005). 1X2 improves in all four leagues. Thirteen variants were
tried, so treat the margin with some caution. Every variant is still about 0.022 log loss behind the closing market:
better ratings, not an edge.

**Shot proxy:** inside each fit window, goals ~ a*shots on target + b*(other shots), no intercept, fitted on that
window only (about a = 0.35, b = 0). The fit maximises 70% Dixon-Coles goals likelihood + 30% Poisson
likelihood of the proxy.

**Live shadow test:** `out/shadow_<date>.csv` is written and committed before kickoff each week, so the candidate
is scored out of sample against v6.1 and DraftKings without any chance of revision.
