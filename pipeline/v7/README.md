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

## Live from 2026-10-20 (Alcadio's decision)

v7 replaced the v6.1 ratings in the dashboard state (`state/summary` v13, `state/detail_<code>`), and the 38
unplayed week 8 predictions were regenerated with v7 (each keeps `created_at` and gets `regenerated_at` and
`previous_model`). **Prices are unchanged:** blend weight is still 0 everywhere, so every bet price is DraftKings'
own, and the edge plan's primary tests (which use the market-anchored grid) are unaffected. The exploratory
Dixon-Coles-vs-DraftKings comparison starts a new period on 2026-10-20.

- The goal-level factor is folded into attack (attack = fitted attack + ln k), so the Monday fixtures job builds
  v7 predictions from `state/detail_<code>` with its usual formula; no job change was needed.
- Backups: `backup_v61_2026-10-20/` (v6.1 state, week 8 predictions and fixtures). Go-live fit and checks:
  `out/refit_2026-10-20/`. Restoring v6.1 = ArtifactData update of those state docs from the backup.
- `live.py refit` runs every Monday at 12:37 PT ("Alpha Lines — v7 weekly refit"), after grading and before the
  fixtures job. It writes a league only if every guard check passes (converged, home advantage 0-0.5, goal-level
  factor 0.85-1.20, ratings below 1.5, every upcoming team rated, no team strength moving more than 0.35 in a week,
  recent goals within 0.6 of actual); a failed league keeps last week's ratings.
- `live.py shadow` + `score` run Mondays at 15:37 PT ("Alpha Lines — v7 vs v6.1 scorecard"): v7 and a v6.1 shadow
  (the old method refit the same way) are committed before kickoff and scored in `out/scorecard.md`. Week 8's
  v6.1 file is the locked v6.1 dashboard predictions (Oct 12 ratings), not a refit.
- Week 7 retro (fit before Oct 13): 1X2 v7 0.913, v6.1 0.935, DraftKings 0.885; over 2.5 and BTTS were worse for
  v7 that week (0.720 vs 0.694, 0.748 vs 0.727). One week; the 2023-26 walk-forward favours v7 on all three.
