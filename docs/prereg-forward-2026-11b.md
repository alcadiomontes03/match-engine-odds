# Pre-registration 2026-11b: Test A restarted for the v9 rating method

**Registered 2026-11-15 by the git commit that adds this file**, during the November international break. No match
in the sample of `docs/prereg-forward-2026-11.md` had been played, so nothing collected is discarded. Asked for by
Alcadio the same day ("test ideas on the remaining gaps, do everything that needs to be tweaked, fixed or added").

This replaces **only Test A** of `docs/prereg-forward-2026-11.md`. **Test B (rule S1) is unchanged** and keeps its
sample start of 2026-11-10: S1 never used the ratings (its grid comes from DraftKings' own prices), and
`pipeline/picks/s1.py` is untouched. `docs/prereg-sharp-2026-11-01.md` (rule P1) also continues unchanged.
Nothing here changes a price (blend weight 0) or a stake. Everything is paper.

**Sample start (Test A):** `2026-11-16T00:00:00Z` (`pipeline/picks/report.py: TEST_A_START`). **Sample end:** the
last matchday of the 2026-27 seasons. **Scope:** Premier League, La Liga, Bundesliga, Ligue 1.

## What changed: the frozen model method is now v9
Code: `pipeline/v9/method.py` (`CHOSEN`, `S`), called by `pipeline/v7/live.py` (`refit`, `shadow`).
- **Target:** every past match enters the fit as the goals the closing market expected from it: the (lam, mu) whose
  Dixon-Coles grid (rho −0.05) reproduces the Shin-de-vigged closing 1X2 and the fair over-2.5 from football-data
  (Pinnacle close, else market-average close). No closing price in football-data: DraftKings' last pre-kickoff
  moneyline from `data/odds/snapshots.csv`; no price at all: the match's shot proxy, else its goals.
- **Decay:** xi = 0.010 a day (69-day half-life), window 1,100 days, centred ridge 1.
- **Promoted teams:** pulled toward last season's relegated teams' level, P = 5, fade k = 10.
- **Stretch:** strength differences × **1.13**, a constant, folded into attack and defence.
- **rho:** fixed at −0.05. **Goal-level factor** and the **guards** (`pipeline/src/guards.py`) as before.
- **Bug fixed with the change:** matches played on football-data's last day were counted twice in the fit.

The numbers refit every week by the same code; the method and every constant above stay fixed for this sample.

## Why (exploration, not part of the sample)
Walk-forward on 4,506 matches, Aug 2023 – Nov 2026 (`pipeline/v8/gap.py`, `pipeline/v8/out/gap_summary*.md`):
gap to the closing market **+0.0221 (v8) → +0.0127 (v9)**, change −0.0093, 95% −0.0136 to −0.0052. About 15
variants were tried over two rounds, so the size is optimistic; every market-target variant with a 69- to 116-day
half-life gained between −0.0070 and −0.0093. Rest days (league matches) and Club Elo added nothing on top of v9.
The ratings still add nothing on top of the price (pool with the closing market: +0.0006, 95% −0.0002 to +0.0015),
and v9 is built from market prices, so Test A is expected to say "no" again; it now measures a better model.

## Test A (unchanged except for the model and the sample start)
- **Prediction:** `pipeline/v7/out/shadow_<date>.csv` (not `shadow_v61_*`, not `shadow_v8_*`), written by the weekly
  scorecard job with the v9 method, timed by the git commit that added the file.
- **Matched DraftKings price, results, primary contrast, secondary measures, exclusions, decision looks:** exactly as
  in `docs/prereg-forward-2026-11.md` (per-match log loss, model − matched DraftKings; mid-season look at 99%,
  season end at 95%; the ratings "add information" only if the whole interval is below 0).
- **Added, reported only:** v9 against the v8 method on the same matches (`shadow_v8_<date>.csv`,
  `pipeline/v7/out/scorecard.md`).
- **Dependency:** the weekly refit and scorecard scheduled tasks must be running (re-enabled 2026-11-15).

## Code at registration
`pipeline/v9/method.py`, `pipeline/v8/gap.py`, `pipeline/v8/fixes.py`, `pipeline/v7/live.py`,
`pipeline/picks/report.py` (only `TEST_A_START` and labels changed; Test B code paths untouched),
`pipeline/test_v9.py`, `pipeline/src/guards.py`.
