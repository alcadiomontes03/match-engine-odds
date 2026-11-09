# Pre-registration: frozen forward test and priced side-market picks (rule S1)

**Registered 2026-11-09 by the git commit that adds this file**, during the November international break and
before any match in the sample. Approved by Alcadio the same day (threshold 0.03, one pick per match, prices
1.40–3.50; near-kickoff re-pulls capped at 15 a week; robot picks treated as diagnostics). The rules below do not
change for this sample. Any change to a rule, threshold, timing, data source or the frozen model method starts a
**new** registration with a new sample start; data collected under this one is then exploration only.

Runs alongside `docs/prereg-sharp-2026-11-01.md` (DraftKings vs Pinnacle moneyline, rule P1), which continues
unchanged. Nothing here changes the dashboard's prices (blend weight 0) or stakes. Everything is paper.

**Sample start:** `2026-11-10T00:00:00Z` (`pipeline/picks/s1.py: SAMPLE_START`). **Sample end:** the last
matchday of the 2026-27 seasons. **Scope:** Premier League (E0), La Liga (SP1), Bundesliga (D1), Ligue 1 (F1).

## Why
- Audit of 2026-11-09, finding F1: the old "biggest edge vs DraftKings" auto-pick never ran. No DraftKings side
  price existed when picks were written, so every pick fell to the fallback. S1 runs only on real prices and
  records "no priced pick", with the reason, when there is nothing to bet.
- Codex's review of the same day: run a frozen forward test against timestamp-matched DraftKings prices with a
  predeclared pick rule, and don't tune on the record used to judge it.

## Test A — does the model add information? (timestamp-matched forecast test)
- **Model, frozen method:** the v7/v8 rating method in `pipeline/v7/` with the guards in `pipeline/src/guards.py`,
  as of this commit, refit weekly by the same code. The numbers update each week; the method does not.
- **Prediction:** the v7 file written by the weekly scorecard job, `pipeline/v7/out/shadow_<date>.csv`
  (not `shadow_v61_*`), timed by **the git commit that added the file**.
- **Matched DraftKings price:** the latest complete DraftKings h2h quote in `data/odds/snapshots.csv` with
  `pulled_at` at or before that commit, Shin de-vigged (`pipeline/v5/market.shin`).
- **Results:** football-data.co.uk (`data/fd/*_2627.csv`), matched by league, team names and date ± 1 day.
- **In the sample:** fixtures kicking off after the sample start whose file was committed after it.
- **Primary contrast:** per-match log loss, model − matched DraftKings.
- **Secondary (reported, no decisions):** Brier; the pool p ∝ DK^0.90 · model^0.10 against DraftKings (weight fixed
  a priori, as in the Pinnacle test); the minutes between the DraftKings pull and the commit.
- **Excluded and counted:** no git time, a team flagged in `unknown_teams`, no matched price, no result.
- **Dependency:** the weekly scorecard scheduled task must be running for Test A to get data.

Exploration only, before this registration (weeks 8–10, 110 matches, recomputed by `pipeline/picks/report.py`):
model 0.9591 vs matched DraftKings 0.9343, +0.0248 (week-block 95% +0.006 to +0.035, only 3 blocks). Timing did
not explain the model's deficit. This number is context; it is never pooled with the sample.

## Test B — rule S1 on DraftKings side markets
**When:** in the odds workflow, straight after a match's **first** side-market pull. One decision per match,
using only prices from that pull. The moneyline comes from the same pull, or the latest earlier pull.
Code: `pipeline/picks/s1.py`.

1. **De-vig:** Shin de-vig of DraftKings' 1X2.
2. **Main total:** the totals line (half or whole) whose Shin over-probability is closest to 0.5. Plain `totals`
   prices win over `alternate_totals` on the same line.
3. **Grid:** solve (λ, μ) so the Dixon-Coles grid with ρ = −0.05 matches that 1X2 and main total
   (`market.implied_lambdas`). With no total, BTTS is the anchor. With neither: "no priced pick (no anchor)".
4. **Eligible lines:** half-goal totals (over/under), half-goal home/away spreads (keyed by the home handicap) and
   BTTS yes/no, priced **1.40–3.50**. Quarter lines, whole lines and the anchor line are not eligible.
5. **EV** = p_grid × DK price − 1. The highest-EV eligible line is **always recorded** (ties: closest to 50%).
   It is a **pick** if EV ≥ **0.03**. At most one pick per match. Flat 1 unit, paper.
6. **Output:** one append-only row per match in `data/picks/s1.csv`, committed in the same workflow run as the
   odds. It holds:
   - the pull times;
   - the DraftKings 1X2;
   - the anchor and λ, μ;
   - the line's market, DraftKings market key, line, side, both prices, p_grid, DraftKings' Shin two-way
     probability and EV;
   - `decision` (`pick` / `no priced pick`) with the reason;
   - the rule and thresholds;
   - the code commit.

   Rows are never edited.

**Closing price:** for each pick only, the odds bot re-pulls that one market inside the last 65 minutes before
kickoff (`capture_odds.py: pull_clv`). There is a hard cap of **15 re-pulls per odds-bot week** (Monday
15:00 UTC to Monday). Rows go to `data/odds/clv.csv`, never `snapshots.csv`. Picks over the cap, or with no
re-pull, have no CLV and are counted.

**Primary endpoint:**
- Mean CLV = DK price taken × p_close − 1.
- p_close is the Shin two-way probability of the re-pulled DraftKings price on the same line.
- Uncertainty from a bootstrap over whole odds-bot weeks.
- Caveat: this is CLV against the same (soft) book, so a pass leads to a sharp-book check, not to real money.

**Secondary (reported, no decisions):** flat ROI and hit rate (week-block intervals); picks per week, league and
market; mean EV; the share of matches with no priced pick and why; matches at secondary thresholds 0.00 and 0.05.

## Looks and decisions
- **Weekly report:** `data/picks/reports/<date>.md`, written by the fd-history workflow on Mondays. Descriptive
  only; a weekly look never triggers a decision.
- **Test B decision looks:** at **100, 200 and 400** picks with CLV, using **99%** intervals; **95%** at season end.
  - *Promising:* the 99% interval for mean CLV is entirely above 0. Next step: a sharp-book check. Real money
    stays Alcadio's decision.
  - *Stop:* at 200 or more, the 95% interval is entirely below 0. S1 is retired; no re-tuning on this sample.
  - *Otherwise:* continue.
- **Test A decision:** mid-season (after the January break, 99%) and season end (95%). The ratings "add information"
  only if the interval for the primary contrast is entirely below 0. Otherwise they stay display-only.

## The dashboard's robot picks
They keep logging for continuity, are labelled diagnostics on the page and in reports, and never count toward any
test here.

## Expected size and cost
- **Test A:** about 37 matches a week, roughly 400 by the January break.
- **Test B:** the pick rate is unknown. Replaying weeks 6–10 through S1 (exploration, not in the sample) gave
  **0 picks in 172 matches**. The best line averaged EV −0.025, and 80 of the matches only had BTTS, the anchor.
  Expect few picks. A long run of zero picks is itself an answer: DraftKings' side lines are consistent with its
  own moneyline and main total.
- **Credits:** side pulls cost about 3 per match (about 110 a week). Re-pulls cost at most about 15–30 a week.
  The bot's credit tiers can cut side pulls late in a month; those matches are recorded as "no priced pick (no side
  prices)" or "only the anchor was priced", not dropped.

## Code at registration
- `pipeline/picks/s1.py`
- `pipeline/picks/report.py`
- `capture_odds.py` (`pull_clv`, `s1_picks`, `soon`)
- `.github/workflows/odds.yml` (runs S1 after each new side pull)
- `.github/workflows/fd-history.yml` (weekly report)
- `pipeline/test_s1.py`
- `pipeline/v7/` and `pipeline/src/guards.py` (the frozen model method)
