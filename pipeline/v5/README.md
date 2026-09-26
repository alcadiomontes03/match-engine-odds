# Pulsematch v5 — market-anchored model

**Status (2026-09-25): built and backtested. Gate closed, paper mode.**

v5 changes what the model is for. It no longer predicts matches from scratch and
competes with the bookmakers. The sharp market price is the starting point, and the
model only learns small corrections to it, shrunk to zero unless the data clearly
supports them. The same goals grid then prices derivative markets (spreads, totals,
BTTS) so they can be compared with DraftKings.

## Pipeline

| Step | File | What it does |
|---|---|---|
| 0. Data | `../../fetch_fd.py`, `.github/workflows/fd-history.yml` | GitHub Actions job pulls football-data.co.uk results, shots and Pinnacle/market-average opening and closing odds (1X2, O/U 2.5, Asian handicap) for PL, La Liga, Bundesliga, Ligue 1, 2017-18 to now, into `data/fd/`. Completed seasons once; current season every Monday 14:17 UTC, before the weekly refit. |
| 1. Market layer | `market.py` | Shin de-vig (fixes the favourite-longshot bias of proportional de-vig), market-implied expected goals (lam, mu) from 1X2 + O/U 2.5, derivative prices from any (lam, mu). |
| 2. Gap features | `features.py` | Walk-forward weekly `dc4` fits on goals and on shots on target (an xG stand-in with full league/season coverage), with a newcomer prior for promoted clubs. Features: ratings supremacy gap vs market, shots gap, shot volume, rest-day difference, early-season interaction. |
| 3. Adjustment model | `model.py` | Offset logistic models: `log p_market + b·x` (1X2, softmax) and `logit p_market + b·x` (O/U 2.5). L2 on every coefficient including intercepts; penalty picked on the last two seasons with the one-standard-error rule (heaviest shrinkage that is statistically as good as the best). |
| 4. Backtest + gate | `backtest.py` | Walk-forward by season, scored against the closing line; CLV gate; quarter-Kelly bankroll path; Monte Carlo. |
| 5. Structure checks | `checks.py` | Soft-vs-sharp price test and derivative-pricing accuracy vs Pinnacle's Asian handicap. |
| 6. Live pricing | `runtime.py` | `price_fixture()` → fair 1X2, totals, spreads, BTTS from the sharp line; `stake_plan()` → quarter Kelly, 2%/bet, 3%/match cap, marked paper-only while the gate is closed. |

Rebuild: `python pipeline/v5/build_dataset.py && python pipeline/v5/backtest.py && python pipeline/v5/checks.py` (about 90 seconds).

## Gate

Open only when the v5 paper bets over the two most recent completed seasons number at
least 300, have mean CLV > 0 against the closing line, and a CLV t-statistic ≥ 2.
Profit is not the gate: it needs thousands of bets to separate skill from luck.

## Results (walk-forward, 2020-21 to 2025-26, four leagues, 8,454 matches)

Log loss (lower is better). "Open" is the pre-match price v5 starts from; "close" is the closing line.

| Season | Close source | 1X2 open | 1X2 v5 | 1X2 close | 1X2 ratings only | O/U open | O/U v5 | O/U close |
|---|---|---|---|---|---|---|---|---|
| 2020-21 | Pinnacle | 0.9950 | 0.9950 | 0.9894 | 1.0184 | 0.6764 | 0.6765 | 0.6736 |
| 2021-22 | Pinnacle | 0.9774 | 0.9774 | 0.9766 | 0.9938 | 0.6739 | 0.6739 | 0.6706 |
| 2022-23 | Pinnacle | 0.9769 | 0.9769 | 0.9747 | 0.9995 | 0.6720 | 0.6720 | 0.6723 |
| 2023-24 | Pinnacle | 0.9508 | 0.9508 | 0.9486 | 0.9708 | 0.6561 | 0.6561 | 0.6535 |
| 2024-25 | Pinnacle | 0.9668 | 0.9668 | 0.9631 | 0.9859 | 0.6653 | 0.6653 | 0.6613 |
| 2025-26 | Pinnacle / average | 0.9775 | 0.9775 | 0.9776 | 1.0009 | 0.6702 | 0.6702 | 0.6655 |
| 2026-27 (200 so far) | average | 0.9946 | 0.9944 | 0.9888 | 1.0177 | 0.6308 | 0.6306 | 0.6314 |

- **Adjustment model:** every season the one-SE rule chose maximum shrinkage for both
  1X2 and totals. The ratings gap, shots gap, rest days and early-season flag carry no
  information the Pinnacle pre-match line doesn't already have. v5 therefore equals the
  market and places **no** paper bets. Gate: **closed**.
- **Ratings only (v4 approach, blend weight 1):** 14,415 paper bets at a 2% edge
  threshold, mean CLV **-4.0%** (t = -56.9), ROI -6.7%. This confirms the blend-weight-0 decision.
- **Earlier v5 drafts, kept as a warning:** an unpenalised intercept, and penalty
  selection on a single season, each produced 500+ bets at about -2% CLV. Both were removed.
- **Derivative pricing** (2,216 recent half-goal Asian handicap lines): the engine,
  given only the closing 1X2 + O/U 2.5, matches Pinnacle's own closing handicap price
  within 0.74 points on average (90th percentile 1.74 points), and its Brier score is
  identical to Pinnacle's (0.2499 vs 0.2498, t = 0.67). So the engine can price
  spreads, totals and BTTS from the main lines.
- **Soft vs sharp:** the *market-average* price beat Pinnacle's fair price by 2% only
  7 times in 12,008 matches, and those lost CLV. An average of books is not a soft
  book. The test that matters, a single recreational book (DraftKings) against the
  sharp line, needs DraftKings history, which the odds repo is now collecting.

## What this means for live use

v5's only credible source of edge is a recreational book lagging the sharp line,
especially on derivative markets it prices off its main line. Testing that directly
would need a sharp reference book in the odds capture (free in The Odds API, since up to
10 bookmakers bill as one region). That option was considered and **not adopted**: the
capture stays DraftKings only (next section), so v5 tests the narrower question of
whether DraftKings' side markets agree with its own main lines.

## DraftKings-only mode (decided 2026-09-25)

No second book is added. The odds job pulls DraftKings moneyline, spreads and totals
in each bulk call (3 credits) and BTTS once per match at its first pre-kickoff pull
(1 credit, skipped whenever credits fall to 100 or below). After every pull that brings
new odds, `paper_picks.py` de-vigs DraftKings' own moneyline and main total line (any
half or whole line, whichever is closest to 50/50), prices its spreads, other total
lines and BTTS from the resulting goals grid, and logs any price >= 2% above fair to
`data/v5/paper_picks.csv`. CLV is scored against DraftKings' last pre-kickoff price for
the same selection, or, when that snapshot doesn't carry it (BTTS, moved lines),
against the closing snapshot's goals grid (`clv_basis` = "grid").
`data/v5/summary.json` holds the running scorecard and uses the same gate rule.
Because the reference line is DraftKings itself, this can only find DraftKings pricing
its side markets out of line with its own main lines, not moneyline edges.

Credit estimate at the recent pull rate: about 300 a month for the three bulk markets
plus about 150 for BTTS, against 500 free. The BTTS reserve keeps the main markets running.

## Dashboard and schedule

- The dashboard has a **v5** page (sidebar / bottom bar) that reads one database
  document, `state/v5`, built by `dashboard_doc.py`: gate status, the season table,
  the pricing-engine check and the latest DraftKings paper picks.
- The scheduled task **"Match Engine — v5 weekly"** runs Mondays 17:10 UTC (after the
  v4 refit): clones this repo, runs build_dataset → backtest → checks → paper_picks →
  dashboard_doc, and writes `state/v5`. It touches nothing else.
- The `fd-history` GitHub Action refreshes `data/fd/` Mondays 14:17 UTC.
- A copy of this folder is kept in Drive under "Match Engine — Pipeline Code/v5";
  GitHub is what the v5 job runs.

## Known limitations

- Rest days count league matches only. Midweek Champions League or cup games are not
  in football-data, so the rotation effect is only partly captured.
- "Open" in football-data is the price collected on Friday (Tuesday for midweek), not the
  true opening line.
- 2025-26 (second half) and 2026-27 use the market-average close, which is softer than
  Pinnacle's, so recent-season CLV numbers are not comparable to earlier seasons.
