# Pre-registration: DraftKings vs Pinnacle (moneyline), registered 2026-11-01

Written before any Pinnacle data existed. The first Pinnacle rows arrive with the weekly pull on Mon 2026-11-02.
The git commit that adds this file is the registration time. The rules below do not change for this sample.
Any change (threshold, de-vig, timing, data) becomes a **new** registration with a new sample start date. Data
collected under this one may then only be used for exploration.

Background:
- **The ratings.** v7 trails the closing market by about 0.022 log loss (walk-forward 2023-26). Two calibration fixes tried on 2026-11-01 found no gain.
- **Codex review.** Its advice (2026-11-01) was to test bet selection against a sharp book, on future matches only.
- **Live rules.** Ratings, prices and stakes on the dashboard do not change because of this test.

## 1. Data
- **DraftKings:** `data/odds/snapshots.csv`, rows with `bookmaker == "draftkings"` and `market == "h2h"`.
- **Pinnacle:** `data/odds/sharp.csv`, rows with `bookmaker == "pinnacle"` and `market == "h2h"`.
  - Both books come from the **same API call**, so a pull's `pulled_at` is shared.
- **Scope:** comps PL, LaLiga, Bundesliga and Ligue1 only.
- **Matching:** events match across books by the API `event_id`. There is no name matching between books.
- **Quote:** a book's three prices (home, draw, away) from one pull. A quote missing any of the three is not a quote.
- **Results:** from the dashboard `results` collection, matched through `fixtures.odds_event_id`. Postponed and unmatched matches are excluded and counted.
- **Sample start:** events kicking off after 2026-11-02T15:00Z. Sample end: the last matchday of the 2026-27 seasons.

## 2. Timing
- **Bet time (T_bet):** the event's **first `late` pull** that has a complete quote from both books. That is within 8 hours of kickoff, under the odds bot's current schedule.
- **Close:** the event's **last** pull, of any kind, strictly after T_bet and before kickoff, that has a complete Pinnacle quote.
  - If there is none (only one late pull happened), the match has **no CLV**.
  - These matches are counted and reported, and left out of the CLV endpoint.
- **Freshness:** a Pinnacle quote is **stale** if `pulled_at` minus `book_last_update` is more than 60 minutes. The API warns its Pinnacle feed may be delayed.
  - A stale quote at bet time: the match is not eligible.
  - A stale quote at the close: no CLV.
  - Both are counted.

## 3. Fair probability
- **Primary:** Shin de-vig of the Pinnacle three-way quote (`pipeline/v5/market.shin`).
- **Sensitivity only:** proportional de-vig is reported next to it. It is never used for a decision.

## 4. Rule P1 (the one primary betting rule)
- At T_bet, for each outcome i: EV_i = DK_price_i × p_pin_i − 1.
- A match is a **candidate** if max EV_i ≥ **0.02**.
- The bet is the outcome with the largest EV, at most one per match.
- Flat 1-unit stakes, paper only.
- **Secondary thresholds:** 0.00 and 0.05 are reported. No decisions are made on them.

## 5. Endpoints
**Primary endpoint (decisions use only this)**
- Mean CLV of P1 candidates, where CLV = DK_price_taken × p_pin_close − 1, with p_pin_close the Shin fair probability at the close.
- **Uncertainty:** a bootstrap that resamples whole betting weeks. Weeks are the odds bot's week key: Monday 15:00 UTC to Monday.

**Secondary (reported, no decisions)**
- Realised flat ROI and hit rate of P1 candidates, with week-block intervals.
- Candidates per week and per league.
- Mean EV at bet time.
- Pinnacle calibration at T_bet in 10-point buckets.

**Forecast test F1** (every match with complete, fresh bet-time quotes from both books, a v7 prediction and a result)
- Log loss of:
  - a) Pinnacle Shin at T_bet
  - b) DraftKings Shin at T_bet
  - c) v7 `x1x2` (the dashboard prediction, locked before kickoff)
  - d) the pool p ∝ pin^0.90 · v7^0.10, renormalised
- Primary F1 contrast: d − a. Also reported: b − a and c − a. Week-block bootstrap.
- **Why w = 0.10:** it was fixed a priori, not tuned.
  - On the 2023-26 walk-forward (4,392 matches), the best log-linear weight on v7 against the closing market was **0.000** in each of the three full seasons.
  - It was 0.29 on the 276 matches of 2026-27 so far, scored against the market-average close.
  - 0.10 is a small, pre-set test of whether v7 adds anything at bet time, which is earlier than the close.

## 6. Looks and decisions
- **Weekly report:** every Monday, descriptive only.
- **Decision looks:** when P1 reaches **100, 200 and 400** candidates with CLV, use **99%** intervals. At the end of the season, use 95%.
- **Promising:** at a decision look, the 99% interval for mean P1 CLV is entirely above 0.
  - Alcadio may then choose to place real flat 1-unit singles on P1 candidates only.
  - Paper tracking continues unchanged.
- **Stop:** at 200 or more candidates, the 95% interval is entirely below 0.
  - P1 is retired.
  - No re-tuning of threshold, timing or de-vig on this sample.
- **Otherwise:** continue.
- **F1:** v7 "adds information" only if the end-of-season 95% interval for d − a is entirely below 0. Otherwise the ratings stay display-only.

## 7. Exploratory only (never decides anything)
- EV at the Monday weekly pull (early prices).
- Side markets.
- CLV of Alcadio's hand-placed bets against the Pinnacle close (kept separate from P1).
- Staleness distribution.
- DraftKings vs Pinnacle margin.
- Results by league.

## 8. Expected size
- About 37 matches a week. The P1 candidate rate is unknown.
- **Why CLV is the endpoint:** per-bet CLV usually has an SD of about 4-8%, so 100 candidates give a standard error of about 0.6%.
- **ROI would need far more:** at even money its standard error is about 100/√n %.
