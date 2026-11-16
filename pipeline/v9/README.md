# Alpha Lines v9 (live from 2026-11-15)

`method.py` holds the method and its frozen constants; `pipeline/v7/live.py` runs it (`refit`, `shadow`, `score`),
so the scheduled jobs' commands did not change. Registration: `docs/prereg-forward-2026-11b.md`. Evidence:
`pipeline/v8/gap.py` and `pipeline/v8/out/gap_summary*.md`. Tests: `pipeline/test_v9.py`.

In one line: the ratings are now fitted to what the closing market expected from each past match (not to the goals
scored), recent matches count much more (69-day half-life), promoted teams start near the relegated teams' level,
and strength gaps are widened by 13%. Backtest gap to the closing market +0.0221 -> +0.0127. Prices are unchanged
(blend weight 0); the ratings are display numbers, not a pick source.

To go back to v8: in `pipeline/v7/live.py` set `CONFIGS["v9"] = dict(C.CHOSEN)` and restore the old MODEL string,
then run the weekly refit; that needs a new registration.
