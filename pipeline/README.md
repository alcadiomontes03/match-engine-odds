# Match Engine — pipeline code (mirror)

This folder mirrors the model code that the scheduled jobs run. **The source of truth is the
Google Drive folder "Match Engine — Pipeline Code"**: the weekly jobs download each file from
Drive by file ID, not from here. When a file changes in Drive, update this mirror in the same
session so the two never drift.

Everything here runs in **paper mode**. Nothing sizes or places a real stake.

Last synced: 2026-09-25 (v4.10; `pipeline.py`, `loader.py`, `backtest.py` and `dixon_coles.py` checked against Drive and identical). **v5 (2026-09-25) lives in `pipeline/v5/` — see `pipeline/v5/README.md`.**

## Layout (`src/`)

| File | What it does | Used by |
|---|---|---|
| `dixon_coles.py` | Live goals model: Dixon-Coles with rho bounded to ±0.25 and ridge shrinkage (v4.5). | Monday refit, Sunday fixtures |
| `markets.py` | De-vig, Brier, blend, and `build_prediction()`, which turns one scoreline grid into every dashboard field (totals, spread, BTTS, grid4, top score, market de-vig, bet prob, EV). v4.7: takes an optional `shots=`. | Sunday fixtures |
| `shots.py` | **New in v4.7.** Shot-attempt and shots-on-target model (see below). | Monday refit (fit), Sunday fixtures (predict) |
| `dc4.py` | v4 fitter with analytic gradient, warm start, newcomer priors, and the joint multi-competition fit for cups. v4.7 adds the two pending audited edits. | `shots.py`, cup research, `price4.py`, `gate4.py` |
| `loader.py` | Parses the football-data.co.uk CSVs that the job saves to `data/raw/`. | Monday refit |
| `backtest.py` | Walk-forward Brier gate (model vs de-vigged market, best blend weight). | Monday refit |
| `pipeline.py` | Weekly runtime: load, fit, validate, write `state.json`. | Monday refit |
| `championship_supplement.py` | Name map and parser for the worldfootball.net Championship results supplement. | Monday refit |
| `xg.py` | Premier League xG overlay from xgstat.com (display only, not part of the fit). | Monday refit |
| `cups.py`, `cupgate.py`, `names.py` | FA Cup, EFL Cup and Conference League walk-forward research. These are ratings-fit only and never produce bets. | Monday refit step 4d |
| `gate4.py`, `shots4.py`, `price4.py` | v4 research scripts: the multi-season market gate, the shots study, and the fair-odds sheet. | Research only |

## v4.7 changes (2026-09-23)

1. **Shots model in the pipeline.** `shots.fit()` fits two Poisson rating models with rho locked
   at 0, one on attempts (HS/AS) and one on shots on target (HST/AST). It also estimates a
   negative-binomial dispersion for each. `shots.predict()` returns the `shots` field the
   dashboard's Goal attempts card already reads: expected attempts and shots on target per team,
   over/under on match attempts (the exact convolution of the two team distributions), and
   team shots-on-target lines. The Monday job stores the fit at `state/shots_<league>`, and the
   Sunday job passes the prediction through `build_prediction(..., shots=...)`.
   *Honest status:* in the shots4 backtest, team shots on target clearly beat the base rate
   (home o3.5 log loss 0.570 vs 0.623 in the PL), while match attempt totals only just beat it.
   No free shot-odds history exists, so none of this has been checked against a bookmaker.
   It is for predictions only.
2. **`dc4.py` fixes that were never merged.** When `dc4.py.new` was promoted on 2026-09-23, two
   audited edits were still missing:
   - `fit(..., rho_bounds=)` from `dc4_rho_bounds.patch.txt`
   - `lambdas(..., fallback=)` from `V4_AUDIT.md`: unseen clubs now fall back to the newcomer
     prior or the league-mean att/def instead of (0, 0).

   `price4.py` and `shots4.py` call both, so they raised a TypeError against the promoted file.
   Default behaviour is unchanged. `lambdas_joint` still sends ungrouped unseen clubs to 0, as
   V4_AUDIT noted; that fix is still open.
3. `markets.build_prediction()` gained `shots=None`. When shots are passed they are written as
   the prediction's `shots` field, and when they are absent the field is left out, so the card
   keeps its placeholder.

## Running the tests

```
pip install numpy pandas scipy
python pipeline/test_shots.py
```

The research notes (`V4_NOTES*.md`, `V4_AUDIT.md`) stay in the Drive folder.
