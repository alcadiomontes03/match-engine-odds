# Real xG test, walk-forward (2023-08 to 2026-05)

4,116 matches with a closing price (98.6% have Understat xG in the archive); 28-day refits; lower is better. Gap = model minus closing market. 95% intervals resample whole 28-day folds.

## 1. xG as a rating target (all with the walk-forward stretch)

| Ratings | Gap to market | Change vs v9 ratings | 95% |
|---|---|---|---|
| v8_live | +0.0224 | +0.0088 | +0.0045 to +0.0131 |
| fix_nov11 | +0.0211 | +0.0075 | +0.0037 to +0.0113 |
| goals_xg | +0.0197 | +0.0061 | +0.0025 to +0.0095 |
| xg_only_x2 | +0.0170 | +0.0034 | -0.0002 to +0.0069 |
| xg_only_x6 | +0.0180 | +0.0044 | -0.0010 to +0.0096 |
| mkt_x10_prom5 | +0.0136 | +0.0000 | +0.0000 to +0.0000 |
| mkt85_xg15_x10 | +0.0122 | -0.0013 | -0.0025 to -0.0003 |
| mkt70_xg30_x10 | +0.0119 | -0.0017 | -0.0039 to +0.0004 |

Real xG against the shot proxy, same method otherwise (goals_xg minus fix_nov11): -0.0014 (-0.0024 to -0.0004).

## 2. xG form as an add-on (+xgform minus +stretch on the same ratings, matches that have the feature)

| Ratings | Matches | Change | 95% | Median fitted coefficient |
|---|---|---|---|---|
| v8_live | 3,966 | -0.0003 | -0.0021 to +0.0016 | +0.121 |
| fix_nov11 | 3,966 | -0.0004 | -0.0023 to +0.0015 | +0.126 |
| goals_xg | 3,966 | -0.0001 | -0.0017 to +0.0016 | +0.109 |
| xg_only_x2 | 3,966 | +0.0005 | -0.0006 to +0.0017 | +0.064 |
| xg_only_x6 | 3,966 | +0.0001 | -0.0008 to +0.0010 | -0.066 |
| mkt_x10_prom5 | 3,966 | -0.0018 | -0.0044 to +0.0008 | +0.182 |
| mkt85_xg15_x10 | 3,966 | -0.0004 | -0.0023 to +0.0015 | +0.128 |
| mkt70_xg30_x10 | 3,966 | +0.0004 | -0.0008 to +0.0017 | +0.073 |

## v9 ratings + xG form, by league (change vs v9 ratings, matches that have the feature)

| League | Matches | Change | 95% |
|---|---|---|---|
| E0 | 1,101 | -0.0029 | -0.0076 to +0.0026 |
| SP1 | 1,104 | -0.0014 | -0.0057 to +0.0030 |
| D1 | 876 | -0.0027 | -0.0077 to +0.0022 |
| F1 | 885 | -0.0001 | -0.0063 to +0.0061 |
