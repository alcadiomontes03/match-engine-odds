# Match Engine — odds capture

Scheduled DraftKings odds capture from The Odds API (free tier, 500 credits a month) for the
Premier League, La Liga, Bundesliga and Ligue 1. The model that uses these prices is described in
[`docs/v6.md`](docs/v6.md) (current version) and [`pipeline/v5/README.md`](pipeline/v5/README.md).

Markets: moneyline, spreads and totals in one call per league per pull, billed 3 credits a call
(one per market requested). Both-teams-to-score is an "additional market" the API only serves per
match, so it is pulled once per match at its first late pull (1 credit), and only while more than
100 credits remain.

Cadence (the workflow runs every 15 minutes and spends credits only when a pull is due):
- **Weekly** - first run after Monday 15:00 UTC (8am Pacific in summer, 7am in winter):
  one call per league covering every match in the next 7 days. GitHub often starts it late
  (it landed at 17:01 UTC on 2026-09-28), so the Monday jobs that read it wait for it.
- **Late** - first run within 3 hours of kickoff, plus one refresh inside the last 65 minutes
  (after team news): DraftKings' near-closing line, used for CLV. At most two late pulls per
  match; one call covers every match in that league that is due in that run.
- Sharp closing lines (Pinnacle / market average) come free from football-data.co.uk
  (`fetch_fd.py`, Mondays 14:17 UTC), so no credits are spent on them.

Credit budget: a full matchweek in all four leagues can use roughly 100-140 credits
(12 for the weekly pulls, 3 per late call, 1 per match for BTTS), so a month with four or five
matchweeks can run past the 500-credit free tier. Guards: BTTS stops below 100 credits left
(`BTTS_RESERVE`) and every pull stops below 40 (`CREDIT_RESERVE`); late in a busy month that
can mean missing closing prices. `data/odds/quota.csv` logs the balance after each paid call.

- `capture_odds.py` - the script (standard-library Python only)
- `.github/workflows/odds.yml` - runs it every 15 minutes
- `data/odds/snapshots.csv` - every captured price, one row per outcome (`kind` is weekly, late or slate)
- `data/odds/raw/` - raw API responses
- `data/odds/quota.csv` - credits remaining after each paid call
- `state/pulled.json` - which matches already had their late pull, and (under `_meta`) which weeks had their weekly pull

The API key lives only in the repository secret `ODDS_API_KEY`.
Nothing in this repo contains it.

Settings:
- `ODDS_BOOKMAKERS` (`draftkings`), `ODDS_MARKETS` (`h2h,spreads,totals`) and `ODDS_BTTS` (`true`)
  are set in `.github/workflows/odds.yml`, so an old repository variable can't override them.
- Repository variables (Settings → Secrets and variables → Actions → Variables):
  - `INCLUDE_CUPS` - `true` to add the FA Cup and Conference League (off: cups are out of scope)
  - `ODDS_TEAM_FILTER` - optional JSON, e.g. `{"LaLiga": ["Real Madrid", "Barcelona"]}`;
    only matches involving a listed team are pulled for that league
