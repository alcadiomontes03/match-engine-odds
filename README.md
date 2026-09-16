# Match Engine — odds capture

Scheduled DraftKings odds capture from The Odds API (free tier, 500 credits/month).

Cadence (runs every 15 minutes, spends credits only when a pull is due):
- **Weekly** - first run after Monday 15:00 UTC (8am Pacific in summer):
  one call per league covering every match in the next 7 days.
- **Late** - 10-70 minutes before each kickoff, after team news:
  DraftKings' near-closing line, used for CLV.
- Sharp closing lines (Pinnacle / market average) come free from
  football-data.co.uk, so no credits are spent on them.

Expected use: about 285 credits/month across PL, La Liga and Championship.

- `capture_odds.py` — the script (standard-library Python only)
- `.github/workflows/odds.yml` — runs it every 15 minutes
- `data/odds/snapshots.csv` — every captured price, one row per outcome
- `data/odds/raw/` — raw API responses
- `data/odds/quota.csv` — credits remaining after each paid call
- `state/pulled.json` — which matches already had their late pull, and which weeks had their weekly pull

The API key lives only in the repository secret `ODDS_API_KEY`.
Nothing in this repo contains it.

Settings (Settings → Secrets and variables → Actions → Variables):
- `ODDS_BOOKMAKERS` — default `draftkings` (only DraftKings is pulled)
- `INCLUDE_CUPS` — `true` to add FA Cup and Conference League
- `ODDS_TEAM_FILTER` — optional JSON, e.g. `{"LaLiga": ["Real Madrid", "Barcelona"]}`;
  only matches involving a listed team are pulled for that league
