# Match Engine — odds capture

Scheduled two-pull odds capture from The Odds API (free tier, 500 credits/month).

- `capture_odds.py` — the script (standard-library Python only)
- `.github/workflows/odds.yml` — runs it every 15 minutes
- `data/odds/snapshots.csv` — every captured price, one row per outcome
- `data/odds/raw/` — raw API responses
- `data/odds/quota.csv` — credits remaining after each paid call
- `state/pulled.json` — which matches already had their slate / late pull

The API key lives only in the repository secret `ODDS_API_KEY`.
Nothing in this repo contains it.

Settings (Settings → Secrets and variables → Actions → Variables):
- `ODDS_REGIONS` — default `eu`; set from the probe result
- `INCLUDE_CUPS` — `true` to add FA Cup and Conference League
