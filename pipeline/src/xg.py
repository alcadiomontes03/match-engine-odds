"""
xG (expected goals) overlay — Premier League only, for now.

Coverage note: this session tried three free xG sources before finding one that
actually verifies against the current season:
  - understat.com: blocks WebFetch outright (robots.txt disallows it — even
    fetching /robots.txt itself was refused).
  - fbref.com: first attempt returned a plausible-looking table that turned out
    to be for a different, older season (teams like Watford/Norwich/Leicester/
    Sheffield Utd that aren't in the 2026-27 top flight) — discarded rather than
    trusted. A retry with an explicit season URL got a 403.
  - footystats.org: same failure mode as the first fbref attempt — returned a
    table with Leicester/Southampton/Ipswich/Wolves in the Premier League, which
    is last-but-one season's cast, not this one. Discarded.
  - xgstat.com: verified — https://www.xgstat.com/competitions/premier-league/2026-2027/standings
    returned a team list matching the real current promoted/relegated set (Hull,
    Sunderland, Leeds, Coventry present; no Leicester/Southampton/Ipswich/Watford)
    with a plausible small match count for early September. This is the source
    wired up below. It has NO La Liga or Championship coverage — checked its own
    nav/footer, which lists Premier League only — so those two leagues still have
    no xG signal.

This is intentionally NOT blended into the Dixon-Coles fit itself (that would
need real per-match shot-level xG, which none of these team-standings pages
give — they're season-to-date aggregates). It's surfaced as a separate
over/underperformance check: teams whose actual points/goal difference are
running well ahead of their xG-implied points/GD are candidates for point-total
regression, which is exactly the kind of thing worth flagging next to a
results-only model without pretending it's been formally incorporated.

## Fetch procedure (assistant does this each pipeline run; see loader.py's
## docstring for why Python itself can't reach the URL)

WebFetch https://www.xgstat.com/competitions/premier-league/2026-2027/standings
(bump the season in the URL each year) with this prompt:
  "First tell me what season and how many matches played this page shows, and
  list 6 team names, so I can verify it's the current season and not a stale/
  cached page. Then extract the full team xG standings table as CSV using the
  actual column headers shown. If this requires login or shows an error/
  different season, say so explicitly."
Verify the team list against the season's actual promoted/relegated clubs
(cross-check a couple of names against the football-data.co.uk E0 file already
fetched this run) before trusting it — that check is what caught fbref and
footystats serving the wrong season, and it costs nothing to repeat every time.
Save the parsed table to data/raw/E0_xg.csv, columns:
Rank,xRank,Team,Pts,xPts,DeltaPts,P,W,D,L,GF,GA,GD,xGF,xGA,xGD,DeltaGD,Form
using football-data.co.uk's short team names (Man City not Manchester City,
Nott'm Forest not Nottingham Forest, etc — see loader.py's teams for the
convention) so the keys line up with the Dixon-Coles ratings dict.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_xg(code: str) -> dict | None:
    """Returns {team: {pts, xpts, delta_pts, gf, ga, gd, xgf, xga, xgd, matches_played}}
    or None if no xG file exists for this league code yet."""
    p = RAW_DIR / f"{code}_xg.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    out = {}
    for _, r in df.iterrows():
        out[r["Team"]] = {
            "matches_played": int(r["P"]),
            "pts": int(r["Pts"]), "xpts": float(r["xPts"]), "delta_pts": float(r["DeltaPts"]),
            "gf": int(r["GF"]), "ga": int(r["GA"]), "gd": int(r["GD"]),
            "xgf": float(r["xGF"]), "xga": float(r["xGA"]), "xgd": float(r["xGD"]),
        }
    return out
