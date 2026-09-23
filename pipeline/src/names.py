"""
Match Engine v4 — team-name reconciliation.

Canonical names = football-data.co.uk / xgabora short names (what the league fits
use). openfootball cup names ("Manchester City", "FC Lugano (SUI)") are mapped onto
them within the same country. Clubs from countries with no domestic data keep
their openfootball name, prefixed with the country code, and are rated from their
European matches plus a prior.
"""
from __future__ import annotations
import difflib
import re
import unicodedata

DIV_COUNTRY = {
    **{d: "ENG" for d in ["E0", "E1", "E2", "E3", "EC"]},
    **{d: "SCO" for d in ["SC0", "SC1", "SC2", "SC3"]},
    "D1": "GER", "D2": "GER", "I1": "ITA", "I2": "ITA", "SP1": "ESP", "SP2": "ESP",
    "F1": "FRA", "F2": "FRA", "N1": "NED", "B1": "BEL", "P1": "POR", "T1": "TUR",
    "G1": "GRE", "SWE": "SWE", "NOR": "NOR", "IRL": "IRL", "RUS": "RUS", "POL": "POL",
    "DEN": "DEN", "ROM": "ROU", "AUT": "AUT", "SUI": "SUI", "FIN": "FIN",
}
# clubs that play in another country's league
PLAYS_IN = {"MCO": "FRA", "LIE": "SUI"}  # (Welsh UEFA entrants play in the Welsh league)

# xgabora internal inconsistencies
DOM_FIX = {"Nottm Forest": "Nott'm Forest", "King\x92s Lynn": "King's Lynn"}

ALIASES = {  # openfootball name -> canonical, where fuzzy matching is unreliable
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Nottingham Forest": "Nott'm Forest", "Sheffield Wednesday": "Sheffield Weds",
    "Wolverhampton Wanderers": "Wolves", "West Bromwich Albion": "West Brom",
    "Queens Park Rangers": "QPR", "Brighton & Hove Albion": "Brighton",
    "Tottenham Hotspur": "Tottenham", "Newcastle United": "Newcastle",
    "Peterborough United": "Peterboro", "Bristol Rovers": "Bristol Rvs",
    "Dagenham & Redbridge": "Dag and Red", "Boston United": "Boston Utd",
    "AFC Wimbledon": "AFC Wimbledon", "Sheffield United": "Sheffield United",
    "West Ham United": "West Ham", "Leeds United": "Leeds", "Leicester City": "Leicester",
    "Oxford City": "Oxford City", "Oxford United": "Oxford",
    "Bayern München": "Bayern Munich", "Borussia Mönchengladbach": "M'gladbach",
    "Paris Saint-Germain": "Paris SG", "Internazionale": "Inter", "Inter Milan": "Inter",
    "Atlético Madrid": "Ath Madrid", "Athletic Club": "Ath Bilbao",
    "Sporting CP": "Sp Lisbon", "Sporting Lisboa": "Sp Lisbon", "Sporting Braga": "Sp Braga",
    "SC Braga": "Sp Braga", "Real Betis": "Betis", "Real Sociedad": "Sociedad",
    "Celta Vigo": "Celta", "Bayer 04 Leverkusen": "Leverkusen", "Bayer Leverkusen": "Leverkusen",
    "Eintracht Frankfurt": "Ein Frankfurt", "1. FC Union Berlin": "Union Berlin",
    "FC København": "FC Copenhagen", "FC Copenhagen": "FC Copenhagen",
    "PSV Eindhoven": "PSV Eindhoven", "AZ Alkmaar": "AZ Alkmaar",
    "Crvena Zvezda": "Crvena Zvezda",
    "Paris Saint-Germain FC": "Paris SG", "Club Atlético de Madrid": "Ath Madrid",
    "Stade Rennais": "Rennes", "Union Saint-Gilloise": "St. Gilloise",
    "İstanbul Başakşehir": "Buyuksehyr", "Sporting Clube de Braga": "Sp Braga",
    "Sporting Clube de Portugal": "Sp Lisbon", "FC Internazionale Milano": "Inter",
    "Lazio Roma": "Lazio", "Spartak Moskva": "Spartak Moscow",
    "Heart of Midlothian": "Hearts", "Solihull Moors": "Solihull",
    "Rapid Wien": "SK Rapid", "Olympique Lyonnais": "Lyon", "Stade Brestois 29": "Brest",
    "Lokomotiv Moskva": "Lokomotiv Moscow", "Austria Wien": "Austria Vienna", "PSV": "PSV Eindhoven", "FK Bodø/Glimt": "Bodo/Glimt",
}

_AFFIX = re.compile(r"^(?:FC|AFC|CF|SC|FK|SK|NK|AC|AS)\s+|\s+(?:FC|AFC|CF|SC|FK|SK|NK|KV)$")

_STOP = {"fc", "afc", "cf", "sc", "fk", "sk", "nk", "ac", "as", "ss", "sv", "if", "bk",
         "ff", "cd", "ud", "rc", "rcd", "sd", "ogc", "osc", "losc", "vfb", "vfl", "tsg",
         "bsc", "fsv", "gnk", "hnk", "pfc", "ofk", "fci", "1", "calcio", "club", "de", "the", "town", "city", "united",
         "utd", "athletic", "rovers", "county", "hotspur", "wanderers", "albion"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = [t for t in s.split() if t not in _STOP and not re.fullmatch(r"\d{2,4}", t)]
    return " ".join(toks)


def build_map(cup_names_by_country: dict, dom_names_by_country: dict, cutoff=0.88):
    """Returns ({(country, of_name): canonical}, [unmatched (country, of_name)])."""
    mapping, unmatched = {}, []
    for ctry, names in cup_names_by_country.items():
        league_ctry = PLAYS_IN.get(ctry, ctry)
        pool = dom_names_by_country.get(league_ctry)
        for n in names:
            bare = _AFFIX.sub("", n).strip()
            alias = ALIASES.get(n) or ALIASES.get(bare)
            if alias and (pool is None or alias in pool):
                mapping[(ctry, n)] = alias; continue
            if not pool:
                # no domestic data: key on the normalised name so "FK Astana" and
                # "Astana" become one club
                mapping[(ctry, n)] = f"{ctry}:{norm(n) or n}"; continue
            if n in pool or bare in pool:
                mapping[(ctry, n)] = n if n in pool else bare; continue
            nn = norm(n); normed = {norm(p): p for p in pool}
            if nn in normed:
                mapping[(ctry, n)] = normed[nn]; continue
            # whole-token containment (domestic short name's tokens all appear in the
            # cup name), then strict fuzzy. Substring matching mapped
            # "Manchester United" -> "Man City" and "Chester" -> "Colchester".
            nt = set(nn.split())
            cands = [p for k, p in normed.items() if k and set(k.split()) <= nt]
            if len(cands) == 1:
                mapping[(ctry, n)] = cands[0]; continue
            best = difflib.get_close_matches(nn, list(normed), n=1, cutoff=cutoff)
            if best:
                mapping[(ctry, n)] = normed[best[0]]; continue
            unmatched.append((ctry, n))
            mapping[(ctry, n)] = f"{ctry}:{n}"
    return mapping, unmatched
