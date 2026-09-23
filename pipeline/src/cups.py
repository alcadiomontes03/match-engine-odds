"""
Match Engine v4 — cup competition loader (FA Cup, EFL Cup, UEFA CL/EL/Conference League).

Source: github.com/openfootball (results only, no odds)
  england/master/<YYYY-YY>/facup.txt    FA Cup from Round 1 proper
  england/master/<YYYY-YY>/eflcup.txt   EFL Cup (League Cup / Carabao Cup) from Round 1
  champions-league/master/<YYYY-YY>/{cl,el,conf}.txt   group/league phase onward
Qualifying rounds are NOT in these files.

Betting markets settle on 90 minutes, so every match is stored with its
90-minute score: for "3-1 a.e.t. (1-1, 1-0)" the 90-minute score is 1-1.
A single-leg tie (FA Cup or EFL Cup) that went to extra time must have been
level after 90; rows failing that check are dropped rather than guessed.

EFL Cup semi-finals are two-legged (home and away), but openfootball already
lists each leg as its own separate match row (not an aggregate score), so no
special two-leg handling is needed beyond treating them like any other round
— unlike a UEFA-style aggregate line, there is nothing to unpack. Only the
EFL Cup Final is at a neutral venue (Wembley); the semi-final legs are played
at each club's own ground.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

EXT = Path(__file__).resolve().parent.parent / "data" / "ext"
MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
DATE_RE = re.compile(r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
MATCH_RE = re.compile(r"^\s*(?:\d{1,2}:\d{2}\s+)?(?P<h>.+?)\s+v\s+(?P<a>.+?)\s+(?P<s>\d+-\d+.*)$")
TEAM_RE = re.compile(r"^(?P<n>.+?)\s*\((?P<c>[A-Z]{3})\)$")
PAIR_RE = re.compile(r"(\d+)-(\d+)")
# Only stages actually played at a neutral venue. EFL Cup semis are home/away
# legs at each club's own ground, so "Semifinals" is deliberately absent here.
NEUTRAL_STAGES = {"facup": {"Semifinals", "Final"}, "eflcup": {"Final"}, "uefa": {"Final"}}
# Single-leg competitions: a tie that reaches extra time must have been level
# after 90 minutes, so the (h, a) pair inside the "a.e.t." parenthesis is the
# real 90-minute score. Two-legged competitions (UEFA ties reported as an
# aggregate line) don't have that constraint — but EFL Cup's two legs are
# already separate rows, so it belongs in this set too, not with UEFA.
SINGLE_LEG = {"FAC", "EFC"}


def _split_team(t):
    m = TEAM_RE.match(t.strip())
    return (m["n"].strip(), m["c"]) if m else (t.strip(), "ENG")


def _ninety(s, single_leg):
    main = PAIR_RE.search(s)
    if "a.e.t" in s:
        paren = s[s.find("("):] if "(" in s else ""
        p = PAIR_RE.search(paren)
        # single-leg ties only reach extra time from a level 90 minutes;
        # two-legged UEFA aggregate ties can reach it from any second-leg score
        if not p or (single_leg and p[1] != p[2]):
            return None
        return int(p[1]), int(p[2]), True
    return int(main[1]), int(main[2]), False


def parse(path: Path, comp: str, season: str):
    y0 = int(season[:4]); rows = []; stage = ""; cur = None
    kind = "facup" if comp == "FAC" else ("eflcup" if comp == "EFC" else "uefa")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("▪"):
            stage = line[1:].strip(); continue
        d = DATE_RE.match(line)
        if d:
            mo = MONTHS[d[1]]
            yr = int(d[3]) if d[3] else (y0 if mo >= 7 else y0 + 1)
            cur = pd.Timestamp(yr, mo, int(d[2])); continue
        m = MATCH_RE.match(line)
        if not m or cur is None:
            continue
        sc = _ninety(m["s"], single_leg=(comp in SINGLE_LEG))
        if sc is None:
            continue
        (h, hc), (a, ac) = _split_team(m["h"]), _split_team(m["a"])
        rows.append({"Date": cur, "Comp": comp, "Season": season, "Stage": stage,
                     "HomeTeam": h, "AwayTeam": a, "HomeCountry": hc, "AwayCountry": ac,
                     "FTHG": sc[0], "FTAG": sc[1], "WentET": sc[2],
                     "Neutral": stage in NEUTRAL_STAGES[kind]})
    return pd.DataFrame(rows)


def load_cups(seasons=("2021-22", "2022-23", "2023-24", "2024-25")):
    fr = []
    for s in seasons:
        for comp, fn in [("FAC", f"facup_{s}.txt"), ("EFC", f"eflcup_{s}.txt"),
                         ("UCL", f"cl_{s}.txt"), ("UEL", f"el_{s}.txt"), ("UECL", f"conf_{s}.txt")]:
            p = EXT / fn
            if p.exists():
                fr.append(parse(p, comp, s))
    df = pd.concat(fr, ignore_index=True)
    df["FTR"] = (df.FTHG > df.FTAG).map({True: "H"}).fillna((df.FTHG < df.FTAG).map({True: "A"})).fillna("D")
    return df.sort_values("Date").reset_index(drop=True)
