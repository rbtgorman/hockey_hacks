"""Metric registry for the team and player profiles, standard library only.

export/profiles.py does the database work and fills one dict of counts per
team-season, skater-season and goalie-season (the COUNTS lists below). Every
number on site/team.html and site/player.html is a METRICS entry computed from
those counts, and the pages draw whatever this list holds. Adding a metric
later (say an NHL EDGE stat once those tables fill) is one entry here, plus the
count it needs in profiles.py if that count isn't already there.

A METRICS entry:
    key       stable id, used in the JSON
    entity    'team' | 'skater' | 'goalie'
    group     heading the page puts it under
    label     short name on the page
    define    one plain-English sentence; the page shows it under the number
    fmt       'int' | 'dec1' | 'dec2' | 'pct' | 'signed' | 'min'
    sample    the count this number stands on (shown next to it)
    min       below this sample the page greys the number out
    league    'pooled' = the metric computed on everyone's counts added up,
              'mean' = average of every team's (or player's) value, None = no
              league figure (plain counts like games played)
    better    'high' | 'low' | None, for colouring against the league figure
    calc      counts -> value, or None when it can't be computed

Like export/period_math.py, this file imports nothing outside the standard
library so tests/test_profile_metrics.py runs in CI with no dependencies.
"""
from __future__ import annotations

import math

SCHEMA_VERSION = 1

# ------------------------------------------------------------- strength
def parse_strength(state: str | None) -> tuple[int, int] | None:
    """'5v4' -> (5, 4): skaters for the shooting team, then the other team.

    shots.strength_state is from the shooter's side. Penalty shots come
    through as '1v0' / '0v1'.
    """
    if not state or "v" not in state:
        return None
    a, _, b = state.partition("v")
    if not (a.isdigit() and b.isdigit()):
        return None
    return int(a), int(b)


def strength_bucket(state: str | None) -> str | None:
    """'5v5', 'pp' (shooting team has more skaters), 'pk' (fewer), or None.

    None covers 4v4, 3v3, an extra attacker for a pulled goalie (6v5, 5v6),
    penalty shots and anything unreadable: those count in 'all situations'
    only. A pulled goalie isn't a power play, so 6v5 is not 'pp'.
    """
    s = parse_strength(state)
    if s is None:
        return None
    a, b = s
    if a == 5 and b == 5:
        return "5v5"
    if not (3 <= a <= 5 and 3 <= b <= 5):
        return None
    if a > b:
        return "pp"
    if a < b:
        return "pk"
    return None


# ----------------------------------------------------------------- zones
# Where an unblocked shot came from, net at (89, 0), shooter attacking +x.
# Rules apply in order; the first match wins.
ZONES = [
    ("behind", "Behind the net", "From behind the goal line."),
    ("crease", "Crease", "Within 10 feet of the net."),
    ("slot", "Inner slot", "Within 30 feet, and no more than 30 degrees off the middle."),
    ("low", "Down low, wide", "Within 30 feet but off to the side, below the faceoff dots."),
    ("high_slot", "High slot", "30 to 50 feet out, no more than 25 degrees off the middle."),
    ("circles", "Faceoff circles", "30 to 50 feet out and off to the side."),
    ("point", "Point", "Farther than 50 feet but still inside the blue line."),
    ("long", "Outside the zone", "From the blue line or farther back."),
]
ZONE_KEYS = [z[0] for z in ZONES]
BLUE_LINE_X = 25.0
GOAL_LINE_X = 89.0


def shot_zone(x: float | None, distance: float | None, angle: float | None) -> str | None:
    """Zone key for one shot, or None if its location is missing."""
    if x is None or distance is None or angle is None:
        return None
    if any(isinstance(v, float) and math.isnan(v) for v in (x, distance, angle)):
        return None
    if x > GOAL_LINE_X:
        return "behind"
    if distance <= 10:
        return "crease"
    if distance <= 30:
        return "slot" if angle <= 30 else "low"
    if distance <= 50:
        return "high_slot" if angle <= 25 else "circles"
    if x >= BLUE_LINE_X:
        return "point"
    return "long"


# ---------------------------------------------------------------- counts
# What profiles.py fills for every entity-season. TOI fields are minutes.
TEAM_COUNTS = [
    "gp", "gf", "ga", "xgf", "xga", "af", "aa",
    "gf_5v5", "ga_5v5", "xgf_5v5", "xga_5v5", "toi_5v5",
    "gf_pp", "xgf_pp", "toi_pp",
    "ga_pk", "xga_pk", "toi_pk",
]
SKATER_COUNTS = [
    "gp", "toi", "toi_5v5", "att", "goals", "ixg",
    "on_gf_5v5", "on_ga_5v5", "on_xgf_5v5", "on_xga_5v5",
]
GOALIE_COUNTS = ["gp", "toi", "sa", "ga", "xga"]
COUNTS = {"team": TEAM_COUNTS, "skater": SKATER_COUNTS, "goalie": GOALIE_COUNTS}


def div(a: float, b: float) -> float | None:
    return None if not b else a / b


def per60(x: float, minutes: float) -> float | None:
    return None if not minutes else x * 60.0 / minutes


def share(a: float, b: float) -> float | None:
    return None if not (a + b) else a / (a + b)


def _m(key, entity, group, label, define, fmt, sample, min_sample, league, better, calc):
    return {"key": key, "entity": entity, "group": group, "label": label,
            "define": define, "fmt": fmt, "sample": sample, "min": min_sample,
            "league": league, "better": better, "calc": calc}


METRICS = [
    # ------------------------------------------------------------ teams
    _m("gp", "team", "Basics", "Games", "Regular-season games played.",
       "int", "gp", 0, None, None, lambda c: c["gp"]),
    _m("gf_pg", "team", "Basics", "Goals per game",
       "Goals scored per game, every situation, overtime included. Shootouts don't count.",
       "dec2", "gp", 10, "pooled", "high", lambda c: div(c["gf"], c["gp"])),
    _m("ga_pg", "team", "Basics", "Goals against per game",
       "Goals allowed per game, same rules.", "dec2", "gp", 10, "pooled", "low",
       lambda c: div(c["ga"], c["gp"])),
    _m("xgf_pg", "team", "Shot quality", "xG per game",
       "Expected goals from the team's unblocked shots, per game. How many goals its chances were worth.",
       "dec2", "gp", 10, "pooled", "high", lambda c: div(c["xgf"], c["gp"])),
    _m("xga_pg", "team", "Shot quality", "xG against per game",
       "Expected goals from the other team's unblocked shots, per game.",
       "dec2", "gp", 10, "pooled", "low", lambda c: div(c["xga"], c["gp"])),
    _m("xgf_pct", "team", "Shot quality", "Share of the xG",
       "Of all the expected goals in its games, the share that came from this team's shots. 50% is even.",
       "pct", "gp", 10, "pooled", "high", lambda c: share(c["xgf"], c["xga"])),
    _m("g_minus_xg", "team", "Shot quality", "Goals minus xG",
       "Goals scored minus expected goals. Above zero means it scored more than its chances were worth: good shooters, or good luck.",
       "signed", "gp", 10, "mean", "high", lambda c: c["gf"] - c["xgf"]),
    _m("ga_minus_xga", "team", "Shot quality", "Goals against minus xG against",
       "Goals allowed minus expected goals allowed. Below zero means its goalies stopped more than the chances said they would.",
       "signed", "gp", 10, "mean", "low", lambda c: c["ga"] - c["xga"]),
    _m("xgf60_5v5", "team", "5-on-5", "xG per 60",
       "Expected goals for every 60 minutes at 5-on-5.", "dec2", "toi_5v5", 300,
       "pooled", "high", lambda c: per60(c["xgf_5v5"], c["toi_5v5"])),
    _m("xga60_5v5", "team", "5-on-5", "xG against per 60",
       "Expected goals against for every 60 minutes at 5-on-5.", "dec2", "toi_5v5", 300,
       "pooled", "low", lambda c: per60(c["xga_5v5"], c["toi_5v5"])),
    _m("xgf_pct_5v5", "team", "5-on-5", "Share of the xG",
       "Share of the 5-on-5 expected goals that came from this team's shots.",
       "pct", "toi_5v5", 300, "pooled", "high", lambda c: share(c["xgf_5v5"], c["xga_5v5"])),
    _m("gf_pct_5v5", "team", "5-on-5", "Share of the goals",
       "Share of the 5-on-5 goals this team scored. Next to the xG share, it shows how much the bounces helped or hurt.",
       "pct", "toi_5v5", 300, "pooled", "high", lambda c: share(c["gf_5v5"], c["ga_5v5"])),
    _m("toi_pp", "team", "Special teams", "Power-play minutes",
       "Minutes with more skaters on the ice than the other team.", "min", "toi_pp", 0,
       "mean", None, lambda c: c["toi_pp"]),
    _m("xgf60_pp", "team", "Special teams", "Power play: xG per 60",
       "Expected goals for every 60 minutes of power play.", "dec2", "toi_pp", 60,
       "pooled", "high", lambda c: per60(c["xgf_pp"], c["toi_pp"])),
    _m("gf60_pp", "team", "Special teams", "Power play: goals per 60",
       "Goals for every 60 minutes of power play.", "dec2", "toi_pp", 60,
       "pooled", "high", lambda c: per60(c["gf_pp"], c["toi_pp"])),
    _m("toi_pk", "team", "Special teams", "Penalty-kill minutes",
       "Minutes with fewer skaters on the ice than the other team.", "min", "toi_pk", 0,
       "mean", None, lambda c: c["toi_pk"]),
    _m("xga60_pk", "team", "Special teams", "Penalty kill: xG against per 60",
       "Expected goals allowed for every 60 minutes of penalty kill.", "dec2", "toi_pk", 60,
       "pooled", "low", lambda c: per60(c["xga_pk"], c["toi_pk"])),
    _m("ga60_pk", "team", "Special teams", "Penalty kill: goals against per 60",
       "Goals allowed for every 60 minutes of penalty kill.", "dec2", "toi_pk", 60,
       "pooled", "low", lambda c: per60(c["ga_pk"], c["toi_pk"])),

    # ---------------------------------------------------------- skaters
    _m("gp", "skater", "Basics", "Games", "Regular-season games with at least one shift.",
       "int", "gp", 0, None, None, lambda c: c["gp"]),
    _m("toi_pg", "skater", "Basics", "Minutes per game", "Average ice time per game, every situation.",
       "dec1", "gp", 5, "pooled", None, lambda c: div(c["toi"], c["gp"])),
    _m("goals", "skater", "Shooting", "Goals", "Goals scored, every situation. Shootouts don't count.",
       "int", "gp", 0, None, None, lambda c: c["goals"]),
    _m("ixg", "skater", "Shooting", "xG", "Expected goals from the player's own unblocked shots: what those chances were worth.",
       "dec1", "gp", 0, None, None, lambda c: c["ixg"]),
    _m("g_minus_ixg", "skater", "Shooting", "Goals minus xG",
       "Goals minus the player's own expected goals. A big number either way over one season is mostly luck; over several it starts to mean finishing skill.",
       "signed", "att", 50, None, "high", lambda c: c["goals"] - c["ixg"]),
    _m("att60", "skater", "Shooting", "Unblocked shots per 60",
       "The player's unblocked shots for every 60 minutes played.", "dec2", "toi", 100,
       "pooled", "high", lambda c: per60(c["att"], c["toi"])),
    _m("ixg60", "skater", "Shooting", "xG per 60",
       "The player's own expected goals for every 60 minutes played.", "dec2", "toi", 100,
       "pooled", "high", lambda c: per60(c["ixg"], c["toi"])),
    _m("goals60", "skater", "Shooting", "Goals per 60",
       "Goals for every 60 minutes played.", "dec2", "toi", 100,
       "pooled", "high", lambda c: per60(c["goals"], c["toi"])),
    _m("toi_5v5", "skater", "On the ice at 5-on-5", "5-on-5 minutes",
       "Minutes played with five skaters and a goalie on each side.", "min", "toi_5v5", 0,
       None, None, lambda c: c["toi_5v5"]),
    _m("on_xgf_pct_5v5", "skater", "On the ice at 5-on-5", "Team's share of the xG",
       "With the player on the ice at 5-on-5, the share of the expected goals that came from the player's team. 50% is even.",
       "pct", "toi_5v5", 100, "pooled", "high",
       lambda c: share(c["on_xgf_5v5"], c["on_xga_5v5"])),
    _m("on_xgf60_5v5", "skater", "On the ice at 5-on-5", "Team xG per 60",
       "The player's team's expected goals for every 60 minutes with the player on the ice at 5-on-5.",
       "dec2", "toi_5v5", 100, "pooled", "high", lambda c: per60(c["on_xgf_5v5"], c["toi_5v5"])),
    _m("on_xga60_5v5", "skater", "On the ice at 5-on-5", "Team xG against per 60",
       "The other team's expected goals for every 60 minutes with the player on the ice at 5-on-5.",
       "dec2", "toi_5v5", 100, "pooled", "low", lambda c: per60(c["on_xga_5v5"], c["toi_5v5"])),
    _m("on_gf_pct_5v5", "skater", "On the ice at 5-on-5", "Team's share of the goals",
       "Of the 5-on-5 goals scored with the player on the ice, the share the player's team scored.",
       "pct", "toi_5v5", 100, "pooled", "high",
       lambda c: share(c["on_gf_5v5"], c["on_ga_5v5"])),

    # ---------------------------------------------------------- goalies
    _m("gp", "goalie", "Basics", "Games", "Regular-season games played in net.",
       "int", "gp", 0, None, None, lambda c: c["gp"]),
    _m("toi", "goalie", "Basics", "Minutes", "Minutes in net.", "min", "toi", 0,
       None, None, lambda c: c["toi"]),
    _m("sa", "goalie", "Shots faced", "Unblocked shots faced",
       "Shots on goal plus misses faced while in net. Empty-net shots don't count.",
       "int", "sa", 0, None, None, lambda c: c["sa"]),
    _m("xga", "goalie", "Shots faced", "xG faced",
       "What those shots were worth in expected goals.", "dec1", "sa", 0,
       None, None, lambda c: c["xga"]),
    _m("ga", "goalie", "Shots faced", "Goals allowed", "Goals allowed while in net.", "int", "sa", 0,
       None, None, lambda c: c["ga"]),
    _m("gsax", "goalie", "Stopping them", "Goals saved above expected",
       "xG faced minus goals allowed. Above zero means more stops than an average goalie would have made on the same shots.",
       "signed", "sa", 300, None, "high", lambda c: c["xga"] - c["ga"]),
    _m("gsax100", "goalie", "Stopping them", "Goals saved above expected per 100 shots",
       "The same thing, for every 100 unblocked shots, so busy and quiet goalies compare fairly.",
       "signed", "sa", 300, "pooled", "high", lambda c: None if not c["sa"] else 100.0 * (c["xga"] - c["ga"]) / c["sa"]),
    _m("fsv", "goalie", "Stopping them", "Save rate on unblocked shots",
       "Share of the unblocked shots faced that didn't go in. Misses count as stops, so this runs higher than save percentage.",
       "pct", "sa", 300, "pooled", "high", lambda c: None if not c["sa"] else 1.0 - c["ga"] / c["sa"]),
    _m("xfsv", "goalie", "Stopping them", "Expected save rate",
       "What that save rate would have been if goals went in exactly as often as xG says.",
       "pct", "sa", 300, "pooled", None, lambda c: None if not c["sa"] else 1.0 - c["xga"] / c["sa"]),
]


def metrics_for(entity: str) -> list[dict]:
    return [m for m in METRICS if m["entity"] == entity]


def compute(entity: str, counts: dict) -> dict:
    """{metric key: value} for one entity-season."""
    return {m["key"]: m["calc"](counts) for m in metrics_for(entity)}


def league_values(entity: str, count_rows: list[dict]) -> dict:
    """{metric key: league figure} over every entity-season in count_rows."""
    if not count_rows:
        return {}
    total = {k: sum(r[k] for r in count_rows) for k in COUNTS[entity]}
    out = {}
    for m in metrics_for(entity):
        if m["league"] == "pooled":
            out[m["key"]] = m["calc"](total)
        elif m["league"] == "mean":
            vals = [m["calc"](r) for r in count_rows]
            vals = [v for v in vals if v is not None]
            out[m["key"]] = sum(vals) / len(vals) if vals else None
    return out


def registry_json() -> dict:
    """The registry as the pages read it: everything but calc."""
    return {
        entity: [{k: v for k, v in m.items() if k not in ("calc", "entity")}
                 for m in metrics_for(entity)]
        for entity in COUNTS
    }
