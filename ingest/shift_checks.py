"""Is the goalie on? Play-by-play vs shift chart, stdlib only (CI tests it).

The play-by-play's situationCode says at every event whether each net has a
goalie: four digits, away goalie, away skaters, home skaters, home goalie
("1551" is 5-on-5 with both goalies in; "1560" is the home net empty). A shift
covers second t when start < t <= end, the rule export/profiles.py uses.
ingest/fix_shifts.py uses these to find goalies the shift chart lost.
"""
from __future__ import annotations

# Events that sit on a shift change, where start < t <= end can go either way.
EDGE_EVENTS = {"period-start", "period-end", "game-end", "shootout-complete",
               "faceoff", "stoppage", "game-official"}


def play_second(play: dict) -> tuple[int, int]:
    """(period, game_seconds) for a play-by-play event: 1200 * (period - 1) + elapsed."""
    period = int(play["periodDescriptor"]["number"])
    m, s = play["timeInPeriod"].split(":")
    return period, 1200 * (period - 1) + int(m) * 60 + int(s)


def goalie_in_seconds(plays: list[dict], side: str) -> list[int]:
    """Game seconds of the non-edge events in periods 1-4 where situationCode says
    `side`'s ('home' or 'away') goalie is in net."""
    pos = 3 if side == "home" else 0
    out = []
    for p in plays:
        code = p.get("situationCode")
        if p.get("typeDescKey") in EDGE_EVENTS or not code or len(code) != 4:
            continue
        period, t = play_second(p)
        if period >= 5:
            continue
        if code[pos] == "1":
            out.append(t)
    return out


def uncovered(seconds: list[int], intervals: list[tuple[int, int]]) -> list[int]:
    """The seconds that no (start, end) interval covers, start < t <= end."""
    return [t for t in seconds if not any(a < t <= b for a, b in intervals)]
