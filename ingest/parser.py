"""Parse NHL play-by-play JSON into DB-ready rows.

Event schema (observed in api-web.nhle.com/v1/gamecenter/{id}/play-by-play):

Top-level keys we use:
  id                 -> int game id
  season             -> int YYYYYYYY
  gameType           -> int (2=reg, 3=playoff)
  gameDate           -> 'YYYY-MM-DD'
  venue.default      -> str
  homeTeam.{id, abbrev, score}
  awayTeam.{id, abbrev, score}
  gameState          -> 'FINAL' | 'OFF' | 'LIVE' | 'FUT' | ...
  plays[]            -> list of event dicts

Each play:
  eventId            -> int, unique per game
  sortOrder          -> int, monotonic
  periodDescriptor.{number, periodType}
  timeInPeriod       -> 'MM:SS' elapsed
  timeRemaining      -> 'MM:SS'
  situationCode      -> 4-char string, e.g. '1551'
                        chars[0]=away goalie, chars[1]=away skaters,
                        chars[2]=home skaters, chars[3]=home goalie
                        (verified against known games; NHL doc is thin)
  typeDescKey        -> 'shot-on-goal' | 'goal' | 'missed-shot' | 'blocked-shot'
                        | 'faceoff' | 'hit' | 'giveaway' | 'takeaway' | 'penalty' | ...
  details            -> dict with event-specific fields:
    xCoord, yCoord   -> -100..100 x, -42.5..42.5 y (origin = center ice)
    eventOwnerTeamId -> team credited with event
    shootingPlayerId -> int (on shots)
    scoringPlayerId  -> int (on goals)
    goalieInNetId    -> int or missing (empty net)
    shotType         -> 'wrist' | 'snap' | 'slap' | 'backhand' | 'tip-in' | 'deflected' | 'wrap-around'
    zoneCode         -> 'O' | 'N' | 'D' (from shooter's perspective)
    homeScore, awayScore (state after the event — we adjust for 'before' on goals)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

# --- constants ---
SHOT_EVENTS = {"shot-on-goal", "goal", "missed-shot", "blocked-shot"}
SOG_EVENTS = {"shot-on-goal", "goal"}
CONTEXT_EVENTS = {
    "faceoff", "hit", "giveaway", "takeaway", "penalty",
    "stoppage", "period-start", "period-end",
    "shot-on-goal", "goal", "missed-shot", "blocked-shot",  # we store these in plays_context too,
    # because "last event before shot X" needs to be able to reference prior shots
}

# Goal is at x = +89 (NHL rink: 200ft long, goals 11ft from each end)
GOAL_X = 89.0
GOAL_Y = 0.0


def _parse_mmss(s: str) -> int:
    """'MM:SS' -> total seconds. Returns 0 on malformed input."""
    if not s or ":" not in s:
        return 0
    try:
        m, sec = s.split(":")
        return int(m) * 60 + int(sec)
    except (ValueError, AttributeError):
        return 0


def _normalize_coords(x: float | None, y: float | None,
                       attacks_positive: bool) -> tuple[
                       Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Return (x_raw, y_raw, x_norm, y_norm).

    `attacks_positive` is determined empirically per (game, period, team) from the
    distribution of that team's shot x-coords — see _infer_attack_sides().

    If attacks_positive is True: shot was taken attacking +x, keep as-is.
    If False: flip both axes so the shot ends up on the +x side.
    """
    if x is None or y is None:
        return x, y, None, None

    if attacks_positive:
        return float(x), float(y), float(x), float(y)
    else:
        return float(x), float(y), float(-x), float(-y)


def _infer_attack_sides(plays: list, home_team_id: int, away_team_id: int) -> dict:
    """Determine which x-side each team attacked in each period.

    Returns dict: (period, team_id) -> attacks_positive_bool

    Method: for each (period, team), take the mean xCoord of their shot attempts.
    Real shots cluster tightly in the offensive zone (x > 25 or x < -25 typically),
    so the sign of the mean is a reliable indicator. We use only SOG/goal/missed
    (not blocked, because blocks can happen anywhere).

    Fallback: if a team has 0 shots in a period, we use the opposite of the home
    team's attack side (since they're at opposite ends), and if we don't know either,
    we default to home-attacks-positive — but this only matters for edge cases like
    a shutout period with no shots from one side, which is rare enough to accept.
    """
    # Collect shot x-coords per (period, team)
    buckets: dict[tuple[int, int], list[float]] = {}
    for play in plays:
        if play.get("typeDescKey") not in ("shot-on-goal", "goal", "missed-shot"):
            continue
        details = play.get("details") or {}
        x = details.get("xCoord")
        team = details.get("eventOwnerTeamId")
        period = (play.get("periodDescriptor") or {}).get("number")
        if x is None or team is None or period is None:
            continue
        buckets.setdefault((int(period), int(team)), []).append(float(x))

    # Decide per (period, team)
    decisions: dict[tuple[int, int], bool] = {}
    for (period, team), xs in buckets.items():
        if not xs:
            continue
        mean_x = sum(xs) / len(xs)
        # If mean x is positive, team attacked +x this period
        decisions[(period, team)] = mean_x > 0

    # Fill gaps: if one team in a period has no shots, it must be attacking
    # the opposite side of whichever team does.
    all_periods = {p for (p, _t) in buckets.keys()}
    for p in all_periods:
        home_known = (p, home_team_id) in decisions
        away_known = (p, away_team_id) in decisions
        if home_known and not away_known:
            decisions[(p, away_team_id)] = not decisions[(p, home_team_id)]
        elif away_known and not home_known:
            decisions[(p, home_team_id)] = not decisions[(p, away_team_id)]
        # If neither known, leave missing — caller falls back to a default

    return decisions


def _distance_angle(x_norm: float | None, y_norm: float | None) -> tuple[
                    Optional[float], Optional[float]]:
    """Euclidean distance (ft) and absolute angle (deg) from shot location to goal."""
    if x_norm is None or y_norm is None:
        return None, None
    dx = GOAL_X - x_norm
    dy = GOAL_Y - y_norm
    dist = math.hypot(dx, dy)
    # angle: 0 = straight on, 90 = from the goal line
    if dist == 0:
        return 0.0, 0.0
    angle = math.degrees(math.atan2(abs(dy), max(dx, 0.01)))
    return dist, angle


def _parse_situation(code: str | None, shooter_is_home: bool | None = None) -> tuple[
                      str, bool, int, int]:
    """Parse 4-char situationCode.

    Returns (strength_state, empty_net_somewhere, home_skaters, away_skaters).

    Char layout: [away_goalie][away_skaters][home_skaters][home_goalie]
    Each char is a digit. Goalie positions: 1 = in net, 0 = pulled.
    Skater positions: number of skaters (typically 3-6).

    Examples:
      '1551' -> 5v5 with both goalies
      '1541' -> home has 5 skaters, away has 4 (home on PP)
      '1451' -> home has 5, away has 4 wait no — '1451' = away 4, home 5. Same as above? No:
                chars are a_g=1, a_s=4, h_s=5, h_g=1 — home PP (5v4)
                vs '1541': a_g=1, a_s=5, h_s=4, h_g=1 — away PP (home shorthanded 4v5)

    strength_state is reported from SHOOTER'S perspective: "{shooter_skaters}v{defense_skaters}"
    If shooter_is_home is None (we don't know yet), defaults to home-POV.
    """
    if not code or len(code) != 4:
        return "unknown", False, 0, 0

    try:
        away_goalie = int(code[0])
        away_skaters = int(code[1])
        home_skaters = int(code[2])
        home_goalie = int(code[3])
    except ValueError:
        return "unknown", False, 0, 0

    if shooter_is_home is True:
        strength = f"{home_skaters}v{away_skaters}"
    elif shooter_is_home is False:
        strength = f"{away_skaters}v{home_skaters}"
    else:
        # No shooter context (e.g. for faceoff context rows) — use home POV
        strength = f"{home_skaters}v{away_skaters}"

    empty_net = (away_goalie == 0) or (home_goalie == 0)
    return strength, empty_net, home_skaters, away_skaters


@dataclass
class ShotRow:
    game_id: int
    event_idx: int
    period: int
    period_type: str
    period_seconds: int
    game_seconds: int
    event_type: str
    is_goal: bool
    is_sog: bool
    x_raw: Optional[float]
    y_raw: Optional[float]
    x_norm: Optional[float]
    y_norm: Optional[float]
    distance_ft: Optional[float]
    angle_deg: Optional[float]
    shot_type: Optional[str]
    zone_code: Optional[str]
    shooter_id: Optional[int]
    goalie_id: Optional[int]
    shooter_team_id: int
    defending_team_id: int
    is_home_shot: bool
    situation_code: Optional[str]
    strength_state: str
    empty_net: bool
    home_score_before: Optional[int]
    away_score_before: Optional[int]
    raw_details_json: dict


@dataclass
class ContextRow:
    game_id: int
    event_idx: int
    period: int
    game_seconds: int
    event_type: str
    x: Optional[float]
    y: Optional[float]
    team_id: Optional[int]


@dataclass
class GameRow:
    game_id: int
    season: int
    game_type: int
    game_date: str
    home_team_id: int
    away_team_id: int
    home_team_abbr: str
    away_team_abbr: str
    venue: Optional[str]
    home_score: Optional[int]
    away_score: Optional[int]
    game_state: str


@dataclass
class ParseResult:
    game: GameRow
    shots: list[ShotRow] = field(default_factory=list)
    contexts: list[ContextRow] = field(default_factory=list)
    parse_warnings: list[str] = field(default_factory=list)


def parse_pbp(payload: dict) -> ParseResult:
    """Parse a full play-by-play JSON into DB-ready rows."""
    game_id = int(payload["id"])
    home = payload["homeTeam"]
    away = payload["awayTeam"]
    home_id = int(home["id"])
    away_id = int(away["id"])

    game = GameRow(
        game_id=game_id,
        season=int(payload.get("season", 0)),
        game_type=int(payload.get("gameType", 0)),
        game_date=payload.get("gameDate", ""),
        home_team_id=home_id,
        away_team_id=away_id,
        home_team_abbr=home.get("abbrev", ""),
        away_team_abbr=away.get("abbrev", ""),
        venue=(payload.get("venue") or {}).get("default"),
        home_score=home.get("score"),
        away_score=away.get("score"),
        game_state=payload.get("gameState", ""),
    )

    result = ParseResult(game=game)
    plays = payload.get("plays", [])

    # PRE-PASS: figure out which x-side each team attacked in each period.
    # This replaces the unreliable "home team attacks +x in odd periods" convention.
    attack_sides = _infer_attack_sides(plays, home_id, away_id)

    # Running score state (so we can report score BEFORE each shot)
    home_running = 0
    away_running = 0

    for i, play in enumerate(plays):
        try:
            period = int((play.get("periodDescriptor") or {}).get("number", 0))
            period_type = (play.get("periodDescriptor") or {}).get("periodType", "")
            t_in_period = _parse_mmss(play.get("timeInPeriod", "0:00"))
            # game_seconds: periods 1-3 are 20min = 1200s each; OT varies
            prior_period_secs = 1200 * min(period - 1, 3)
            # OT in regular season is 5min; playoffs are 20min. Just sum.
            if period > 3:
                prior_period_secs += (period - 4) * 1200  # rough but we rarely care past OT1
            game_seconds = prior_period_secs + t_in_period

            event_type = play.get("typeDescKey", "")
            details = play.get("details") or {}
            situation = play.get("situationCode")

            x = details.get("xCoord")
            y = details.get("yCoord")
            team_id = details.get("eventOwnerTeamId")

            # --- Shots ---
            if event_type in SHOT_EVENTS:
                if team_id is None:
                    result.parse_warnings.append(f"shot at idx {i} missing eventOwnerTeamId")
                    continue

                is_home = (team_id == home_id)
                defending = away_id if is_home else home_id

                # Look up which side this team attacked this period (inferred from data).
                # Fallback if this team had no shots (extremely rare): assume the
                # team attacks the side opposite what home defends, with home-attacks-+x
                # as the ultimate default. Won't matter in practice since this path
                # only triggers for teams with 0 shots in a period.
                attacks_positive = attack_sides.get(
                    (period, team_id),
                    # fallback: flip the other team's known side if we can
                    not attack_sides.get(
                        (period, defending),
                        period % 2 == 0  # last resort matches old convention
                    )
                )

                x_raw, y_raw, x_norm, y_norm = _normalize_coords(x, y, attacks_positive)
                dist, ang = _distance_angle(x_norm, y_norm)
                strength, empty_net_global, _hs, _as = _parse_situation(situation, shooter_is_home=is_home)

                # Empty net specifically against this shot: shooter's team scores
                # into an empty net when *defending* goalie is pulled
                if situation and len(situation) == 4:
                    if is_home:
                        en_against = situation[0] == "0"   # away goalie pulled
                    else:
                        en_against = situation[3] == "0"   # home goalie pulled
                else:
                    en_against = False

                shooter_id = details.get("shootingPlayerId") or details.get("scoringPlayerId")
                goalie_id = details.get("goalieInNetId")

                is_goal = event_type == "goal"
                is_sog = event_type in SOG_EVENTS

                result.shots.append(ShotRow(
                    game_id=game_id,
                    event_idx=i,
                    period=period,
                    period_type=period_type,
                    period_seconds=t_in_period,
                    game_seconds=game_seconds,
                    event_type=event_type,
                    is_goal=is_goal,
                    is_sog=is_sog,
                    x_raw=x_raw, y_raw=y_raw,
                    x_norm=x_norm, y_norm=y_norm,
                    distance_ft=dist, angle_deg=ang,
                    shot_type=details.get("shotType"),
                    zone_code=details.get("zoneCode"),
                    shooter_id=shooter_id,
                    goalie_id=goalie_id,
                    shooter_team_id=team_id,
                    defending_team_id=defending,
                    is_home_shot=is_home,
                    situation_code=situation,
                    strength_state=strength,
                    empty_net=en_against,
                    home_score_before=home_running,
                    away_score_before=away_running,
                    raw_details_json=details,
                ))

                # update running score AFTER recording the "before" state
                if is_goal:
                    if is_home:
                        home_running += 1
                    else:
                        away_running += 1

            # --- Context events (all events, including shots, for temporal context) ---
            if event_type in CONTEXT_EVENTS:
                result.contexts.append(ContextRow(
                    game_id=game_id,
                    event_idx=i,
                    period=period,
                    game_seconds=game_seconds,
                    event_type=event_type,
                    x=float(x) if x is not None else None,
                    y=float(y) if y is not None else None,
                    team_id=team_id,
                ))

        except (KeyError, ValueError, TypeError) as e:
            result.parse_warnings.append(f"play idx {i}: {type(e).__name__}: {e}")
            continue

    return result