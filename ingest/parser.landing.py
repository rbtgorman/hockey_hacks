"""Parse NHL gamecenter/{id}/landing response for roster and starter info.

The landing endpoint returns a rich payload. We care about:
  - matchup.homeTeam.forwards, defensemen, goalies
  - matchup.awayTeam.forwards, defensemen, goalies
  - summary.iceSurface.homeTeam.forwards / awayTeam.* (starting lineup info)
  - matchup.goalieComparison.homeTeam.starter (who's starting in net)

Response varies slightly by game state (FINAL vs LIVE vs FUT). We parse
defensively and only emit rows for fields we actually find.

Expected fields per player record:
  playerId, sweaterNumber, position, firstName.default, lastName.default,
  ...plus optional: isCaptain, isAlternateCaptain, etc.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


POSITION_MAP = {
    # sometimes the API returns "R" or "Right Wing" etc.; normalize
    "C": "C", "L": "L", "R": "R", "D": "D", "G": "G",
    "CENTER": "C", "LEFTWING": "L", "RIGHTWING": "R",
    "LEFT WING": "L", "RIGHT WING": "R",
    "DEFENSE": "D", "DEFENCE": "D", "GOALIE": "G",
    "GOAL": "G",
}


def _norm_position(p: str | None) -> str | None:
    if not p:
        return None
    return POSITION_MAP.get(p.upper().strip(), p.upper().strip())


@dataclass
class RosterRow:
    game_id: int
    player_id: int
    team_id: int
    sweater: Optional[int]
    position: Optional[str]
    is_starter: bool
    is_scratch: bool


@dataclass
class PlayerUpdate:
    """Backfill for the players table."""
    player_id: int
    full_name: Optional[str]
    position: Optional[str]


@dataclass
class LandingParseResult:
    game_id: int
    rosters: list[RosterRow] = field(default_factory=list)
    player_updates: list[PlayerUpdate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _iter_team_players(team_block: dict | None):
    """Yield player dicts from a team block (has 'forwards', 'defensemen', 'goalies')."""
    if not isinstance(team_block, dict):
        return
    for group in ("forwards", "defensemen", "goalies"):
        lst = team_block.get(group)
        if isinstance(lst, list):
            for p in lst:
                yield p, group


def parse_landing(payload: dict, game_id: int) -> LandingParseResult:
    result = LandingParseResult(game_id=game_id)

    # --- Rosters come from summary.iceSurface (more reliable than matchup.* sections) ---
    ice_surface = (payload.get("summary") or {}).get("iceSurface") or {}
    home_team_block = ice_surface.get("homeTeam")
    away_team_block = ice_surface.get("awayTeam")

    home_team_id = (payload.get("homeTeam") or {}).get("id")
    away_team_id = (payload.get("awayTeam") or {}).get("id")

    if home_team_id is None or away_team_id is None:
        result.warnings.append(
            f"missing team IDs in payload: home={home_team_id} away={away_team_id}"
        )
        return result

    # Figure out starting goalies from matchup.goalieComparison if present
    starting_goalies: set[int] = set()
    goalie_cmp = (payload.get("matchup") or {}).get("goalieComparison") or {}
    for side in ("homeTeam", "awayTeam"):
        starter = (goalie_cmp.get(side) or {}).get("starter")
        if isinstance(starter, dict) and starter.get("playerId"):
            starting_goalies.add(int(starter["playerId"]))

    # If not available, fall back to "first goalie in team.goalies" as starter
    # (better than nothing; parser warnings will note this)
    for team_block, team_id in [(home_team_block, home_team_id),
                                 (away_team_block, away_team_id)]:
        if team_block is None:
            continue

        goalies_list = team_block.get("goalies") or []
        if not starting_goalies and goalies_list:
            # fallback: first goalie listed for each team
            if isinstance(goalies_list[0], dict) and goalies_list[0].get("playerId"):
                starting_goalies.add(int(goalies_list[0]["playerId"]))

        for p, group in _iter_team_players(team_block):
            if not isinstance(p, dict):
                continue
            pid = p.get("playerId")
            if not pid:
                continue
            pid = int(pid)

            # position
            pos = _norm_position(p.get("positionCode") or p.get("position"))
            if pos is None:
                # group fallback: forwards -> F, defensemen -> D, goalies -> G
                pos = {"forwards": "F", "defensemen": "D", "goalies": "G"}.get(group)

            sweater = p.get("sweaterNumber")
            sweater = int(sweater) if sweater is not None else None

            is_goalie = (pos == "G") or (group == "goalies")
            is_starter = (pid in starting_goalies) if is_goalie else False

            result.rosters.append(RosterRow(
                game_id=game_id,
                player_id=pid,
                team_id=int(team_id),
                sweater=sweater,
                position=pos,
                is_starter=is_starter,
                is_scratch=bool(p.get("isScratch", False)),
            ))

            # Player backfill
            first = (p.get("firstName") or {}).get("default") if isinstance(p.get("firstName"), dict) else p.get("firstName")
            last = (p.get("lastName") or {}).get("default") if isinstance(p.get("lastName"), dict) else p.get("lastName")
            full_name = " ".join(filter(None, [first, last])) or None
            result.player_updates.append(PlayerUpdate(
                player_id=pid,
                full_name=full_name,
                position=pos,
            ))

    return result