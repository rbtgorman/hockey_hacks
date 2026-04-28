"""Parse NHL gamecenter/{id}/boxscore response for roster and starter info.

History: originally built for /landing, but as of the 2024-25 NHL API
overhaul /landing no longer contains player data — it only has scoring,
penalties, three-stars. Rosters live in /boxscore.

Expected structure of /boxscore (verified against live 2024-25 data):
{
  "id": 2024020001,
  "homeTeam": {"id": int, "abbrev": str, "score": int, ...},
  "awayTeam": {"id": int, "abbrev": str, ...},
  "playerByGameStats": {
    "homeTeam": {
      "forwards": [
        {"playerId": 8477492, "sweaterNumber": 29,
         "name": {"default": "N. MacKinnon"},
         "position": "C", "goals": 0, "assists": 1, ...},
        ...
      ],
      "defense": [...],              // NOTE: "defense", not "defensemen"
      "goalies": [...]
    },
    "awayTeam": {... same shape ...}
  },
  ...
}

Starter detection: goalies in /boxscore typically have `starter: true` on the
starting goalie. If that field is absent, we fall back to "first goalie listed"
heuristic (usually correct but not guaranteed).

Player names: in /boxscore they're under `name.default` not `firstName.default`
+ `lastName.default` like /landing used to be. Format is "F. Lastname"
(e.g., "N. MacKinnon") — we use it as-is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


POSITION_MAP = {
    "C": "C", "L": "L", "R": "R", "D": "D", "G": "G",
    "LW": "L", "RW": "R",
    "CENTER": "C", "LEFT WING": "L", "RIGHT WING": "R",
    "DEFENSE": "D", "DEFENCE": "D", "GOALIE": "G", "GOAL": "G",
}


def _norm_position(p: str | None) -> str | None:
    if not p:
        return None
    return POSITION_MAP.get(str(p).upper().strip(), str(p).upper().strip())


def _extract_name(p: dict) -> str | None:
    """Handle name field variants across endpoints/seasons."""
    # /boxscore 2024-25 uses name.default = "F. Lastname"
    name_block = p.get("name")
    if isinstance(name_block, dict) and name_block.get("default"):
        return name_block["default"]
    # Older /landing format used firstName + lastName
    fn = p.get("firstName")
    ln = p.get("lastName")
    first = fn.get("default") if isinstance(fn, dict) else fn
    last = ln.get("default") if isinstance(ln, dict) else ln
    combined = " ".join(filter(None, [first, last]))
    return combined or None


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
    player_id: int
    full_name: Optional[str]
    position: Optional[str]


@dataclass
class LandingParseResult:
    game_id: int
    rosters: list[RosterRow] = field(default_factory=list)
    player_updates: list[PlayerUpdate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _iter_players(team_block: dict | None):
    """Yield (player_dict, group_name) pairs from a playerByGameStats team block."""
    if not isinstance(team_block, dict):
        return
    # /boxscore uses "defense" not "defensemen"; handle both for safety.
    for group in ("forwards", "defense", "defensemen", "goalies"):
        lst = team_block.get(group)
        if isinstance(lst, list):
            normalized = "D" if group in ("defense", "defensemen") else group
            for p in lst:
                yield p, normalized


def parse_landing(payload: dict, game_id: int) -> LandingParseResult:
    """Parse a /boxscore response. Function name kept for compatibility."""
    result = LandingParseResult(game_id=game_id)

    home_team = payload.get("homeTeam") or {}
    away_team = payload.get("awayTeam") or {}
    home_team_id = home_team.get("id")
    away_team_id = away_team.get("id")

    if home_team_id is None or away_team_id is None:
        result.warnings.append(
            f"missing team IDs in payload: home={home_team_id} away={away_team_id}"
        )
        return result

    pbgs = payload.get("playerByGameStats")
    if not isinstance(pbgs, dict):
        result.warnings.append(
            f"no 'playerByGameStats' in payload. Top-level keys: {list(payload.keys())}"
        )
        return result

    home_block = pbgs.get("homeTeam")
    away_block = pbgs.get("awayTeam")

    for team_block, team_id in [(home_block, home_team_id), (away_block, away_team_id)]:
        if team_block is None:
            result.warnings.append(f"no playerByGameStats block for team_id={team_id}")
            continue

        # Identify starter(s)
        goalies_list = team_block.get("goalies") or []
        explicit_starter_ids: set[int] = set()
        for g in goalies_list:
            if not isinstance(g, dict):
                continue
            if g.get("starter") is True:
                pid = g.get("playerId")
                if pid is not None:
                    explicit_starter_ids.add(int(pid))

        fallback_starter_id: int | None = None
        if not explicit_starter_ids and goalies_list:
            first = goalies_list[0]
            if isinstance(first, dict) and first.get("playerId") is not None:
                fallback_starter_id = int(first["playerId"])
                result.warnings.append(
                    f"team {team_id}: no explicit goalie starter flag, "
                    f"assuming first goalie listed (playerId={fallback_starter_id})"
                )

        for p, group in _iter_players(team_block):
            if not isinstance(p, dict):
                continue
            pid = p.get("playerId")
            if not pid:
                continue
            pid = int(pid)

            pos = _norm_position(p.get("position") or p.get("positionCode"))
            if pos is None:
                pos = {"forwards": "F", "D": "D", "goalies": "G"}.get(group)

            sweater = p.get("sweaterNumber")
            sweater = int(sweater) if sweater is not None else None

            is_goalie = (pos == "G") or (group == "goalies")
            if is_goalie:
                is_starter = (pid in explicit_starter_ids) or (pid == fallback_starter_id)
            else:
                is_starter = False

            result.rosters.append(RosterRow(
                game_id=game_id,
                player_id=pid,
                team_id=int(team_id),
                sweater=sweater,
                position=pos,
                is_starter=is_starter,
                is_scratch=bool(p.get("isScratch", False)),
            ))

            result.player_updates.append(PlayerUpdate(
                player_id=pid,
                full_name=_extract_name(p),
                position=pos,
            ))

    if not result.rosters:
        result.warnings.append(
            "parsed 0 rosters despite having playerByGameStats block"
        )

    return result