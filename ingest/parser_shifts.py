"""Parse NHL shiftcharts API response.

Endpoint shape (api.nhle.com/stats/rest/en/shiftcharts):
{
  "data": [
    {
      "id": 12345,
      "gameId": 2024020500,
      "playerId": 8477492,
      "teamId": 21,
      "teamAbbrev": "COL",
      "firstName": "Nathan",
      "lastName": "MacKinnon",
      "period": 1,
      "shiftNumber": 3,
      "startTime": "02:15",       -- MM:SS elapsed in period
      "endTime": "03:02",
      "duration": "00:47",
      "hexValue": "#6F263D",
      "eventDescription": null,
      "eventDetails": null,
      "typeCode": 517,
      ...
    },
    ...
  ],
  "total": 750
}

Note: eventDescription/eventDetails/typeCode fields appear on SOME entries;
we ignore them and only parse the core shift data.

Edge cases handled:
- OT shifts have period 4+, and times still report within the period
- Shootout "shifts" (period 5) are ignored — they're not real hockey time
- startTime after endTime happens occasionally on bad records — skip those
- Some players have player_id 0 or missing — skip those
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


def _parse_mmss(s: str | None) -> int | None:
    """'MM:SS' -> total seconds. Returns None on malformed input."""
    if not s or ":" not in s:
        return None
    try:
        parts = s.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return None
    except (ValueError, AttributeError):
        return None


@dataclass
class ShiftRow:
    game_id: int
    player_id: int
    team_id: int
    period: int
    start_seconds: int
    end_seconds: int
    shift_number: Optional[int]
    raw_json: dict


@dataclass
class ShiftParseResult:
    game_id: int
    shifts: list[ShiftRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def parse_shifts(payload: dict, game_id: int) -> ShiftParseResult:
    """Parse shift chart JSON into normalized ShiftRow list.

    game_id passed explicitly — we cross-check against payload entries
    because some games have shifts listed for multiple games in one response
    (very rare, probably a NHL data issue on a given day).
    """
    result = ShiftParseResult(game_id=game_id)
    data = payload.get("data")
    if not isinstance(data, list):
        result.warnings.append(f"shiftcharts payload has no 'data' array: keys={list(payload.keys())}")
        return result

    for i, rec in enumerate(data):
        try:
            rec_game_id = rec.get("gameId")
            if rec_game_id is not None and int(rec_game_id) != game_id:
                # skip entries that don't match the game we asked for
                continue

            player_id = rec.get("playerId")
            team_id = rec.get("teamId")
            period = rec.get("period")
            if not player_id or not team_id or period is None:
                result.warnings.append(f"record {i} missing required fields")
                continue

            # Skip shootout rounds
            if int(period) >= 5:
                continue

            # Period times map to game_seconds as:
            #   period 1: 0 + t
            #   period 2: 1200 + t
            #   period 3: 2400 + t
            #   period 4 (OT): 3600 + t
            prior = 1200 * min(int(period) - 1, 3)
            if int(period) > 3:
                prior = 3600 + (int(period) - 4) * 1200

            start_raw = _parse_mmss(rec.get("startTime"))
            end_raw = _parse_mmss(rec.get("endTime"))
            if start_raw is None or end_raw is None:
                result.warnings.append(f"record {i} unparseable times: "
                                        f"start={rec.get('startTime')!r} end={rec.get('endTime')!r}")
                continue
            if end_raw < start_raw:
                # Data glitch — skip
                result.warnings.append(f"record {i} end<start: {start_raw}->{end_raw}")
                continue

            start_seconds = prior + start_raw
            end_seconds = prior + end_raw

            result.shifts.append(ShiftRow(
                game_id=game_id,
                player_id=int(player_id),
                team_id=int(team_id),
                period=int(period),
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                shift_number=rec.get("shiftNumber"),
                raw_json=rec,
            ))
        except (KeyError, ValueError, TypeError) as e:
            result.warnings.append(f"record {i}: {type(e).__name__}: {e}")
            continue

    return result