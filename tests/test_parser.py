"""Smoke test for the parser using a synthetic play-by-play payload.

Validates:
  - shot events are extracted
  - coordinates are normalized correctly
  - distance/angle are sane
  - strength state parsing handles 5v5, PP, empty net
  - goal events update the running score correctly
  - context events are captured
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.parser import parse_pbp

# Synthetic payload modeled on the real NHL API schema.
# Two teams, period 1, with a variety of shot types and situations.
PAYLOAD = {
    "id": 2024020500,
    "season": 20242025,
    "gameType": 2,
    "gameDate": "2025-01-15",
    "venue": {"default": "Ball Arena"},
    "gameState": "FINAL",
    "homeTeam": {"id": 21, "abbrev": "COL", "score": 4},
    "awayTeam": {"id": 22, "abbrev": "EDM", "score": 3},
    "plays": [
        # Opening faceoff — center ice
        {
            "eventId": 1, "sortOrder": 1,
            "periodDescriptor": {"number": 1, "periodType": "REG"},
            "timeInPeriod": "00:00", "timeRemaining": "20:00",
            "situationCode": "1551",
            "typeDescKey": "faceoff",
            "details": {"xCoord": 0, "yCoord": 0, "eventOwnerTeamId": 21},
        },
        # Home shot from in close, right side — should have small distance, non-zero angle
        # Home attacks +x in period 1, so +x coords stay positive after normalization
        {
            "eventId": 2, "sortOrder": 2,
            "periodDescriptor": {"number": 1, "periodType": "REG"},
            "timeInPeriod": "02:15",
            "situationCode": "1551",
            "typeDescKey": "shot-on-goal",
            "details": {
                "xCoord": 75, "yCoord": 10,
                "shotType": "wrist",
                "zoneCode": "O",
                "shootingPlayerId": 8477492,
                "goalieInNetId": 8478048,
                "eventOwnerTeamId": 21,
            },
        },
        # Home GOAL — slot shot
        {
            "eventId": 3, "sortOrder": 3,
            "periodDescriptor": {"number": 1, "periodType": "REG"},
            "timeInPeriod": "05:42",
            "situationCode": "1551",
            "typeDescKey": "goal",
            "details": {
                "xCoord": 82, "yCoord": 2,
                "shotType": "snap",
                "zoneCode": "O",
                "scoringPlayerId": 8477492,
                "goalieInNetId": 8478048,
                "eventOwnerTeamId": 21,
            },
        },
        # Away shot — away shoots toward -x in period 1 (their offensive side),
        # raw xCoord will be negative (~-80). After normalization (flipping for
        # the away team) it should become positive.
        {
            "eventId": 4, "sortOrder": 4,
            "periodDescriptor": {"number": 1, "periodType": "REG"},
            "timeInPeriod": "08:30",
            "situationCode": "1541",  # home PP (5v4), but this is an away shot while SH
            "typeDescKey": "shot-on-goal",
            "details": {
                "xCoord": -70, "yCoord": -15,
                "shotType": "slap",
                "zoneCode": "O",
                "shootingPlayerId": 8478402,
                "goalieInNetId": 8476412,
                "eventOwnerTeamId": 22,
            },
        },
        # Home blocked shot
        {
            "eventId": 5, "sortOrder": 5,
            "periodDescriptor": {"number": 1, "periodType": "REG"},
            "timeInPeriod": "12:00",
            "situationCode": "1551",
            "typeDescKey": "blocked-shot",
            "details": {
                "xCoord": 60, "yCoord": 5,
                "shotType": "wrist",
                "zoneCode": "O",
                "shootingPlayerId": 8474564,
                "eventOwnerTeamId": 21,
            },
        },
        # Empty-net goal — away team pulls goalie late, home scores
        {
            "eventId": 6, "sortOrder": 6,
            "periodDescriptor": {"number": 3, "periodType": "REG"},
            "timeInPeriod": "19:30",
            "situationCode": "0651",  # away goalie pulled, 6 away skaters, 5 home, home goalie in
            "typeDescKey": "goal",
            "details": {
                "xCoord": -80,  # home team attacks -x in period 3 (odd period, same as P1)... wait
                "yCoord": 0,
                "shotType": "wrist",
                "zoneCode": "O",
                "scoringPlayerId": 8479420,
                "eventOwnerTeamId": 21,
                # no goalieInNetId -> empty net
            },
        },
        # Penalty on away team
        {
            "eventId": 7, "sortOrder": 7,
            "periodDescriptor": {"number": 1, "periodType": "REG"},
            "timeInPeriod": "14:22",
            "situationCode": "1551",
            "typeDescKey": "penalty",
            "details": {"xCoord": 40, "yCoord": 20, "eventOwnerTeamId": 22},
        },
    ],
}


def assert_eq(name, got, want):
    status = "OK" if got == want else "FAIL"
    print(f"  [{status}] {name}: got={got!r} want={want!r}")
    return got == want


def assert_close(name, got, want, tol=0.5):
    if got is None:
        print(f"  [FAIL] {name}: got None")
        return False
    ok = abs(got - want) <= tol
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: got={got:.2f} want={want:.2f} (tol {tol})")
    return ok


def main():
    result = parse_pbp(PAYLOAD)
    failures = 0

    print("=== Game metadata ===")
    if not assert_eq("game_id", result.game.game_id, 2024020500): failures += 1
    if not assert_eq("season", result.game.season, 20242025): failures += 1
    if not assert_eq("home_abbr", result.game.home_team_abbr, "COL"): failures += 1
    if not assert_eq("away_abbr", result.game.away_team_abbr, "EDM"): failures += 1

    print(f"\n=== Shots ({len(result.shots)} extracted) ===")
    # We expect 5 shot events: 2 SOGs, 2 goals, 1 blocked shot
    if not assert_eq("shot_count", len(result.shots), 5): failures += 1

    # Shot 0: home wrist shot from (75, 10), period 1
    s0 = result.shots[0]
    print("\n-- Shot 0 (home wrist, P1) --")
    if not assert_eq("s0 event_type", s0.event_type, "shot-on-goal"): failures += 1
    if not assert_eq("s0 is_goal", s0.is_goal, False): failures += 1
    if not assert_eq("s0 is_home_shot", s0.is_home_shot, True): failures += 1
    if not assert_eq("s0 x_norm (P1 home attacks +x)", s0.x_norm, 75.0): failures += 1
    if not assert_close("s0 distance (sqrt(14^2+10^2))", s0.distance_ft, 17.2, tol=0.5): failures += 1
    if not assert_eq("s0 strength_state", s0.strength_state, "5v5"): failures += 1
    if not assert_eq("s0 score_before (0-0)",
                     (s0.home_score_before, s0.away_score_before), (0, 0)): failures += 1

    # Shot 1: home goal — running score should become (1, 0) AFTER this shot
    s1 = result.shots[1]
    print("\n-- Shot 1 (home goal) --")
    if not assert_eq("s1 is_goal", s1.is_goal, True): failures += 1
    if not assert_eq("s1 score_before still (0,0)",
                     (s1.home_score_before, s1.away_score_before), (0, 0)): failures += 1

    # Shot 2: away shot. Raw coords (-70, -15). Away attacks -x in P1 (because home attacks +x).
    # So in normalized space (shooter always attacks +x), we flip: -(-70)=+70, -(-15)=+15.
    s2 = result.shots[2]
    print("\n-- Shot 2 (away slap, P1) --")
    if not assert_eq("s2 is_home_shot", s2.is_home_shot, False): failures += 1
    if not assert_eq("s2 x_norm (flipped from -70)", s2.x_norm, 70.0): failures += 1
    if not assert_eq("s2 y_norm (flipped from -15)", s2.y_norm, 15.0): failures += 1
    if not assert_eq("s2 strength (home PP means away SH: 5v4 home)",
                     s2.strength_state, "5v4"): failures += 1

    # The score_before for s2: home goal already scored (s1), so home_before = 1
    if not assert_eq("s2 score_before (1, 0)",
                     (s2.home_score_before, s2.away_score_before), (1, 0)): failures += 1

    # Shot 3: blocked shot
    s3 = result.shots[3]
    print("\n-- Shot 3 (blocked) --")
    if not assert_eq("s3 event_type", s3.event_type, "blocked-shot"): failures += 1
    if not assert_eq("s3 is_sog", s3.is_sog, False): failures += 1

    # Shot 4: empty-net goal. situation '0651' -> away goalie pulled (char[0]=0)
    s4 = result.shots[4]
    print("\n-- Shot 4 (empty net goal, P3) --")
    if not assert_eq("s4 is_goal", s4.is_goal, True): failures += 1
    if not assert_eq("s4 empty_net (away goalie pulled, home shot)", s4.empty_net, True): failures += 1
    if not assert_eq("s4 goalie_id (none)", s4.goalie_id, None): failures += 1

    print(f"\n=== Context events ({len(result.contexts)} captured) ===")
    # We expect: faceoff + 5 shot-family + penalty = 7 context rows
    if not assert_eq("context_count", len(result.contexts), 7): failures += 1

    # Types present
    types = [c.event_type for c in result.contexts]
    for t in ("faceoff", "penalty", "shot-on-goal", "goal", "blocked-shot"):
        if not assert_eq(f"contains {t}", t in types, True): failures += 1

    print(f"\n=== Warnings ({len(result.parse_warnings)}) ===")
    for w in result.parse_warnings:
        print(f"  - {w}")

    # =================================================================
    # Regression test for the real bug we hit in prod:
    # game 2022020001 had home team shooting at the NEGATIVE x net in P1.
    # Build a payload where home's shots are all at negative x — the parser
    # must auto-detect this and normalize correctly.
    # =================================================================
    print("\n=== REGRESSION: home attacks -x in P1 ===")
    INVERTED = {
        "id": 9999999999, "season": 20222023, "gameType": 2,
        "gameDate": "2022-10-07", "gameState": "FINAL",
        "homeTeam": {"id": 21, "abbrev": "HOM", "score": 1},
        "awayTeam": {"id": 22, "abbrev": "AWY", "score": 0},
        "plays": [
            # Home team takes 3 shots at negative-x net in P1
            {"eventId": 1, "periodDescriptor": {"number": 1, "periodType": "REG"},
             "timeInPeriod": "02:00", "situationCode": "1551",
             "typeDescKey": "shot-on-goal",
             "details": {"xCoord": -74, "yCoord": -5, "shotType": "wrist",
                         "shootingPlayerId": 1, "goalieInNetId": 2,
                         "eventOwnerTeamId": 21}},
            {"eventId": 2, "periodDescriptor": {"number": 1, "periodType": "REG"},
             "timeInPeriod": "04:00", "situationCode": "1551",
             "typeDescKey": "shot-on-goal",
             "details": {"xCoord": -81, "yCoord": 15, "shotType": "snap",
                         "shootingPlayerId": 1, "goalieInNetId": 2,
                         "eventOwnerTeamId": 21}},
            {"eventId": 3, "periodDescriptor": {"number": 1, "periodType": "REG"},
             "timeInPeriod": "06:00", "situationCode": "1551",
             "typeDescKey": "goal",
             "details": {"xCoord": -85, "yCoord": 0, "shotType": "wrist",
                         "scoringPlayerId": 1, "goalieInNetId": 2,
                         "eventOwnerTeamId": 21}},
            # Away team takes 2 shots at positive-x net in P1
            {"eventId": 4, "periodDescriptor": {"number": 1, "periodType": "REG"},
             "timeInPeriod": "08:00", "situationCode": "1551",
             "typeDescKey": "shot-on-goal",
             "details": {"xCoord": 72, "yCoord": 2, "shotType": "slap",
                         "shootingPlayerId": 3, "goalieInNetId": 4,
                         "eventOwnerTeamId": 22}},
            {"eventId": 5, "periodDescriptor": {"number": 1, "periodType": "REG"},
             "timeInPeriod": "10:00", "situationCode": "1551",
             "typeDescKey": "shot-on-goal",
             "details": {"xCoord": 76, "yCoord": -2, "shotType": "wrist",
                         "shootingPlayerId": 3, "goalieInNetId": 4,
                         "eventOwnerTeamId": 22}},
        ],
    }
    r2 = parse_pbp(INVERTED)
    # All five shots should have reasonable distances (< 50 ft, since they're
    # all in close). Before the fix, the home shots would register as ~170 ft.
    for idx, s in enumerate(r2.shots):
        ok = s.distance_ft is not None and s.distance_ft < 60
        mark = "OK" if ok else "FAIL"
        print(f"  [{mark}] inv shot {idx}: team={s.shooter_team_id} "
              f"x_raw={s.x_raw} x_norm={s.x_norm} dist={s.distance_ft:.1f}")
        if not ok:
            failures += 1

    print(f"\n{'=' * 40}")
    if failures == 0:
        print(f"ALL CHECKS PASSED")
    else:
        print(f"{failures} CHECK(S) FAILED")
    return failures


if __name__ == "__main__":
    sys.exit(main())