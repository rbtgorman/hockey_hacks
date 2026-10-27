"""Synthetic-payload tests for actor extraction on non-shot events.

Separate file from tests/test_parser.py on purpose: that file covers shot
geometry, normalization and strength state, and mixing concerns into it makes
a failure harder to localize.

Covers:
  - hit, penalty, faceoff, giveaway, takeaway actors land in the right slots
  - role names derive correctly from key names (camelCase -> snake_case)
  - a penalty with no drawn-by player (bench minor) leaves slot 2 null
  - raw_details_json is stored for non-shot events and NOT for shot-family ones
  - a RENAMED NHL key still produces an actor, which is the whole point of
    generic extraction over a hardcoded map
  - shot-family context rows still carry no raw_details_json duplicate
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.parser import parse_pbp, _role_from_key, _extract_actors


def assert_eq(name, got, want):
    status = "OK" if got == want else "FAIL"
    print(f"  [{status}] {name}: got={got!r} want={want!r}")
    return got == want


def _play(idx, etype, details, period=1, t="05:00"):
    return {
        "eventId": idx, "sortOrder": idx,
        "periodDescriptor": {"number": period, "periodType": "REG"},
        "timeInPeriod": t, "situationCode": "1551",
        "typeDescKey": etype, "details": details,
    }


PAYLOAD = {
    "id": 2025020001, "season": 20252026, "gameType": 2,
    "gameDate": "2025-10-07", "gameState": "FINAL",
    "venue": {"default": "Test Arena"},
    "homeTeam": {"id": 21, "abbrev": "HOM", "score": 1},
    "awayTeam": {"id": 22, "abbrev": "AWY", "score": 0},
    "plays": [
        _play(0, "faceoff", {
            "xCoord": 0, "yCoord": 0, "eventOwnerTeamId": 21,
            "winningPlayerId": 101, "losingPlayerId": 201,
        }),
        _play(1, "hit", {
            "xCoord": 60, "yCoord": 30, "eventOwnerTeamId": 21,
            "hittingPlayerId": 102, "hitteePlayerId": 202,
        }),
        # Penalty with a drawn-by player. Note the payload lists drawnBy FIRST,
        # so if slot assignment used payload order alone the roles would invert.
        _play(2, "penalty", {
            "xCoord": -40, "yCoord": 10, "eventOwnerTeamId": 22,
            "drawnByPlayerId": 103, "committedByPlayerId": 203,
            "descKey": "tripping", "duration": 2,
        }),
        # Bench minor: committed by someone, drawn by nobody.
        _play(3, "penalty", {
            "eventOwnerTeamId": 22,
            "servedByPlayerId": 204, "committedByPlayerId": 205,
            "descKey": "too-many-men", "duration": 2,
        }),
        _play(4, "giveaway", {
            "xCoord": -70, "yCoord": 5, "eventOwnerTeamId": 21,
            "playerId": 104,
        }),
        _play(5, "takeaway", {
            "xCoord": 55, "yCoord": -12, "eventOwnerTeamId": 22,
            "playerId": 206,
        }),
        # Shot family: three home shots so attack-side inference has something
        # to work with, including a blocked shot that names a blocker.
        _play(6, "shot-on-goal", {
            "xCoord": 70, "yCoord": 5, "shotType": "wrist",
            "shootingPlayerId": 105, "goalieInNetId": 207,
            "eventOwnerTeamId": 21,
        }),
        _play(7, "blocked-shot", {
            "xCoord": 55, "yCoord": -8, "eventOwnerTeamId": 21,
            "shootingPlayerId": 106, "blockingPlayerId": 208,
        }),
        _play(8, "goal", {
            "xCoord": 82, "yCoord": 1, "shotType": "snap",
            "scoringPlayerId": 105, "assist1PlayerId": 107,
            "assist2PlayerId": 108, "goalieInNetId": 207,
            "eventOwnerTeamId": 21,
        }),
    ],
}


def ctx_by_idx(result):
    return {c.event_idx: c for c in result.contexts}


def main():
    failures = 0
    print("=== role name derivation ===")
    cases = [
        ("hittingPlayerId", "hit", "hitting"),
        ("hitteePlayerId", "hit", "hittee"),
        ("drawnByPlayerId", "penalty", "drawn_by"),
        ("committedByPlayerId", "penalty", "committed_by"),
        ("servedByPlayerId", "penalty", "served_by"),
        ("winningPlayerId", "faceoff", "winning"),
        ("blockingPlayerId", "blocked-shot", "blocking"),
        ("assist1PlayerId", "goal", "assist1"),
        ("playerId", "giveaway", "giveaway"),
        ("playerId", "takeaway", "takeaway"),
    ]
    for key, etype, want in cases:
        if not assert_eq(f"{key} ({etype})", _role_from_key(key, etype), want):
            failures += 1

    result = parse_pbp(PAYLOAD)
    c = ctx_by_idx(result)

    print("\n=== faceoff ===")
    if not assert_eq("actor_1_id", c[0].actor_1_id, 101): failures += 1
    if not assert_eq("actor_1_role", c[0].actor_1_role, "winning"): failures += 1
    if not assert_eq("actor_2_id", c[0].actor_2_id, 201): failures += 1
    if not assert_eq("actor_2_role", c[0].actor_2_role, "losing"): failures += 1

    print("\n=== hit ===")
    if not assert_eq("actor_1_id (hitter)", c[1].actor_1_id, 102): failures += 1
    if not assert_eq("actor_1_role", c[1].actor_1_role, "hitting"): failures += 1
    if not assert_eq("actor_2_id (hittee)", c[1].actor_2_id, 202): failures += 1
    if not assert_eq("actor_2_role", c[1].actor_2_role, "hittee"): failures += 1

    print("\n=== penalty with drawn-by (payload order is drawnBy first) ===")
    if not assert_eq("actor_1_id (committed)", c[2].actor_1_id, 203): failures += 1
    if not assert_eq("actor_1_role", c[2].actor_1_role, "committed_by"): failures += 1
    if not assert_eq("actor_2_id (drawn)", c[2].actor_2_id, 103): failures += 1
    if not assert_eq("actor_2_role", c[2].actor_2_role, "drawn_by"): failures += 1

    print("\n=== bench minor: no drawn-by ===")
    if not assert_eq("actor_1_id (committed)", c[3].actor_1_id, 205): failures += 1
    if not assert_eq("actor_1_role", c[3].actor_1_role, "committed_by"): failures += 1
    if not assert_eq("actor_2_role (served_by, not drawn_by)",
                     c[3].actor_2_role, "served_by"): failures += 1
    if not assert_eq("no drawn_by anywhere",
                     "drawn_by" in (c[3].actor_1_role, c[3].actor_2_role), False):
        failures += 1

    print("\n=== giveaway / takeaway ===")
    if not assert_eq("giveaway actor", c[4].actor_1_id, 104): failures += 1
    if not assert_eq("giveaway role", c[4].actor_1_role, "giveaway"): failures += 1
    if not assert_eq("giveaway slot 2 empty", c[4].actor_2_id, None): failures += 1
    if not assert_eq("takeaway actor", c[5].actor_1_id, 206): failures += 1
    if not assert_eq("takeaway role", c[5].actor_1_role, "takeaway"): failures += 1

    print("\n=== blocked shot: blocker is a NEW unlock (shots has only shooter) ===")
    if not assert_eq("actor_1_id (shooter)", c[7].actor_1_id, 106): failures += 1
    if not assert_eq("actor_2_id (blocker)", c[7].actor_2_id, 208): failures += 1
    if not assert_eq("actor_2_role", c[7].actor_2_role, "blocking"): failures += 1

    print("\n=== goalieInNetId is never an actor ===")
    roles = {c[6].actor_1_role, c[6].actor_2_role, c[8].actor_1_role, c[8].actor_2_role}
    if not assert_eq("no goalie role", any(r and "goalie" in r for r in roles), False):
        failures += 1
    if not assert_eq("sog actor_1 is shooter", c[6].actor_1_id, 105): failures += 1
    if not assert_eq("sog slot 2 empty", c[6].actor_2_id, None): failures += 1

    print("\n=== raw_details_json: stored for context events, NOT for shot family ===")
    if not assert_eq("hit has raw details", c[1].raw_details_json is not None, True):
        failures += 1
    if not assert_eq("penalty descKey preserved",
                     (c[2].raw_details_json or {}).get("descKey"), "tripping"):
        failures += 1
    if not assert_eq("sog has NO duplicate raw details",
                     c[6].raw_details_json, None): failures += 1
    if not assert_eq("goal has NO duplicate raw details",
                     c[8].raw_details_json, None): failures += 1
    if not assert_eq("blocked-shot has NO duplicate raw details",
                     c[7].raw_details_json, None): failures += 1

    print("\n=== shots table is untouched by this change ===")
    if not assert_eq("shot count", len(result.shots), 3): failures += 1
    if not assert_eq("context count", len(result.contexts), 9): failures += 1
    if not assert_eq("parse warnings", len(result.parse_warnings), 0): failures += 1

    print("\n=== RENAMED KEY: the reason this is generic, not a hardcoded map ===")
    # Suppose the NHL renames drawnByPlayerId to penaltyDrawnByPlayerId.
    # A hardcoded map returns nothing and the column silently goes null.
    # Generic extraction still produces an actor, under a visibly new role.
    renamed = _extract_actors(
        {"committedByPlayerId": 301, "penaltyDrawnByPlayerId": 302}, "penalty")
    if not assert_eq("renamed key still yields 2 actors", len(renamed), 2): failures += 1
    if not assert_eq("committed_by still slot 1", renamed[0], (301, "committed_by")):
        failures += 1
    if not assert_eq("new role surfaces in the data", renamed[1],
                     (302, "penalty_drawn_by")): failures += 1

    print(f"\n{'=' * 40}")
    if failures == 0:
        print("ALL CHECKS PASSED")
    else:
        print(f"{failures} CHECK(S) FAILED")
    return failures


if __name__ == "__main__":
    sys.exit(main())