"""Test parser_landing against synthetic /boxscore payload."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.parser_landing import parse_landing

GAME_ID = 2024020500

PAYLOAD = {
    "id": GAME_ID,
    "homeTeam": {"id": 21, "abbrev": "COL"},
    "awayTeam": {"id": 22, "abbrev": "FLA"},
    "playerByGameStats": {
        "homeTeam": {
            "forwards": [
                {"playerId": 8477492, "position": "C", "sweaterNumber": 29,
                 "name": {"default": "N. MacKinnon"}},
                {"playerId": 8478492, "position": "L", "sweaterNumber": 13,
                 "name": {"default": "V. Nichushkin"}},
            ],
            # NOTE: /boxscore uses "defense" not "defensemen"
            "defense": [
                {"playerId": 8480039, "position": "D", "sweaterNumber": 8,
                 "name": {"default": "C. Makar"}},
            ],
            "goalies": [
                # starter: true on Georgiev
                {"playerId": 8476412, "position": "G", "sweaterNumber": 40,
                 "name": {"default": "A. Georgiev"}, "starter": True},
                # Backup, no starter flag
                {"playerId": 8476999, "position": "G", "sweaterNumber": 1,
                 "name": {"default": "B. Goalie"}},
            ],
        },
        "awayTeam": {
            "forwards": [
                {"playerId": 8478402, "position": "C", "sweaterNumber": 16,
                 "name": {"default": "A. Barkov"}},
            ],
            "defense": [],
            "goalies": [
                # No explicit starter flag — parser should fall back to first listed
                {"playerId": 8478048, "position": "G", "sweaterNumber": 72,
                 "name": {"default": "S. Bobrovsky"}},
            ],
        },
    },
}


def assert_eq(name, got, want):
    ok = got == want
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    return ok


def main():
    r = parse_landing(PAYLOAD, GAME_ID)
    failures = 0

    print(f"Parsed: {len(r.rosters)} rosters, {len(r.player_updates)} player updates")
    for w in r.warnings:
        print(f"  warning: {w}")

    # Home: 2F + 1D + 2G = 5, Away: 1F + 0D + 1G = 2, Total: 7
    if not assert_eq("total roster rows", len(r.rosters), 7): failures += 1

    by_pid = {x.player_id: x for x in r.rosters}

    # Georgiev: explicit starter
    gg = by_pid.get(8476412)
    if gg:
        if not assert_eq("Georgiev is goalie", gg.position, "G"): failures += 1
        if not assert_eq("Georgiev is starter (explicit flag)", gg.is_starter, True): failures += 1
        if not assert_eq("Georgiev team_id", gg.team_id, 21): failures += 1

    # Bobrovsky: fallback starter (first goalie listed for away)
    bb = by_pid.get(8478048)
    if bb:
        if not assert_eq("Bobrovsky is starter (fallback)", bb.is_starter, True): failures += 1

    # Backup should NOT be starter
    backup = by_pid.get(8476999)
    if backup:
        if not assert_eq("backup goalie not starter", backup.is_starter, False): failures += 1

    # Makar: position D from "defense" group (boxscore naming)
    makar = by_pid.get(8480039)
    if makar:
        if not assert_eq("Makar position (from 'defense' group)", makar.position, "D"): failures += 1

    # MacKinnon
    mac = by_pid.get(8477492)
    if mac:
        if not assert_eq("MacKinnon position", mac.position, "C"): failures += 1
        if not assert_eq("MacKinnon not starter", mac.is_starter, False): failures += 1
        if not assert_eq("MacKinnon sweater", mac.sweater, 29): failures += 1

    # Player updates use name.default
    name_by_pid = {u.player_id: u.full_name for u in r.player_updates}
    if not assert_eq("MacKinnon name", name_by_pid.get(8477492), "N. MacKinnon"): failures += 1
    if not assert_eq("Bobrovsky name", name_by_pid.get(8478048), "S. Bobrovsky"): failures += 1

    # Verify fallback warning was emitted for Bobrovsky (no explicit starter)
    had_fallback_warning = any("fallback" in w.lower() or "assuming first" in w.lower()
                                for w in r.warnings)
    if not assert_eq("fallback starter warning emitted", had_fallback_warning, True): failures += 1

    print(f"\n{'='*40}")
    if failures == 0:
        print("ALL LANDING CHECKS PASSED")
    else:
        print(f"{failures} FAILED")
    return failures


if __name__ == "__main__":
    sys.exit(main())