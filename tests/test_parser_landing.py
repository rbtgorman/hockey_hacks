"""Test parser_landing against synthetic gamecenter/landing payload."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.parser_landing import parse_landing

GAME_ID = 2024020500

PAYLOAD = {
    "id": GAME_ID,
    "homeTeam": {"id": 21, "abbrev": "COL"},
    "awayTeam": {"id": 22, "abbrev": "FLA"},
    "matchup": {
        "goalieComparison": {
            "homeTeam": {
                "starter": {"playerId": 8476412, "firstName": {"default": "Alexandar"},
                            "lastName": {"default": "Georgiev"}}
            },
            "awayTeam": {
                "starter": {"playerId": 8478048, "firstName": {"default": "Sergei"},
                            "lastName": {"default": "Bobrovsky"}}
            },
        }
    },
    "summary": {
        "iceSurface": {
            "homeTeam": {
                "forwards": [
                    {"playerId": 8477492, "positionCode": "C", "sweaterNumber": 29,
                     "firstName": {"default": "Nathan"}, "lastName": {"default": "MacKinnon"}},
                    {"playerId": 8478492, "positionCode": "L", "sweaterNumber": 13,
                     "firstName": {"default": "Valeri"}, "lastName": {"default": "Nichushkin"}},
                ],
                "defensemen": [
                    {"playerId": 8480039, "positionCode": "D", "sweaterNumber": 8,
                     "firstName": {"default": "Cale"}, "lastName": {"default": "Makar"}},
                ],
                "goalies": [
                    {"playerId": 8476412, "positionCode": "G", "sweaterNumber": 40,
                     "firstName": {"default": "Alexandar"}, "lastName": {"default": "Georgiev"}},
                    # backup
                    {"playerId": 8476999, "positionCode": "G", "sweaterNumber": 1,
                     "firstName": {"default": "Backup"}, "lastName": {"default": "Goalie"}},
                ],
            },
            "awayTeam": {
                "forwards": [
                    {"playerId": 8478402, "positionCode": "C", "sweaterNumber": 16,
                     "firstName": {"default": "Aleksander"}, "lastName": {"default": "Barkov"}},
                ],
                "defensemen": [],
                "goalies": [
                    {"playerId": 8478048, "positionCode": "G", "sweaterNumber": 72,
                     "firstName": {"default": "Sergei"}, "lastName": {"default": "Bobrovsky"}},
                ],
            },
        }
    }
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

    by_pid = {r.player_id: r for r in r.rosters}

    # Georgiev is home starting goalie
    gg = by_pid.get(8476412)
    if gg:
        if not assert_eq("Georgiev is goalie", gg.position, "G"): failures += 1
        if not assert_eq("Georgiev is starter", gg.is_starter, True): failures += 1
        if not assert_eq("Georgiev team_id", gg.team_id, 21): failures += 1

    # Bobrovsky is away starting goalie
    bb = by_pid.get(8478048)
    if bb:
        if not assert_eq("Bobrovsky is starter", bb.is_starter, True): failures += 1

    # Backup is NOT a starter
    backup = by_pid.get(8476999)
    if backup:
        if not assert_eq("backup goalie not starter", backup.is_starter, False): failures += 1

    # MacKinnon is a C (not starter, he's not a goalie)
    mac = by_pid.get(8477492)
    if mac:
        if not assert_eq("MacKinnon position", mac.position, "C"): failures += 1
        if not assert_eq("MacKinnon not starter", mac.is_starter, False): failures += 1
        if not assert_eq("MacKinnon sweater", mac.sweater, 29): failures += 1

    # Player updates include names
    name_by_pid = {u.player_id: u.full_name for u in r.player_updates}
    if not assert_eq("MacKinnon name", name_by_pid.get(8477492), "Nathan MacKinnon"): failures += 1
    if not assert_eq("Bobrovsky name", name_by_pid.get(8478048), "Sergei Bobrovsky"): failures += 1

    print(f"\n{'='*40}")
    if failures == 0:
        print("ALL LANDING CHECKS PASSED")
    else:
        print(f"{failures} FAILED")
    return failures


if __name__ == "__main__":
    sys.exit(main())