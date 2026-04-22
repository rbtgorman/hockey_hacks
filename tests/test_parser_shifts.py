"""Test parser_shifts against a synthetic payload matching the documented
NHL shiftcharts response shape.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.parser_shifts import parse_shifts

GAME_ID = 2024020500

PAYLOAD = {
    "data": [
        # Shift 1: MacKinnon, period 1, first shift
        {
            "id": 1,
            "gameId": GAME_ID,
            "playerId": 8477492,
            "teamId": 21,
            "period": 1,
            "shiftNumber": 1,
            "startTime": "00:00",
            "endTime": "00:45",
            "duration": "00:45",
        },
        # Shift 2: MacKinnon, period 1, second shift
        {
            "id": 2,
            "gameId": GAME_ID,
            "playerId": 8477492,
            "teamId": 21,
            "period": 1,
            "shiftNumber": 2,
            "startTime": "02:10",
            "endTime": "02:57",
            "duration": "00:47",
        },
        # Shift 3: MacKinnon, period 2 (time maps to 1200+125 = 1325)
        {
            "id": 3,
            "gameId": GAME_ID,
            "playerId": 8477492,
            "teamId": 21,
            "period": 2,
            "shiftNumber": 7,
            "startTime": "02:05",  # -> game_seconds 1325
            "endTime": "02:58",    # -> game_seconds 1378
            "duration": "00:53",
        },
        # Shift 4: OT shift (period 4 regulation 5min OT)
        {
            "id": 4,
            "gameId": GAME_ID,
            "playerId": 8477492,
            "teamId": 21,
            "period": 4,
            "shiftNumber": 20,
            "startTime": "00:15",  # -> 3600 + 15 = 3615
            "endTime": "01:02",    # -> 3600 + 62 = 3662
            "duration": "00:47",
        },
        # Shift 5: SHOOTOUT (period 5) — should be skipped
        {
            "id": 5,
            "gameId": GAME_ID,
            "playerId": 8477492,
            "teamId": 21,
            "period": 5,
            "shiftNumber": 1,
            "startTime": "00:00",
            "endTime": "00:05",
        },
        # Shift 6: goalie (Bobrovsky), period 1, full period
        {
            "id": 6,
            "gameId": GAME_ID,
            "playerId": 8478048,
            "teamId": 22,
            "period": 1,
            "shiftNumber": 1,
            "startTime": "00:00",
            "endTime": "20:00",
            "duration": "20:00",
        },
        # Shift 7: malformed times
        {
            "id": 7,
            "gameId": GAME_ID,
            "playerId": 8477492,
            "teamId": 21,
            "period": 1,
            "startTime": "abc",
            "endTime": "00:30",
        },
        # Shift 8: end before start (data glitch)
        {
            "id": 8,
            "gameId": GAME_ID,
            "playerId": 8477492,
            "teamId": 21,
            "period": 1,
            "startTime": "05:00",
            "endTime": "03:00",
        },
        # Shift 9: wrong game_id — should be ignored
        {
            "id": 9,
            "gameId": 9999999,
            "playerId": 8477492,
            "teamId": 21,
            "period": 1,
            "startTime": "10:00",
            "endTime": "10:30",
        },
    ],
    "total": 9,
}


def assert_eq(name, got, want):
    ok = got == want
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    return ok


def main():
    r = parse_shifts(PAYLOAD, GAME_ID)
    failures = 0

    print(f"=== Parsed shifts: {len(r.shifts)} ===")
    print(f"=== Warnings: {len(r.warnings)} ===")
    for w in r.warnings:
        print(f"    - {w}")

    # We expect: shifts 1, 2, 3, 4, 6 (five valid shifts); skip 5, 7, 8, 9
    if not assert_eq("total shifts kept", len(r.shifts), 5): failures += 1

    # Verify specific shifts
    by_id = {f"{s.player_id}-p{s.period}-{s.start_seconds}": s for s in r.shifts}

    # Shift 1: start=0, end=45
    s1 = by_id.get("8477492-p1-0")
    if s1 is None:
        print("  [FAIL] shift 1 not found")
        failures += 1
    else:
        if not assert_eq("shift1 end_seconds", s1.end_seconds, 45): failures += 1
        if not assert_eq("shift1 shift_number", s1.shift_number, 1): failures += 1

    # Shift 2: start=130, end=177
    s2 = by_id.get("8477492-p1-130")
    if s2 is None:
        print("  [FAIL] shift 2 not found")
        failures += 1
    else:
        if not assert_eq("shift2 end_seconds", s2.end_seconds, 177): failures += 1

    # Shift 3: period 2, start=1325, end=1378 (1200 + 2:05 = 1325)
    s3 = by_id.get("8477492-p2-1325")
    if s3 is None:
        print("  [FAIL] shift 3 not found")
        failures += 1
    else:
        if not assert_eq("shift3 end_seconds", s3.end_seconds, 1378): failures += 1

    # Shift 4: period 4 OT, start=3615, end=3662
    s4 = by_id.get("8477492-p4-3615")
    if s4 is None:
        print("  [FAIL] shift 4 (OT) not found")
        failures += 1
    else:
        if not assert_eq("shift4 end_seconds", s4.end_seconds, 3662): failures += 1

    # Goalie shift
    sg = by_id.get("8478048-p1-0")
    if sg is None:
        print("  [FAIL] goalie shift not found")
        failures += 1
    else:
        if not assert_eq("goalie team_id", sg.team_id, 22): failures += 1
        if not assert_eq("goalie end_seconds (full period)", sg.end_seconds, 1200): failures += 1

    # Check that bad records produced warnings
    if not assert_eq("shootout filtered", "8477492-p5-0" in by_id, False): failures += 1
    if not assert_eq("wrong game_id filtered",
                     any(s for s in r.shifts if s.game_id != GAME_ID), False): failures += 1

    print(f"\n{'='*40}")
    if failures == 0:
        print("ALL SHIFT CHECKS PASSED")
    else:
        print(f"{failures} FAILED")
    return failures


if __name__ == "__main__":
    sys.exit(main())