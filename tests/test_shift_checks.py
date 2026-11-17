"""Test ingest/shift_checks.py (stdlib only): which play-by-play events say a
goalie is in net, and which of them no goalie shift covers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.shift_checks import goalie_in_seconds, play_second, uncovered


def play(period, clock, kind, code):
    return {"periodDescriptor": {"number": period}, "timeInPeriod": clock,
            "typeDescKey": kind, "situationCode": code}


PLAYS = [
    play(1, "00:00", "faceoff", "1551"),        # edge event: never counted
    play(1, "05:00", "shot-on-goal", "1551"),   # t=300, both goalies in
    play(3, "18:00", "hit", "1560"),            # t=3480, home net empty
    play(3, "19:10", "missed-shot", "0651"),    # t=3550, away net empty
    play(4, "00:30", "blocked-shot", "1331"),   # t=3630, OT 3-on-3
    play(4, "02:00", "stoppage", "1331"),       # edge event
    play(5, "00:00", "shot-on-goal", "1010"),   # regular-season shootout: skipped
    play(2, "01:00", "giveaway", None),         # no code: skipped
]


def assert_eq(name, got, want):
    ok = got == want
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    return ok


def main():
    failures = 0
    if not assert_eq("play_second, 3rd period", play_second(PLAYS[2]), (3, 3480)): failures += 1
    if not assert_eq("play_second, OT", play_second(PLAYS[4]), (4, 3630)): failures += 1

    home = goalie_in_seconds(PLAYS, "home")
    away = goalie_in_seconds(PLAYS, "away")
    if not assert_eq("home goalie in (last digit)", home, [300, 3550, 3630]): failures += 1
    if not assert_eq("away goalie in (first digit)", away, [300, 3480, 3630]): failures += 1

    # start < t <= end: a shift ending at 300 covers 300, one starting at 300 doesn't.
    if not assert_eq("end is inside", uncovered([300], [(0, 300)]), []): failures += 1
    if not assert_eq("start is outside", uncovered([300], [(300, 600)]), [300]): failures += 1
    # The chart lost the home goalie after 3500: the OT event at 3630 has no shift.
    shifts = [(0, 1200), (1200, 2400), (2400, 3500)]
    if not assert_eq("lost goalie found", uncovered(home, shifts), [3550, 3630]): failures += 1
    if not assert_eq("no shifts at all", uncovered(home, []), home): failures += 1

    print(f"\n{'='*40}")
    print("ALL SHIFT CHECKS PASSED" if failures == 0 else f"{failures} FAILED")
    return failures


if __name__ == "__main__":
    sys.exit(main())
