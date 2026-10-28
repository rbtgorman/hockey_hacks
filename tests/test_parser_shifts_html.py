"""Test parser_shifts_html against a synthetic snippet matching the NHL HTML
TOI report layout (nested tables, playerHeading cells, 7-cell summary rows).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.parser_shifts_html import parse_shifts_html

GAME_ID = 2024021235
TEAM_ID = 7

HEADER = """
<tr>
<td class="heading + lborder + bborder">Shift #</td>
<td class="heading + lborder + bborder">Per</td>
<td class="heading + lborder + bborder">Start of Shift<br>Elapsed / Game</td>
<td class="heading + lborder + bborder">End of Shift<br>Elapsed / Game</td>
<td class="heading + lborder + bborder">Duration</td>
<td class="heading + lborder + bborder + rborder">Event<br><span class="note">G=Goal<br>P=Penalty</span></td>
</tr>
"""


def shift_row(n, per, start, end, dur, event="&nbsp;"):
    return (f'<tr class="oddColor"><td class="lborder + bborder">{n}</td>'
            f'<td class="lborder + bborder">{per}</td>'
            f'<td class="lborder + bborder">{start}</td>'
            f'<td class="lborder + bborder">{end}</td>'
            f'<td class="lborder + bborder">{dur}</td>'
            f'<td class="lborder + bborder + rborder">{event}</td></tr>')


SUMMARY = """
<tr><td>Per</td><td>SHF</td><td>AVG</td><td>TOI</td><td>EV TOT</td><td>PP TOT</td><td>SH TOT</td></tr>
<tr><td>1</td><td>2</td><td>00:42</td><td>01:24</td><td>01:24</td><td>00:00</td><td>00:00</td></tr>
"""


def player_block(heading, rows):
    return (f'<tr><td><table>'
            f'<tr><td class="playerHeading + border" colspan="8">{heading}</td></tr>'
            f'{HEADER}{"".join(rows)}{SUMMARY}</table></td></tr>')


HTML = f"""<html><body><table>
<tr><td class="tablewidth"><table><tr><td class="teamHeading + border">BUFFALO SABRES</td></tr></table></td></tr>
{player_block("4 BYRAM, BOWEN", [
    shift_row(1, 1, "0:36 / 19:24", "1:16 / 18:44", "00:40"),
    shift_row(2, 2, "2:05 / 17:55", "2:58 / 17:02", "00:53", "G"),
    shift_row(3, "OT", "0:00 / 5:00", "1:02 / 3:58", "01:02"),
    shift_row(4, "SO", "0:00 / 0:00", "0:00 / 0:00", "00:00"),
    shift_row(5, 3, "5:00 / 15:00", "3:00 / 17:00", "-02:00"),
])}
{player_block("9 BENSON, ZACH", [
    shift_row(1, 3, "19:30 / 0:30", "20:00 / 0:00", "00:30"),
    shift_row(1, 3, "19:30 / 0:30", "20:00 / 0:00", "00:30"),
])}
{player_block("99 NOBODY, NOT ROSTERED", [
    shift_row(1, 1, "1:00 / 19:00", "1:30 / 18:30", "00:30"),
])}
</table></body></html>"""

SWEATERS = {4: 8481524, 9: 8484145}


def assert_eq(name, got, want):
    ok = got == want
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    return ok


def main():
    r = parse_shifts_html(HTML, GAME_ID, TEAM_ID, SWEATERS)
    failures = 0

    print(f"=== Parsed shifts: {len(r.shifts)} ===")
    print(f"=== Warnings: {len(r.warnings)} ===")
    for w in r.warnings:
        print(f"    - {w}")

    # Kept: Byram 1, 2, OT; Benson once (duplicate dropped).
    # Skipped: shootout, end<start, unrostered sweater, summary rows.
    if not assert_eq("total shifts kept", len(r.shifts), 4): failures += 1
    if not assert_eq("warnings (end<start, unrostered)", len(r.warnings), 2): failures += 1

    by_key = {(s.player_id, s.period, s.start_seconds): s for s in r.shifts}

    s1 = by_key.get((8481524, 1, 36))
    if s1 is None:
        print("  [FAIL] Byram shift 1 not found"); failures += 1
    else:
        if not assert_eq("shift1 end_seconds", s1.end_seconds, 76): failures += 1
        if not assert_eq("shift1 team_id", s1.team_id, TEAM_ID): failures += 1
        if not assert_eq("shift1 source", s1.raw_json["source"], "htmlreport"): failures += 1

    # Period 2: 1200 + 2:05 = 1325, end 1200 + 2:58 = 1378 (same as API parser)
    s2 = by_key.get((8481524, 2, 1325))
    if s2 is None:
        print("  [FAIL] Byram shift 2 not found"); failures += 1
    else:
        if not assert_eq("shift2 end_seconds", s2.end_seconds, 1378): failures += 1
        if not assert_eq("shift2 event", s2.raw_json["event"], "G"): failures += 1

    # OT is period 4: 3600 + 0 .. 3600 + 62
    s3 = by_key.get((8481524, 4, 3600))
    if s3 is None:
        print("  [FAIL] Byram OT shift not found"); failures += 1
    else:
        if not assert_eq("OT end_seconds", s3.end_seconds, 3662): failures += 1

    # Period 3 end: 2400 + 19:30 = 3570 .. 3600
    s4 = by_key.get((8484145, 3, 3570))
    if s4 is None:
        print("  [FAIL] Benson shift not found"); failures += 1
    else:
        if not assert_eq("period-end end_seconds", s4.end_seconds, 3600): failures += 1

    if not assert_eq("no shootout rows", any(s.period >= 5 for s in r.shifts), False): failures += 1

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    return failures


if __name__ == "__main__":
    sys.exit(main())
