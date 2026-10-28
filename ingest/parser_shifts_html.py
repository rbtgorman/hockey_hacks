"""Parse NHL HTML TOI shift reports (TH = home, TV = visitor).

WHY THIS EXISTS
---------------
The shiftcharts API returns {"data": [], "total": 0} for 57 games of 2024-25
(2024021235-2024021291, April 8-15 2025), live and cached. The HTML reports at
nhl.com/scores/htmlreports/<season>/T{H,V}0<game number>.HTM still carry those
shifts, so this parser recovers them into the same ShiftRow the API parser
produces.

Report shape (one team per file):

    <td class="playerHeading + border">4 BYRAM, BOWEN</td>
    <tr> Shift # | Per | Start (Elapsed / Game) | End (Elapsed / Game) | Duration | Event </tr>
    <tr> 1 | 1 | 0:36 / 19:24 | 1:16 / 18:44 | 00:40 | &nbsp; </tr>
    ...
    <tr> Per | SHF | AVG | TOI | EV TOT | PP TOT | SH TOT </tr>   -- summary, 7 cells, skipped

The report names players by sweater number, not player id, so the caller
passes a sweater -> player_id map for that game and team (from game_rosters).

Times use the same period-elapsed convention as the API, so game_seconds come
out identical: 1200 * (period - 1) + elapsed. "OT" is period 4. Shootout rows
are skipped, as in parser_shifts.

The API's shift data also carries zero-length goal and penalty records
(typeCode 505). The HTML report has none, so compare on typeCode 517 rows only.
"""
from __future__ import annotations

from html.parser import HTMLParser

from .parser_shifts import ShiftParseResult, ShiftRow, _parse_mmss


class _RowCollector(HTMLParser):
    """Flatten every <tr> into a list of (td class, text) cells, in document order.

    Rows and cells are kept on stacks because the report nests tables inside
    cells; text goes only to the innermost open cell.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, str]]] = []
        self._row_stack: list[list[tuple[str, str]]] = []
        self._cell_stack: list[tuple[str, list[str]]] = []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row_stack.append([])
        elif tag == "td":
            self._cell_stack.append((dict(attrs).get("class") or "", []))
        elif tag == "br" and self._cell_stack:
            self._cell_stack[-1][1].append(" ")

    def handle_endtag(self, tag):
        if tag == "td" and self._cell_stack:
            cls, parts = self._cell_stack.pop()
            text = " ".join("".join(parts).replace("\xa0", " ").split())
            if self._row_stack:
                self._row_stack[-1].append((cls, text))
        elif tag == "tr" and self._row_stack:
            self.rows.append(self._row_stack.pop())

    def handle_data(self, data):
        if self._cell_stack:
            self._cell_stack[-1][1].append(data)


def _period_num(label: str) -> int | None:
    """'1'..'3' -> int, 'OT' -> 4, 'SO' or anything else -> None."""
    label = label.strip().upper()
    if label.isdigit():
        return int(label)
    if label == "OT":
        return 4
    return None


def parse_shifts_html(html_text: str, game_id: int, team_id: int,
                      sweater_to_player: dict[int, int]) -> ShiftParseResult:
    """Parse one team's TOI report into ShiftRow list."""
    result = ShiftParseResult(game_id=game_id)
    collector = _RowCollector()
    collector.feed(html_text)
    collector.close()

    seen: set[tuple[int, int, int]] = set()
    sweater: int | None = None
    name = ""

    for i, row in enumerate(collector.rows):
        if len(row) == 1 and "playerHeading" in row[0][0]:
            head = row[0][1]
            num, _, name = head.partition(" ")
            try:
                sweater = int(num)
            except ValueError:
                sweater = None
                result.warnings.append(f"row {i}: unparseable player heading {head!r}")
            continue

        cells = [text for _, text in row]
        if len(cells) != 6 or not cells[0].isdigit() or "/" not in cells[2]:
            continue
        if sweater is None:
            result.warnings.append(f"row {i}: shift row before any player heading")
            continue

        player_id = sweater_to_player.get(sweater)
        if player_id is None:
            result.warnings.append(f"row {i}: sweater {sweater} ({name}) not on roster")
            continue

        period = _period_num(cells[1])
        if period is None:
            if cells[1].strip().upper() != "SO":
                result.warnings.append(f"row {i}: unknown period {cells[1]!r}")
            continue
        if period >= 5 and str(game_id)[4:6] == "02":
            continue  # regular-season shootout

        start_raw = _parse_mmss(cells[2].split("/")[0].strip())
        end_raw = _parse_mmss(cells[3].split("/")[0].strip())
        if start_raw is None or end_raw is None:
            result.warnings.append(f"row {i}: unparseable times {cells[2]!r} {cells[3]!r}")
            continue
        if end_raw < start_raw:
            result.warnings.append(f"row {i}: end<start: {start_raw}->{end_raw}")
            continue

        prior = 1200 * (period - 1)
        key = (player_id, period, prior + start_raw)
        if key in seen:
            continue  # page breaks can repeat a player's block
        seen.add(key)

        result.shifts.append(ShiftRow(
            game_id=game_id,
            player_id=player_id,
            team_id=team_id,
            period=period,
            start_seconds=prior + start_raw,
            end_seconds=prior + end_raw,
            shift_number=int(cells[0]),
            raw_json={
                "source": "htmlreport",
                "sweater": sweater,
                "name": name,
                "shiftNumber": int(cells[0]),
                "period": cells[1],
                "start": cells[2],
                "end": cells[3],
                "duration": cells[4],
                "event": cells[5],
            },
        ))

    return result
