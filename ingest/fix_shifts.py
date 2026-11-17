"""Fix three known problems in the stored shift charts (data audit, 2026-11-11).

1. Playoff 2nd+ overtimes. The API shift parser used to drop every period 5 and
   later as a shootout, but only the regular season has one: in the playoffs
   period 5 is a 2nd overtime. parser_shifts now keeps them; this step re-parses
   the cached shiftcharts JSON (no downloads) for playoff games that went past
   the 1st overtime and have no shifts there yet.

2. Lost goalies. Now and then the NHL's shift chart drops a team's goalie for
   part of a game (2022020567 loses the home goalie for the whole 3rd). The
   play-by-play's situationCode says at every event whether each net has a
   goalie, so that's the check: an event saying "goalie in" with no goalie
   shift on (start < t <= end) is a miss. Faceoffs and stoppages sit on shift
   edges, so they don't count. For each regular-season team-game with a miss,
   read the NHL's HTML shift report (as run_shifts_html does for empty charts);
   if its goalie shifts cover more of those events, they replace the team's API
   goalie shifts (raw_json.source = 'htmlreport'). The skaters' rows stay.
   A team-game with STOP_AT or more misses that the HTML report can't fix
   either stops the script (exit 1) with nothing written.
   Long gaps where the play-by-play agrees the net was empty are real goalie
   pulls and are left alone.

3. Shifts that run past their period's end. 2024021158's chart keeps three
   players on until 5:00 of an overtime that ended at 0:59. A shift's end is
   clipped to its period's last 'period-end' event, never below its start.

Safe to rerun: each step only acts where the problem is still there.

Usage:
    python -m ingest.fix_shifts --season 20262027   # what export/refresh.sh runs
    python -m ingest.fix_shifts                     # every season
    python -m ingest.fix_shifts --dry-run           # report only, writes nothing
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import RAW_DATA_DIR
from .db import insert_shifts, pg_conn
from .parser_shifts import parse_shifts
from .run_shifts_html import parse_game
from .shift_checks import goalie_in_seconds, uncovered

# Misses that neither source can cover, at or above which the script stops.
# 1-2 is shift-edge timing (the HTML report matches the API to the second there).
STOP_AT = 3


# ------------------------------------------------------------------ steps
def season_filter(seasons: list[int] | None) -> tuple[str, tuple]:
    if not seasons:
        return "", ()
    return " AND g.season = ANY(%s)", (seasons,)


def fix_playoff_ot(cur, seasons) -> list[tuple[int, int]]:
    """Step 1. Returns (game_id, rows added) per game."""
    where, params = season_filter(seasons)
    cur.execute(f"""
        SELECT g.game_id, g.season FROM games g
        WHERE g.game_type = 3{where}
          AND EXISTS (SELECT 1 FROM plays_context p WHERE p.game_id = g.game_id
                      AND p.event_type = 'period-end' AND p.period >= 5)
          AND NOT EXISTS (SELECT 1 FROM shifts sh WHERE sh.game_id = g.game_id
                          AND sh.period >= 5)
        ORDER BY 1""", params)
    done = []
    for gid, season in cur.fetchall():
        cache = RAW_DATA_DIR / f"shifts/{season}/{gid}.json"
        if not cache.exists():
            print(f"  {gid}: no cached shift chart, skipped")
            continue
        r = parse_shifts(json.loads(cache.read_text()), gid)
        done.append((gid, insert_shifts(cur, [s for s in r.shifts if s.period >= 5])))
    return done


def fix_lost_goalies(cur, seasons):
    """Step 2. Returns (replaced, edge, stop): lists of dicts per team-game."""
    where, params = season_filter(seasons)
    cur.execute(f"""
        SELECT g.game_id, g.season, g.home_team_id, g.away_team_id FROM games g
        WHERE g.game_type = 2 AND g.game_state = 'OFF'{where} ORDER BY 1""", params)
    games = cur.fetchall()
    cur.execute(f"""
        SELECT sh.game_id, sh.team_id, sh.start_seconds, sh.end_seconds
        FROM shifts sh JOIN games g USING (game_id)
        JOIN game_rosters r ON r.game_id = sh.game_id AND r.player_id = sh.player_id
        WHERE r.position = 'G' AND g.game_type = 2{where}""", params)
    api = {}
    for gid, tid, s, e in cur.fetchall():
        api.setdefault((gid, tid), []).append((s, e))

    replaced, edge, stop = [], [], []
    for gid, season, home, away in games:
        plays = json.loads((RAW_DATA_DIR / f"pbp/{season}/{gid}.json").read_text())["plays"]
        html = None
        for side, tid in (("home", home), ("away", away)):
            secs = goalie_in_seconds(plays, side)
            miss_api = uncovered(secs, api.get((gid, tid), []))
            if not miss_api:
                continue
            if html is None:
                shifts, _ = parse_game(cur, gid)
                cur.execute("SELECT player_id FROM game_rosters WHERE game_id = %s "
                            "AND position = 'G'", (gid,))
                goalies = {row[0] for row in cur.fetchall()}
                html = [s for s in shifts if s.player_id in goalies]
            h = [s for s in html if s.team_id == tid]
            miss_html = uncovered(secs, [(s.start_seconds, s.end_seconds) for s in h])
            row = dict(game_id=gid, side=side, team_id=tid, misses_api=len(miss_api),
                       misses_html=len(miss_html),
                       api_s=sum(e - s for s, e in api.get((gid, tid), [])),
                       html_s=sum(s.end_seconds - s.start_seconds for s in h))
            if len(miss_html) < len(miss_api):
                cur.execute("""
                    DELETE FROM shifts sh USING game_rosters r
                    WHERE sh.game_id = %s AND sh.team_id = %s
                      AND r.game_id = sh.game_id AND r.player_id = sh.player_id
                      AND r.position = 'G'""", (gid, tid))
                row["deleted"] = cur.rowcount
                row["inserted"] = insert_shifts(cur, h)
                replaced.append(row)
            elif len(miss_html) >= STOP_AT:
                stop.append(row)
            else:
                edge.append(row)
    return replaced, edge, stop


def clip_past_period_end(cur, seasons) -> list[tuple]:
    """Step 3. Returns the clipped rows (game, period, player, old end, new end)."""
    where, params = season_filter(seasons)
    cur.execute(f"""
        WITH pe AS (
            SELECT p.game_id, p.period, MAX(p.game_seconds) AS pend
            FROM plays_context p JOIN games g USING (game_id)
            WHERE p.event_type = 'period-end'{where}
            GROUP BY 1, 2)
        UPDATE shifts sh SET end_seconds = GREATEST(sh.start_seconds, pe.pend)
        FROM pe, shifts old
        WHERE sh.game_id = pe.game_id AND sh.period = pe.period
          AND old.shift_id = sh.shift_id
          AND sh.end_seconds > GREATEST(sh.start_seconds, pe.pend)
        RETURNING sh.game_id, sh.period, sh.player_id, old.end_seconds, sh.end_seconds""",
                params)
    return sorted(cur.fetchall())


# ------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--season", type=int, action="append",
                    help="Season id, e.g. 20262027 (repeatable). Default: every season.")
    ap.add_argument("--dry-run", action="store_true", help="Report only; roll back.")
    args = ap.parse_args()
    seasons = args.season

    print("Shift fixes:", "seasons " + ", ".join(map(str, seasons)) if seasons
          else "every season", "(dry run)" if args.dry_run else "")
    with pg_conn() as conn, conn.cursor() as cur:
        ot = fix_playoff_ot(cur, seasons)
        print(f"1. Playoff 2nd+ overtimes re-parsed: {len(ot)} games, "
              f"{sum(n for _, n in ot)} shift rows added")
        for gid, n in ot:
            print(f"     {gid}: +{n}")

        replaced, edge, stop = fix_lost_goalies(cur, seasons)
        print(f"2. Lost goalies: {len(replaced)} team-games took their goalie shifts from "
              f"the HTML report; {len(edge)} with 1-{STOP_AT - 1} shift-edge misses left as is")
        for r in replaced:
            print(f"     {r['game_id']} {r['side']}: goalie {r['api_s']} s -> {r['html_s']} s, "
                  f"'goalie in' events with no goalie shift {r['misses_api']} -> "
                  f"{r['misses_html']} (rows -{r['deleted']} +{r['inserted']})")

        clipped = clip_past_period_end(cur, seasons)
        print(f"3. Shifts clipped to their period's end: {len(clipped)}")
        for gid, per, pid, old, new in clipped:
            print(f"     {gid} period {per} player {pid}: end {old} -> {new}")

        if stop:
            conn.rollback()
            print(f"\nSTOP: {len(stop)} team-game(s) where the play-by-play says the goalie "
                  f"was in net at {STOP_AT}+ events and neither the API chart nor the HTML "
                  "report has a goalie on. Nothing was written. Look at these by hand:")
            for r in stop:
                print(f"     {r['game_id']} {r['side']}: misses api {r['misses_api']}, "
                      f"html {r['misses_html']}")
            return 1
        if args.dry_run:
            conn.rollback()
            print("Dry run: rolled back.")
        else:
            conn.commit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
