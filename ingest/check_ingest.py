"""Checks that stop a refresh when the ingested data doesn't add up.

export/refresh.sh runs this after the ingest steps and before the shot table is
built. Any problem prints a STOP line and exits with code 1, which
`set -euo pipefail` turns into a stopped refresh, so nothing gets scored or
exported. These are the audit's checks 1 and 2 (diagnostics/check_data_audit.py,
results/checks/data_audit.txt) made permanent, plus the parse-warning check
(gap 6 in docs/DATA_PIPELINE.md).

1. Completeness
   - every game in `games` is official (OFF);
   - no ingest_log row is anything but 'ok'. A play-by-play row that failed only
     because the game wasn't official yet is printed but doesn't stop: the next
     refresh fetches it again;
   - every game has an 'ok' play-by-play, shifts and rosters row;
   - each finished season has regular-season games 0001-1312, no more, no less;
   - the season in progress has every number from 1 up to its highest. A gap is
     looked up on the NHL API: a game that isn't official yet (postponed, say)
     is printed, one that is official but missing stops the refresh.
   Playoff games aren't counted: their numbers depend on how series go.
2. Goals add up: for every game, play-by-play goals outside the shootout, plus 1
   for the shootout winner, must equal home_score / away_score in `games`.
3. Parse warnings: any game whose play-by-play parse dropped a shot stops the
   refresh. Other warnings are only printed (--report-warnings, at the end of
   refresh.sh).

Usage (repo root):
    .venv/bin/python -m ingest.check_ingest                     # checks 1-3
    .venv/bin/python -m ingest.check_ingest --report-warnings   # print warnings only
    .venv/bin/python -m ingest.check_ingest --backfill-warnings # fill counts for
                                                                # games logged before
                                                                # migration 004
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from .config import PBP_URL, RAW_DATA_DIR, SEASONS
from .db import ensure_warning_columns, log_parse_warnings, pg_conn
from .http_client import FetchError, fetch_json
from .parser import parse_pbp

SEASON_GAMES = 1312  # regular-season games in a 32-team season


def rows(cur, sql: str, params=None) -> list[tuple]:
    cur.execute(sql, params)
    return cur.fetchall()


def in_progress_seasons() -> set[int]:
    today = date.today().isoformat()
    return {sid for sid, _start, end in SEASONS if end >= today}


# ---------------------------------------------------------------------------
def check_completeness(cur) -> list[str]:
    stops = []

    not_off = rows(cur, "SELECT game_id, game_state FROM games WHERE game_state <> 'OFF'")
    for gid, state in not_off:
        stops.append(f"game {gid} is in `games` but not official (state {state})")

    for kind, key, status, err, when in rows(cur, """
            SELECT kind, target_key, status, error_msg, completed_at
            FROM ingest_log WHERE status <> 'ok' ORDER BY kind, target_key"""):
        if kind == "pbp" and (err or "").startswith("game not official yet"):
            print(f"  note: {kind} {key} not official yet when last tried ({when:%Y-%m-%d}); "
                  "the next refresh tries again")
        else:
            stops.append(f"ingest_log {kind} {key} is '{status}': {err}")

    for kind in ("pbp", "shifts", "rosters"):
        missing = rows(cur, """
            SELECT g.game_id FROM games g
            WHERE NOT EXISTS (SELECT 1 FROM ingest_log l
                              WHERE l.kind = %s AND l.status = 'ok'
                                AND l.target_key = g.game_id::text)
            ORDER BY 1""", (kind,))
        if missing:
            ids = [r[0] for r in missing]
            stops.append(f"{len(ids)} game(s) with no 'ok' {kind} row, e.g. {ids[:10]}")

    current = in_progress_seasons()
    by_season: dict[int, set[int]] = {}
    for season, gid in rows(cur, "SELECT season, game_id FROM games WHERE game_type = 2"):
        by_season.setdefault(season, set()).add(gid % 10000)
    for season in sorted(by_season):
        have = by_season[season]
        extra = sorted(n for n in have if n > SEASON_GAMES)
        if extra:
            stops.append(f"{season}: regular-season game numbers above {SEASON_GAMES}: {extra[:10]}")
        if season not in current:
            missing = sorted(set(range(1, SEASON_GAMES + 1)) - have)
            if missing:
                stops.append(f"{season}: {len(missing)} regular-season game(s) missing, "
                             f"e.g. {missing[:10]}")
            print(f"  {season}: {len(have)} regular-season games")
            continue
        top = max(have)
        gaps = sorted(set(range(1, top + 1)) - have)
        for n in gaps:
            gid = int(f"{str(season)[:4]}02{n:04d}")
            try:
                state = fetch_json(PBP_URL.format(game_id=gid)).get("gameState", "?")
            except FetchError as e:
                stops.append(f"{season}: game {gid} is missing and the API lookup failed: {e}")
                continue
            if state == "OFF":
                stops.append(f"{season}: game {gid} is official on the NHL API but not "
                             "in `games`")
            else:
                print(f"  note: {season} game {gid} isn't in `games` and isn't official "
                      f"yet (state {state}), probably postponed")
        print(f"  {season}: {len(have)} regular-season games, numbers 1..{top}"
              f"{', in progress' if season in current else ''}")
    return stops


# ---------------------------------------------------------------------------
GOALS_SQL = """
SELECT g.game_id, g.home_score, g.away_score,
  COUNT(*) FILTER (WHERE s.is_goal AND s.period_type <> 'SO'
                         AND s.shooter_team_id = g.home_team_id) AS h_goals,
  COUNT(*) FILTER (WHERE s.is_goal AND s.period_type <> 'SO'
                         AND s.shooter_team_id = g.away_team_id) AS a_goals,
  COUNT(*) FILTER (WHERE s.is_goal AND s.period_type = 'SO'
                         AND s.shooter_team_id = g.home_team_id) AS h_so,
  COUNT(*) FILTER (WHERE s.is_goal AND s.period_type = 'SO'
                         AND s.shooter_team_id = g.away_team_id) AS a_so,
  COUNT(*) FILTER (WHERE s.period_type = 'SO') AS so_attempts
FROM games g LEFT JOIN shots s ON s.game_id = g.game_id
GROUP BY 1, 2, 3
ORDER BY 1
"""


def check_goals(cur) -> list[str]:
    stops = []
    games = rows(cur, GOALS_SQL)
    for gid, hs, as_, hg, ag, hso, aso, so in games:
        exp_h = hg + (1 if so and hso > aso else 0)
        exp_a = ag + (1 if so and aso > hso else 0)
        if (exp_h, exp_a) != (hs, as_):
            stops.append(f"game {gid}: final score {hs}-{as_}, play-by-play goals "
                         f"{hg}-{ag}, shootout goals {hso}-{aso}")
        elif so and (hg != ag or hso == aso):
            stops.append(f"game {gid}: has a shootout but goals before it were "
                         f"{hg}-{ag} and shootout goals {hso}-{aso}")
    print(f"  goals checked in {len(games):,} games")
    return stops


# ---------------------------------------------------------------------------
def check_dropped_shots(cur) -> list[str]:
    ensure_warning_columns(cur)
    return [f"game {key}: the parser dropped {n} shot(s): {sample}"
            for key, n, sample in rows(cur, """
                SELECT target_key, shots_dropped, warning_sample FROM ingest_log
                WHERE kind = 'pbp' AND shots_dropped > 0 ORDER BY target_key""")]


def report_warnings(cur) -> None:
    ensure_warning_columns(cur)
    games, warns, dropped, unknown = rows(cur, """
        SELECT COUNT(*), COALESCE(SUM(parse_warnings), 0),
               COALESCE(SUM(shots_dropped), 0),
               COUNT(*) FILTER (WHERE parse_warnings IS NULL)
        FROM ingest_log WHERE kind = 'pbp' AND status = 'ok'""")[0]
    print(f"Parse warnings: {warns} in {games:,} games, {dropped} shot(s) dropped"
          + (f"; {unknown} game(s) logged before the counts existed "
             "(--backfill-warnings fills them)" if unknown else ""))
    for key, n, d, sample, when in rows(cur, """
            SELECT target_key, parse_warnings, shots_dropped, warning_sample, completed_at
            FROM ingest_log
            WHERE kind = 'pbp' AND parse_warnings > 0
            ORDER BY completed_at DESC, target_key LIMIT 30"""):
        print(f"  game {key} ({when:%Y-%m-%d}): {n} warning(s), {d} shot(s) dropped: {sample}")


def backfill_warnings() -> None:
    """Parse every cached play-by-play file again (no writes to shots) and
    store its warning counts on the matching ingest_log row."""
    with pg_conn() as conn, conn.cursor() as cur:
        ensure_warning_columns(cur)
        todo = {int(r[0]) for r in rows(cur, """
            SELECT target_key FROM ingest_log
            WHERE kind = 'pbp' AND status = 'ok' AND parse_warnings IS NULL""")}
        done = 0
        for path in sorted((RAW_DATA_DIR / "pbp").glob("*/*.json")):
            gid = int(path.stem)
            if gid not in todo:
                continue
            with path.open() as f:
                result = parse_pbp(json.load(f))
            log_parse_warnings(cur, gid, result.parse_warnings, result.shots_dropped)
            done += 1
        conn.commit()
    print(f"Backfilled parse-warning counts for {done:,} of {len(todo):,} games.")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report-warnings", action="store_true",
                    help="Only print the parse-warning summary.")
    ap.add_argument("--backfill-warnings", action="store_true",
                    help="Fill parse-warning counts for games logged before migration 004.")
    args = ap.parse_args()

    if args.backfill_warnings:
        backfill_warnings()
        return
    with pg_conn() as conn, conn.cursor() as cur:
        if args.report_warnings:
            report_warnings(cur)
            conn.commit()
            return
        print("Ingest checks: completeness, goals add up, dropped shots")
        stops = check_completeness(cur) + check_goals(cur) + check_dropped_shots(cur)
        conn.commit()
    if stops:
        print(f"\nINGEST CHECKS FAILED ({len(stops)}). Stopping (nothing after this should run):")
        for s in stops[:50]:
            print("  STOP " + s)
        if len(stops) > 50:
            print(f"  ... and {len(stops) - 50} more")
        sys.exit(1)
    print("Ingest checks passed.")


if __name__ == "__main__":
    main()
