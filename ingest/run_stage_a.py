"""Stage A orchestrator: fetch shift charts and game rosters for games
already in the database.

Usage:
    python -m ingest.run_stage_a --init-schema               # create tables first
    python -m ingest.run_stage_a --season 20242025           # one season (recommended first)
    python -m ingest.run_stage_a --season 20242025 --max-games 20   # smoke test
    python -m ingest.run_stage_a --seasons all               # backfill everything

The script reads game_ids from the `games` table (no re-walking the schedule),
so it only pulls shifts/rosters for games we've already successfully ingested
play-by-play for.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from .config import SHIFTS_URL, BOXSCORE_URL, RAW_DATA_DIR
from .http_client import fetch_json, FetchError
from .parser_shifts import parse_shifts
from .parser_landing import parse_landing
from .db import (
    pg_conn, init_schema, insert_shifts, insert_rosters,
    upsert_player_updates, log_ingest,
    already_ingested_shifts, already_ingested_rosters,
)


def games_in_db(conn, season: int | None = None) -> list[tuple[int, int]]:
    """Return list of (game_id, season) for games we've already ingested pbp for."""
    sql = """
        SELECT g.game_id, g.season
        FROM games g
        JOIN ingest_log l
          ON l.kind = 'pbp' AND l.target_key = g.game_id::text AND l.status = 'ok'
    """
    params = ()
    if season is not None:
        sql += " WHERE g.season = %s"
        params = (season,)
    sql += " ORDER BY g.game_date, g.game_id"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def ingest_shifts_for_game(game_id: int, season: int) -> tuple[bool, int, str | None]:
    """Fetch + parse + insert shifts for one game."""
    cache = RAW_DATA_DIR / f"shifts/{season}/{game_id}.json"
    try:
        payload = fetch_json(SHIFTS_URL.format(game_id=game_id), cache_path=cache)
    except FetchError as e:
        return (False, 0, f"fetch: {e}")

    try:
        result = parse_shifts(payload, game_id)
    except Exception as e:
        return (False, 0, f"parse: {type(e).__name__}: {e}")

    if not result.shifts:
        return (False, 0, f"no shifts parsed ({len(result.warnings)} warnings)")

    try:
        with pg_conn() as conn, conn.cursor() as cur:
            n = insert_shifts(cur, result.shifts)
            log_ingest(cur, "shifts", str(game_id), "ok", shots_parsed=n)
            conn.commit()
    except Exception as e:
        return (False, 0, f"db: {type(e).__name__}: {e}")

    return (True, len(result.shifts), None)


def ingest_rosters_for_game(game_id: int, season: int) -> tuple[bool, int, str | None]:
    """Fetch + parse + insert roster for one game via /boxscore endpoint.

    NOTE: This endpoint replaces /landing which stopped carrying roster data
    in the 2024-25 API refresh. Cache goes to raw_data/boxscore/ to avoid
    collision with old (empty-for-rosters) /landing JSONs.
    """
    cache = RAW_DATA_DIR / f"boxscore/{season}/{game_id}.json"
    try:
        payload = fetch_json(BOXSCORE_URL.format(game_id=game_id), cache_path=cache)
    except FetchError as e:
        return (False, 0, f"fetch: {e}")

    try:
        result = parse_landing(payload, game_id)
    except Exception as e:
        return (False, 0, f"parse: {type(e).__name__}: {e}")

    if not result.rosters:
        warn_summary = "; ".join(result.warnings[:3]) if result.warnings else "no warnings"
        return (False, 0, f"no rosters parsed ({warn_summary})")

    try:
        with pg_conn() as conn, conn.cursor() as cur:
            n_r = insert_rosters(cur, result.rosters)
            upsert_player_updates(cur, result.player_updates)
            log_ingest(cur, "rosters", str(game_id), "ok", shots_parsed=n_r)
            conn.commit()
    except Exception as e:
        return (False, 0, f"db: {type(e).__name__}: {e}")

    return (True, len(result.rosters), None)


def run(season: int | None, max_games: int | None, skip_rosters: bool, skip_shifts: bool):
    with pg_conn() as conn:
        all_games = games_in_db(conn, season)
        done_shifts = already_ingested_shifts(conn) if not skip_shifts else set()
        done_rosters = already_ingested_rosters(conn) if not skip_rosters else set()

    if not all_games:
        season_str = f"season {season}" if season else "any season"
        print(f"No games in DB for {season_str}. Run play-by-play ingest first.")
        sys.exit(1)

    todo_shifts = [(g, s) for g, s in all_games if g not in done_shifts]
    todo_rosters = [(g, s) for g, s in all_games if g not in done_rosters]

    if max_games:
        todo_shifts = todo_shifts[:max_games]
        todo_rosters = todo_rosters[:max_games]

    print(f"Total games in DB: {len(all_games)}")
    if not skip_shifts:
        print(f"Shifts: {len(done_shifts)} done, {len(todo_shifts)} to fetch")
    if not skip_rosters:
        print(f"Rosters: {len(done_rosters)} done, {len(todo_rosters)} to fetch")

    if not skip_rosters and todo_rosters:
        ok = fail = total = 0
        for gid, sid in tqdm(todo_rosters, desc="rosters", unit="game"):
            success, n, err = ingest_rosters_for_game(gid, sid)
            if success:
                ok += 1
                total += n
            else:
                fail += 1
                tqdm.write(f"  FAIL roster {gid}: {err}")
                try:
                    with pg_conn() as conn, conn.cursor() as cur:
                        log_ingest(cur, "rosters", str(gid), "fail", error_msg=err)
                        conn.commit()
                except Exception:
                    pass
        print(f"Rosters: {ok} ok, {fail} failed, {total} roster rows stored.")

    if not skip_shifts and todo_shifts:
        ok = fail = total = 0
        for gid, sid in tqdm(todo_shifts, desc="shifts", unit="game"):
            success, n, err = ingest_shifts_for_game(gid, sid)
            if success:
                ok += 1
                total += n
            else:
                fail += 1
                tqdm.write(f"  FAIL shifts {gid}: {err}")
                try:
                    with pg_conn() as conn, conn.cursor() as cur:
                        log_ingest(cur, "shifts", str(gid), "fail", error_msg=err)
                        conn.commit()
                except Exception:
                    pass
        print(f"Shifts: {ok} ok, {fail} failed, {total} shift rows stored.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-schema", action="store_true",
                    help="Apply schema_stage_a.sql (safe to re-run)")
    ap.add_argument("--season", type=int, default=None,
                    help="One season id, e.g. 20242025")
    ap.add_argument("--seasons", default=None,
                    help="'all' to run every season in DB")
    ap.add_argument("--max-games", type=int, default=None,
                    help="Cap games (smoke testing)")
    ap.add_argument("--skip-shifts", action="store_true")
    ap.add_argument("--skip-rosters", action="store_true")
    args = ap.parse_args()

    if args.init_schema:
        schema_path = Path(__file__).resolve().parent.parent / "db" / "schema_stage_a.sql"
        print(f"Applying schema from {schema_path}")
        init_schema(str(schema_path))
        print("Stage A schema ready.")

    season = None
    if args.seasons == "all":
        season = None
    elif args.season:
        season = args.season

    if args.init_schema and not (args.season or args.seasons):
        # Just apply schema, no ingest
        return

    run(season, args.max_games, args.skip_rosters, args.skip_shifts)


if __name__ == "__main__":
    main()