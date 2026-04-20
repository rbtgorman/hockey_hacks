"""Main ingestion orchestrator.

Usage:
    python -m ingest.run --init-schema
    python -m ingest.run --seasons all
    python -m ingest.run --seasons 20242025 --max-games 50  # quick test
    python -m ingest.run --resume                            # pick up where it left off

Flow per season:
  1. Walk dates from start to end, hit /schedule/{date}, collect game_ids
  2. For each game_id not already in ingest_log, fetch /gamecenter/{id}/play-by-play
  3. Parse -> upsert game, insert shots + contexts, log success
  4. Raw JSONs cached to raw_data/{season}/{game_id}.json
"""
from __future__ import annotations

import argparse
import sys
import traceback
from datetime import date, timedelta
from pathlib import Path

from tqdm import tqdm

from .config import SEASONS, SCHEDULE_URL, PBP_URL, RAW_DATA_DIR
from .http_client import fetch_json, FetchError
from .parser import parse_pbp
from .db import (
    pg_conn, init_schema, upsert_game, insert_shots, insert_contexts,
    log_ingest, already_ingested_games,
)


def iter_dates(start: str, end: str):
    """Yield YYYY-MM-DD strings from start to end inclusive."""
    d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    while d <= end_d:
        yield d.isoformat()
        d += timedelta(days=1)


def collect_game_ids_for_season(season_id: int, start: str, end: str) -> list[tuple[int, str]]:
    """Return list of (game_id, game_date) for one season.

    The /schedule/{date} endpoint returns a 'gameWeek' with ~7 days of games,
    but we call it per-date and dedupe because week boundaries are unstable.
    """
    seen: dict[int, str] = {}
    dates = list(iter_dates(start, end))

    for d in tqdm(dates, desc=f"schedule {season_id}", unit="day"):
        cache = RAW_DATA_DIR / f"schedule/{season_id}/{d}.json"
        try:
            data = fetch_json(SCHEDULE_URL.format(date=d), cache_path=cache)
        except FetchError as e:
            # Most schedule fetches succeed; log and continue
            tqdm.write(f"  schedule fail {d}: {e}")
            continue

        # Response shape: { "gameWeek": [ { "date": ..., "games": [ {id, gameType, ...} ] } ] }
        for week_day in data.get("gameWeek", []) or []:
            wd_date = week_day.get("date")
            # Only take games dated exactly on d (the week may overflow)
            if wd_date != d:
                continue
            for g in week_day.get("games", []) or []:
                # Only regular + playoff (skip preseason=1, all-star=4 etc)
                if g.get("gameType") not in (2, 3):
                    continue
                gid = g.get("id")
                if gid and gid not in seen:
                    seen[int(gid)] = wd_date

    return sorted(seen.items())


def ingest_one_game(game_id: int, season_id: int) -> tuple[bool, int, str | None]:
    """Fetch + parse + write a single game. Returns (ok, shots_parsed, err_msg)."""
    cache = RAW_DATA_DIR / f"pbp/{season_id}/{game_id}.json"
    try:
        payload = fetch_json(PBP_URL.format(game_id=game_id), cache_path=cache)
    except FetchError as e:
        return (False, 0, f"fetch: {e}")

    try:
        result = parse_pbp(payload)
    except Exception as e:
        return (False, 0, f"parse: {type(e).__name__}: {e}")

    # Skip games that haven't happened yet (state FUT / PRE)
    if result.game.game_state in ("FUT", "PRE"):
        return (False, 0, f"game not played yet (state={result.game.game_state})")

    # Write to DB in a single transaction per game
    try:
        with pg_conn() as conn, conn.cursor() as cur:
            upsert_game(cur, result.game, raw_json_path=str(cache.relative_to(RAW_DATA_DIR.parent)))
            n_shots = insert_shots(cur, result.shots)
            insert_contexts(cur, result.contexts)
            log_ingest(cur, "pbp", str(game_id), "ok", shots_parsed=n_shots)
            conn.commit()
    except Exception as e:
        return (False, 0, f"db: {type(e).__name__}: {e}")

    return (True, len(result.shots), None)


def run(seasons: list[tuple[int, str, str]], max_games: int | None = None,
        resume: bool = True) -> None:
    # Load already-done set once
    if resume:
        with pg_conn() as conn:
            done = already_ingested_games(conn)
        print(f"Resume: {len(done)} games already ingested.")
    else:
        done = set()

    for season_id, start, end in seasons:
        print(f"\n=== Season {season_id} ({start} -> {end}) ===")
        game_list = collect_game_ids_for_season(season_id, start, end)
        print(f"Found {len(game_list)} games in schedule.")

        to_do = [(gid, gdate) for (gid, gdate) in game_list if gid not in done]
        if max_games:
            to_do = to_do[:max_games]
        print(f"{len(to_do)} new games to ingest ({len(game_list) - len(to_do)} already done).")

        ok = 0
        fail = 0
        total_shots = 0
        for gid, gdate in tqdm(to_do, desc=f"pbp {season_id}", unit="game"):
            success, n_shots, err = ingest_one_game(gid, season_id)
            if success:
                ok += 1
                total_shots += n_shots
            else:
                fail += 1
                tqdm.write(f"  FAIL game {gid}: {err}")
                # Log the failure so we can skip or retry intentionally
                try:
                    with pg_conn() as conn, conn.cursor() as cur:
                        log_ingest(cur, "pbp", str(gid), "fail", error_msg=err)
                        conn.commit()
                except Exception:
                    pass

        print(f"Season {season_id}: {ok} ok, {fail} failed, {total_shots} shots stored.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init-schema", action="store_true",
                   help="Create tables (safe to run on existing DB).")
    p.add_argument("--seasons", default="all",
                   help="'all' or comma-list of season_ids, e.g. '20232024,20242025'")
    p.add_argument("--max-games", type=int, default=None,
                   help="Cap games per season (for testing).")
    p.add_argument("--no-resume", action="store_true",
                   help="Re-ingest games even if already logged.")
    args = p.parse_args()

    if args.init_schema:
        schema_path = Path(__file__).resolve().parent.parent / "db" / "schema.sql"
        print(f"Initializing schema from {schema_path}")
        init_schema(str(schema_path))
        print("Schema ready.")

    if args.seasons == "all":
        seasons = SEASONS
    else:
        wanted = {int(s.strip()) for s in args.seasons.split(",")}
        seasons = [s for s in SEASONS if s[0] in wanted]
        if not seasons:
            print(f"No matching seasons in config. Available: {[s[0] for s in SEASONS]}")
            sys.exit(1)

    run(seasons, max_games=args.max_games, resume=not args.no_resume)


if __name__ == "__main__":
    main()
