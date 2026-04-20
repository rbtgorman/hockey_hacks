"""Re-parse cached play-by-play JSONs through the current parser.

Use this after a parser fix to rebuild the DB without re-downloading.
Wipes the shots + plays_context tables first (but keeps games and ingest_log,
so the resume logic still works).

Usage:
    python -m ingest.reparse                      # re-parse everything in raw_data/pbp/
    python -m ingest.reparse --season 20222023    # just one season
    python -m ingest.reparse --dry-run            # show what would happen, no writes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from .config import RAW_DATA_DIR
from .parser import parse_pbp
from .db import pg_conn, upsert_game, insert_shots, insert_contexts, log_ingest


def wipe_derived_tables(cur) -> None:
    """Clear shots, plays_context, and ingest_log so re-parse is a clean rebuild.

    We keep `games` — it's harmless because upsert_game will refresh each row.
    """
    print("Wiping shots, plays_context, and ingest_log...")
    cur.execute("TRUNCATE TABLE shots RESTART IDENTITY CASCADE;")
    cur.execute("TRUNCATE TABLE plays_context CASCADE;")
    cur.execute("DELETE FROM ingest_log WHERE kind = 'pbp';")


def iter_cached_games(season_filter: int | None = None):
    """Yield (season_id, game_id, path) for every cached pbp JSON on disk."""
    pbp_root = RAW_DATA_DIR / "pbp"
    if not pbp_root.exists():
        return
    for season_dir in sorted(pbp_root.iterdir()):
        if not season_dir.is_dir():
            continue
        try:
            season_id = int(season_dir.name)
        except ValueError:
            continue
        if season_filter is not None and season_id != season_filter:
            continue
        for json_file in sorted(season_dir.glob("*.json")):
            try:
                game_id = int(json_file.stem)
            except ValueError:
                continue
            yield season_id, game_id, json_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=None,
                   help="Only re-parse one season (e.g. 20222023)")
    p.add_argument("--dry-run", action="store_true",
                   help="Don't write to DB; just count and report")
    args = p.parse_args()

    cached = list(iter_cached_games(args.season))
    print(f"Found {len(cached)} cached pbp JSONs on disk.")
    if not cached:
        print("Nothing to do. Did you run `python -m ingest.run` first?")
        sys.exit(0)

    if args.dry_run:
        print("DRY RUN — not wiping or writing. Summary:")
        by_season = {}
        for s, _, _ in cached:
            by_season[s] = by_season.get(s, 0) + 1
        for s, n in sorted(by_season.items()):
            print(f"  Season {s}: {n} games")
        return

    # Wipe in its own transaction
    if args.season is None:
        with pg_conn() as conn, conn.cursor() as cur:
            wipe_derived_tables(cur)
            conn.commit()
    else:
        # Partial wipe: only this season's games
        with pg_conn() as conn, conn.cursor() as cur:
            print(f"Wiping season {args.season} from shots, plays_context, ingest_log...")
            cur.execute("""
                DELETE FROM shots
                WHERE game_id IN (SELECT game_id FROM games WHERE season = %s)
            """, (args.season,))
            cur.execute("""
                DELETE FROM plays_context
                WHERE game_id IN (SELECT game_id FROM games WHERE season = %s)
            """, (args.season,))
            cur.execute("""
                DELETE FROM ingest_log
                WHERE kind='pbp'
                  AND CAST(target_key AS BIGINT) IN
                      (SELECT game_id FROM games WHERE season = %s)
            """, (args.season,))
            conn.commit()

    ok = 0
    fail = 0
    total_shots = 0
    for season_id, game_id, path in tqdm(cached, desc="reparse", unit="game"):
        try:
            with path.open("r") as f:
                payload = json.load(f)
            result = parse_pbp(payload)

            with pg_conn() as conn, conn.cursor() as cur:
                upsert_game(cur, result.game,
                            raw_json_path=str(path.relative_to(RAW_DATA_DIR.parent)))
                n_shots = insert_shots(cur, result.shots)
                insert_contexts(cur, result.contexts)
                log_ingest(cur, "pbp", str(game_id), "ok", shots_parsed=n_shots)
                conn.commit()

            ok += 1
            total_shots += len(result.shots)
        except Exception as e:
            fail += 1
            tqdm.write(f"  FAIL {game_id}: {type(e).__name__}: {e}")

    print(f"\nReparse complete: {ok} ok, {fail} failed, {total_shots} shots stored.")


if __name__ == "__main__":
    main()