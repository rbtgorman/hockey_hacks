"""Database writer — batch upserts with psycopg2."""
from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Iterable

import psycopg2
from psycopg2.extras import execute_values, Json

from .config import PG_DSN
from .parser import GameRow, ShotRow, ContextRow


@contextmanager
def pg_conn():
    conn = psycopg2.connect(PG_DSN)
    try:
        yield conn
    finally:
        conn.close()


def init_schema(schema_path: str) -> None:
    with open(schema_path) as f:
        sql = f.read()
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()


def upsert_game(cur, g: GameRow, raw_json_path: str | None = None) -> None:
    cur.execute("""
        INSERT INTO games (
            game_id, season, game_type, game_date,
            home_team_id, away_team_id, home_team_abbr, away_team_abbr,
            venue, home_score, away_score, game_state, raw_json_path
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (game_id) DO UPDATE SET
            home_score = EXCLUDED.home_score,
            away_score = EXCLUDED.away_score,
            game_state = EXCLUDED.game_state,
            ingested_at = NOW()
    """, (
        g.game_id, g.season, g.game_type, g.game_date,
        g.home_team_id, g.away_team_id, g.home_team_abbr, g.away_team_abbr,
        g.venue, g.home_score, g.away_score, g.game_state, raw_json_path,
    ))


def insert_shots(cur, shots: Iterable[ShotRow]) -> int:
    """Bulk insert shots. Returns count inserted.

    Uses ON CONFLICT DO NOTHING on (game_id, event_idx) so re-runs are idempotent.
    """
    rows = [
        (
            s.game_id, s.event_idx, s.period, s.period_type,
            s.period_seconds, s.game_seconds, s.event_type,
            s.is_goal, s.is_sog,
            s.x_raw, s.y_raw, s.x_norm, s.y_norm,
            s.distance_ft, s.angle_deg,
            s.shot_type, s.zone_code,
            s.shooter_id, s.goalie_id,
            s.shooter_team_id, s.defending_team_id, s.is_home_shot,
            s.situation_code, s.strength_state, s.empty_net,
            s.home_score_before, s.away_score_before,
            Json(s.raw_details_json),
        )
        for s in shots
    ]
    if not rows:
        return 0
    execute_values(cur, """
        INSERT INTO shots (
            game_id, event_idx, period, period_type,
            period_seconds, game_seconds, event_type,
            is_goal, is_sog,
            x_raw, y_raw, x_norm, y_norm,
            distance_ft, angle_deg,
            shot_type, zone_code,
            shooter_id, goalie_id,
            shooter_team_id, defending_team_id, is_home_shot,
            situation_code, strength_state, empty_net,
            home_score_before, away_score_before,
            raw_details_json
        ) VALUES %s
        ON CONFLICT (game_id, event_idx) DO NOTHING
    """, rows)
    return cur.rowcount


def insert_contexts(cur, ctxs: Iterable[ContextRow]) -> int:
    rows = [
        (c.game_id, c.event_idx, c.period, c.game_seconds,
         c.event_type, c.x, c.y, c.team_id)
        for c in ctxs
    ]
    if not rows:
        return 0
    execute_values(cur, """
        INSERT INTO plays_context (
            game_id, event_idx, period, game_seconds,
            event_type, x, y, team_id
        ) VALUES %s
        ON CONFLICT (game_id, event_idx) DO NOTHING
    """, rows)
    return cur.rowcount


def log_ingest(cur, kind: str, target_key: str, status: str,
               shots_parsed: int | None = None,
               error_msg: str | None = None,
               http_status: int | None = None) -> None:
    cur.execute("""
        INSERT INTO ingest_log (kind, target_key, status, shots_parsed, error_msg, http_status)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (kind, target_key) DO UPDATE SET
            status = EXCLUDED.status,
            shots_parsed = EXCLUDED.shots_parsed,
            error_msg = EXCLUDED.error_msg,
            http_status = EXCLUDED.http_status,
            completed_at = NOW()
    """, (kind, target_key, status, shots_parsed, error_msg, http_status))


def already_ingested_games(conn) -> set[int]:
    """Return set of game_ids already successfully ingested (for skip on resume)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT CAST(target_key AS BIGINT) FROM ingest_log
            WHERE kind = 'pbp' AND status = 'ok'
        """)
        return {r[0] for r in cur.fetchall()}
