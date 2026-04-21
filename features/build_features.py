"""Build the shot_features table for v1 xG training.

Design decisions:
  - Uses plays_context (ALL events) for previous-event lookup, not just shots.
    This is the whole reason plays_context exists — rebounds happen across
    faceoffs and hits, not just shot-to-shot.
  - Per-shot features only; no player/goalie/team identity priors yet (v2).
  - "Previous event" is strictly temporal (ORDER BY game_seconds, event_idx)
    within the same period. Cross-period leakage is avoided.
  - Rebounds are defined as: previous event was a shot attempt by ANY team
    within 3 seconds AND in the same offensive zone area (x_norm > 25).
    The "same offensive zone" filter avoids labeling an away-team shot
    attempt that came right before a home-team shot at the other end as
    a "rebound" (the detected 3s gap is an artifact of period transitions).
  - Rushes are defined as: shot in the offensive zone (x_norm > 25), where the
    most recent event was a takeaway/giveaway/faceoff outside the OZ (x < 25
    in the event's *own* raw frame = neutral or defensive zone), within 5s.
  - Score state is capped at ±3 (beyond that, teams stop caring).
  - Strength state is kept as raw text (model will one-hot encode).

Run:
    python -m features.build_features
    python -m features.build_features --verify   # print distribution sanity checks
"""
from __future__ import annotations

import argparse

from ingest.db import pg_conn


# The main feature-builder SQL.
# We build two CTEs:
#   1. shot_prev: for each shot, find the single most recent event in
#      plays_context from the SAME game + period that strictly precedes it.
#   2. shot_prev_shot_same_team: for rebound detection, we specifically want
#      the previous SHOT by the same team (not any event) so we can compute
#      sec_since_prev_team_shot.
# Then we compose the final table.
BUILD_SQL = r"""
DROP TABLE IF EXISTS shot_features;

CREATE TABLE shot_features AS
WITH
-- Step 1: For each shot, find its immediately preceding event (any type)
-- in the same game + period, based on game_seconds and event_idx.
-- Uses DISTINCT ON for Postgres-native "top 1 per partition".
shot_prev_any AS (
    SELECT DISTINCT ON (s.game_id, s.event_idx)
        s.game_id,
        s.event_idx,
        c.event_type       AS prev_event_type,
        c.game_seconds     AS prev_game_seconds,
        c.x                AS prev_x_raw,
        c.y                AS prev_y_raw,
        c.team_id          AS prev_team_id
    FROM shots s
    LEFT JOIN plays_context c
        ON c.game_id = s.game_id
       AND c.period  = s.period
       AND (c.game_seconds <  s.game_seconds
            OR (c.game_seconds = s.game_seconds AND c.event_idx < s.event_idx))
    ORDER BY s.game_id, s.event_idx, c.game_seconds DESC, c.event_idx DESC
),

-- Step 2: For rebound detection specifically, find the previous SHOT BY THE
-- SAME TEAM in the same period.
shot_prev_same_team AS (
    SELECT DISTINCT ON (s.game_id, s.event_idx)
        s.game_id,
        s.event_idx,
        s2.game_seconds    AS prev_shot_game_seconds,
        s2.x_norm          AS prev_shot_x_norm,
        s2.y_norm          AS prev_shot_y_norm,
        s2.event_type      AS prev_shot_event_type
    FROM shots s
    LEFT JOIN shots s2
        ON s2.game_id = s.game_id
       AND s2.period  = s.period
       AND s2.shooter_team_id = s.shooter_team_id
       AND (s2.game_seconds <  s.game_seconds
            OR (s2.game_seconds = s.game_seconds AND s2.event_idx < s.event_idx))
    ORDER BY s.game_id, s.event_idx, s2.game_seconds DESC, s2.event_idx DESC
)

SELECT
    -- Identifiers (kept for join-back and debugging)
    s.shot_id,
    s.game_id,
    s.event_idx,
    s.period,
    s.game_seconds,

    -- Target
    s.is_goal,

    -- Core geometry (already normalized in the shots table)
    s.distance_ft,
    s.angle_deg,
    s.x_norm,
    s.y_norm,

    -- Shot type (null -> 'unknown' for encoding stability)
    COALESCE(s.shot_type, 'unknown') AS shot_type,

    -- Game state
    s.strength_state,
    s.empty_net,
    s.period AS period_num,
    GREATEST(0, 1200 - s.period_seconds) AS seconds_remaining_in_period,

    -- Score state from shooter's perspective, capped at [-3, +3]
    LEAST(3, GREATEST(-3,
        CASE
            WHEN s.is_home_shot THEN s.home_score_before - s.away_score_before
            ELSE s.away_score_before - s.home_score_before
        END
    )) AS score_diff,

    -- Temporal context: time since last event (any type)
    COALESCE(s.game_seconds - sp.prev_game_seconds, 1200) AS seconds_since_last_event,
    COALESCE(sp.prev_event_type, 'none') AS last_event_type,

    -- Rebound: previous shot by SAME TEAM, within 3 seconds, AND previous
    -- shot was NOT a goal (goals stop play and cause a faceoff — anything
    -- that looks like a "rebound after a goal" is a data artifact).
    CASE
        WHEN spt.prev_shot_game_seconds IS NOT NULL
         AND spt.prev_shot_event_type IN ('shot-on-goal', 'missed-shot')
         AND (s.game_seconds - spt.prev_shot_game_seconds) <= 3
         AND (s.game_seconds - spt.prev_shot_game_seconds) >= 0
        THEN TRUE
        ELSE FALSE
    END AS is_rebound,

    -- Rush: shot in offensive zone (x_norm > 25) AND previous non-stoppage
    -- event was a turnover or faceoff *outside* the offensive zone within 5 sec.
    -- Using raw x here because plays_context is not normalized.
    --
    -- Why the mixed conventions: rush is "we just transitioned zones quickly."
    -- The previous event being near center ice or in the shooter's defensive
    -- zone is what matters, regardless of which side of the raw coordinate
    -- system "defensive zone" is on for this team. We approximate with
    -- |prev_x_raw| < 25, which covers the neutral zone in either orientation.
    CASE
        WHEN s.x_norm > 25
         AND sp.prev_event_type IN ('takeaway', 'giveaway', 'faceoff', 'hit')
         AND sp.prev_x_raw IS NOT NULL
         AND ABS(sp.prev_x_raw) < 25
         AND (s.game_seconds - sp.prev_game_seconds) <= 5
         AND (s.game_seconds - sp.prev_game_seconds) >= 0
        THEN TRUE
        ELSE FALSE
    END AS is_rush,

    -- Distance the puck "moved" since last event. Useful signal even absent
    -- the rebound/rush flags. Null if we have no previous event or it had
    -- no coords (stoppage, period-start).
    CASE
        WHEN sp.prev_x_raw IS NOT NULL AND sp.prev_y_raw IS NOT NULL
        THEN SQRT(POWER(s.x_raw - sp.prev_x_raw, 2) + POWER(s.y_raw - sp.prev_y_raw, 2))
        ELSE NULL
    END AS distance_from_last_event_ft

FROM shots s
LEFT JOIN shot_prev_any       sp  ON sp.game_id = s.game_id AND sp.event_idx = s.event_idx
LEFT JOIN shot_prev_same_team spt ON spt.game_id = s.game_id AND spt.event_idx = s.event_idx
WHERE s.is_sog = TRUE;   -- only train on SOG + goals, not misses or blocks

-- Indexes for the training script
CREATE INDEX shot_features_game_idx ON shot_features(game_id);
CREATE INDEX shot_features_goal_idx ON shot_features(is_goal) WHERE is_goal = TRUE;
"""


VERIFY_QUERIES = [
    ("Total rows and label balance",
     """
     SELECT COUNT(*) AS total_shots,
            COUNT(*) FILTER (WHERE is_goal) AS goals,
            ROUND(100.0 * COUNT(*) FILTER (WHERE is_goal) / COUNT(*), 2) AS goal_pct
     FROM shot_features;
     """),
    ("Rebound / rush distribution",
     """
     SELECT is_rebound, is_rush, COUNT(*) AS n,
            ROUND(100.0 * COUNT(*) FILTER (WHERE is_goal) / COUNT(*), 2) AS goal_pct
     FROM shot_features
     GROUP BY 1, 2
     ORDER BY 1, 2;
     """),
    ("Goal rate by shot type",
     """
     SELECT shot_type, COUNT(*) AS n,
            ROUND(100.0 * COUNT(*) FILTER (WHERE is_goal) / COUNT(*), 2) AS goal_pct
     FROM shot_features
     GROUP BY shot_type
     ORDER BY goal_pct DESC;
     """),
    ("Goal rate by strength state",
     """
     SELECT strength_state, COUNT(*) AS n,
            ROUND(100.0 * COUNT(*) FILTER (WHERE is_goal) / COUNT(*), 2) AS goal_pct
     FROM shot_features
     WHERE strength_state != 'unknown'
     GROUP BY strength_state
     ORDER BY n DESC
     LIMIT 8;
     """),
    ("Sanity: seconds_since_last_event distribution",
     """
     SELECT CASE
              WHEN seconds_since_last_event <= 3  THEN '0-3s'
              WHEN seconds_since_last_event <= 10 THEN '3-10s'
              WHEN seconds_since_last_event <= 30 THEN '10-30s'
              ELSE '30s+'
            END AS bucket,
            COUNT(*) AS n,
            ROUND(100.0 * COUNT(*) FILTER (WHERE is_goal) / COUNT(*), 2) AS goal_pct
     FROM shot_features
     GROUP BY 1
     ORDER BY MIN(seconds_since_last_event);
     """),
    ("Rebound sanity: rebound goal rate should be 2-3x baseline",
     """
     SELECT is_rebound, COUNT(*) AS n,
            ROUND(100.0 * COUNT(*) FILTER (WHERE is_goal) / COUNT(*), 2) AS goal_pct
     FROM shot_features
     GROUP BY is_rebound;
     """),
    ("Score state effect (trailing teams shoot more, often score less)",
     """
     SELECT score_diff, COUNT(*) AS n,
            ROUND(100.0 * COUNT(*) FILTER (WHERE is_goal) / COUNT(*), 2) AS goal_pct
     FROM shot_features
     GROUP BY score_diff ORDER BY score_diff;
     """),
]


def build():
    with pg_conn() as conn, conn.cursor() as cur:
        print("Building shot_features table (this reads plays_context + shots)...")
        cur.execute(BUILD_SQL)
        conn.commit()
        cur.execute("SELECT COUNT(*) FROM shot_features;")
        n = cur.fetchone()[0]
        print(f"Done. {n:,} rows in shot_features.")


def verify():
    with pg_conn() as conn, conn.cursor() as cur:
        for title, q in VERIFY_QUERIES:
            print(f"\n--- {title} ---")
            cur.execute(q)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            widths = [max(len(str(c)), max((len(str(r[i])) for r in rows), default=0))
                      for i, c in enumerate(cols)]
            print("  " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols)))
            print("  " + "-+-".join("-" * w for w in widths))
            for r in rows:
                print("  " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(cols))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verify-only", action="store_true",
                   help="Skip the build, just run the sanity queries.")
    p.add_argument("--no-verify", action="store_true",
                   help="Build but skip the sanity queries.")
    args = p.parse_args()

    if not args.verify_only:
        build()
    if not args.no_verify:
        verify()


if __name__ == "__main__":
    main()
