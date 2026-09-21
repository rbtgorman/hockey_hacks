"""Build shot_features_fenwick: xG features on unblocked shot attempts.

WHY THIS EXISTS
---------------
features/build_features.py builds shot_features from shots on goal only
(is_sog = TRUE). Starting in 2023-24 the NHL records many pucks the goalie
touched on their way wide as missed shots. From 2022-23 to 2024-25,
conversion per shot on goal rose 6-9% in each distance band under 60 ft,
while conversion per unblocked attempt stayed flat
(db/checks/c4_shot_regime_by_season.sql). Goals per unblocked attempt do not
move when a shot is relabeled from on-net to missed, so that is the
population the model trains on from here.

build_features.py stays frozen so the shots-on-goal leaderboard (v1-v2.3)
remains reproducible. This builder writes its own table and never touches
shot_features. A rebuild drops and recreates shot_features_fenwick only.

DIFFERENCES FROM build_features.py
----------------------------------
1. Population: shot-on-goal, goal and missed-shot (was is_sog only).
2. Shootout attempts are excluded (regular-season period >= 5).
   Playoff overtime periods are real play and are kept.
3. Two event types exist in only part of the 2022-25 window:
       teammate-blocked      logged from 2023-24
       failed-bank-attempt   logged from 2024-25
   Both are removed from the population and from every previous-event
   lookup. Left in, a teammate block between a shot and its rebound turns
   is_rebound off, and one between a takeaway and a shot turns is_rush off,
   in 2023-24 onward only.
4. last_event_type merges shot-on-goal and missed-shot into
   'unblocked-shot', because the relabel moves pucks between those two
   labels across seasons. is_rebound already accepted both.
5. event_type is kept as a column for diagnostics and the Stage D on-net
   layer. It is not a model feature.
6. shot_type merges wrist and snap into 'wrist-snap'. In 2024-25 about
   7,500 attempts moved from one label to the other: wrist fell from 54.5%
   to 48.0% of attempts in a single season and snap rose from 14.1% to
   21.1%, while the two combined held at 68.3% / 68.6% / 69.1%. Shooting
   doesn't change like that in one season; the labelling did.
7. A missing shot type becomes 'wrist-snap', the most common type. The
   legacy table's 'unknown' appears only on goals (100% goal rate in every
   season), so as its own category it tells the model the answer. Leaving
   it NULL would not help: LightGBM learns a direction for missing values,
   and it would learn the same thing.

Every other feature is defined exactly as in build_features.py. The first
verify query checks that claim: on 2022-23, where neither regime-only event
exists, the two tables must agree on every shared row.

RUN
---
    python3 -m features.build_features_fenwick
    python3 -m features.build_features_fenwick --verify-only
"""
from __future__ import annotations

import argparse

from ingest.db import pg_conn


BUILD_SQL = r"""
DROP TABLE IF EXISTS shot_features_fenwick;

CREATE TABLE shot_features_fenwick AS
WITH
-- Events that exist in only part of the 2022-25 window. Removed from the
-- population and from every previous-event lookup, so all three seasons
-- share one event stream. shots and plays_context use the same event_idx
-- (verified: 7,738 of 7,738 teammate blocks matched).
regime_only AS (
    SELECT game_id, event_idx
    FROM shots
    WHERE raw_details_json::jsonb ->> 'reason'
          IN ('teammate-blocked', 'failed-bank-attempt')
),

shots_kept AS (
    SELECT s.shot_id, s.game_id, s.event_idx, s.period, s.period_seconds,
           s.game_seconds, s.event_type, s.is_goal,
           s.distance_ft, s.angle_deg, s.x_raw, s.y_raw, s.x_norm, s.y_norm,
           s.shot_type, s.strength_state, s.empty_net, s.is_home_shot,
           s.home_score_before, s.away_score_before, s.shooter_team_id
    FROM shots s
    WHERE NOT EXISTS (
        SELECT 1 FROM regime_only r
        WHERE r.game_id = s.game_id AND r.event_idx = s.event_idx
    )
),

ctx_kept AS (
    SELECT c.game_id, c.event_idx, c.period, c.game_seconds,
           c.event_type, c.x, c.y
    FROM plays_context c
    WHERE NOT EXISTS (
        SELECT 1 FROM regime_only r
        WHERE r.game_id = c.game_id AND r.event_idx = c.event_idx
    )
),

-- Population: unblocked attempts. Shootout attempts are out
-- (regular season is game type 02; its period 5 is the shootout).
pop AS (
    SELECT *
    FROM shots_kept
    WHERE event_type IN ('shot-on-goal', 'goal', 'missed-shot')
      AND NOT (substr(game_id::text, 5, 2) = '02' AND period >= 5)
),

-- Most recent event of any type in the same game and period.
shot_prev_any AS (
    SELECT DISTINCT ON (s.game_id, s.event_idx)
        s.game_id,
        s.event_idx,
        c.event_type   AS prev_event_type,
        c.game_seconds AS prev_game_seconds,
        c.x            AS prev_x_raw,
        c.y            AS prev_y_raw
    FROM pop s
    LEFT JOIN ctx_kept c
        ON c.game_id = s.game_id
       AND c.period  = s.period
       AND (c.game_seconds <  s.game_seconds
            OR (c.game_seconds = s.game_seconds AND c.event_idx < s.event_idx))
    ORDER BY s.game_id, s.event_idx, c.game_seconds DESC, c.event_idx DESC
),

-- Most recent shot of any type by the same team in the same period.
shot_prev_same_team AS (
    SELECT DISTINCT ON (s.game_id, s.event_idx)
        s.game_id,
        s.event_idx,
        s2.game_seconds AS prev_shot_game_seconds,
        s2.event_type   AS prev_shot_event_type
    FROM pop s
    LEFT JOIN shots_kept s2
        ON s2.game_id = s.game_id
       AND s2.period  = s.period
       AND s2.shooter_team_id = s.shooter_team_id
       AND (s2.game_seconds <  s.game_seconds
            OR (s2.game_seconds = s.game_seconds AND s2.event_idx < s.event_idx))
    ORDER BY s.game_id, s.event_idx, s2.game_seconds DESC, s2.event_idx DESC
)

SELECT
    -- Identifiers
    s.shot_id,
    s.game_id,
    s.event_idx,
    s.period,
    s.game_seconds,

    -- Population label (diagnostics / on-net layer only, not a feature)
    s.event_type,

    -- Target
    s.is_goal,

    -- Geometry (normalized in the shots table)
    s.distance_ft,
    s.angle_deg,
    s.x_norm,
    s.y_norm,

    CASE
        WHEN s.shot_type IS NULL
          OR s.shot_type IN ('wrist', 'snap', 'unknown') THEN 'wrist-snap'
        ELSE s.shot_type
    END AS shot_type,

    -- Game state
    s.strength_state,
    s.empty_net,
    s.period AS period_num,
    GREATEST(0, 1200 - s.period_seconds) AS seconds_remaining_in_period,
    LEAST(3, GREATEST(-3,
        CASE
            WHEN s.is_home_shot THEN s.home_score_before - s.away_score_before
            ELSE s.away_score_before - s.home_score_before
        END
    )) AS score_diff,

    -- Temporal context
    COALESCE(s.game_seconds - sp.prev_game_seconds, 1200) AS seconds_since_last_event,

    CASE
        WHEN sp.prev_event_type IN ('shot-on-goal', 'missed-shot') THEN 'unblocked-shot'
        ELSE COALESCE(sp.prev_event_type, 'none')
    END AS last_event_type,

    -- Rebound: previous same-team shot was on goal or missed, within 3 s.
    CASE
        WHEN spt.prev_shot_game_seconds IS NOT NULL
         AND spt.prev_shot_event_type IN ('shot-on-goal', 'missed-shot')
         AND (s.game_seconds - spt.prev_shot_game_seconds) <= 3
         AND (s.game_seconds - spt.prev_shot_game_seconds) >= 0
        THEN TRUE
        ELSE FALSE
    END AS is_rebound,

    -- Rush: offensive-zone shot within 5 s of a turnover, faceoff or hit
    -- near center ice (|raw x| < 25; plays_context is not normalized).
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

    CASE
        WHEN sp.prev_x_raw IS NOT NULL AND sp.prev_y_raw IS NOT NULL
        THEN SQRT(POWER(s.x_raw - sp.prev_x_raw, 2) + POWER(s.y_raw - sp.prev_y_raw, 2))
        ELSE NULL
    END AS distance_from_last_event_ft

FROM pop s
LEFT JOIN shot_prev_any       sp  ON sp.game_id  = s.game_id AND sp.event_idx  = s.event_idx
LEFT JOIN shot_prev_same_team spt ON spt.game_id = s.game_id AND spt.event_idx = s.event_idx;

CREATE INDEX shot_features_fenwick_game_idx       ON shot_features_fenwick (game_id);
CREATE INDEX shot_features_fenwick_game_event_idx ON shot_features_fenwick (game_id, event_idx);
"""


VERIFY_QUERIES = [
    ("1. Parity with legacy shot_features on shared rows "
     "(2022 must be all zeros; 2023/2024 nonzero only in the context columns)",
     """
     SELECT substr(n.game_id::text, 1, 4) AS season_start,
            COUNT(*) AS rows_compared,
            COUNT(*) FILTER (WHERE n.distance_ft IS DISTINCT FROM o.distance_ft
                                OR n.angle_deg IS DISTINCT FROM o.angle_deg
                                OR n.shot_type IS DISTINCT FROM
                                   CASE WHEN o.shot_type IN ('wrist', 'snap', 'unknown')
                                        THEN 'wrist-snap'
                                        ELSE o.shot_type END
                                OR n.strength_state IS DISTINCT FROM o.strength_state
                                OR n.score_diff IS DISTINCT FROM o.score_diff
                                OR n.seconds_remaining_in_period
                                   IS DISTINCT FROM o.seconds_remaining_in_period)
                AS static_cols,
            COUNT(*) FILTER (WHERE n.seconds_since_last_event
                                   IS DISTINCT FROM o.seconds_since_last_event) AS secs_since_last,
            COUNT(*) FILTER (WHERE n.distance_from_last_event_ft
                                   IS DISTINCT FROM o.distance_from_last_event_ft) AS dist_from_last,
            COUNT(*) FILTER (WHERE n.last_event_type IS DISTINCT FROM
                                   CASE WHEN o.last_event_type IN ('shot-on-goal', 'missed-shot')
                                        THEN 'unblocked-shot'
                                        ELSE o.last_event_type END) AS last_event,
            COUNT(*) FILTER (WHERE n.is_rebound IS DISTINCT FROM o.is_rebound) AS is_rebound,
            COUNT(*) FILTER (WHERE n.is_rush IS DISTINCT FROM o.is_rush) AS is_rush
     FROM shot_features_fenwick n
     JOIN shot_features o
       ON o.game_id = n.game_id AND o.event_idx = n.event_idx
     GROUP BY 1
     ORDER BY 1;
     """),
    ("2. Legacy rows missing from the new table (expected: only game type 02, period 5)",
     """
     SELECT substr(o.game_id::text, 1, 4) AS season_start,
            substr(o.game_id::text, 5, 2) AS game_type,
            o.period,
            COUNT(*) AS legacy_rows_missing
     FROM shot_features o
     WHERE NOT EXISTS (
         SELECT 1 FROM shot_features_fenwick n
         WHERE n.game_id = o.game_id AND n.event_idx = o.event_idx
     )
     GROUP BY 1, 2, 3
     ORDER BY 1, 2, 3;
     """),
    ("3. Rows and goals by season and event type (missed-shot goals must be 0)",
     """
     SELECT substr(f.game_id::text, 1, 4) AS season_start,
            f.event_type,
            COUNT(*) AS n,
            COUNT(*) FILTER (WHERE f.is_goal) AS goals
     FROM shot_features_fenwick f
     GROUP BY 1, 2
     ORDER BY 1, 2;
     """),
    ("4. Goal rate by distance band, regular season, no empty net "
     "(each band should be flat across seasons)",
     """
     SELECT CASE
                WHEN f.distance_ft IS NULL THEN 'e_null'
                WHEN f.distance_ft < 20 THEN 'a_lt20'
                WHEN f.distance_ft < 40 THEN 'b_20_40'
                WHEN f.distance_ft < 60 THEN 'c_40_60'
                ELSE 'd_60plus'
            END AS band,
            substr(f.game_id::text, 1, 4) AS season_start,
            COUNT(*) AS n,
            ROUND(100.0 * COUNT(*) FILTER (WHERE f.is_goal) / COUNT(*), 2) AS goal_pct
     FROM shot_features_fenwick f
     WHERE substr(f.game_id::text, 5, 2) = '02'
       AND NOT COALESCE(f.empty_net, false)
     GROUP BY 1, 2
     ORDER BY 1, 2;
     """),
    ("5. Attempts from behind the goal line per game, regular season "
     "(flat if dropping failed-bank-attempt was right)",
     """
     SELECT substr(f.game_id::text, 1, 4) AS season_start,
            ROUND((COUNT(*) FILTER (WHERE f.x_norm > 89))::numeric
                  / COUNT(DISTINCT f.game_id), 3) AS behind_line_per_game
     FROM shot_features_fenwick f
     WHERE substr(f.game_id::text, 5, 2) = '02'
     GROUP BY 1
     ORDER BY 1;
     """),
    ("6. Goal rate by model shot_type and season, regular season, no empty net "
     "(each type should be roughly flat)",
     """
     SELECT f.shot_type,
            substr(f.game_id::text, 1, 4) AS season_start,
            COUNT(*) AS n,
            ROUND(100.0 * COUNT(*) FILTER (WHERE f.is_goal) / COUNT(*), 2) AS goal_pct
     FROM shot_features_fenwick f
     WHERE substr(f.game_id::text, 5, 2) = '02'
       AND NOT COALESCE(f.empty_net, false)
     GROUP BY 1, 2
     ORDER BY 1, 2;
     """),
    ("7. Raw wrist vs snap by season (why they are merged)",
     """
     SELECT s.shot_type AS raw_shot_type,
            substr(f.game_id::text, 1, 4) AS season_start,
            COUNT(*) AS n,
            ROUND(100.0 * COUNT(*) FILTER (WHERE f.is_goal) / COUNT(*), 2) AS goal_pct,
            ROUND(AVG(f.distance_ft)::numeric, 1) AS avg_ft
     FROM shot_features_fenwick f
     JOIN shots s ON s.game_id = f.game_id AND s.event_idx = f.event_idx
     WHERE substr(f.game_id::text, 5, 2) = '02'
       AND NOT COALESCE(f.empty_net, false)
       AND s.shot_type IN ('wrist', 'snap')
     GROUP BY 1, 2
     ORDER BY 1, 2;
     """),
    ("8. Categorical values that give away the answer (expected: 0 rows). "
     "Flags any value that is always a goal (n >= 10) or never a goal (n >= 200)",
     """
     WITH v AS (
         SELECT 'shot_type' AS feature, shot_type AS value, is_goal
         FROM shot_features_fenwick
         UNION ALL
         SELECT 'strength_state', strength_state, is_goal
         FROM shot_features_fenwick
         UNION ALL
         SELECT 'last_event_type', last_event_type, is_goal
         FROM shot_features_fenwick
     )
     SELECT feature,
            value,
            COUNT(*) AS n,
            COUNT(*) FILTER (WHERE is_goal) AS goals
     FROM v
     GROUP BY 1, 2
     HAVING (COUNT(*) >= 10  AND COUNT(*) FILTER (WHERE is_goal) = COUNT(*))
         OR (COUNT(*) >= 200 AND COUNT(*) FILTER (WHERE is_goal) = 0)
     ORDER BY 1, 3 DESC;
     """),
]


def build() -> None:
    with pg_conn() as conn, conn.cursor() as cur:
        print("Building shot_features_fenwick (reads shots + plays_context)...")
        cur.execute(BUILD_SQL)
        conn.commit()
        cur.execute("SELECT COUNT(*) FROM shot_features_fenwick;")
        n = cur.fetchone()[0]
        print(f"Done. {n:,} rows in shot_features_fenwick.")


def verify() -> None:
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
            if not rows:
                print("  (0 rows)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build shot_features_fenwick (unblocked attempts)."
    )
    p.add_argument("--verify-only", action="store_true",
                   help="Skip the build, just run the checks.")
    args = p.parse_args()
    if not args.verify_only:
        build()
    verify()


if __name__ == "__main__":
    main()