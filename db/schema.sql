-- HockeyViz Pro — database schema
-- Designed for 3 seasons of NHL play-by-play + EDGE season snapshots
-- Postgres 13+

-- =====================================================================
-- GAMES
-- =====================================================================
CREATE TABLE IF NOT EXISTS games (
    game_id         BIGINT PRIMARY KEY,           -- NHL game id, e.g. 2024020500
    season          INT NOT NULL,                 -- e.g. 20242025
    game_type       SMALLINT NOT NULL,            -- 2 = reg season, 3 = playoff
    game_date       DATE NOT NULL,
    home_team_id    INT NOT NULL,
    away_team_id    INT NOT NULL,
    home_team_abbr  TEXT NOT NULL,
    away_team_abbr  TEXT NOT NULL,
    venue           TEXT,
    home_score      INT,
    away_score      INT,
    game_state      TEXT,                         -- FINAL, LIVE, OFF, etc.
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    raw_json_path   TEXT                          -- relative path to cached JSON
);
CREATE INDEX IF NOT EXISTS games_season_idx ON games(season);
CREATE INDEX IF NOT EXISTS games_date_idx ON games(game_date);

-- =====================================================================
-- PLAYERS (dim table — we fill as we see them in play-by-play)
-- =====================================================================
CREATE TABLE IF NOT EXISTS players (
    player_id       BIGINT PRIMARY KEY,
    full_name       TEXT,
    position        TEXT,                         -- F, D, G (when we learn it)
    shoots_catches  TEXT,                         -- L / R
    first_seen      DATE,
    last_seen       DATE
);

-- =====================================================================
-- SHOTS — one row per shot attempt (SOG, goal, miss, block)
-- This is the training table for xG
-- =====================================================================
CREATE TABLE IF NOT EXISTS shots (
    shot_id             BIGSERIAL PRIMARY KEY,
    game_id             BIGINT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    event_idx           INT NOT NULL,             -- position in plays[] array
    period              SMALLINT NOT NULL,
    period_type         TEXT,                     -- REG, OT, SO
    period_seconds      INT NOT NULL,             -- seconds elapsed in period
    game_seconds        INT NOT NULL,             -- total seconds elapsed
    event_type          TEXT NOT NULL,            -- shot-on-goal | goal | missed-shot | blocked-shot
    is_goal             BOOLEAN NOT NULL,         -- TARGET for xG
    is_sog              BOOLEAN NOT NULL,         -- SOG (goal or saved)
    -- geometry (normalized so shooter always attacks the +x net)
    x_raw               DOUBLE PRECISION,
    y_raw               DOUBLE PRECISION,
    x_norm              DOUBLE PRECISION,         -- always positive side
    y_norm              DOUBLE PRECISION,
    distance_ft         DOUBLE PRECISION,         -- from shot location to goal (89, 0)
    angle_deg           DOUBLE PRECISION,         -- absolute angle off center line
    -- shot context
    shot_type           TEXT,                     -- wrist, snap, slap, backhand, tip-in, deflected, wrap-around
    zone_code           TEXT,                     -- O, N, D (from shooter's perspective)
    -- actors
    shooter_id          BIGINT,                   -- NOT FK — fill as we see
    goalie_id           BIGINT,                   -- may be null (empty net)
    shooter_team_id     INT NOT NULL,
    defending_team_id   INT NOT NULL,
    is_home_shot        BOOLEAN NOT NULL,         -- shooter's team = home team
    -- strength state (from situationCode: 4-char string e.g. "1551")
    situation_code      TEXT,                     -- raw 4-char
    strength_state      TEXT,                     -- '5v5','5v4','4v5','5v3','3v5','4v4','3v3','EN_for','EN_against'
    empty_net           BOOLEAN NOT NULL DEFAULT FALSE,
    -- score state at time of shot
    home_score_before   INT,
    away_score_before   INT,
    -- raw
    raw_details_json    JSONB,                    -- stash original for later re-parsing
    UNIQUE (game_id, event_idx)
);
CREATE INDEX IF NOT EXISTS shots_game_idx ON shots(game_id);
CREATE INDEX IF NOT EXISTS shots_shooter_idx ON shots(shooter_id);
CREATE INDEX IF NOT EXISTS shots_goalie_idx ON shots(goalie_id);
CREATE INDEX IF NOT EXISTS shots_shooter_goalie_idx ON shots(shooter_id, goalie_id);  -- matchup queries
CREATE INDEX IF NOT EXISTS shots_goal_idx ON shots(is_goal) WHERE is_goal = TRUE;

-- =====================================================================
-- NON-SHOT EVENTS — needed for context features (rebound, rush, prior event)
-- Kept lean: we only store what feature engineering needs
-- =====================================================================
CREATE TABLE IF NOT EXISTS plays_context (
    game_id         BIGINT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    event_idx       INT NOT NULL,
    period          SMALLINT NOT NULL,
    game_seconds    INT NOT NULL,
    event_type      TEXT NOT NULL,                -- faceoff, hit, giveaway, takeaway, penalty, stoppage, etc.
    x               DOUBLE PRECISION,
    y               DOUBLE PRECISION,
    team_id         INT,
    PRIMARY KEY (game_id, event_idx)
);
CREATE INDEX IF NOT EXISTS plays_context_game_time_idx ON plays_context(game_id, game_seconds);

-- =====================================================================
-- EDGE SNAPSHOTS (season-level) — priors to join onto shots at training time
-- =====================================================================
CREATE TABLE IF NOT EXISTS edge_skater_season (
    player_id               BIGINT NOT NULL,
    season                  INT NOT NULL,
    game_type               SMALLINT NOT NULL,
    top_shot_speed_mph      DOUBLE PRECISION,
    avg_shot_speed_mph      DOUBLE PRECISION,
    shots_100_plus          INT,
    shots_90_100            INT,
    shots_80_90             INT,
    shots_70_80             INT,
    top_skating_speed_mph   DOUBLE PRECISION,
    bursts_over_22mph       INT,
    total_skating_dist_mi   DOUBLE PRECISION,
    off_zone_time_pct       DOUBLE PRECISION,
    def_zone_time_pct       DOUBLE PRECISION,
    raw_json                JSONB,
    fetched_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (player_id, season, game_type)
);

CREATE TABLE IF NOT EXISTS edge_goalie_season (
    player_id               BIGINT NOT NULL,
    season                  INT NOT NULL,
    game_type               SMALLINT NOT NULL,
    gaa                     DOUBLE PRECISION,
    save_pct                DOUBLE PRECISION,
    games_above_900         INT,
    goal_diff_per_60        DOUBLE PRECISION,
    -- save % by shot-location bin (NHL provides 16 bins; we flatten the key ones)
    sv_pct_high_danger      DOUBLE PRECISION,
    sv_pct_mid_danger       DOUBLE PRECISION,
    sv_pct_low_danger       DOUBLE PRECISION,
    raw_json                JSONB,
    fetched_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (player_id, season, game_type)
);

-- =====================================================================
-- INGESTION LEDGER — know what we've pulled, skip on re-run
-- =====================================================================
CREATE TABLE IF NOT EXISTS ingest_log (
    id              BIGSERIAL PRIMARY KEY,
    kind            TEXT NOT NULL,                -- 'schedule' | 'pbp' | 'edge_skater' | 'edge_goalie'
    target_key      TEXT NOT NULL,                -- date / game_id / player_id-season
    status          TEXT NOT NULL,                -- 'ok' | 'fail' | 'empty'
    shots_parsed    INT,
    error_msg       TEXT,
    http_status     INT,
    completed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (kind, target_key)
);

-- =====================================================================
-- Helpful views for downstream
-- =====================================================================
CREATE OR REPLACE VIEW v_shot_rate_by_season AS
SELECT
    g.season,
    COUNT(*) FILTER (WHERE s.is_sog) AS sog_total,
    COUNT(*) FILTER (WHERE s.is_goal) AS goals_total,
    COUNT(*) FILTER (WHERE s.is_goal)::FLOAT / NULLIF(COUNT(*) FILTER (WHERE s.is_sog), 0) AS shooting_pct,
    COUNT(DISTINCT s.game_id) AS games
FROM shots s
JOIN games g ON g.game_id = s.game_id
GROUP BY g.season
ORDER BY g.season;
