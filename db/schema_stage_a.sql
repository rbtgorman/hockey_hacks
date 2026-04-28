-- HockeyViz Pro — Stage A schema additions
-- Shift data and game rosters. Safe to run alongside existing schema.sql.
-- Run this AFTER schema.sql has already been applied.

-- =====================================================================
-- SHIFTS — one row per player-shift
-- Joined to shots later via (game_id, period, game_seconds in [start, end])
-- =====================================================================
CREATE TABLE IF NOT EXISTS shifts (
    shift_id        BIGSERIAL PRIMARY KEY,
    game_id         BIGINT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    player_id       BIGINT NOT NULL,
    team_id         INT NOT NULL,
    period          SMALLINT NOT NULL,
    start_seconds   INT NOT NULL,   -- game_seconds (0-3600 regulation, higher in OT)
    end_seconds     INT NOT NULL,
    duration_sec    INT GENERATED ALWAYS AS (end_seconds - start_seconds) STORED,
    shift_number    INT,            -- nth shift for this player this game
    raw_json        JSONB,          -- stash source record for later re-parse
    UNIQUE (game_id, player_id, period, start_seconds)
);
CREATE INDEX IF NOT EXISTS shifts_game_idx     ON shifts(game_id);
CREATE INDEX IF NOT EXISTS shifts_player_idx   ON shifts(player_id);
-- The critical index for shot-to-shift joins:
CREATE INDEX IF NOT EXISTS shifts_game_period_time_idx
    ON shifts(game_id, period, start_seconds, end_seconds);

-- =====================================================================
-- GAME ROSTERS — who was dressed, who started, who's the goalie
-- Sourced from gamecenter/{id}/landing (richer than play-by-play actor IDs alone)
-- =====================================================================
CREATE TABLE IF NOT EXISTS game_rosters (
    game_id         BIGINT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    player_id       BIGINT NOT NULL,
    team_id         INT NOT NULL,
    sweater         SMALLINT,
    position        TEXT,           -- C/L/R/D/G
    is_starter      BOOLEAN NOT NULL DEFAULT FALSE,
    is_scratch      BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (game_id, player_id)
);
CREATE INDEX IF NOT EXISTS game_rosters_player_idx ON game_rosters(player_id);
CREATE INDEX IF NOT EXISTS game_rosters_starting_goalies
    ON game_rosters(game_id, team_id)
    WHERE is_starter = TRUE AND position = 'G';

-- =====================================================================
-- PLAYERS backfill — positions, names, handedness
-- We already have the players table from schema.sql; this extends it.
-- (ALTER statements are idempotent because of IF NOT EXISTS pattern)
-- =====================================================================
-- Note: PostgreSQL doesn't support "ADD COLUMN IF NOT EXISTS" on older versions;
-- we use a DO block to be safe across PG 12+
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='players' AND column_name='height_in') THEN
        ALTER TABLE players ADD COLUMN height_in SMALLINT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='players' AND column_name='weight_lb') THEN
        ALTER TABLE players ADD COLUMN weight_lb SMALLINT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='players' AND column_name='birth_date') THEN
        ALTER TABLE players ADD COLUMN birth_date DATE;
    END IF;
END $$;

-- =====================================================================
-- Sanity view — shots joined to on-ice skaters
-- Useful for Stage B; validates the temporal join works
--
-- Uses half-open interval [start_seconds, end_seconds) for the shift join:
-- the shift INCLUDES its start second but EXCLUDES its end second. Without
-- this, line changes get double-counted because both the leaving and arriving
-- players are recorded as "active" at the transition second.
-- =====================================================================
CREATE OR REPLACE VIEW v_shots_with_onice AS
SELECT
    s.shot_id,
    s.game_id,
    s.period,
    s.game_seconds,
    s.is_goal,
    s.shooter_id,
    s.goalie_id,
    s.shooter_team_id,
    s.defending_team_id,
    COUNT(*) FILTER (WHERE sh.team_id = s.shooter_team_id) AS n_shooter_team_onice,
    COUNT(*) FILTER (WHERE sh.team_id = s.defending_team_id) AS n_defending_team_onice
FROM shots s
LEFT JOIN shifts sh
    ON sh.game_id = s.game_id
   AND sh.period  = s.period
   AND sh.start_seconds <= s.game_seconds
   AND sh.end_seconds   >  s.game_seconds   -- half-open: excludes the end second
GROUP BY s.shot_id, s.game_id, s.period, s.game_seconds, s.is_goal,
         s.shooter_id, s.goalie_id, s.shooter_team_id, s.defending_team_id;