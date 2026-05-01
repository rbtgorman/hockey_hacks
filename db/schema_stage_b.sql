-- HockeyViz Pro — Stage B schema additions
-- Empirical Bayes player priors for shooting % and goals-against rate.
-- Run AFTER schema.sql and schema_stage_a.sql.
--
-- Two tables (skater_priors, goalie_priors) plus two lookup views.
-- All shrunken rates are computed by features/build_priors.py.

-- =====================================================================
-- SKATER PRIORS
-- One row per (player, strength_state). Strength buckets: '5v5', 'PP', 'PK', 'all'.
-- '4v4' and '3v3' are folded into '5v5' upstream (see build_priors.py:shooter_strength).
-- =====================================================================
CREATE TABLE IF NOT EXISTS skater_priors (
    player_id               BIGINT NOT NULL,
    strength_state          TEXT NOT NULL,        -- '5v5', 'PP', 'PK', 'all'

    -- Sample (5/4/3 recency-weighted across 2024-25, 2023-24, 2022-23)
    weighted_shots          DOUBLE PRECISION NOT NULL,
    weighted_goals          DOUBLE PRECISION NOT NULL,
    raw_shots               INTEGER NOT NULL,
    raw_goals               INTEGER NOT NULL,

    -- Population Beta(alpha, beta) prior fit on this strength bucket
    prior_alpha             DOUBLE PRECISION NOT NULL,
    prior_beta              DOUBLE PRECISION NOT NULL,
    prior_concentration     DOUBLE PRECISION NOT NULL,  -- alpha + beta = K
    prior_mean              DOUBLE PRECISION NOT NULL,  -- alpha / K = league avg shooting %

    -- Outputs
    raw_shooting_pct        DOUBLE PRECISION,            -- weighted_goals / weighted_shots
    shrunken_shooting_pct   DOUBLE PRECISION NOT NULL,   -- (weighted_goals + alpha) / (weighted_shots + K)

    computed_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (player_id, strength_state)
);
CREATE INDEX IF NOT EXISTS skater_priors_strength_idx ON skater_priors(strength_state);

-- =====================================================================
-- GOALIE PRIORS
-- One row per (goalie, danger_zone, strength_state).
--
-- Danger zones (distance from goal):
--   'high' : <= 20 ft
--   'mid'  : 20-40 ft
--   'low'  : > 40 ft
--   'all'  : aggregate across zones (for backward-compat / quick lookup)
--
-- Strength state is FROM THE GOALIE'S POINT OF VIEW (inverse of shot.strength_state):
--   '5v5'         : even strength
--   'PP_against'  : goalie's team is shorthanded (PK), shooter is on PP
--   'PK_against'  : goalie's team is on PP, shooter is shorthanded
--   'all'         : aggregate across strength
-- =====================================================================
CREATE TABLE IF NOT EXISTS goalie_priors (
    player_id               BIGINT NOT NULL,
    danger_zone             TEXT NOT NULL,        -- 'high', 'mid', 'low', 'all'
    strength_state          TEXT NOT NULL,        -- '5v5', 'PP_against', 'PK_against', 'all'

    -- Sample
    weighted_shots_against  DOUBLE PRECISION NOT NULL,
    weighted_goals_against  DOUBLE PRECISION NOT NULL,
    raw_shots_against       INTEGER NOT NULL,
    raw_goals_against       INTEGER NOT NULL,

    -- Population Beta(alpha, beta) prior on goals-against rate
    prior_alpha             DOUBLE PRECISION NOT NULL,
    prior_beta              DOUBLE PRECISION NOT NULL,
    prior_concentration     DOUBLE PRECISION NOT NULL,
    prior_mean              DOUBLE PRECISION NOT NULL,  -- league avg GA rate (1 - sv%)

    -- Outputs (we report goals-against rate, not save %, to keep direction consistent
    -- with skater_priors and to avoid sign confusion in downstream features)
    raw_ga_rate             DOUBLE PRECISION,
    shrunken_ga_rate        DOUBLE PRECISION NOT NULL,   -- save % = 1 - this

    computed_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (player_id, danger_zone, strength_state)
);
CREATE INDEX IF NOT EXISTS goalie_priors_zone_idx ON goalie_priors(danger_zone, strength_state);

-- =====================================================================
-- Convenience views for joining priors to shots in v2 feature build.
-- =====================================================================
CREATE OR REPLACE VIEW v_shooter_prior_5v5 AS
SELECT player_id,
       shrunken_shooting_pct AS prior_shooting_pct,
       weighted_shots         AS prior_sample_size
FROM skater_priors
WHERE strength_state = '5v5';

CREATE OR REPLACE VIEW v_goalie_prior_overall AS
SELECT player_id,
       shrunken_ga_rate       AS prior_ga_rate,
       weighted_shots_against AS prior_sample_size
FROM goalie_priors
WHERE danger_zone = 'all' AND strength_state = 'all';