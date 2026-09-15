-- Stage C3: Expanding-window goalie priors
--
-- Same architecture as skater_priors_expanding (schema_stage_c2.sql): for each
-- date a goalie played, compute their goals-against rate using only shots from
-- the trailing window [game_date - 2 years, game_date).
--
-- Two differences from the skater table:
--
-- 1. The grouping key adds danger_zone, so the grid is 3 zones x 3 strength
--    states instead of 3 strength states. Nine cells instead of three, from a
--    smaller population (~90 goalies vs ~900 skaters). Cells get thin.
--
-- 2. evidence_level records how much history backed each row. It is descriptive
--    only and never changes the estimate: shrinkage already scales with evidence
--    continuously. The column exists so you can check afterwards whether the
--    9-cell grid is too fine for a ~90-goalie population, which would show up as
--    'sparse' dominating the row counts.

CREATE TABLE IF NOT EXISTS goalie_priors_expanding (
    goalie_id              BIGINT NOT NULL,
    game_date              DATE   NOT NULL,
    danger_zone            TEXT   NOT NULL,          -- 'high' (<=20ft), 'mid' (20-40ft), 'low' (>40ft)
    strength_state         TEXT   NOT NULL,          -- goalie POV: '5v5', 'PP_against', 'PK_against'
    weighted_shots         DOUBLE PRECISION NOT NULL,
    weighted_goals         DOUBLE PRECISION NOT NULL,
    raw_shots              INTEGER NOT NULL,
    raw_goals              INTEGER NOT NULL,
    prior_alpha            DOUBLE PRECISION NOT NULL,
    prior_beta             DOUBLE PRECISION NOT NULL,
    prior_concentration    DOUBLE PRECISION NOT NULL,  -- K, fit once per cell on full population
    prior_mean             DOUBLE PRECISION NOT NULL,  -- league GA rate for this cell
    raw_ga_rate            DOUBLE PRECISION,
    shrunken_ga_rate       DOUBLE PRECISION NOT NULL,
    evidence_level         TEXT NOT NULL,              -- 'sufficient' | 'sparse' | 'none' (descriptive only)
    window_start_date      DATE NOT NULL,              -- inclusive
    window_end_date        DATE NOT NULL,              -- exclusive (= game_date)
    computed_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (goalie_id, game_date, danger_zone, strength_state)
);

CREATE INDEX IF NOT EXISTS goalie_priors_expanding_date_idx
    ON goalie_priors_expanding (game_date);
CREATE INDEX IF NOT EXISTS goalie_priors_expanding_goalie_idx
    ON goalie_priors_expanding (goalie_id);
CREATE INDEX IF NOT EXISTS goalie_priors_expanding_join_idx
    ON goalie_priors_expanding (goalie_id, game_date, danger_zone, strength_state);