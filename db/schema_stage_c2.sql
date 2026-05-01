-- Stage C2: Expanding-window skater priors
--
-- Per-(player, game_date, strength_bucket) priors using a trailing 2-season window.
-- For each shot a player took, we compute their prior using only data from the
-- window [game_date - 2 years, game_date).
--
-- This avoids the leakage path of season-pooled priors, where a 2024-25 shot's
-- prior was informed by 2024-25 outcomes.

CREATE TABLE IF NOT EXISTS skater_priors_expanding (
    player_id              BIGINT NOT NULL,
    game_date              DATE   NOT NULL,
    strength_state         TEXT   NOT NULL,          -- '5v5', 'PP', 'PK'
    weighted_shots         DOUBLE PRECISION NOT NULL,
    weighted_goals         DOUBLE PRECISION NOT NULL,
    raw_shots              INTEGER NOT NULL,
    raw_goals              INTEGER NOT NULL,
    prior_alpha            DOUBLE PRECISION NOT NULL,
    prior_beta             DOUBLE PRECISION NOT NULL,
    prior_concentration    DOUBLE PRECISION NOT NULL,  -- K, fit once on full population
    prior_mean             DOUBLE PRECISION NOT NULL,  -- league mean for this bucket
    raw_shooting_pct       DOUBLE PRECISION,
    shrunken_shooting_pct  DOUBLE PRECISION NOT NULL,
    window_start_date      DATE NOT NULL,              -- inclusive
    window_end_date        DATE NOT NULL,              -- exclusive (= game_date)
    computed_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_id, game_date, strength_state)
);

CREATE INDEX IF NOT EXISTS skater_priors_expanding_date_idx
    ON skater_priors_expanding (game_date);
CREATE INDEX IF NOT EXISTS skater_priors_expanding_player_idx
    ON skater_priors_expanding (player_id);
CREATE INDEX IF NOT EXISTS skater_priors_expanding_strength_idx
    ON skater_priors_expanding (strength_state);