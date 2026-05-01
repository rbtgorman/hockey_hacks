-- HockeyViz Pro — Stage C support: train-only priors tables
--
-- These mirror skater_priors / goalie_priors but are populated from train+val
-- seasons only (typically 20222023 and 20232024), excluding the test season
-- (20242025). Used as features for v2 xG training to avoid leaking test-season
-- shooter/goalie performance into the model.
--
-- Run AFTER schema_stage_b.sql.

CREATE TABLE IF NOT EXISTS skater_priors_train (
    LIKE skater_priors INCLUDING ALL
);

CREATE TABLE IF NOT EXISTS goalie_priors_train (
    LIKE goalie_priors INCLUDING ALL
);

-- Convenience views matching the production view names with _train suffix
CREATE OR REPLACE VIEW v_shooter_prior_5v5_train AS
SELECT player_id,
       shrunken_shooting_pct AS prior_shooting_pct,
       weighted_shots         AS prior_sample_size
FROM skater_priors_train
WHERE strength_state = '5v5';

CREATE OR REPLACE VIEW v_goalie_prior_overall_train AS
SELECT player_id,
       shrunken_ga_rate       AS prior_ga_rate,
       weighted_shots_against AS prior_sample_size
FROM goalie_priors_train
WHERE danger_zone = 'all' AND strength_state = 'all';