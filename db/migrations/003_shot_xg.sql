-- 003_shot_xg.sql
-- Stores per-shot xG from a saved model, so published numbers name their model.
--
-- WHY
-- The site needs xG by team, game and period. Until now predictions existed only
-- in memory inside each training run, so nothing outside a run could use them.
-- model/score_fenwick.py scores every row of shot_features_fenwick and writes
-- them here, one row per (model_version, shot_id).
--
-- SAFE TO RUN ON A LIVE DATABASE. CREATE TABLE IF NOT EXISTS is additive and
-- idempotent; no existing table is read, rewritten or deleted. The scoring script
-- runs this file itself before writing.
--
-- shot_id is shots.shot_id. ingest/reparse.py restarts that identity, so after a
-- reparse rerun model/score_fenwick.py before trusting this table.

CREATE TABLE IF NOT EXISTS shot_xg (
    model_version TEXT NOT NULL,
    shot_id       BIGINT NOT NULL,
    xg            DOUBLE PRECISION NOT NULL,
    scored_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (model_version, shot_id)
);
