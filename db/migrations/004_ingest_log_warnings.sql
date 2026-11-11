-- 004_ingest_log_warnings.sql
-- Parse warnings per game in ingest_log.
--
-- WHY
-- ingest/parser.py notes every play it skips (a shot with no team, a play that
-- raises), but those notes used to be thrown away, so a dropped shot left no
-- trace (gap 6 in docs/DATA_PIPELINE.md). ingest/run.py now writes the counts
-- here for each play-by-play row, and ingest/check_ingest.py prints them at the
-- end of a refresh and stops it if any game dropped a shot.
--
--   parse_warnings  every warning the parser gave for the game
--   shots_dropped   shot events among them that never reached `shots`
--   warning_sample  the first three warnings, for reading
--
-- NULL means the game was logged before this migration and not backfilled;
-- `python -m ingest.check_ingest --backfill-warnings` fills them from the
-- cached raw files.
--
-- SAFE TO RUN ON A LIVE DATABASE: ADD COLUMN IF NOT EXISTS only. ingest/run.py
-- runs this file itself before ingesting.

ALTER TABLE ingest_log ADD COLUMN IF NOT EXISTS parse_warnings INT;
ALTER TABLE ingest_log ADD COLUMN IF NOT EXISTS shots_dropped  INT;
ALTER TABLE ingest_log ADD COLUMN IF NOT EXISTS warning_sample TEXT;
