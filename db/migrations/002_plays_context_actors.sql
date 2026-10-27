-- 002_plays_context_actors.sql
-- Adds player attribution to non-shot events in plays_context.
--
-- WHY
-- plays_context stored event_type, coordinates and eventOwnerTeamId and nothing
-- else, so every hit, penalty, giveaway, takeaway and faceoff in the database was
-- anonymous. "Who draws penalties" was unanswerable. The player ids were always in
-- the payload; ingest/parser.py discarded them. This migration makes room for them.
--
-- SAFE TO RUN ON A LIVE TABLE. Every statement is additive and idempotent. No rows
-- are read, rewritten or deleted. Existing columns, the primary key and the foreign
-- key to games are untouched. Adding a nullable column without a default is a
-- catalog-only change in PostgreSQL 11+, so this does not rewrite the table.
--
-- Columns are populated by `python -m ingest.reparse`, which rebuilds from the
-- cached JSON in raw_data/. Until that runs, the new columns are all NULL.
--
-- Apply with:
--     psql -d hockeyviz -f db/migrations/002_plays_context_actors.sql

BEGIN;

ALTER TABLE plays_context ADD COLUMN IF NOT EXISTS actor_1_id       BIGINT;
ALTER TABLE plays_context ADD COLUMN IF NOT EXISTS actor_1_role     TEXT;
ALTER TABLE plays_context ADD COLUMN IF NOT EXISTS actor_2_id       BIGINT;
ALTER TABLE plays_context ADD COLUMN IF NOT EXISTS actor_2_role     TEXT;
ALTER TABLE plays_context ADD COLUMN IF NOT EXISTS raw_details_json JSONB;

COMMENT ON COLUMN plays_context.actor_1_id IS
    'Player who performed the event (hitter, penalty committer, faceoff winner).';
COMMENT ON COLUMN plays_context.actor_2_id IS
    'Player the event was performed on (hittee, penalty drawer, faceoff loser). NULL where the event names only one player, e.g. a giveaway or a bench minor.';
COMMENT ON COLUMN plays_context.raw_details_json IS
    'Full details dict for NON-shot events. NULL on shot-family rows because shots.raw_details_json already holds those, keyed on the same (game_id, event_idx).';

-- Two generic slots with a role label, not one named column per event type.
-- Named columns would mean hitting_player_id, hittee_player_id,
-- committed_by_player_id, drawn_by_player_id, giveaway_player_id and more, each
-- NULL on every row of every other event type. The role string carries the same
-- semantics without that sparsity, and a renamed NHL key lands in a column with a
-- visibly new role rather than silently vanishing.

CREATE INDEX IF NOT EXISTS plays_context_actor_1_idx
    ON plays_context (actor_1_id) WHERE actor_1_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS plays_context_actor_2_idx
    ON plays_context (actor_2_id) WHERE actor_2_id IS NOT NULL;

-- event_type alone is low-cardinality and rarely worth an index, but every actor
-- query filters on it alongside a role, so the composite pays for itself.
CREATE INDEX IF NOT EXISTS plays_context_type_role1_idx
    ON plays_context (event_type, actor_1_role);
CREATE INDEX IF NOT EXISTS plays_context_type_role2_idx
    ON plays_context (event_type, actor_2_role);

-- Unpivot the two slots into one row per (event, player, role).
--
-- This is what makes slot assignment an implementation detail. Without it every
-- query would have to know whether drawn_by lands in slot 1 or slot 2 and write
-- "actor_1_role = 'drawn_by' OR actor_2_role = 'drawn_by'". Through the view the
-- question is one predicate on role, and it keeps working if slot priorities are
-- ever reordered in the parser.
CREATE OR REPLACE VIEW v_event_actors AS
SELECT game_id, event_idx, period, game_seconds, event_type, team_id, x, y,
       actor_1_id AS player_id, actor_1_role AS role, 1 AS slot
FROM plays_context
WHERE actor_1_id IS NOT NULL
UNION ALL
SELECT game_id, event_idx, period, game_seconds, event_type, team_id, x, y,
       actor_2_id AS player_id, actor_2_role AS role, 2 AS slot
FROM plays_context
WHERE actor_2_id IS NOT NULL;

COMMENT ON VIEW v_event_actors IS
    'One row per (event, player, role). Query actors through this, not through the slot columns directly.';

COMMIT;