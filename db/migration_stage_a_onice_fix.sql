-- Stage A bugfix: v_shots_with_onice was using inclusive [start, end] join
-- which double-counted players during shift changes. A player whose shift
-- ends at second X and a player whose shift starts at second X were BOTH
-- being counted as "on ice" when a shot also occurred at second X.
--
-- The fix: change to half-open interval [start, end) — i.e., the shift
-- includes its start second but EXCLUDES its end second. This correctly
-- captures the "leaving" player as already off-ice and the "arriving"
-- player as on-ice.
--
-- Empirical evidence from game 2024020001 shot_id 335690 (period 1, sec 519):
--   5 players had shifts ending at sec 519 (the OLD line)
--   Those 5 + the 5 new players + a goalie = 11 phantom on-ice for one team
--
-- 2026-11-17: switched to the other half-open edge, start < t <= end, so this
-- view matches export/profiles.py. The data audit (2026-11-11, check 3) found
-- start < t <= end agrees with the play-by-play's skater counts about 10x as
-- often as start <= t < end. On all 704,933 shots the two rules give a different
-- total on the ice for 10,141 (1.44%), and the new rule leaves fewer shots with
-- a total outside 9-13 players (9,181 vs 9,479). Only
-- diagnostics/check_onice_overcounts.sql reads this view.

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
    COUNT(*) FILTER (WHERE sh.team_id = s.shooter_team_id)   AS n_shooter_team_onice,
    COUNT(*) FILTER (WHERE sh.team_id = s.defending_team_id) AS n_defending_team_onice
FROM shots s
LEFT JOIN shifts sh
    ON sh.game_id = s.game_id
   AND sh.period  = s.period
   AND sh.start_seconds <  s.game_seconds   -- start < t <= end, as in export/profiles.py
   AND sh.end_seconds   >= s.game_seconds
GROUP BY s.shot_id, s.game_id, s.period, s.game_seconds, s.is_goal,
         s.shooter_id, s.goalie_id, s.shooter_team_id, s.defending_team_id;