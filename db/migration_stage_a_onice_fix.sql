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
   AND sh.start_seconds <= s.game_seconds
   AND sh.end_seconds   >  s.game_seconds   -- HALF-OPEN: was >=, now >
GROUP BY s.shot_id, s.game_id, s.period, s.game_seconds, s.is_goal,
         s.shooter_id, s.goalie_id, s.shooter_team_id, s.defending_team_id;