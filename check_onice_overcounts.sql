-- Find a shot with > 14 on-ice and see what's going on
WITH bad_shot AS (
  SELECT * FROM v_shots_with_onice
  WHERE n_shooter_team_onice + n_defending_team_onice >= 16
  LIMIT 1
)
SELECT
  bs.shot_id,
  bs.game_id,
  bs.period,
  bs.game_seconds,
  bs.shooter_team_id,
  bs.defending_team_id,
  bs.n_shooter_team_onice,
  bs.n_defending_team_onice,
  s.player_id,
  s.team_id,
  s.start_seconds,
  s.end_seconds,
  s.period AS shift_period
FROM bad_shot bs
JOIN shifts s
  ON s.game_id = bs.game_id
 AND s.period = bs.period
 AND s.start_seconds <= bs.game_seconds
 AND s.end_seconds   >= bs.game_seconds
ORDER BY s.team_id, s.start_seconds
LIMIT 30;