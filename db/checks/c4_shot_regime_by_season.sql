-- Regular season, no shootout, no empty net (sog_ptg lines up with Hockey-Reference SA).
WITH base AS (
    SELECT gm.season,
           s.game_id,
           s.event_type,
           s.is_goal::int AS goal,
           CASE
               WHEN s.distance_ft IS NULL THEN 'e_null'
               WHEN s.distance_ft < 20   THEN 'a_lt20'
               WHEN s.distance_ft < 40   THEN 'b_20_40'
               WHEN s.distance_ft < 60   THEN 'c_40_60'
               ELSE 'd_60plus'
           END AS band
    FROM shots s
    JOIN games gm ON gm.game_id = s.game_id
    WHERE substr(s.game_id::text, 5, 2) = '02'
      AND s.period < 5
      AND NOT COALESCE(s.empty_net, false)
),
ng AS (
    SELECT season, COUNT(DISTINCT game_id) AS games
    FROM base
    GROUP BY season
)
SELECT b.season,
       COALESCE(b.band, '0_all') AS band,
       ROUND((COUNT(*) FILTER (WHERE b.event_type IN ('shot-on-goal','goal')))::numeric
             / (2 * MAX(ng.games)), 2) AS sog_ptg,
       ROUND((COUNT(*) FILTER (WHERE b.event_type IN ('shot-on-goal','goal','missed-shot')))::numeric
             / (2 * MAX(ng.games)), 2) AS fenwick_ptg,
       ROUND((COUNT(*) FILTER (WHERE b.event_type = 'missed-shot'))::numeric
             / NULLIF(COUNT(*) FILTER (WHERE b.event_type IN ('shot-on-goal','goal','missed-shot')), 0), 3) AS missed_share,
       ROUND((COUNT(*) FILTER (WHERE b.event_type = 'blocked-shot'))::numeric
             / (2 * MAX(ng.games)), 2) AS blocked_ptg,
       ROUND(SUM(b.goal)::numeric / (2 * MAX(ng.games)), 3) AS ga_ptg,
       ROUND((SUM(b.goal) FILTER (WHERE b.event_type IN ('shot-on-goal','goal')))::numeric
             / NULLIF(COUNT(*) FILTER (WHERE b.event_type IN ('shot-on-goal','goal')), 0), 4) AS gr_sog,
       ROUND((SUM(b.goal) FILTER (WHERE b.event_type IN ('shot-on-goal','goal','missed-shot')))::numeric
             / NULLIF(COUNT(*) FILTER (WHERE b.event_type IN ('shot-on-goal','goal','missed-shot')), 0), 4) AS gr_fenwick
FROM base b
JOIN ng ON ng.season = b.season
GROUP BY GROUPING SETS ((b.season, b.band), (b.season))
ORDER BY b.season, band;
