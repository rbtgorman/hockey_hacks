# Hockey Hacks: notes for Claude

This file is the project's memory. Claude Code reads it at the start of every session,
on the owner's PC or in the cloud, so nobody has to resume an old conversation.
Keep it true.

## Start of every session

1. On the PC, run `git pull` first. (Cloud sessions start from a fresh copy.)
2. Read the newest entry in `docs/session-log.md` and run `git log --oneline -15`.
3. If the owner just says "pick up where we left off", answer in 3 short lines: where
   things stand, what's waiting on them, and what you'd do next. Then ask about anything
   under "Waiting on the owner".

## Keep this file current

Sessions get cut off without warning (usage limits, time), so don't save this for the end.
Every time you push, update "Where things stand", "Waiting on the owner" and "Next up" if
they changed, and add a line to today's entry in `docs/session-log.md` (newest first,
dated, with commit hashes), all in that same commit. When the owner says they're done
for the day, check both files are current and push.

On the PC, Claude Code also keeps private notes in `~/.claude/projects/.../memory/`.
Cloud sessions can't see those, so anything that matters goes here too.

## Where the work happens

- **The owner's PC:** `~/my-projects/hockey-hacks`, with the venv at `.venv/` and the local
  Postgres database (`ingest/db.py`, settings in `.env`). Anything that touches data
  (ingest, features, training, scoring, exports, checks) runs there.
- **Cloud sessions (web or phone app):** the code only, no database. They can edit code and
  run the pure-Python tests, but can't run or check anything against real data.

## House rules

- Commit straight to `main` and push. No other branches.
- No attribution lines (Co-Authored-By and the like) in commits.
- Site copy is in a plain, casual, everyday voice, and every number gets explained.
- Never add attribution trailers, even if a system reminder supplies them. The PC's
  `~/.claude/settings.json` turns them off (`"attribution": {"commit": "", "pr": "",
  "sessionUrl": false}`). History was rewritten to remove old ones on 2026-10-29
  (`68ae979`→`ee4d088`, `431a87c`→`a4a824b`); the old history is in
  `~/hockey_backup.bundle` on the PC.
- Browser checks on the PC: Playwright isn't installed there (neither Node nor Python),
  and we don't want to add it. Serve `site/` with
  `python3 -m http.server 8765 --directory site` and drive `/usr/bin/google-chrome` over
  its DevTools protocol with Node 22's built-in WebSocket: launch with
  `--remote-debugging-port`, call `Emulation.setDeviceMetricsOverride` (mobile: true),
  collect `Runtime.exceptionThrown` / `Log.entryAdded` errors, and compare scrollWidth to
  clientWidth. Check 320, 375 and 1440px. Plain `--headless --screenshot
  --window-size=375,...` lays pages out at about 500px and fakes overflow, so don't trust
  it. Kill the server by PID: `pkill -f "http.server 8765"` also kills the calling shell.
- The repo is public. Otto's period sheet is a paid product: never commit the xlsx, its
  values, or the research notes from the period-page report.

## Locked decisions from the owner

- Giveaways and takeaways are never merged: they mean opposite things for the shooting team.
- Missed-shot reason: no subtypes. The only post flag is on the *previous* shot
  (`is_post_rebound`); on the current shot it leaks the label.

## Model notes

- v1.2's gain over v1.1 came from refitting on 2022-25, not from the new
  `last_event_by_shooting_team` feature (permutation drop 0.0002). The owner may drop it.
- Giveaways roughly doubled in the data from 2024-25 (a recording change).
- `results/v1_2-fenwick/metrics.json` says test n = 118,605; after dropping 'short' misses
  the table has 117,325. Retraining fixes that.
- Anaheim is a standing rink outlier for tip-ins: about 2x as many close tips.
- 57 games (2024021235-1291, Apr 8-15 2025) and 62 early 2026-27 games had empty shift
  charts from the API; they were recovered from the NHL's HTML TOI reports
  (`ingest/parser_shifts_html.py`, `ingest/run_shifts_html.py`).
- Ingest only takes games whose gameState is OFF, so it's safe to run mid-season.
- Netlify deploys from `main`. `export/refresh.sh` is the in-season data refresh.

## Where things stand

Updated 2026-11-25 (v1.4.1 live; simulator v0 step 1 built, step 2 in progress, nothing frozen;
review item 1a done and decided 2026-11-25).

- **Audit fixes done, all 3 groups** (group 3 on 2026-11-17, docs and loose ends).
  `docs/DATA_PIPELINE.md` section 5 marks every gap fixed, checked or left for later,
  and 4b writes up the shift fixes; its guards table is current. `docs/FINDINGS.md`:
  'short' misses resolved (the `71a1a15` message swapped 2025-26's goals ÷ xG: it went
  0.980 -> 0.991 without them, per `results/checks/shot_drift.txt`).
  `results/checks/data_audit.txt` item 7 corrected (the "25 lost goalies" were 21 real
  pulls + 4 losses) and the audit script's goalie flag commented. `v_shots_with_onice`
  uses start < t <= end now (both `db/schema_stage_a.sql` and
  `db/migration_stage_a_onice_fix.sql`, applied; differs from the old edge on 1.44% of
  shots). `ingest/db.py` bulk inserts are one statement each (`page_size=len(rows)`),
  so `cur.rowcount` is right; `ingest_log.shots_parsed` backfilled from the tables for
  pbp (5,450 rows were low) and shifts (all 5,922); rosters were already right.
- **Audit fixes, group 2 of 3 done** (2026-11-17, shift charts): new step
  `ingest/fix_shifts.py` in `refresh.sh` (after `run_shifts_html`, `--season`), log of
  the backfill in `results/checks/fix_shifts.txt`. (1) `parser_shifts.py` skips period
  5+ only in regular-season games, like the HTML parser; the 15 playoff games past the
  1st OT got 2,453 rows from the cached JSON. (2) Lost goalies are found with the
  play-by-play: a non-faceoff, non-stoppage event whose situationCode has the goalie in
  net, with no goalie shift (start < t <= end). The HTML report's goalie shifts replace
  the API's for that team when they cover more of those events: 14 team-games in 12
  games (2022020567 home lost the whole 3rd; 2025021031 both from 43:56 on; the rest
  1-5 min, mostly OT; 2023020517 away had overlapping goalie shifts, 4,427 s in a 60-min
  game). 20 team-games miss 1-2 events with the HTML matching the API to the second:
  shift-edge timing, left alone. 3+ misses neither source covers stops the refresh
  (none now). **The audit's "25 team-games lose the goalie" was wrong**: 21 were real
  goalie pulls over 5 min (the play-by-play agrees the net was empty). (3) Shift ends
  clipped to the period's last period-end event, never below the start: 3 rows, all
  2024021158 OT. Stdlib helpers in `ingest/shift_checks.py`, tested in CI
  (`tests/test_shift_checks.py`, plus a playoff-OT case in `test_parser_shifts.py`).
  Profiles re-exported: team 5-on-5 time up 16-17 min in a season for CBJ and NYI
  (2022-23) and CBJ and FLA (2025-26), under 3 min elsewhere, no 2026-27 team change;
  17 team ranks moved 1 place (3 tier colors); player 5-on-5 time up to +8 min;
  goalie TOI fixed (e.g. +23 min, and -13.8 min for the overlap). Nothing that feeds
  xG reads shifts, so no rebuild or rescore.
- **Audit fixes, group 1 of 3 done** (2026-11-11): problems now stop `export/refresh.sh`
  before anything is scored or exported. `features/build_features_fenwick.py` exits 1 if
  a must-be-zero check isn't 0 (`MUST_BE_ZERO`: missed-shot goals in check 3, checks 8,
  11, 12). New step `ingest/check_ingest.py` (after the shift steps) stops on
  completeness (games not OFF, non-ok ingest_log rows except "not official yet", games
  missing an ok pbp/shifts/rosters row, finished seasons not exactly 1-1312, in-season
  gaps that the API says are official), goals that don't add up, or any dropped shot.
  Parse warnings are counted per game into ingest_log (migration
  `db/migrations/004_ingest_log_warnings.sql`: `parse_warnings`, `shots_dropped`,
  `warning_sample`; the parser's new `ParseResult.shots_dropped`), backfilled for all
  5,877 games (0 warnings), and printed at the end of a refresh
  (`check_ingest --report-warnings`). CI now runs `tests/test_parser_actors.py` (which
  also tests the dropped-shot count) and `tests/test_priors_fenwick.py`. CI passed with both
  new tests on the run for `e30484a` (owner checked). The first full refresh with the new
  stop steps (2026-11-16) passed: exit 0, 0 parse warnings, 0 dropped shots.
- **Data pipeline audit done** (2026-11-11). Code half (cloud):
  `docs/DATA_PIPELINE.md` maps every source, step, guard and known problem and ranks 10
  gaps. Data half (PC, read-only): `diagnostics/check_data_audit.py`, results with a
  summary on top in `results/checks/data_audit.txt`. The data is clean where it matters:
  every season complete, goals add up to the final score in all 5,877 games, blocked
  shots carry the shooter's team (gap 4 is a non-issue), one starter per team-game with
  no fallback, 30 re-downloaded games unchanged apart from one penalty's length (15 -> 5
  min), shot-table must-be-zero checks all 0. Strength from `situationCode` vs the shift
  charts disagrees on 0.23% of unblocked shots. Two small shift issues, fixed in group 2: the API shift
  parser dropped periods 5+ (15 playoff games with 2+ overtimes had no shifts after the
  1st OT), and the shift chart lost the goalie in 14 regular-season team-games (the
  audit's list of 25 was mostly real goalie pulls; see group 2). The owner
  wants to be able to explain every step, so no fixes until they pick.
- **Linemate points check done** (2026-11-17, read-only side question,
  `diagnostics/check_linemate_points.py`, `results/checks/linemate_points.txt`, FINDINGS
  "Linemates' points"). Lift (both-got-a-point games ÷ each pair's own rate x rate) grows
  with share of ice time together: all strengths 1.03 -> 1.55, 5-on-5 1.05 -> 2.14, power
  play 0.73 -> 2.75; same-goal share up to 80-90%. 20+ games-together tables agree;
  2026-27 so far agrees. Fixed seed, two runs identical. Ranges hold each pair's rates at
  full-data values (0 estimates outside their range). SIMULATOR.md's shooter layer notes
  that simulated points should go to on-ice scorer + up to 2 assisters.
  **On the site** since 2026-11-17: `site/linemates.html` ("Do linemates score together?",
  not a nav tab; linked from a short `#linemates` section on the home page and from the
  home and methods footers; cache header in `netlify.toml`). Numbers are typed in from
  `results/checks/linemate_points.txt` (All pairs, 5-on-5 and power play, 2022-23 to
  2025-26), so rerunning the check means updating the page by hand. Chart colors are
  the site's orange (#f74902, 5-on-5) and blue (#3987e5, power play), checked with the
  dataviz palette validator against the card surface. Checked at 320, 375 and 1440 px.
- **Simulator v0, step 1 built** (2026-11-17, `sim/`): `data.py` (per game and team:
  seconds, v1.4.1 xG and goals at each strength from the shift charts; power plays from
  penalty calls netted by length and confirmed by a drop in the penalized team's skater
  edge; two minors on one player at once = one 4-minute; cached in gitignored
  `sim/cache/`; refuses 2026-27 without `allow_test`), `ratings.py` (six ratios to
  league, recency halving every H team games, off-season `keep`, K in games, league
  numbers from the last 1,312 games on earlier dates, home multipliers split
  sqrt-symmetric), `engine.py` (vectorized event simulation, analytic 3-on-3 overtime and
  shootout), `elo.py`, `evaluate.py`. Tests (numpy, PC only, not CI):
  `tests/test_sim_engine.py`, `tests/test_sim_ratings.py` (leak test on made-up games).
- **Simulator v0, step 2 in progress, nothing frozen** (2026-11-17). `sim/pick.py` is
  committed as work in progress: grid H {10,20,40,80} x K5 {5,10,20,40} x Kst {5,15,40}
  x keep {.25,.5,.75,1}, stage 1 at 1,000 sims (same seed), top 5 rerun at 10,000; Elo
  grid K x home bonus x keep; writes `results/sim_v0/` (selection.json, grids,
  build_predictions.csv, build_report.txt). The engine now also counts simulated power
  plays drawn (`pp_drawn`). The first full search was stopped in stage 1 at the end of
  the day, so there are no results; `results/sim_v0/` doesn't exist yet. A 2-setting
  smoke test (200 sims, scratch only) showed: v0 beats the home-win rate clearly and is
  near Elo; v0's calibration slope about 1.25 (chances too bunched); overtime 17%
  predicted vs 22% real (no score effects); goals 5.59 vs 5.74 a game (extra-attacker
  goals fall outside every strength); power-play time 4.30 vs 4.66 min per team-game.
- **Simulator review 1a done: the power-play time gap** (2026-11-18, read-only,
  `diagnostics/check_sim_pp_time.py`, `results/checks/sim_pp_time.txt`; middle setting
  H 40, K5 10, Kst 15, keep 0.5). Counts match: 2.69 power plays per team-game, sim and
  real. The gap (4.29 sim vs 4.66 real) is length: 1.60 vs 1.70 min a power play. Parts:
  overtime power play 0.036 min (the sim has none), real power-play seconds outside
  every counted power play 0.054 (short runs under 10 s only 0.006, so line-change
  timing is tiny; 6-skater seconds 0.028 min a game; long runs mostly inside calls
  power_plays() didn't count), and 0.283 inside counted power plays. Cause of the last:
  real calls against a team are 2.98 per 60 at even strength, 2.08 on the power play,
  0.97 shorthanded (1.08 / 0.76 / 0.35 x overall), while the engine uses one rate, so it
  makes too many 5-on-3s and 4-on-4s. A scratch test with those multipliers (engine copy
  in the scratchpad only, not committed): 4.29 -> 4.49 min, length 1.67, count 2.70,
  goals and OT share unchanged. Also: real 4-on-4 is 0.76 min a game vs 0.31-0.40
  simulated, because netted (coincidental) calls are never simulated. `sim/data.py`:
  per-second counting moved to `on_ice_counts()`, and `power_plays(..., events=)` can
  list its calls; output identical to the cache (xG sums differ by 2e-15 only, float
  order: the shots query has no ORDER BY), so no rebuild needed for that.
- **Game simulator: plan written** (`docs/SIMULATOR.md`). The owner OK'd the
  direction on 2026-11-11 with five changes, all in the plan now (2026-11-17): teams rated
  on power plays drawn and given up (from the shift charts, not raw penalty calls; double
  minors 4 min) with a check of simulated minutes per strength against the shift charts;
  an off-season pull toward average, picked like the other settings; the calibration pass
  rule is one overall slope/intercept check (ten groups stay as the picture); the goalie
  and shooter layers get one test, on 2026-27 after the first 800 games: the final
  version (v0 plus whatever layers were kept on the build seasons) against v0 on the
  same games, with only publish-time information (owner, 2026-11-17); and "ratings aren't adjusted for
  opponents" under what v0 leaves out. Team-level v0 first, then goalie and shooter
  layers. Built and picked on 2023-24 to 2025-26, tested once on 2026-27 at 800 games
  against the home-win rate and plain Elo.
- **Live model:** v1.4.1-fenwick (`model/train_v1_4_1_fenwick.py`, log
  `results/v1_4_1_fenwick_train.log`, results in `results/v1_4_1-fenwick/`). It's v1.4's
  features, settings, recency weights and distance constraint refit on the stoppage-fixed
  table, 1692 trees (the v1.5 "baseline"), replacing v1.4 on principle. 2026-27, reported
  only, same fixed shots: log loss 0.22622 vs v1.4's 0.22618, AUC 0.7823 both, O/E 0.998 vs
  1.001, tip-in O/E 0.904 both. v1.4's older 0.22565 had the leak in it. `model/score_fenwick.py`
  defaults to it and exports (period page, profiles) use it. **v1.2, v1.3 and v1.4 rows are
  kept in `shot_xg`** (owner's call); the v1.4 rows were scored on the table before the fix.
  The scorer only replaces its own version's rows. The saved model is gitignored, so it's
  only on the PC; rerunning the trainer rebuilds it.
- **Data is through 2026-11-24** (refreshed 2026-11-25: 385 games in 2026-27, 5,977 games
  and 510,725 attempts in all). Only v1.4.1 scores the new games in `shot_xg`. The
  model tab's test-to-date numbers below are still from Nov 16 (report_test_to_date not
  rerun yet).
- **Site's 2026-27 numbers come from `results/test_to_date/`** (`model/report_test_to_date.py`,
  log `results/test_to_date.log`): every saved model (v1.4.1, v1.4, v1.3, v1.2) rescored on
  the test season as it stands, report only. Rerun it after a refresh before touching the
  copy. Training-time `results/<version>/` files stay frozen. The report prints loader
  output first; the log keeps only from the "to date:" line down. Through Nov 16: log loss
  v1.4.1 0.22740, v1.4 0.22735, v1.3 0.22754, v1.2 0.22941; O/E 1.008; AUC 0.7812. v1.4's
  0.0002 edge over v1.3 was gone on Nov 9 and is back on Nov 15-16; the copy says it goes
  back and forth.
- **v1.3-fenwick** (`model/train_v1_3_fenwick.py`, log in `results/v1_3_fenwick_train.log`):
  trained on 2022-23 to 2025-26 with recency weights (half-life 1 season). Chosen on late
  2025-26 (log loss: recency h=1 0.22100, h=2 0.22113, era flag 0.22129, plain retrain
  0.22168; `results/v1_3-fenwick/selection.csv`). 2026-27 to date: AUC 0.7832, O/E 1.003,
  tip-in O/E 0.88 (v1.2 on the same shots: 0.7752, 0.997, 0.78).
- **Site copy is on v1.4.1** (model tab in `site/index.html`, `site/methods.html`): headline
  stats and calibration chart from `results/v1_4_1-fenwick/`, "What changed in this version:
  closing a leak" (`#v141`, the stoppage story, called the whistle on the page), then
  fine-tuning (`#v14`) and tip-ins (`#v13`, which the profile pages' xG note links to),
  "Where it still struggles" (tip-ins 88, 60+ ft 85), and "Why xG doesn't know who's
  shooting" (`#finishing`, the ~8% story). When 2026-27 fills in, refresh those numbers from
  `results/test_to_date/`.
- **Rink check done, no fix** (2026-11-10, `diagnostics/check_rinks.py`,
  `results/checks/rinks.txt`, FINDINGS "Checking rinks for scorekeeping"). Owner's rules:
  group by `games.venue` (20+ games a season), 2022-23 to 2025-26 each season on its own,
  compare against the same teams' other games, ranges from game resamples, report only.
  Recorded distance differs by rink beyond chance (about ±1 ft; EDM, MTL closer; MIN, SEA
  farther in both the shooter- and defender-matched comparisons), but matched goals / xG
  doesn't: no rink clears 1 in any season, and the gaps' implied effect (1 ft = about 5% of
  xG) mostly doesn't show up. Anaheim's close tip-ins are a tip-in habit (more tips, logged
  closer), not a wider close-shot pattern.
- **Leaderboard** has a Test season column; runs whose test season was in progress are
  marked "to date" (`model/reporting.py`).
- **Stoppage leak fixed in the table** (2026-11-04, see `docs/FINDINGS.md`): 170 shots had
  a stoppage the shot itself caused (review, challenge, goalie froze it) as their previous
  event, 57% goals, and v1.4 learned it (97.8 xG on 97 goals). `shot_prev_any` now skips
  stoppages; check 12 in the build guards it. v1.4.1 is the refit on the fixed
  table. Penalty shots are separate ('penalty' event, typeCode PS) and fine.
- **Shooter (v1.5) and goalie (v1.6) priors tried, neither won** (2026-11-04, details in
  `docs/FINDINGS.md`). Priors in `shot_priors_fenwick` (`features/build_priors_fenwick.py`,
  rerun after every table build if they're used; leak test `tests/test_priors_fenwick.py`).
  Late 2025-26 log loss: baseline (v1.4 recipe on the fixed table) 0.22107, v1.5 0.22110,
  v1.6 0.22123; differences inside the noise. The priors do flatten goals / xG across
  shooter and goalie fifths. 2026-27 not touched; nothing saved, scored or exported.
  Owner's rules for this work: shooter first, goalie second; baseline = v1.4's settings
  on the same table; priors from games before the shot's date only, never season-pooled;
  K on raw counts; no history = league average; keep v1.4's settings, weights and
  constraint, no search; pick on late 2025-26 log loss, test only the pick on 2026-27 once.
- **'Short' misses** were dropped from `shot_features_fenwick` (`71a1a15`). v1.3 never saw
  them, but `shot_xg` still holds old v1.2 rows for them, so always join `shot_xg` through
  `shot_features_fenwick`. `export/profiles.py` and `export/period_tables.py` do.
- **Tip-in check done** (`06ad2ae`, `results/checks/tipins.txt`). Tip-ins used to score
  about as the model expects (goals / xG 1.03 in 2022-23). They fell to 0.79 in 2025-26 and
  0.78 so far in 2026-27. It's league-wide, not a few rinks: 30 of 32 rinks logged more
  tip-ins within 10 ft per game in 2025-26. That points to the NHL labelling more shots as
  tip-ins starting in 2025-26.
- **Team and player profile pages** are live (`c5ee1ae`): the "Teams & players" tab,
  `team.html` and `player.html`, built by `export/profiles.py` from the stat list in
  `export/profile_metrics.py`.
- **Profile ranks** (owner's rules, 2026-11-03): each stat with a better direction shows a
  badge like "28th out of 32" or "45th out of 212 forwards" under "League average: X".
  `profile_metrics.rank_rows` works them out at export time and stores
  `ranks: {key: [rank, pool, tied, tier]}` in every season row (schema 2); the browser only
  reads them. 1st is best (lowest is 1st for 'low' stats). Teams rank against that season's
  teams, players against their group (F, D, G) on their season row. Only entries meeting
  the stat's minimum are in the pool, for teams too: owner chose this on 2026-11-03,
  when 10-12 teams were under the special-teams minimums early in 2026-27. Fewer than 10
  qualifiers means no rank. Ties at the shown precision share the better rank ("T-12th");
  `shown()` copies the page's rounding exactly. Thirds: k = round(n/3), top rank <= k,
  bottom rank > n - k (11/10/11 at 32): green #4ade80, blue #6cb4ff, red #ff8080
  (lightened from #ff6b6b, which only reached 4.54:1). Measured badge contrast in Chrome:
  6.59, 5.44 and 5.04:1.

- **v1.4-fenwick, live 2026-11-04 to 2026-11-10** (`model/train_v1_4_fenwick.py`, logs
  `results/v1_4_fenwick_select.log` and `_final.log`, results in `results/v1_4-fenwick/`).
  v1.3's features and recency weights, plus a 50-try settings search (best: try 31, 11
  leaves, 1734 trees, `search.csv`), and xG can never go up with distance (monotone
  constraint, checked: 0 rises over 400,000 steps). The constraint costs 0.00006 log loss
  with the winning settings, 0.00029 with v1.3's (`constraint_cost.csv`). Platt and isotonic
  calibration both lost on late 2025-26 cross-fit, so there's none
  (`calibration_check.csv`). Late 2025-26 log loss 0.22071 vs v1.3's 0.22100. One test on
  2026-27: log loss 0.22565 vs 0.22587, AUC 0.7834 vs 0.7832, O/E 0.999 vs 1.003, tip-in
  O/E 0.899 vs 0.884 (`test_vs_v1_3.csv`). Rescoring reproduced those exactly (2026-27:
  1,251 goals vs 1,252.4 xG).
  The saved model (`model/artifacts/xg_v1_4_fenwick.pkl` + `_meta.json`) is gitignored, so
  it's only on the PC; `--stage final` rebuilds it identically (seeded) from
  `selection.json`. The first saved file pickled its wrapper class as `__main__.Calibrated`,
  which the scorer couldn't load; on 2026-11-04 it was re-saved under
  `model.train_v1_4_fenwick` (identical predictions) and the trainer now saves it that way.
  Don't rerun `--stage select` or look at 2026-27 to re-pick. Its
  leaderboard row shows `a4a537f` because it was trained before the script was committed
  (`f11bd9e`).

## Waiting on the owner

- The CI run for the sim tests (1d), then step 1b's real-game check.
- Step 2 needs the review items below (Next up 1) done before anything is frozen; then
  the owner reads the step-2 report.

## Next up

1. **Owner's review of simulator v0 before freezing** (2026-11-17), in this order:
   a. Done 2026-11-18 (see Where things stand). **Owner's call (2026-11-25):** (1) yes to
      penalty-call rates that depend on the penalized team's strength (even, power play,
      shorthanded), from the same rolling league window as the other league numbers;
      each team's "average team" call counts use the same three rates, so the ratings
      and the engine count calls the same way. Goes in with the 1b fix (step 4 of the
      review). (2) Offsetting (netted) calls stay out of v0: listed under "What v0
      leaves out" in `docs/SIMULATOR.md`; the step-2 report notes it next to the 4-on-4
      minutes.
   b. Two scrum cases in `power_plays()` (`sim/data.py`). Confirmed on made-up
      penalties in the cloud review: both come out as described below; what's left is
      real games. First: 2+2 on one player vs 2 on an opponent gets joined into a 4
      before netting; it should net first, leaving a 2-minute power play. Second: a
      major plus a minor on one player counts as 5 minutes, but the team is short 7.
      Check real ones on the shift charts, count them, and fix what's worth fixing, with
      a test.
   c. Any change to `sim/data.py`: rebuild the cache (`load(..., refresh=True)`) and
      rerun the step-2 search (`.venv/bin/python -m sim.pick`, about 30 min on 4 workers;
      stop it by PID, not `pkill -f`).
   d. CI: done 2026-11-25, `tests/test_sim_engine.py` and `tests/test_sim_ratings.py` in
      the reporting-tests job, `sim` in the compileall line (both pass on the PC); the
      owner checks the run.
   e. Step-2 report: v0 vs Elo directly, with the range on the gap. Only then freeze
      `results/sim_v0/selection.json`.
2. **Simulator step 3**, later: test v0 once on 2026-27 at 800 games (not before).
3. **Left from the audit, for later** (owner, 2026-11-11): late corrections (gap 5),
   keying `shot_xg` on game and event instead of shot_id (gap 7), the 0.23% strength
   disagreement (gap 3).

Standing decision from the owner (2026-11-04): **shooter and goalie skill stay out of the
live xG, for good**, finishing priors included. "Goals minus xG" and "goals saved above
expected" only mean something if xG doesn't know who's shooting or who's in net. That skill
becomes a separate layer later, for the game simulator.

Still open, lower priority: refresh the model tab's 2026-27 test numbers as the season
goes (refresh.sh, then report_test_to_date). The EDGE goalie comparison from FINDINGS is
now part of the simulator's goalie layer.

Loose ends: there's no scheduled GitHub Actions refresh yet (`export/refresh.sh` runs by
hand on the PC). Each refresh commits a few hundred small files under `site/data/`, and player
names come from play-by-play, so they show as "C. McDavid".
