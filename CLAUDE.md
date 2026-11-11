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

Updated 2026-11-11 (v1.4.1 live; fixing what the data audit found, before the simulator).

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
  also tests the dropped-shot count) and `tests/test_priors_fenwick.py`.
- **Data pipeline audit done** (2026-11-11). Code half (cloud):
  `docs/DATA_PIPELINE.md` maps every source, step, guard and known problem and ranks 10
  gaps. Data half (PC, read-only): `diagnostics/check_data_audit.py`, results with a
  summary on top in `results/checks/data_audit.txt`. The data is clean where it matters:
  every season complete, goals add up to the final score in all 5,877 games, blocked
  shots carry the shooter's team (gap 4 is a non-issue), one starter per team-game with
  no fallback, 30 re-downloaded games unchanged apart from one penalty's length (15 -> 5
  min), shot-table must-be-zero checks all 0. Strength from `situationCode` vs the shift
  charts disagrees on 0.23% of unblocked shots. Two small shift issues: the API shift
  parser drops periods 5+ (15 playoff games with 2+ overtimes have no shifts after the
  1st OT), and in 25 regular-season team-games the NHL's shift chart loses the goalie
  partway through a period, so profiles miss up to 20 min of 5-on-5 there. The owner
  wants to be able to explain every step, so no fixes until they pick.
- **Game simulator: plan written, not built** (`docs/SIMULATOR.md`). The owner OK'd the
  direction on 2026-11-11 with five changes (Next up, item 2) but chose to audit the data
  first. Team-level v0 first, then goalie and shooter layers. Built and picked on 2023-24
  to 2025-26, tested once on 2026-27 at 800 games against the home-win rate and plain Elo.
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
- **Data is through 2026-11-09** (refreshed 2026-11-10: 285 games in 2026-27, 502,509
  attempts in all). Only v1.4.1 scores the new games in `shot_xg`.
- **Site's 2026-27 numbers come from `results/test_to_date/`** (`model/report_test_to_date.py`,
  log `results/test_to_date.log`): every saved model (v1.4.1, v1.4, v1.3, v1.2) rescored on
  the test season as it stands, report only. Rerun it after a refresh before touching the
  copy. Training-time `results/<version>/` files stay frozen. Through Nov 9: log loss
  v1.4.1 0.22955, v1.4 0.22951, v1.3 0.22951, v1.2 0.23147; O/E 1.014; AUC 0.7797. v1.4's
  early edge over v1.3 is gone, and the copy says so.
- **v1.3-fenwick** (`model/train_v1_3_fenwick.py`, log in `results/v1_3_fenwick_train.log`):
  trained on 2022-23 to 2025-26 with recency weights (half-life 1 season). Chosen on late
  2025-26 (log loss: recency h=1 0.22100, h=2 0.22113, era flag 0.22129, plain retrain
  0.22168; `results/v1_3-fenwick/selection.csv`). 2026-27 to date: AUC 0.7832, O/E 1.003,
  tip-in O/E 0.88 (v1.2 on the same shots: 0.7752, 0.997, 0.78).
- **Site copy is on v1.4.1** (model tab in `site/index.html`, `site/methods.html`): headline
  stats and calibration chart from `results/v1_4_1-fenwick/`, "What changed in this version:
  closing a leak" (`#v141`, the stoppage story, called the whistle on the page), then
  fine-tuning (`#v14`) and tip-ins (`#v13`, which the profile pages' xG note links to),
  "Where it still struggles" (tip-ins 90, 60+ ft 86), and "Why xG doesn't know who's
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

- **Say go for group 2.** Stopped for the day on 2026-11-11 right after group 1
  (`7f02682`, `9f77da4`). Group 2 is not started: no code, data or files touched for it.
  The owner wants each group committed, explained in plain words, and stopped before
  the next.
- Two things from group 1 not yet seen working: the CI run for `7f02682` (`gh` on the
  PC says "Bad credentials"; the owner can run `! gh auth login`), and a full
  `export/refresh.sh` with the new stop steps (not run, so no new games were pulled;
  data is still through 2026-11-09). The next refresh is that test.

## Next up

1. **Audit fixes, groups 2 and 3** (owner's list, 2026-11-11), one group per commit, then
   stop and explain what changed, which lines and why:
   - Group 2, shift charts: the API shift parser skips period 5 only in regular-season
     games (the shootout); in the playoffs it's a 2nd OT, so re-parse the 15 playoff
     games. For the 25 team-games where the shift chart loses the goalie mid-period, use
     the HTML report like the empty-chart games; if the HTML report lacks the goalie
     too, stop and tell the owner. Clip shifts that run past their period's end
     (2024021158). Re-export profiles and report how much 5-on-5 time changed.
   - Group 3, docs: in `docs/DATA_PIPELINE.md` mark gaps 2, 3, 4 and 8 checked with the
     audit's numbers, and gaps 1, 6 and 9 fixed. `docs/FINDINGS.md`: 'short' misses are
     no longer open. Make the old on-ice view in `db/` use the profiles' start < t <= end
     rule.
   - Left for later, just noted: late corrections (gap 5), keying `shot_xg` on game and
     event instead of shot_id (gap 7), the 0.23% strength disagreement.
2. **Simulator plan changes** the owner agreed on 2026-11-11. Put them in
   `docs/SIMULATOR.md`, then push:
   - Penalties: rate teams on power plays drawn and given up (when the shift charts show
     one side short-handed), not on raw penalty calls; fighting majors and matching
     minors usually cancel out. Double minors are 4 minutes. Check that simulated minutes
     per game at 5-on-5, power play and penalty kill match the shift charts.
   - Off-season: the team ratings get a pull back toward average between seasons, like
     Elo, as one more setting picked on the build seasons.
   - Calibration pass rule: one overall check (calibration slope near 1 and intercept
     near 0, each range covering that value) instead of "every group within its range",
     which fails a perfectly calibrated model about 40% of the time with ten groups. Keep
     the ten groups as the picture and the other two pass rules as they are. Test at 800
     games of 2026-27 stays.
   - Goalie and shooter layers: test them on the rest of 2026-27 after the first 800
     games, compared with v0 on the same games, using only what would be known when the
     site publishes (starters and lineups come late).
   - Add to "What v0 leaves out": ratings aren't adjusted for the opponents faced.
3. **Game simulator v0**, only when the owner says go: follow `docs/SIMULATOR.md` (build
   in `sim/`, pick on 2023-24 to 2025-26, freeze `results/sim_v0/selection.json`,
   report). Don't look at simulator output on 2026-27 until it has 800 games, then test
   once.

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
