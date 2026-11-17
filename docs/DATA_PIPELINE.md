# How the data gets from the NHL to the site

This is a map of the whole pipeline: where every number on the site comes from, what
happens to it on the way, what checks guard each step, and what nothing checks yet. It
was written on 2026-11-11 from the code alone, in a session without the database, so
anything the code can't prove is marked **not confirmed from code**. The checks that
need the real data are in the prompt at the end.

File references look like `ingest/run.py:48-50` (file, then line numbers).

**Updated 2026-11-17.** The data checks ran on the PC on 2026-11-11
(`results/checks/data_audit.txt`), and the gaps were fixed in three groups: group 1
made problems stop the refresh (`7f02682`), group 2 fixed the shift charts
(`53f1810`, section 4b), group 3 is these docs. Section 5 marks each gap checked,
fixed or left for later.

## 1. Where the data comes from

Everything comes from the NHL's free public sites. Nothing is bought, scraped from a
third party or typed in by hand.

| Source | What it gives us | Code | Lands in |
|---|---|---|---|
| Schedule, `api-web.nhle.com/v1/schedule/{date}` | Which games were played each day. Only regular season (type 2) and playoffs (type 3) are kept. | `ingest/run.py:61-96` | the list of games to fetch |
| Play-by-play, `api-web.nhle.com/v1/gamecenter/{id}/play-by-play` | Every event in a game: time, type, x/y location, players, the strength code, the score | `ingest/run.py:99-129`, `ingest/parser.py:382-539` | `games`, `shots`, `plays_context` |
| Box score, `api-web.nhle.com/v1/gamecenter/{id}/boxscore` | Rosters, positions, starting goalies | `ingest/run_stage_a.py:80-110`, `ingest/parser_landing.py:110-190` | `game_rosters`, `players` |
| Shift charts, `api.nhle.com/stats/rest/en/shiftcharts` | Every shift: who, which period, when it started and ended | `ingest/run_stage_a.py:51-78`, `ingest/parser_shifts.py` | `shifts` |
| HTML time-on-ice reports, `www.nhl.com/scores/htmlreports/{season}/TH…/TV….HTM` | A backup for games whose shift chart comes back empty. Rows are tagged `htmlreport` so they can always be told apart. | `ingest/run_shifts_html.py:1-11, 27-38` | `shifts` |

Defined but not used yet: the `landing` endpoint (`ingest/config.py:16`) and the NHL EDGE
endpoints (`ingest/config.py:19-20`). The EDGE tables exist in `db/schema.sql`, but
nothing fills them.

**How fetching works** (`ingest/http_client.py`):
- It retries network errors and server errors (5xx) up to 4 times, waiting longer each
  time (lines 37-49). A 4xx error, such as a game that doesn't exist, isn't retried.
- It waits 0.25 seconds between calls (`ingest/config.py:51`) and sends a User-Agent
  that says who it is (line 52).
- Every raw response is saved to `raw_data/` on the PC (lines 52-92), written to a temp
  file first so a crash can't leave half a file. `raw_data/` is git-ignored, so the raw
  files and the database live only on the PC.

**Only official games get in.** The NHL marks a game `OFF` once it's official; until
then, even a "final" game can still be corrected. Ingest skips anything not `OFF`
(`ingest/run.py:113-116`), caches a play-by-play file only once it's `OFF`
(lines 101-104), caches a schedule day only once every game that day is `OFF`
(lines 53-58), and never looks past yesterday (lines 69-70). That's why it's safe to run
in the middle of a season.

## 2. From raw data to the website, step by step

**Step 1: play-by-play into tables** (`ingest/parser.py`, `ingest/db.py`)
- Every shot event, blocked shots included, becomes a row in `shots`. Faceoffs, hits,
  giveaways, takeaways, penalties, stoppages and shots also go into `plays_context`, so
  later steps can ask "what happened just before this shot?" (`parser.py:65-72`).
- **Which net a team is attacking** isn't reliable by period number, so it's worked out
  per team, per period, from where that team's shots cluster (`parser.py:197-246`).
  Every shot is then flipped so it attacks the net at x = +89 (`parser.py:177-194`).
  This fixed an early bug that put about 28% of shots more than 100 feet from the net.
  A test guards it (`tests/test_parser.py:223-228`).
- **Distance and angle** are measured from the shot to the middle of the net at
  (89, 0) (`parser.py:249-261`).
- **Strength** (5-on-5, power play and so on) comes from the NHL's `situationCode`, a
  four-digit code: away goalie, away skaters, home skaters, home goalie
  (`parser.py:264-304`). A shot counts as empty-net when the defending goalie is off
  (`parser.py:463-471`).
- **The score before each shot** is a running count of goals as the events go by
  (`parser.py:412-414, 507-512`).
- Each game is written in one database transaction. A row that's already there is left
  alone, matched on game and event number (`db.py:51-121`).

**Step 2: shifts and rosters** (`ingest/run_stage_a.py`, `ingest/run_shifts_html.py`)
- Shift charts and box scores are fetched only for games already in the database.
  Regular-season shootout "shifts" and broken records (start after end, no player) are
  skipped (`parser_shifts.py:33-39, 106-109`); in the playoffs period 5 and later are
  more overtimes and are kept. Games with an empty shift chart get the HTML report
  instead.
- `ingest/fix_shifts.py` then fixes goalies the shift chart lost (from the HTML report)
  and clips shifts that run past their period's end. See section 4b.

**Step 3: the shot table the model learns from** (`features/build_features_fenwick.py`)
- Kept: unblocked attempts, meaning shots on goal, goals and misses (analytics people
  call this "Fenwick"), including playoffs (`lines 151-152`).
- Removed: blocked shots, regular-season shootout attempts, and three labels the NHL only
  used for part of the four seasons: 'teammate-blocked', 'short' misses (fanned shots)
  and 'failed-bank-attempt' (`lines 111-145`; why in the docstring, lines 25-36).
- Built: the 17 inputs (listed in the README under "How the model works"), including
  rebounds (a shot within 3 seconds of a teammate's), post rebounds, rushes, and what happened just before. That lookup
  skips stoppages, because a stoppage the shot itself caused can be listed first
  (`lines 155-173`).
- The whole table is rebuilt from scratch every time (about a minute).

**Step 4: the xG model** (`model/train_v1_4_1_fenwick.py` and the scripts it builds on)
- LightGBM, a model built from many small decision trees, predicts how likely each shot
  is to be a goal from its 17 inputs (`model/train_v1.py:58-80`, plus the post-rebound
  flag from v1.1 and whose play came before from v1.2).
- Trained on 2022-23 to 2025-26, with each older season counting half as much as the
  one after it (`model/train_v1_3_fenwick.py:92-95`). xG is never allowed to go up as a
  shot gets farther away (`model/train_v1_4_fenwick.py:78`). Settings came from a
  50-try search judged on late 2025-26.
- The saved model lives in `model/artifacts/` on the PC (git-ignored). Retraining is
  seeded, so it rebuilds the same model.

**Step 5: scoring every shot** (`model/score_fenwick.py`)
- Every row of the shot table gets an xG from the saved model, stored in `shot_xg` with
  the model's name. A run replaces only its own version's rows (lines 61-70) and prints
  goals ÷ xG by season (lines 52-56).

**Step 6: exports** (`export/period_tables.py`, `export/profiles.py`)
- Regular season, official games only. Shots are always joined through the shot table,
  so dropped shots stay out. Time on ice comes from the shift charts, second by second:
  5-on-5 means five skaters and a goalie each side (`export/profiles.py:1-40`).
- The output is JSON files in `site/data/`. Ranks are worked out here, and a file is
  only rewritten when its numbers change.

**Step 7: the site**
- Plain HTML pages read those JSON files. There's no database behind the site. Netlify
  publishes whatever is on `main`.

**In season**, `export/refresh.sh` runs steps 1-6 for the current season (lines 17-27)
and stops at the first command that fails (`set -euo pipefail`, line 13). Three steps
stop it on purpose when the data looks wrong: `ingest/fix_shifts.py`,
`ingest/check_ingest.py` and the shot-table build (section 3).

## 3. What guards each step

| Guard | What it catches | Where | Runs |
|---|---|---|---|
| Play-by-play parser test | Event parsing, coordinate flipping (including the old wrong-net bug), strength, empty net | `tests/test_parser.py` | CI (every push) |
| Player-in-event test | Who hit, who drew the penalty and so on, and the count of dropped shots | `tests/test_parser_actors.py` | CI (since 2026-11-11) |
| Shift parser tests | API and HTML shift parsing, playoff overtimes kept | `tests/test_parser_shifts.py`, `tests/test_parser_shifts_html.py` | CI |
| Goalie in net vs shift chart | Which events say a goalie is in, and which no goalie shift covers | `tests/test_shift_checks.py` | CI (since 2026-11-17) |
| Box score test | Rosters, positions, starting goalies | `tests/test_parser_landing.py` | CI |
| Period page math | The odds math and regression to the mean | `tests/test_period_math.py` | CI |
| Profile stats and ranks | Stat formulas, ties, thirds, minimums | `tests/test_profile_metrics.py` | CI |
| Calibration math | Goals ÷ xG and calibration slope | `tests/test_calibration_stats.py` | CI |
| Prior leak test | A later game can't change an earlier rating | `tests/test_priors_fenwick.py` | CI (since 2026-11-11) |
| Goalie priors | Needs the database | `tests/test_goalie_priors.py` | by hand only |
| Syntax check | Any file that can't even be read by Python | `.github/workflows/tests.yml:81` | CI |
| 12 shot-table checks | Label leaks, leftover dropped labels, shots after a stoppage, goal rates by season | `features/build_features_fenwick.py:293-502` | every build; checks 3 (missed-shot goals), 8, 11 and 12 must be 0 and **stop the build** if not (`MUST_BE_ZERO`, lines 520-575), the rest are printed for reading |
| Ingest check | Games missing or not official, failed downloads, goals that don't add up to the final score, dropped shots; prints parse warnings | `ingest/check_ingest.py` | every refresh, **stops it** |
| Shift fixes | Goalies the shift chart lost, playoff overtimes, shifts past the period's end | `ingest/fix_shifts.py` | every refresh, **stops it** if a goalie can't be found in either source |
| Trainer stops | Leftover stoppage shots, a fit that doesn't reproduce, xG rising with distance | `model/train_v1_4_1_fenwick.py:68, 82, 90` | when the model is trained |
| Goals ÷ xG by season | A model that's suddenly off | `model/score_fenwick.py:52-56` | every scoring run, printed |
| HTML shift check | HTML shifts against the API's, where both exist | `ingest/run_shifts_html.py:75` (`--validate`) | by hand |
| One-off checks | Drift, tip-ins, rinks, NHL field names, failed shifts, on-ice counts | `diagnostics/`, `db/checks/` | by hand |

## 4. Known data problems and how each is handled

The reasons and numbers are in `docs/FINDINGS.md`.

- **Saves recorded as misses since 2023-24.** The model trains on unblocked attempts,
  which counts both (`build_features_fenwick.py:1-16`; FINDINGS "The data problem").
- **'Short' misses since 2023-24** (fans and flubs, never goals): removed. Check 11.
- **Labels that exist for only part of the window**: removed. Check 11.
- **Wrist and snap shots relabelled in 2024-25**: merged into one type. Check 7.
- **A missing shot type that only ever appeared on goals**: it gave the answer away, so
  it's filled with the most common type. Check 8 looks for any value like it.
- **More shots called tip-ins since 2025-26**: recent seasons count more (v1.3 on).
  `results/checks/tipins.txt`.
- **Giveaways doubled in 2024-25**: kept apart from takeaways, and watched by check 10.
- **More posts logged from 2024-25**: the post flag only sits on the *previous* shot,
  because on the shot itself it gives the answer away. Check 9.
- **A stoppage caused by the shot, listed before it**: the lookup skips stoppages.
  Check 12 and the trainer both stop if one is left.
- **Empty shift charts** (57 games in 2024-25, 62 early in 2026-27): rebuilt from the
  HTML reports.
- **Rinks recording distance differently**: checked; about ±1 ft with no effect on
  goals ÷ xG, so no fix. `results/checks/rinks.txt`.
- **Players counted on the ice twice at line changes**: a player is on the ice for a
  shot at second t if their shift has start < t ≤ end (`export/profiles.py:33-35`).
  The audit found this edge agrees with the play-by-play about 10x as often as the
  other one. The older database view `v_shots_with_onice` used start ≤ t < end; since
  2026-11-17 it uses the same rule as the profiles (`db/migration_stage_a_onice_fix.sql`,
  `db/schema_stage_a.sql`). The two rules give a different total on the ice for 1.44%
  of shots. Only `diagnostics/check_onice_overcounts.sql` reads the view.

### 4b. The shift-chart fixes (group 2, 2026-11-17)

All three run in `ingest/fix_shifts.py`, which `refresh.sh` runs after the HTML
recovery step, and which only acts where a problem is still there. The backfill's
output is in `results/checks/fix_shifts.txt`.

- **Playoff overtimes past the 1st.** The API shift parser threw away period 5 and
  later as a shootout. Only the regular season has a shootout; in the playoffs period
  5 is a 2nd overtime. The parser now skips them only in regular-season games, like the
  HTML parser, and the 15 playoff games that went that far got 2,453 shift rows back
  from the saved files. Profiles use the regular season only, so the site didn't move;
  the simulator will use these.
- **Goalies the shift chart lost.** The audit flagged 25 team-games with the goalie on
  the ice more than 5 minutes less than the game, and called them all a lost goalie.
  That was wrong. The play-by-play's `situationCode` says at every event whether each
  net has a goalie, and in 21 of the 25 it agrees the net was empty: real goalie pulls.
  The check is now an event (faceoffs and stoppages left out, since they sit on shift
  changes) where `situationCode` has the goalie in and no goalie shift covers it. That
  finds 14 team-games in 12 games, 10 of them too short for the audit's flag (1-5
  minutes, mostly OT). The worst: 2022020567, the home goalie missing for the whole
  3rd; 2025021031, both goalies from 43:56 through OT; and 2023020517, where overlapping
  shifts gave a goalie 4,427 seconds in a 60-minute game. The HTML report had the
  goalie in every one, so its goalie shifts replaced the API's for those teams
  (skaters left alone). 20 more team-games miss 1-2 events with the HTML report matching
  the API to the second: that's timing at a shift change, left as is. If 3 or more
  such events can't be covered by either source, the step stops the refresh.
- **Shifts past the period's end.** A shift's end is clipped to its period's last
  'period-end' event, never before its start. That changed 3 rows, all in 2024021158,
  where the chart kept three players on until 5:00 of an OT that ended at 0:59.
- **What it moved on the site.** Team 5-on-5 time rose 16-17 minutes over a season for
  CBJ and NYI (2022-23) and CBJ and FLA (2025-26), and under 3 minutes for six other
  team-seasons. No 2026-27 team changed. 17 team ranks moved one place (3 badge
  colors); player 5-on-5 time rose by up to 8 minutes. Shifts don't feed xG, so nothing
  was rebuilt or rescored.
- **Row counts in `ingest_log`** (group 3). `db.py`'s bulk inserts sent rows in pages of
  100, and `cur.rowcount` only counted the last page, so `shots_parsed` was low for
  5,450 of 5,922 play-by-play rows and every shifts row (rosters, under 100 a game, were
  right). Each insert is now one statement, and the old rows were set to the real
  count of rows in `shots` and `shifts` for each game. No data was ever missing.

## 5. Gaps: what nothing checked, and where each stands

Ranked by how much they could hurt the numbers on the site. Status as of 2026-11-17;
the numbers are from the audit (`results/checks/data_audit.txt`) unless noted.

1. **FIXED (group 1, `7f02682`). The 12 shot-table checks printed but never stopped
   anything.** Checks 3 (missed-shot goals), 8, 11 and 12 must be 0, and the build now
   exits with an error if one isn't, which stops `refresh.sh` before scoring
   (`build_features_fenwick.py:520-575`).
2. **CHECKED, and now guarded. Do the goals add up?** Play-by-play goals (plus 1 for the
   shootout winner) matched the final score in all 5,877 games at the audit; every
   finished season has all 1,312 games. `ingest/check_ingest.py` now checks both on
   every refresh and stops it on a mismatch (5,922 games pass through Nov 15).
3. **CHECKED, small; left for later. Strength from two places.** `situationCode` and
   the shift charts disagree on the strength (even, up a skater, down a skater) for
   0.23% of unblocked shots, 0.2-0.3% each regular season. The start < t ≤ end rule is
   the right edge: the other disagrees about 10x as often.
4. **CHECKED, not a problem. Blocked shots' team.** `eventOwnerTeamId` is the
   shooter's team on 100% of blocked shots, every season, checked against the rosters.
5. **CHECKED, left for later. Corrections after a game is official.** 30 random older
   games re-downloaded: shots, goals, scores and coordinates identical. One change in
   30: a penalty's length went from 15 to 5 minutes (2023020232). It doesn't touch xG,
   but it would touch penalty minutes in the simulator.
6. **FIXED (group 1). Parse warnings were thrown away.** They're counted per game in
   `ingest_log` (`db/migrations/004_ingest_log_warnings.sql`), backfilled (0 warnings
   in 5,877 games), printed at the end of each refresh, and a dropped shot stops it.
7. **LEFT FOR LATER. Older xG rows could end up on the wrong shots** after a reparse,
   because `shot_xg` is keyed on `shot_id`. Fix when it matters: key on game and event
   number.
8. **CHECKED, not a problem. The starting goalie as a guess.** Every team-game has
   exactly one starter, none from the "first goalie listed" fallback, and the starter
   was in net for the first shot faced in all but 3 of 11,754 team-games.
9. **FIXED (group 1). Two tests weren't in CI.** `tests/test_parser_actors.py` and
   `tests/test_priors_fenwick.py` run on every push; the run for `e30484a` passed.
10. **FIXED (group 3). FINDINGS called the 'short' misses open.** Marked resolved, with
    the counts and the effect on goals ÷ xG.
11. **FIXED (group 2, `53f1810`). Found by the audit: shift charts** (audit check 7).
    Playoff overtimes past the 1st, goalies the chart lost, and shifts past the
    period's end. See section 4b.

## 6. Read these yourself, in this order

These eight are the backbone. If you can explain each one, you can explain the project.

1. **`ingest/config.py`** (51 lines). Every source and season window. *Question to
   answer: which endpoints feed which tables?*
2. **`ingest/run.py`** (205 lines). How games are found and the official-only rule.
   *Why is a "final" game not good enough?*
3. **`ingest/parser.py`**, mainly `parse_pbp` (382-539), `_infer_attack_sides`
   (197-246) and `_distance_angle` (249-261). Raw JSON to rows. *Why flip coordinates,
   and how do you know which way a team is attacking?*
4. **`features/build_features_fenwick.py`**: the docstring (1-90), then the SQL
   (106-290). Every choice about what's in and out. *Why unblocked attempts, and why
   drop 'short' misses?*
5. **`model/train_v1.py:58-80`** and **`model/train_v1_4_fenwick.py`** (73-110). The 17
   inputs, the distance rule and the settings search. *Why judge on log loss, and why
   pick settings without touching the test season?*
6. **`model/score_fenwick.py`** (75 lines). How xG gets into the database, one version
   at a time. *How do you know which model made a number?*
7. **`export/profiles.py:1-40`** and **`export/period_tables.py:1-46`**. How shots turn
   into what the site shows, and time on ice from shifts. *What does "per 60 at 5-on-5"
   divide by?*
8. **`docs/FINDINGS.md`**. The why behind every fix, including what didn't work.

## 7. Prompt for the PC: the checks that need the data

Ran on 2026-11-11; kept for the record. Its script is `diagnostics/check_data_audit.py`.
It only reads; it changes no code or data.

```
Read-only data audit, from docs/DATA_PIPELINE.md section 5. Don't change any code,
table or cached file. Write the results to results/checks/data_audit.txt with the
script in diagnostics/check_data_audit.py, then update CLAUDE.md and the session log,
commit and push. For each check, give the count, the share, the worst examples, and
whether it's a problem.

1. Completeness: regular-season games per season in `games` (a full season is 1,312),
   any game not 'OFF', and every 'fail' row in ingest_log.
2. Goals add up: for each game, goals in `shots` before any shootout, plus 1 for the
   shootout winner, against home_score / away_score in `games`. List every mismatch.
3. Strength agrees: at each unblocked shot, skaters on the ice from `situationCode`
   against skaters from the shift charts at that second, by season.
4. Blocked shots' team: for blocked shots, is eventOwnerTeamId the shooter's team (from
   game_rosters, using shootingPlayerId) or the blocker's? Share by season.
5. Late corrections: re-download play-by-play for 30 random older games into a temp
   folder (not raw_data/) and compare goals, shots and event counts with the database.
6. Starting goalies: games where a team has zero or two starters in game_rosters, and
   how many starters came from the "first goalie listed" fallback, if that can be told.
7. Shift coverage: games with no shifts for a team, and team time on ice per game that's
   far from 60 minutes, by season, split by API and htmlreport rows.
8. Shot-table checks: run features.build_features_fenwick --verify-only and list any
   check whose "must be 0" isn't 0.
Stop and report. Don't fix anything yet; we'll decide fixes together.
```
