# How the data gets from the NHL to the site

This is a map of the whole pipeline: where every number on the site comes from, what
happens to it on the way, what checks guard each step, and what nothing checks yet. It
was written on 2026-11-11 from the code alone, in a session without the database, so
anything the code can't prove is marked **not confirmed from code**. The checks that
need the real data are in the prompt at the end.

File references look like `ingest/run.py:48-50` (file, then line numbers).

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
  Shootout "shifts" and broken records (start after end, no player) are skipped
  (`parser_shifts.py:30-35`). Games with an empty shift chart get the HTML report
  instead.

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

**In season**, `export/refresh.sh` runs steps 1-6 for the current season (lines 9-15)
and stops at the first command that fails (`set -euo pipefail`, line 5).

## 3. What guards each step

| Guard | What it catches | Where | Runs |
|---|---|---|---|
| Play-by-play parser test | Event parsing, coordinate flipping (including the old wrong-net bug), strength, empty net | `tests/test_parser.py` | CI (every push) |
| Player-in-event test | Who hit, who drew the penalty and so on | `tests/test_parser_actors.py` | by hand only |
| Shift parser tests | API and HTML shift parsing | `tests/test_parser_shifts.py`, `tests/test_parser_shifts_html.py` | CI |
| Box score test | Rosters, positions, starting goalies | `tests/test_parser_landing.py` | CI |
| Period page math | The odds math and regression to the mean | `tests/test_period_math.py` | CI |
| Profile stats and ranks | Stat formulas, ties, thirds, minimums | `tests/test_profile_metrics.py` | CI |
| Calibration math | Goals ÷ xG and calibration slope | `tests/test_calibration_stats.py` | CI |
| Prior leak test | A later game can't change an earlier rating | `tests/test_priors_fenwick.py` | by hand only |
| Goalie priors | Needs the database | `tests/test_goalie_priors.py` | by hand only |
| Syntax check | Any file that can't even be read by Python | `.github/workflows/tests.yml:71` | CI |
| 12 shot-table checks | Label leaks, leftover dropped labels, shots after a stoppage, goal rates by season | `features/build_features_fenwick.py:293-502` | every build, **printed for a person to read** |
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
- **Players counted on the ice twice at line changes**: the profiles count a player on
  the ice for a shot at second t if their shift has start < t ≤ end
  (`export/profiles.py:33-35`). The older database view in
  `db/migration_stage_a_onice_fix.sql` uses the opposite edge (start ≤ t < end); only a
  diagnostic uses it now, but the two rules should match.

## 5. Gaps: what nothing checks yet

Ranked by how much they could hurt the numbers on the site.

1. **The 12 shot-table checks print but never stop anything.** `verify()` prints each
   result (`build_features_fenwick.py:515-529`), and the build exits fine either way, so
   `refresh.sh` carries on to scoring and exports. If check 8, 11 or 12 found a problem
   and nobody read the output, it would reach the site. Fix: make the "must be zero"
   checks exit with an error, which `set -euo pipefail` turns into a stop.
2. **Nothing checks that the goals add up.** Nothing compares the goals in the
   play-by-play with each game's final score in `games`, or the games in the database
   with the games on the schedule. A missing or doubled goal would quietly shift
   goals ÷ xG, the period page and the profiles. (PC check 1 and 2.)
3. **Strength comes from two places that could disagree.** A shot's strength comes from
   the NHL's `situationCode` (`parser.py:264-304`), but time on ice comes from the shift
   charts (`export/profiles.py:1-40`). If they disagree, per-60 rates at 5-on-5, on the
   power play and on the penalty kill are off. **Not confirmed from code** how often
   that happens. (PC check 3.)
4. **Blocked shots might carry the wrong team.** Every shot's team is the NHL's
   `eventOwnerTeamId` (`parser.py:434-443`). If the NHL files blocked shots under the
   *blocking* team, every blocked shot's team is flipped. Blocked shots aren't in xG,
   but they feed "what happened just before" and would matter for anything counting
   all attempts. The tests use made-up payloads, so they can't tell. **Not confirmed
   from code.** (PC check 4.)
5. **Corrections after a game is official are never picked up.** A game that's in the
   database is skipped on every later run (`ingest/run.py:135-147`), the raw file is
   reused (`http_client.py:65-69`), and existing rows are never updated
   (`db.py:88, 119`). FINDINGS item 8 notes the NHL often revises shot data after games.
   (PC check 5.)
6. **Parse warnings are thrown away.** The parser notes skipped events, like a shot with
   no team (`parser.py:379, 439, 538`), but `ingest/run.py:108-129` never looks at them,
   so a dropped shot leaves no trace. Fix: count them into `ingest_log`.
7. **Older xG rows can end up on the wrong shots.** `shot_xg` is keyed on `shot_id`, and
   `ingest/reparse.py:32` restarts those numbers. After a reparse, the kept v1.2-v1.4
   rows would point at different shots until each is rescored. That's written down
   (`db/migrations/003_shot_xg.sql:14-15`) but not enforced. Fix: key on game and event
   number instead.
8. **The starting goalie is sometimes a guess.** When the box score has no starter flag,
   the first goalie listed is used (`ingest/parser_landing.py:151-158`). It doesn't
   affect xG, but the simulator's goalie layer will lean on it. (PC check 6.)
9. **Two tests that need no database aren't in CI**: `tests/test_parser_actors.py` and
   `tests/test_priors_fenwick.py`. Both pass. Fix: add them to the workflow.
10. **FINDINGS still calls the 'short' misses an open issue** (`docs/FINDINGS.md:10, 120`),
    though they were dropped in `71a1a15`. Docs only.

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

Paste this into Claude Code on the PC. It only reads; it changes no code or data.

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
