# Session log

What got done each working day, newest first. Claude adds to today's entry every time it
pushes. Commit hashes point to the details.

## 2026-11-11

- PC: audit fixes group 1, problems stop the refresh (commit below). Shot-table
  must-be-zero checks exit 1; new `ingest/check_ingest.py` in `refresh.sh` stops on
  missing games, goals that don't add up, or dropped shots, and prints parse warnings
  at the end; warnings counted per game in ingest_log (migration 004, backfilled: 0 in
  5,877 games); CI runs the actors and priors tests. Each stop tested by planting
  problems in a rolled-back transaction. Group 2 waits on the owner's go.
- PC: reran the read-only audit; identical to `16c18f2` apart from the timestamp.
- PC: read-only data audit (`16c18f2`): `diagnostics/check_data_audit.py`,
  `results/checks/data_audit.txt`. All OK except two small shift-chart issues: no shifts
  after the 1st OT in 15 playoff games (the API parser drops periods 5+), and 25
  regular-season team-games where the shift chart loses the goalie mid-period. Goals add
  up in all 5,877 games; strength sources disagree on 0.23% of shots; blocked shots carry
  the shooter's team; 30 re-downloaded games unchanged but for one penalty length.
  Nothing fixed; the owner picks.
- Cloud: the owner chose to slow down before building the simulator, so the pipeline
  is easy to explain end to end. Code-only audit in `docs/DATA_PIPELINE.md`: sources,
  every step with file references, guards (CI vs by hand), known data problems, 10
  gaps ranked, a reading list, and a read-only prompt for the PC data checks. Agreed
  simulator plan changes are saved under Next up; nothing built.
- PC: game simulator plan in `docs/SIMULATOR.md` (`0e0a970`), waiting on the owner's OK.
  v0 is team-level (xG rates at 5-on-5, PP, PK and penalty rates, from earlier games,
  recency-weighted and pulled toward average, home ice), played out 10,000 times a game
  with NHL overtime and shootout rules. Picked on 2023-24 to 2025-26, tested once on
  2026-27 at 800 games against the home-win rate and a plain Elo. To size the test I ran a
  rough Elo against the home-win rate on game results, 2026-27 to date included (no
  simulator anything): per-game log loss gap SD 0.13-0.19; Elo's edge 0.018-0.026, except
  0.0019 in 2025-26. README step 4 now points to the plan.

## 2026-11-10

- Cloud: the owner picked the game simulator as what's next and stopped for the day
  before it started. The plan-first instructions and decisions are under Next up in
  `CLAUDE.md`; nothing built yet.
- PC: rink-by-rink shot location check (`88871da`), report only (`diagnostics/check_rinks.py`,
  `results/checks/rinks.txt`, FINDINGS). Rinks do record distance a little differently
  (about ±1 ft, beyond chance; EDM and MTL closer, MIN and SEA farther, in both matched
  comparisons), but goals / xG against the same teams' other games shows nothing beyond
  chance, and the gaps' implied 5%-per-ft effect mostly doesn't appear. No fix suggested.
  Anaheim: more tip-ins logged, and logged closer; its other shots aren't closer.
- PC: ran `export/refresh.sh` (`ceeebc4`): 83 new games (2026-27 now 285 games, through Nov 9), one
  shift chart from the HTML fallback, build checks clean. New report-only
  `model/report_test_to_date.py` rescores 2026-27 with every saved model into
  `results/test_to_date/`; the model tab, methods page and README now take their 2026-27
  numbers from it (23,378 shots, 1,797 goals vs 1,771.8 xG, AUC 0.780, log loss 0.2296 vs
  0.2295 for v1.4 and v1.3). v1.4's edge over v1.3 is gone and the copy says so. Nothing
  re-picked. Pages checked at 320, 375 and 1440px.
- PC: v1.4.1-fenwick is live (`a6332e3`): v1.4's recipe refit on the stoppage-fixed table, 1692 trees,
  replacing v1.4 on principle. It reproduced the v1.5 baseline's late 2025-26 log loss
  (0.221075). 2026-27, reported only: log loss 0.22622 vs v1.4's 0.22618 on the same fixed
  shots (+0.00004, interval -0.00016 to +0.00024), 1,251 goals vs 1,253.5 xG. Rescored
  `shot_xg` (v1.2, v1.3, v1.4 rows kept), re-exported, no refresh of new games (data still
  through Oct 28). Model tab and methods: the whistle-leak section, the ~8% finishing section,
  and "doesn't know players yet" reworded now that it's permanent. Also corrected a stale
  period factor on the methods page (now x 1.0123). Pages checked at 320, 375 and 1440px: no
  errors, no sideways scrolling.

## 2026-11-04

- PC: end of day. The owner's answers are under "Next up" in CLAUDE.md: skill stays out
  of xG (model tab to explain the ~8% finishing story), v1.4.1 = leak-free refit replaces
  v1.4 (2026-27 reported only), then the rink-by-rink location check. None started.
- PC: tried the shooter prior (v1.5) and goalie prior (v1.6) against v1.4's recipe
  retrained on the fixed table (`0ad8a28`). Neither won on late 2025-26 log loss
  (0.22107 baseline, 0.22110, 0.22123), so no model was saved and 2026-27 wasn't touched.
  New `features/build_priors_fenwick.py` (K and league mean from earlier games only),
  `tests/test_priors_fenwick.py`, `model/train_v1_5_fenwick.py`. Written up in FINDINGS;
  next step is the owner's call.
- PC: fixed the stoppage leak in `shot_features_fenwick` (`59b5e2d`). 170 shots had a
  stoppage caused by the shot as their previous event, all at 0 s, 97 goals; v1.4 gave them
  97.8 xG. `shot_prev_any` skips stoppages now; only those 170 rows changed (checked against
  the old rule built side by side). New build check 12; check 1 leaves them out of legacy
  parity. Penalty shots aren't affected and get no flag. Written up in `docs/FINDINGS.md`,
  whose plan now names v1.5 (shooter) and v1.6 (goalie). v1.4 not retrained or rescored.
- PC: v1.4-fenwick is live (`26ae177`). Added to `model/score_fenwick.py` as the default, rescored every
  shot into `shot_xg` (v1.3 and v1.2 rows kept), re-exported the period page and profiles,
  and rewrote the model tab and methods page for v1.4. Checks before pushing: 2026-27 came
  out at 1,251 goals vs 1,252.4 xG (O/E 0.999) and tip-ins 0.899, the same as the test file;
  older seasons within 0.002 O/E of v1.3; the biggest team xGF% move was 0.7 points (VAN).
  The few shots that moved a lot are mostly empty-net shots and a quirk where a shot is
  logged in the same second as a stoppage (170 shots, 57% goals). The saved model had been
  pickled as `__main__.Calibrated`, so it was re-saved under its module (same predictions)
  and the trainer fixed. Pages checked at 320, 375 and 1440px: no errors, no sideways
  scrolling.

## 2026-11-03

- PC: the owner approved v1.4. Making it live is now first under "Next up" in CLAUDE.md;
  not done yet.
- PC: end of day. CLAUDE.md and this log checked; v1.4 trained and tested, not live,
  waiting on the owner.
- PC: trained v1.4-fenwick, not live. 50-try settings search on late 2025-26 log loss,
  xG never rises with distance (checked), calibration tried and dropped. Late 2025-26 log
  loss 0.22071 vs v1.3's 0.22100; one 2026-27 test 0.22565 vs 0.22587 (`f11bd9e`).
  Nothing rescored or exported; making it live waits on the owner's call.
- PC: the profile pages' "About the xG here" note now names v1.3-fenwick, says it was
  retrained in November 2026 for the tip-in labelling change and still doesn't know who's
  shooting or who's in net. Red badge lightened to #ff8080 (5.04:1 contrast, was 4.54). Rank
  badges `e8ecf50`. Checked at 320, 375 and 1440px.
- PC: profile badges are now real ranks ("28th out of 32", "45th out of 212 forwards"),
  colored by third (green, blue, red), worked out in `export/profiles.py` and stored in
  every season row. Only qualifiers are ranked, teams included (the owner's call, since
  10-12 teams are under the special-teams minimums so far in 2026-27). One-line
  explanation on the team, player and methods pages. New rank tests in
  `tests/test_profile_metrics.py`; a Node cross-check of all 31,439 ranks against the page's
  own formatting found 0 mismatches. Checked at 320, 375 and 1440px.
- Cloud: profile stat cards now say "Better than league average" or "Worse than league
  average" in a green or red badge, instead of a small orange or blue "better"/"worse".
  It compares with the league average by any amount; it isn't a rank.
- PC: v1.3-fenwick. Recency weighting (half-life 1 season) beat the era flag on late
  2025-26, then only the winner was tested on 2026-27 (AUC 0.7832 vs v1.2's 0.7752, O/E 1.003,
  tip-ins 88 per 100 expected vs 78). Rescored every shot, re-exported the period page and
  profiles (`1acc9eb`).
- PC: rewrote the model tab and the methods page for v1.3: month-old-test warning, tip-in
  story, v1.3 vs v1.2 table with log loss, updated weak spots. Leaderboard got a Test
  season column and a correct header; v1.3 tagged `1acc9eb`. Checked at 320, 375 and
  1440px: no errors, no sideways scrolling.
- PC: copied everything from Claude's private PC notes into `CLAUDE.md` (house rules,
  locked decisions, model notes), so cloud sessions see it too.
- PC: period page xG now joins through `shot_features_fenwick`, so the dropped 'short'
  misses stay out (`50de0a7`). A cloud session made the same fix on a side branch, which
  was merged as PR #1 (`3a02d82`). Merging both copies joined the same tables twice in
  that query, which would have crashed the next refresh. Fixed back to the `50de0a7`
  version (`c1fc476`).
- PC: tip-in check (`06ad2ae`, `results/checks/tipins.txt`). Tip-in goals / xG fell from
  1.03 (2022-23) to 0.79 (2025-26), and 30 of 32 rinks logged more close tip-ins in
  2025-26. That looks like a league-wide labelling change.
- PC: proposed v1.3 (recent seasons weighted more, train on 2022-23 through 2025-26, test
  on 2026-27). Waiting on the owner.
- Cloud: added `CLAUDE.md` and this log, so any new session can pick up without resuming.

## 2026-11-02

- Dropped the 'short' misses from the shot table, plus two more drift checks (`71a1a15`),
  with before and after output (`b86b7ef`). Model not retrained.
- Team and player profile pages (`c5ee1ae`): `export/profiles.py`,
  `export/profile_metrics.py`, `tests/test_profile_metrics.py` in CI, added to
  `export/refresh.sh`, cache headers in `netlify.toml`.

## 2026-10-30

- Simplified the site and explained every number (`9ed19d2`), casual everyday voice
  (`cca737e`), xG model tab first (`1f3872b`).
- Rewrote the README for where the project is now (`de32f19`).
- Read-only shot drift checks for the model tune-up (`9368e54`).

## 2026-10-29

- Period scoring page, v1.2-fenwick, in-season ingest (`a4a824b`).
- Site redesign: big tab navigation, color-coded results, orange and black theme
  (`0756f5d`, `0c026fa`).

## 2026-10-28

- Backfilled shifts, recovered 57 games from the HTML reports, added v1.1-fenwick
  (`ee4d088`).

## 2026-10-27

- Ingested the 2025-26 season, added the play-context actor keys migration with its checks,
  and parser updates and tests.

## Before 2026-10-27

- September to October 21: Fenwick feature builds, v1-fenwick, a calibration test in CI,
  the v2.3 rescore, the project dashboard and README work. See `git log`.
