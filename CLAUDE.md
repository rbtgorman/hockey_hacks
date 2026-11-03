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

Updated 2026-11-03 (v1.3 live).

- **Live model:** v1.3-fenwick (`model/train_v1_3_fenwick.py`, log in
  `results/v1_3_fenwick_train.log`). It's trained on 2022-23 to 2025-26 with recency
  weights (half-life 1 season), and every shot in `shot_xg` is rescored. The choice was
  made on late 2025-26 (log loss: recency h=1 0.22100, h=2 0.22113, era flag 0.22129,
  plain retrain 0.22168; `results/v1_3-fenwick/selection.csv`). Only the winner was
  tested on 2026-27 to date: AUC 0.7832, O/E 1.003, tip-in O/E 0.88 (v1.2 on the same
  shots: 0.7752, 0.997, 0.78). The exports (period page, profiles) use v1.3.
- **Site copy is on v1.3** (model tab in `site/index.html`, `site/methods.html`): the
  2026-27 test with a "this is only a month" warning, the tip-in story, a v1.3 vs v1.2
  table with log loss, and "Where it still struggles" (tip-ins 88, 60+ ft 85). When
  2026-27 fills in, refresh those numbers from `results/v1_3-fenwick/` and a rerun of the
  v1.2 comparison.
- **Leaderboard** has a Test season column; runs whose test season was in progress are
  marked "to date" (`model/reporting.py`).
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

- **v1.4-fenwick is trained but NOT live** (`model/train_v1_4_fenwick.py`, logs
  `results/v1_4_fenwick_select.log` and `_final.log`, results in `results/v1_4-fenwick/`).
  v1.3's features and recency weights, plus a 50-try settings search (best: try 31, 11
  leaves, 1734 trees, `search.csv`), and xG can never go up with distance (monotone
  constraint, checked: 0 rises over 400,000 steps). The constraint costs 0.00006 log loss
  with the winning settings, 0.00029 with v1.3's (`constraint_cost.csv`). Platt and isotonic
  calibration both lost on late 2025-26 cross-fit, so there's none
  (`calibration_check.csv`). Late 2025-26 log loss 0.22071 vs v1.3's 0.22100. One test on
  2026-27: log loss 0.22565 vs 0.22587, AUC 0.7834 vs 0.7832, O/E 0.999 vs 1.003, tip-in
  O/E 0.899 vs 0.884 (`test_vs_v1_3.csv`). `shot_xg`, exports and site copy still v1.3.

## Waiting on the owner

- Make v1.4 live or not? If yes: add it to `model/score_fenwick.py` MODELS and default,
  rescore, re-export, point the exports and site copy at v1.4.

## Next up

1. The owner's call on v1.4 (above), then the rest of "What's next" in the README:
   shooters and goalies.

Loose ends: there's no scheduled GitHub Actions refresh yet (`export/refresh.sh` runs by
hand on the PC). Each refresh commits a few hundred small files under `site/data/`, and player
names come from play-by-play, so they show as "C. McDavid".
