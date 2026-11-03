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
- Browser checks on the PC: Playwright isn't installed there. Serve `site/` with
  `python3 -m http.server` and drive Google Chrome over its DevTools protocol. Check 320,
  375 and 1440px for console errors and sideways scrolling.
- Netlify deploys from `main`. `export/refresh.sh` is the in-season data refresh.

## Where things stand

Updated 2026-11-03.

- **Live model:** v1.2-fenwick (see the README and `results/leaderboard.md`).
- **'Short' misses** were dropped from `shot_features_fenwick` (`71a1a15`), but the model
  hasn't been retrained without them, and `shot_xg` still holds about 3,740 rows for them.
  Always join `shot_xg` through `shot_features_fenwick`. `export/profiles.py` does, and
  `export/period_tables.py` does as of `50de0a7`.
- **Tip-in check done** (`06ad2ae`, `results/checks/tipins.txt`). Tip-ins used to score
  about as the model expects (goals / xG 1.03 in 2022-23). They fell to 0.79 in 2025-26 and
  0.78 so far in 2026-27. It's league-wide, not a few rinks: 30 of 32 rinks logged more
  tip-ins within 10 ft per game in 2025-26. That points to the NHL labelling more shots as
  tip-ins starting in 2025-26.
- **Team and player profile pages** are live (`c5ee1ae`): the "Teams & players" tab,
  `team.html` and `player.html`, built by `export/profiles.py` from the stat list in
  `export/profile_metrics.py`.

## Waiting on the owner

- **v1.3, yes or no?** On 2026-11-03 the PC session proposed retraining with recent seasons
  weighted more heavily, training on 2022-23 through 2025-26 and keeping 2026-27 as the test.

## Next up

1. Once the owner decides: train v1.3, rescore every shot (`model.score_fenwick`), then rerun
   the export steps from `export/refresh.sh`. The period page and profiles update from that.
2. After that, the rest of "What's next" in the README.

Loose ends: each refresh commits a few hundred small files under `site/data/`, and player
names come from play-by-play, so they show as "C. McDavid".
