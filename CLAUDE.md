# Hockey Hacks: notes for Claude

Read this first in every session. It replaces "resuming" yesterday's conversation:
everything needed to pick up is here, in the README, and in git history.

## Where the work happens

- The owner's PC: `~/my-projects/hockey-hacks`, with the project venv at `.venv/` and the
  local Postgres database (`ingest/db.py`, settings in `.env`). Anything that touches data
  (ingest, features, training, scoring, exports, drift checks) runs there.
- Cloud sessions (web or phone app) have the code but no database. They can edit code and
  run the pure-Python tests, but can't run or check anything against real data.

## House rules

- Commit straight to `main` and push. No other branches.
- No attribution lines (Co-Authored-By and the like) in commits.
- Site copy is in a plain, casual, everyday voice, and every number gets explained.
- Browser checks on the PC: Playwright isn't installed there. Serve `site/` with
  `python3 -m http.server` and drive Google Chrome over its DevTools protocol (Node has
  WebSocket built in). Check 320, 375 and 1440px for console errors and sideways scrolling.
- Netlify deploys from `main`. `export/refresh.sh` is the in-season data refresh.

## Where things stand

Updated 2026-11-03.

- 'Short' misses are dropped from `shot_features_fenwick` (`71a1a15`). The model has NOT
  been retrained without them yet; `shot_xg` still holds about 3,740 rows for them.
- Always join `shot_xg` through `shot_features_fenwick`. `export/profiles.py` does, and
  `export/period_tables.py` does as of 2026-11-03 (before that it would have counted the
  dropped misses on the next refresh).
- Team and player profile pages are live (`c5ee1ae`, 2026-11-02): the "Teams & players"
  tab, `team.html`, `player.html`, built by `export/profiles.py` from the stat list in
  `export/profile_metrics.py`.

## Next up

1. Tip-in check: find out why tip-ins go in less often than the model expects
   (78 for every 100). Needs the PC.
2. Retrain the model on the cleaned shot table, rescore every shot (`model.score_fenwick`),
   then rerun the exports. The period page and profiles update from that.
3. After that, the rest of "What's next" in the README.

Known loose ends: each refresh now commits a few hundred small files under `site/data/`;
player names come from play-by-play, so they show as "C. McDavid".

## Ending a session

When the owner says they're done for the day, update "Where things stand" and "Next up"
above (with the date), commit, and push to `main` before stopping. That's what lets the
next session start cold with plain `claude`.
