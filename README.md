# Hockey Hacks

My personal NHL stats project. It pulls every game from the NHL's free public API, rates every shot with an expected-goals (xG) model, and shows how teams do period by period, with the luck taken into account.

**Site:** [hockeyhacks.netlify.app](https://hockeyhacks.netlify.app). Every number on it is explained on the [How it works](https://hockeyhacks.netlify.app/methods) page.

## What this is (and isn't, yet)

The long-term goal is real game projections, something like what the baseball sites do. Hockey makes that hard. MLB puts pitch-by-pitch tracking data out there for anyone, and each team plays 162 games. The NHL gives you play-by-play, with a location for every shot, but its player-tracking data (NHL EDGE) only comes out as summaries. And 82 games isn't much. For "2+ goals in the 1st period," a team needs about 356 games before its own record tells you as much as the league average does.

So I'm building it in order, and testing each piece before anything gets stacked on it:

1. **Get the data.** Done. 5,794 games (regular season and playoffs) from 2022-23 through October 28, 2026, and 499,443 unblocked shots.
2. **Figure out shot quality.** In progress. The xG model works, but it doesn't know yet who shot or who was in net.
3. **Size up teams fairly.** In progress. That's the [period scoring page](https://hockeyhacks.netlify.app/periods).
4. **Simulate games.** Not started. It waits until the model knows shooters and goalies, and has been tested on a season it never saw.

## How the model works

- **Shots:** every unblocked attempt (goals, saves and misses; analytics people call this Fenwick). Blocked shots and shootouts are out.
- **Model:** LightGBM (gradient-boosted trees) on 17 inputs: distance, angle, where on the ice, shot type, strength (5-on-5, PP and so on), empty net, score, period and time left, what happened right before (the play, which team made it, how long ago and how far away), rebounds, rebounds off the post, and rushes.
- **Split:** always by season, never random. v1.2 trains on 2022-24 and uses 2024-25 to pick how many trees to use, then refits on 2022-25 and gets tested on all of 2025-26. Games from 2026-27 are a forward check.
- **Leak check:** every run confirms the score changes within games. If final scores had leaked into the inputs, it wouldn't.

## Results

The latest versions, from [results/leaderboard.md](results/leaderboard.md):

| Version | What changed | Tested on | AUC | Biggest calibration gap | Goals ÷ predicted |
|---|---|---|---|---|---|
| v1-fenwick | First model on unblocked shots | 2024-25 | 0.7661 | 0.0130 | 1.031 |
| v1.1-fenwick | + rebounds off the post | 2025-26 | 0.7722 | 0.0084 | 0.985 |
| **v1.2-fenwick** | **+ which team made the play before the shot** | **2025-26** | **0.7788** | **0.0038** | **0.980** |

Quick guide to the columns:
- **AUC** is how often a goal gets a higher xG than a shot that didn't go in. 0.5 is a coin flip.
- **Biggest calibration gap:** sort the shots into ten groups by xG. This is the largest difference between predicted and actual scoring in any group.
- **Goals ÷ predicted:** 1.00 is dead on. v1.2 called for about 2% too many goals.

AUC only compares within the same test season. My older models that only used shots on goal (v1 through v2.3) are kept in the leaderboard so they still reproduce.

## Stuff I learned the hard way

- **The way shots get recorded seems to have changed.** The shots-on-goal models all came in 7–8% short on goals in 2024-25. Goals per shot on goal went up, but goals per unblocked shot barely moved. That's what you'd see if some saves were getting written down as misses, which [David Johnson at Hockey Analysis](https://hockeyanalysis.com/2025/03/21/are-changes-in-nhl-shot-data-tracking-hiding-that-teams-are-getting-better-at-defending/) thinks puck tracking might be doing. Switching to unblocked shots cut the miss to 3%.
- **A leak can look like a win.** One shooter-skill version (v2.2) mixed validation-season results into its inputs. Of the three shots-on-goal versions, it had the best validation score and the worst test score.
- **Silent bugs are the worst bugs:**
  - Assuming which way the rink faced put about 28% of shots more than 100 feet from the net.
  - Treating shifts as closed on both ends put impossible numbers of players on the ice.
  - A missing shot type was coded "unknown", and "unknown" only ever showed up on goals, so it gave the answer away.

The full write-up is in [docs/FINDINGS.md](docs/FINDINGS.md).

## What's in here

| Folder | What it does |
|---|---|
| `ingest/` | Pulls play-by-play, shifts and box scores from `api-web.nhle.com` and `api.nhle.com` into PostgreSQL. Only finished games get loaded. |
| `db/` | Schemas, migrations and SQL checks |
| `features/` | Builds the shot tables and the shooter and goalie skill estimates |
| `model/` | Trains each version and saves every shot's xG to `shot_xg` |
| `export/` | Builds the period scoring data in `site/data/` |
| `site/` | The Netlify site: plain HTML, no build step |
| `results/` | Metrics, logs and calibration plots for every version |
| `tests/` | Script-style tests. Run them with `python tests/test_period_math.py` and so on, not pytest. |

## Running it

You need Python 3.10+ and PostgreSQL. Copy `.env.example` to `.env` and fill in your database login. Then:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m ingest.run --init-schema
.venv/bin/python -m ingest.run --seasons all
.venv/bin/python -m ingest.run_stage_a --init-schema
.venv/bin/python -m ingest.run_stage_a --seasons all
```

Also apply `db/migrations/002_plays_context_actors.sql` with `psql`. Then build the shot table, train, score and export:

```bash
.venv/bin/python -m features.build_features_fenwick
.venv/bin/python -m model.train_v1_2_fenwick
.venv/bin/python -m model.score_fenwick
.venv/bin/python -m export.period_tables
```

During the season, `export/refresh.sh` runs the whole update for the current season. Then commit `site/data/` and push to update the site.

## What's next

- **Clean up the data first.** Count the 'short' misses (basically fanned shots and flubs the NHL has logged as misses since 2023-24) and take them out. Find out why tip-ins go in less than the model expects (78 for every 100). And check whether shot locations get recorded differently from rink to rink.
- **Tune the model properly.** Run a real settings search scored on log loss, make sure xG never goes up as a shot gets farther away, and add a calibration step.
- **Add who's shooting and who's in net.** The builders for both already exist.
- **Keep the test clean.** 2025-26 has been looked at a lot now, so 2026-27 becomes the next real test.

## Credits

Data from the NHL's public API. Not affiliated with the NHL. The period stats follow the period charts at Otto's Hockey Portal (@PsuOtto). Nothing here is betting advice. If gambling is a problem, call 1-800-GAMBLER (US).

MIT license.
