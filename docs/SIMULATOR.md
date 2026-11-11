# Game simulator plan

Status: plan only, waiting on the owner's OK. Nothing built yet.

The idea: rate every team from its xG, play each game out many times with those
ratings, and count how often each team wins. Version 0 (v0) is team-level only. A goalie
layer and then a shooter layer come after it, and each stays only if it makes the game
predictions better.

## What goes into v0

Everything comes from games played **before** the game's date, regular season only.

**Team ratings**, each one a rate per 60 minutes of that kind of play:

| Rating | From |
|---|---|
| 5-on-5 attack | xG for at 5-on-5 |
| 5-on-5 defense | xG against at 5-on-5 |
| Power play | xG for on the power play |
| Penalty kill | xG against shorthanded |
| Draws penalties | penalties drawn per 60 |
| Takes penalties | penalties taken per 60 |

- **xG** is v1.4.1 from `shot_xg`, joined through `shot_features_fenwick`, used as it is.
  Empty-net shots are left out (v0 has no empty nets).
- **Time at each strength** comes from the shift charts (skaters on the ice, second by
  second). That gives the per-60 denominators for 5-on-5, the power play and the penalty
  kill.
- **Penalties** come from `plays_context` (`penalty` events): minors and majors count;
  misconducts don't, since they don't change the strength.
- **Recent games count more:** each game's weight halves every H games back, and older
  seasons carry over the same way. H is picked (below).
- **Pulled toward the league average:** each rating gets K minutes of league-average play
  added before it's divided out, so a team with little history sits near average. One K
  for 5-on-5 rates and one for special teams and penalties, both picked.
- **Matchup:** a team's expected xG rate = league rate x (its attack / league) x (the
  opponent's defense / league), the same for power play against penalty kill.
- **Home ice:** one multiplier on the home team's xG rates and one on its penalty draws,
  fit on games before the date (not picked by hand).
- **4-on-4 and 3-on-3** use the league rate for that strength, scaled by each team's
  5-on-5 attack and defense. Too little time to rate them on their own.

**Simulating a game** (10,000 times per game):

1. Three 20-minute periods. Both teams draw penalties at their matchup rates. A minor
   lasts 2 minutes and ends early on a power-play goal; a major lasts 5. Overlapping
   penalties give 5-on-3 and 4-on-4, at league rates scaled as above.
2. Goals arrive at the xG rate for whatever strength is on the ice (Poisson: xG becomes
   goals one for one, with no finishing skill).
3. Tied after 60: 5 minutes of 3-on-3, first goal wins. Still tied: a shootout, which the
   home team wins at the league's shootout home-win rate from earlier seasons.

**Outputs per game:** chance of a home win and an away win, each split into regulation,
overtime and shootout; goal totals (each team and combined, as a spread); goals by period.

## What v0 leaves out

On purpose, to keep v0 simple. Each one is a possible later step, not a promise:

- Score effects (trailing teams shoot more, leading teams sit back)
- Pulled goalies and empty-net goals (so goal totals will run a little low; checks compare
  totals without empty-net goals too)
- Which goalie starts, and shooter skill (the next two layers)
- Rest days, back-to-backs and travel
- Injuries, trades and lineup changes, beyond what shows up in recent games
- Who's on the ice within a game (lines, matchups, deployment)
- Playoffs (overtime there is full periods of 5-on-5 and there's no shootout)
- Team shootout skill (every shootout uses the league home rate)
- Anything that isn't xG and penalties: faceoffs, hits, and so on

## How it's tested

**Score:** log loss of the home team's win chance against what happened, over all
regular-season games. Lower is better. Every number gets a range from resampling games.

**Calibration:** sort games into ten groups by the home team's win chance and compare
predicted and actual wins in each, with resample ranges. Also checked: how often games go
to overtime and a shootout, and goal totals against real totals.

**Baselines,** held to the same before-the-date rule:

- **Home-win rate:** every game gets the league's home-win rate from earlier seasons.
- **Plain Elo:** win/loss only, with a home bonus and a pull back toward average between
  seasons. K, the home bonus and the carry-over are picked the same way and on the same
  seasons as the simulator's settings, so it's a fair fight.

**Build and pick on 2023-24 to 2025-26** (3,936 games). 2022-23 is only history, so
the first games of 2023-24 have something to go on. A small grid over H and the two Ks,
picked on combined log loss. One caveat: v1.4.1 learned from 2022-23 to 2025-26 shots,
so xG is in-sample in the build seasons and they'll flatter it a bit. That's why the test
has to be 2026-27.

**The one real test: 2026-27, once, at 800 regular-season games** (about 61% of the
1,312; it should get there around late January or early February). Why 800:

- In a rough check, the per-game log loss gap between a model and the home-win rate had a
  spread (standard deviation) of about 0.16 to 0.19. At 800 games that makes the range on
  the gap about ±0.012. Elo's edge over the home-win rate was about 0.02 in three of the
  four full seasons since 2022-23, so a real edge that size shows up clearly at 800 and not at 300.
- Ten calibration groups get about 80 games each.
- Waiting for the full season would push the decision past April for little extra
  precision (the range only shrinks to about ±0.010).

That rough check was game results only (a quick Elo against the home-win rate, 2022-23
to 2026-27 to date); nothing about the simulator has touched 2026-27. One honest warning
from it: in 2025-26 Elo barely beat the home-win rate (0.0019). Some seasons are hard to
call, and a narrow result is a real result.

**Passes if,** on those 800 games:

1. Its log loss beats the home-win rate, with the range on the gap entirely above zero;
2. Its log loss beats Elo (range reported either way);
3. No calibration group is off by more than its resample range.

If it fails, it stays off the site and the write-up says so. No re-picking on 2026-27.

## Steps, in order

1. **Build v0** in a new `sim/` folder: ratings, the game engine, the baselines and the
   scoring. Tests: ratings use only earlier games (leak test), and the engine follows the
   rules (overtime is 3-on-3 and sudden death, shootout only after a scoreless overtime, a
   power-play goal ends a minor).
2. **Pick v0's settings and Elo's** on 2023-24 to 2025-26. Report to the owner, with the
   settings frozen in `results/sim_v0/selection.json`.
3. **Test v0 on 2026-27** once it reaches 800 games. Report only.
4. **Goalie layer:** each starting goalie's goals saved above expected, from earlier games
   and pulled toward average, adjusts the xG against. The EDGE goalie save percentages are
   a sanity check on the goalie ratings, not an input. Kept only if game log loss improves
   on the build seasons.
5. **Shooter layer:** the same idea for the skaters in the lineup (goals minus xG, pulled
   hard toward average). Kept only if it improves on top of whatever step 4 kept.
6. **Test the layers** once, together, on the 2026-27 games after the first 800
   (proposal: the owner may prefer to wait for 2027-28).
7. **Site:** only after a passed test. Win chances, goal totals and per-period numbers for
   the period page, in plain words, with no betting language anywhere.
