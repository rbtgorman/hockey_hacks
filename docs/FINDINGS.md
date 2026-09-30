# Hockey Hacks: findings

State of `main` at 40aa2f6 (2026-09-21). Hockey Hacks is a personal NHL expected-goals (xG) model, meant to feed a Monte Carlo game simulator that isn't built yet. Data: 4,198 games, 2022-23 to 2024-25 including playoffs, split by season only: train 2022-23, validate 2023-24, test 2024-25. O/E is observed ÷ expected goals. [not in repo] marks query output that isn't committed yet.

## Summary

- Every model trained on shots on goal (SOG) under-predicted 2024-25. The best, v2.3, had test O/E 1.070: 7% more goals than predicted.
- Scoring didn't rise: league goals per team-game fell from 3.14 to 3.01. What changed is what counts as a shot on goal. From 2023-24, many pucks a goalie stops on their way wide are recorded as missed shots ([Johnson 2025][johnson]). Here, the missed share of unblocked attempts rose from .281 to .338 [not in repo].
- Switching the target population to unblocked attempts cut the 2024-25 level miss from 7.0% to 1.4–3.1% (test O/E 1.014 and 1.031 in two v1-fenwick runs). A save relabelled as a miss is still an unblocked attempt, so the relabel doesn't change this population.
- **That headline is provisional.** 'short' misses, a label that starts in 2023-24, are still in val and test, and removing them should raise O/E.
- Three problems in my own pipeline were fixed along the way: a label leak, a units bug that all but disabled prior shrinkage, and a validation leak from season-pooled priors. No model reaches the 0.78 AUC target.

## The data problem

League goals per team-game fell 3.14 → 3.08 → 3.01 ([Hockey-Reference][hr]). The ingest matches league totals (parentheses below), so neither more scoring nor a counting error explains the miss.

| c4 filters | 2022-23 | 2023-24 | 2024-25 |
|---|---|---|---|
| SOG per team-game (Hockey-Reference) | 31.08 (31.1) | 30.12 (30.1) | 28.08 (28.1) |
| Missed share of unblocked attempts | .281 | .307 | .338 |
| Goals per SOG (1 − league sv%) | .0958 (.096) | .0966 (.097) | .1001 (.100) |
| Goals per unblocked attempt | .0689 | [NEED: 2023-24 value] | .0663 |
| Goals per unblocked attempt, <20 ft | .1232 | .1240 | .1222 |

*Values from `db/checks/c4_shot_regime_by_season.sql` [not in repo]; parentheses from Hockey-Reference. Filters: regular season only, no shootout, no empty net. Failed bank attempts and 'short' misses are kept. The Fenwick table's filters differ, so its goal rates don't compare with these.*

[Johnson (2025)][johnson] reports the mechanism. Puck tracking lets the NHL tell that some stopped pucks would have gone wide, so they are "now being classified as missed shots, even though the goalie stopped the puck." Crease shots missing the net went from 10.5% (2021-22) to 21.1% (2024-25). Fewer saves recorded means the same goals over fewer shots on goal. Goals per SOG rises without better shooting, and a model trained on the old ratio under-predicts.

- The missed share rose in every distance band [not in repo].
- Inside 20 ft, goals per SOG rose .1674 → .1823 [not in repo] while goals per unblocked attempt held. In the Fenwick table, which drops failed bank attempts, the rate is 12.32 / 12.40 / 12.36% (`results/build_features_fenwick.log`, check 4).
- Goals per SOG rose 6–9% from 2022-23 to 2024-25 in every band under 60 ft (`features/build_features_fenwick.py` docstring).
- Overall goals per unblocked attempt fell 3.8% only because attempts moved farther out [not in repo]. 60+ ft attempts (regular season, no empty net) went 10,476 → 13,519 → 14,294 (check 4).
- Applying 2022-23 per-band conversion rates to the later seasons' shot mixes predicts goals within 1% on unblocked attempts (−1.0% / −0.3%). On SOG it misses by +2.8% / +8.2% [not in repo; no committed query].

## Results

O/E compares across the two tables, because both models predict the same season's goals. Every other metric compares only within its own table.

**Shots on goal** (`shot_features`, frozen so v1–v2.3 reproduce): 252,450 shots, including regular-season shootout attempts (about 0.6%). Goal rate is 10.25 / 10.33 / 10.83% for train / val / test (`results/leaderboard.md`).

| Model, test 2024-25 | AUC | Max calibration gap |
|---|---|---|
| v1 | 0.7705 | 0.0244 |
| v2.2 | 0.7666 | 0.0188 |
| v2.3 before priors fix | 0.7706 | 0.0178 |
| v2.3 | 0.7707 | 0.0167 |

v2.3's test O/E of 1.070 exists only as a fixed reference line printed by `model/train_v1_fenwick.py`, because `results/v2_3/metrics.json` predates the O/E field.

**Unblocked attempts** (`shot_features_fenwick`, from `features/build_features_fenwick.py`): 364,074 goals, SOG and missed shots, empty net included. Goal rate is 7.21 / 7.01 / 7.06%. Removed:
- shootout attempts;
- 'teammate-blocked' (from 2023-24; 7,738 matched) and 'failed-bank-attempt' (from 2024-25). Both exist in only part of the window, and both are also dropped from every previous-event lookup.

| v1-fenwick | AUC | O/E | Slope | Max gap |
|---|---|---|---|---|
| Val, current | 0.7713 | 1.015 | 1.028 | 0.0090 |
| Test, current | 0.7661 | 1.031 | 0.986 | 0.0130 |
| Val, pre-merge | 0.7739 | 1.019 | 1.006 | 0.0075 |
| Test, pre-merge | 0.7677 | 1.014 | 0.960 | 0.0138 |

The current run is from 2026-09-21, commit 2ee3bdd (`results/v1-fenwick/metrics.json`). Its test PR-AUC is 0.2438, log loss 0.2208, and Brier 0.0596, against 0.0656 for a constant base-rate forecast. The pre-merge run is from 2026-09-16 (`results/v1_fenwick_train.log`).

Merging wrist and snap into one type cost ranking on both splits and moved test O/E from 1.014 to 1.031. Test slope and max gap improved; val's slipped. One untested reading: before the merge, relabelled wrist shots picked up snap's higher learned conversion, which offset part of the under-prediction. 1.031 is the current figure.

## What worked

- **Changing the population instead of patching the output.** The legacy table stays frozen, so earlier results still reproduce.
- **Eight checks run on every build** (`results/build_features_fenwick.log`). They include:
  - parity with the legacy table on shared rows;
  - regular-season shootout attempts as the only missing legacy rows (548 / 498 / 425);
  - no missed shot scored as a goal;
  - check 8, which flags any shot_type, strength_state or last_event_type value that is always a goal (n ≥ 10) or never one (n ≥ 200). It returns 0 rows.

  Checks 5 and 6 raise open issues (below).
- **Closing a label leak.** The legacy table codes a missing shot type as 'unknown', and 'unknown' appears only on goals, a 100% goal rate every season (`features/build_features_fenwick.py` docstring; 17 / 10 / 20 rows [not in repo]). v1–v2.3 trained with it. The Fenwick table keeps those rows and folds 'unknown' into 'wrist-snap', the most common type. Left NULL it would still leak, because LightGBM learns a direction for missing values.
- **Merging a drifting label.** Snap's share of wrist + snap went 22.6% → 20.4% → 29.9%, with snap attempts up 44% in 2024-25. Wrist + snap held at 68.3 / 68.6 / 69.1% of attempts. The merged 'wrist-snap' converts 6.75 / 6.56 / 6.58% (docstring; build log checks 6–7).
- **Fixing prior shrinkage.** Priors are Beta-Binomial empirical Bayes: w × own rate + (1 − w) × league mean, with w = n / (n + K).
  - The bug: recency weights of 5/4/3 made weighted shots about 4.49× raw shots, while K was fit on raw counts. The posterior claimed about 4.5× the evidence that existed, so shrinkage barely fired.
  - The fix: dividing by W_MAX = 5 fixes the units and leaves the point estimate unchanged. The goalie builder has the same fix.
  - The effect: mean self-weight fell 0.77 → 0.49 at 5v5, 0.46 → 0.18 on the PP and 0.28 → 0.08 on the PK (`features/build_priors_expanding.py`, `results/prepatch/`).
- **Expanding-window priors and a leakage guard.** v2.3 resolves priors per player per game date from a trailing two-year window, so no shot is scored with information from its own date or later. Every training run confirms that score_diff varies within games (4,181 of 4,198, 99.6%). Leaked final scores would make it constant (`results/v1_fenwick_train.log`).

## What didn't, and what it taught

- **Season-pooled priors.** v2.2 pooled priors over train and val.
  - Of v1, v2.2 and v2.3, it had the best validation AUC (0.7875) and the worst test AUC (0.7666).
  - Its val-to-test drop was 0.0209, against 0.0075 for v1 and 0.0074 for v2.3 (`results/{v1,v2_2,v2_3}/metrics.json`).
  - Its priors contained validation outcomes, so the leak looked like success.
  - Earlier, v2 (pooled over all three seasons, test AUC 0.7620, `README.md` at 40aa2f6) and v2.1 (0.7615 [not in repo]) also finished below v1. No cause was recorded.
- **A constant rescale.** ×1.082 on pre-patch SOG v2.3 closed the mean miss but widened the max gap from 0.0178 to 0.0206, because the top deciles were already calibrated (`README.md` at 40aa2f6). The miss wasn't uniform, so a level fix couldn't work.
- **Priors for ranking.** The shooter prior added +0.0002 test AUC (v2.3 0.7707 vs v1 0.7705). A rough ceiling puts even perfectly known shooter and goalie talent at about 0.777 AUC [not in repo; NEED: ceiling inputs and where the calculation lives]. Priors serve calibration and simulator realism, not discrimination.
- **Gain importance.** By gain, seconds_since_last_event ranks 4th and distance_from_last_event_ft 5th (`results/v1_fenwick_train.log`). Permutation on test drops AUC by 0.0029 for the first and by zero or less for the second, against 0.1815 for distance (`results/v1-fenwick/perm_importance.csv`). Gain overstates sequence features.

## Open questions

- **'short' misses.**
  - What's reported: [HockeyStats.com][hs] says that from 2023-24 the API adds missed shots with reason 'short', "much closer to fanned attempts or flubs". They run about 0.65 per game in 2023-24 and 0.92 in 2024-25 and 2025-26, and the label itself is already drifting. HockeyStats.com excludes them from Corsi, Fenwick and xG.
  - Where they sit here: my builder keeps them, so they're in val and test but not in training. Counts: [NEED: 'short' misses by season].
  - Expected effect: removing them should raise val and test O/E, because the model scores these zero-goal attempts like real ones. Size: [NEED: O/E after removal].
- **Where the remaining miss sits** (`results/v1-fenwick/segments.csv`; deciles from the reliability tables in `results/v1_fenwick_train.log`; SE ≈ 1/√goals).
  - Deciles: test under-predicts the 5th–8th (O/E 1.08, 1.09, 1.10, 1.14) and sits at 0.97 in the top decile. Val's miss is in the 8th–9th (1.05, 1.07).
  - Both seasons (val / test): central shots are under-predicted (0–15°: 1.100 / 1.083; 15–30°: 1.038 / 1.086), all four 45°+ cells are over-predicted (0.93–0.95), and wrist-snap is under-predicted (1.034 / 1.069).
  - Test only: 20–40 ft at 1.077, tip-in at 0.901.
  - Reading: the trainer reads a miss in both seasons as the 2023-24 change and a test-only miss as something new in 2024-25 (`model/train_v1_fenwick.py`). The pattern is consistent with a persistent location-recording shift from 2023-24, reported by Knodell ([2024][k1], [2025][k2]). Training on 2022-23 teaches pre-change geometry.
- **Check 5.** Behind-the-line attempts per game run 0.767 / 1.245 / 0.798. The 2023-24 spike is unexplained, and 2023-24 becomes the training season after the roll-forward.
- **Check 6.** Tip-in conversion fell 8.77 → 8.34 → 7.76% and slap 5.16 → 4.94 → 4.69%, where roughly flat is expected.
- **Is 2022-23 clean?** Johnson treats 2021-22 and 2022-23 as transitional and 2023-24 onward as a separate dataset. Knodell (2025) also suggests 2022-23 may be transitional, so the training season may be partly post-change.

## Next steps

1. **Housekeeping and evidence.**
   - Delete the stray file `0`.
   - Add `tests/test_calibration_stats.py` to CI, and drop `tests/test_goalie_priors.py`, which is a copy of the builder rather than a test.
   - Commit the output behind every [not in repo] number.
   - Re-score SOG v2.3 so its O/E sits in `results/`.
   - Update the README.
2. **Population fixes before any retrain.**
   - Count 'short' misses by season. If they start in 2023-24, remove them from the table and its previous-event lookups, add a check, and re-score v1-fenwick.
   - Explain the check-5 spike.
   - Check tip-in + deflected O/E before any merge.
3. **Roll forward.** Ingest 2025-26 and check SOG per team-game against Hockey-Reference's final figure. Then train 2023-24, validate 2024-25 and test 2025-26, all after the recording change. 2022-23 is used only to warm the priors.
4. **Priors cleanup:** PP μ check, PK eligibility threshold, position-specific μ and K.
5. **v2.3-fenwick:** the shooter prior on unblocked attempts.
6. **Goalie priors.** Zone-only, on unblocked attempts, standardized for strength mix, with a real synthetic test. Compare them with NHL EDGE goalie numbers by location, after checking which shots EDGE counts.
7. **Location diagnostics, run alongside 3–6.**
   - EDGE team shot counts by location vs play-by-play, by season.
   - A rink-level check of recorded shot distance.
   - An outside public model's O/E by season, used only if that model wasn't trained on the seasons compared [NEED: confirm the benchmark's terms and training seasons].
8. **Simulator design.**
   - Fenwick xG plus a P(on goal | attempt) layer with trailing calibration.
   - A settle delay for post-game shot revisions, which [AP][ap] reports are now common.
   - Preseason games (type 01) excluded.

[hr]: https://www.hockey-reference.com/leagues/stats.html
[johnson]: https://hockeyanalysis.com/2025/03/21/are-changes-in-nhl-shot-data-tracking-hiding-that-teams-are-getting-better-at-defending/
[k1]: https://puckovertheglass.substack.com/p/something-strange-is-going-on-with
[k2]: https://puckovertheglass.substack.com/p/a-brief-history-of-nhl-play-by-play
[hs]: https://hockeystats.com/methodology/expected-goals
[ap]: https://www.espn.com/nhl/story/_/id/48487239/nhl-goalies-save-percentage-dips-lowest-point-three-decades