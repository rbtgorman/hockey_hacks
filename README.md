# Hockey Hacks  
Data: 252,450 shots across three seasons (2022-23 through 2024-25).

## Pipeline

**Stage A — Ingest**  
Play-by-play, shift charts, and boxscores from NHL APIs (`api-web.nhle.com` and `api.nhle.com`). Cached as raw JSON and parsed into PostgreSQL. Parsers use pure standard library with synthetic-payload tests, keeping CI dependency-free. 
*   **Coordinate normalization:** Assuming conventional rink orientation placed ~28% of shots over 100 ft from the goal. Deriving normalization empirically from the data fixed this.
*   **Shift intervals:** Treating shifts as closed on both ends created impossible on-ice player counts. Switching to half-open `[start, end)` intervals dropped these errors from 5,008 to 14.

**Stage B — Player Priors**  
Fits Beta-Binomial empirical Bayes priors for shooter finishing to handle shrinkage, stratified by strength state (5v5 / PP / PK). `build_priors_expanding.py` resolves a prior per `(player, game_date)` using a trailing window. No shot is ever scored with information from its own date or later.

**Stage C — Model**  
LightGBM classifier. Strict temporal split: train 2022-23, validate 2023-24, test 2024-25. No random splitting.

## Results 

Held-out 2024-25 season. Full artifacts in `results/`.

| Version | Modification | Test AUC | Max Calibration Gap |
| :--- | :--- | :--- | :--- |
| **v1** | Geometric baseline. No player priors. | 0.7705 | 0.0244 |
| **v2.2** | Static per-player prior (pooled train+val). | 0.7666 | 0.0188 |
| **v2.3** | Expanding-window prior (resolved per game-date). | 0.7706 | 0.0178 |

v2.3 is the production model. It matches the baseline AUC but achieves the tightest calibration, which is the hard requirement for the downstream Monte Carlo simulator.

## Key Findings

*   **Pooled priors leak:** v2.2 pooled priors across train and val. It posted the highest val AUC (0.7875) but the lowest test AUC (0.7666)—a val-to-test drop of 0.0209. The model was reading an answer key. Removing this leak in v2.3 cut the `shooter_prior_pct` gain importance in half (from 24,388 to 12,397).
*   **Environment drift vs. model error:** All versions under-predict the test season by ~8%. The training-season goal rate was 10.25%; the test season was 10.83%. Three differently-built models missing by the exact same margin indicates environment drift. 
*   **Flat rescaling fails:** Rescaling predictions by 1.082 to close the mean miss widens the max gap (0.0178 → 0.0206) because the top deciles are already calibrated. Fixing this requires a monotone calibration map, not a constant multiplier.

## Roadmap
*   Goalie priors using the expanding-window design.
*   Rolling in-season recalibration layer.
*   Monte Carlo game simulator.
