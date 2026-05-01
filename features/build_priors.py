"""Build skater and goalie priors with Empirical Bayes (Beta-Binomial) shrinkage.

Outputs two tables:
  - skater_priors:  shooting % stratified by strength state (5v5, PP, PK, all)
  - goalie_priors:  goals-against rate stratified by danger zone x strength state

Method
------
1. Aggregate per-player counts across seasons with 5/4/3 recency weighting
   (most recent season counts ~1.67x as much as the oldest, mirroring BallparkPal).
2. Fit a Beta(alpha, beta) prior to the population using method of moments.
3. Shrunken rate per player = (alpha + obs_goals) / (alpha + beta + obs_shots).
4. Persist with raw + shrunken values so downstream code can verify shrinkage.

Stratification
--------------
Skaters: '5v5', 'PP', 'PK', 'all'  (4v4 and 3v3 are folded into 5v5)
Goalies: zone in {'high', 'mid', 'low', 'all'} x strength in {'5v5', 'PP_against', 'PK_against', 'all'}

Run
---
  python -m features.build_priors --init-schema   # one time, applies schema_stage_b.sql
  python -m features.build_priors                  # populate priors

Leakage warning for downstream consumers
----------------------------------------
This script pools ALL THREE seasons (2022-23, 2023-24, 2024-25). For training the v2
xG model, the test season (2024-25) must be excluded from the priors used at training
time, OR an expanding-window version of this script must be used. See train_v2.py.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------

load_dotenv()

PG_DSN = os.environ.get("PG_DSN") or (
    "host={host} port={port} dbname={db} user={user} password={pw}".format(
        host=os.environ.get("PG_HOST", "localhost"),
        port=os.environ.get("PG_PORT", "5432"),
        db=os.environ.get("PG_DB", "hockeyviz"),
        user=os.environ.get("PG_USER", "postgres"),
        pw=os.environ.get("PG_PASSWORD", ""),
    )
)

# Map season-string to recency weight.  5/4/3 mirrors BallparkPal.
SEASON_WEIGHTS = {
    "20242025": 5.0,
    "20232024": 4.0,
    "20222023": 3.0,
}

# Minimum sample sizes for INCLUSION IN THE BETA FIT (i.e., used to estimate K).
# Players below these thresholds still get priors written to the table (they just
# get heavily shrunk toward the league mean) -- they just don't influence the
# population variance estimate.
#
# Skater min was 50 in v0.1 of this script; that produced K=42 because small-sample
# noise inflated the population variance. 200 is roughly where individual shooting
# % becomes signal-dominated. Players below that threshold are still in the table.
#
# PK has a separate, lower threshold because PK ice time is concentrated in a small
# subset of players -- only ~5 players league-wide hit 200 PK shots over three
# seasons. Using 50 for PK gives a usable population (~50-100 players) for K fitting
# at the cost of some additional variance noise.
MIN_SHOTS_FOR_SKATER_FIT = 200
MIN_SHOTS_FOR_SKATER_FIT_PK = 50
MIN_SHOTS_FOR_GOALIE_FIT_OVERALL = 200
MIN_SHOTS_FOR_GOALIE_FIT_BUCKETED = 100

SCHEMA_FILE = Path(__file__).resolve().parent.parent / "db" / "schema_stage_b.sql"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_priors")


# -------------------------------------------------------------------------
# Strength state helpers
# -------------------------------------------------------------------------

def shooter_strength(state: str | None) -> str:
    """Bucket a shooter-POV strength state into {5v5, PP, PK, unknown}.

    4v4 and 3v3 are treated as 5v5 for prior purposes.  This is a deliberate
    simplification: those situations are rare and have similar shooting rates
    to even strength on aggregate.  Stage B may revisit.
    """
    if not state:
        return "unknown"
    s = state.strip().lower().replace(" ", "")
    # Strength state is shooter-POV: '5v4' -> shooter has 5, defender has 4 -> shooter on PP
    mapping = {
        "5v5": "5v5", "4v4": "5v5", "3v3": "5v5",
        "5v4": "PP",  "5v3": "PP",  "4v3": "PP",
        "6v5": "PP",  "6v4": "PP",  # extra attacker situations
        "4v5": "PK",  "3v5": "PK",  "3v4": "PK",
        "5v6": "PK",  "4v6": "PK",
    }
    return mapping.get(s, "unknown")


def goalie_strength(state: str | None) -> str:
    """Bucket a shot's strength state into a goalie-POV label.

    Inverts the shooter perspective: a 5v4 shot (shooter on PP) means the
    goalie is on the PK -> 'PP_against'.
    """
    s = shooter_strength(state)
    return {"PP": "PP_against", "PK": "PK_against", "5v5": "5v5"}.get(s, "unknown")


def danger_zone(distance_ft: float | None) -> str:
    """Classify a shot's distance into a danger zone.

    Thresholds chosen to roughly match published "high/mid/low danger" zones
    used by Natural Stat Trick and similar.
    """
    if distance_ft is None:
        return "unknown"
    if distance_ft <= 20:
        return "high"
    if distance_ft <= 40:
        return "mid"
    return "low"


# -------------------------------------------------------------------------
# Beta prior fitting
# -------------------------------------------------------------------------

@dataclass
class BetaPriorFit:
    """Fitted Beta(alpha, beta) population prior."""
    alpha: float
    beta: float
    concentration: float    # alpha + beta = K
    mean: float             # alpha / K
    n_players_used: int
    min_sample: int


def fit_beta_prior(
    counts: pd.DataFrame,
    success_col: str,
    total_col: str,
    min_sample: int,
) -> BetaPriorFit:
    """Fit Beta(alpha, beta) to player rates by method of moments.

    Two-step fit:
      1. Estimate the population MEAN from the full eligible population
         (any player with at least `total_col` >= 1 shot). This is the
         league rate and is well-estimated by total goals / total shots.
      2. Estimate the CONCENTRATION K from only high-volume players
         (`total_col` >= min_sample) so small-sample noise doesn't
         inflate the between-player variance.

    Why this matters: in earlier versions, both mean and K were estimated
    from the same min_sample-filtered population. For PK shooting, only
    5 players had >= 200 shots over three seasons, and those 5 are by
    definition the league's PK ice-time leaders -- who are above-average
    shooters. The "mean" came out as the elite-PK-shooter mean, not the
    league PK mean. Decoupling fixes that.

    Method of moments for Beta:
        mean = alpha / (alpha + beta)
        var  = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
    Solving for K = alpha + beta given a target mean and variance:
        K = mean * (1 - mean) / var - 1
    """
    # Step 1: mean from the full population (weight by trial count to get
    # the actual league rate, not the average of player rates which is
    # biased toward low-volume players)
    full_pop = counts[counts[total_col] > 0]
    if len(full_pop) == 0:
        raise RuntimeError(f"No players with any {total_col} for Beta fit")
    total_successes = float(full_pop[success_col].sum())
    total_trials = float(full_pop[total_col].sum())
    m = total_successes / total_trials
    m = float(np.clip(m, 1e-6, 1 - 1e-6))

    # Step 2: K from the high-volume subpopulation (so small-sample noise
    # doesn't dominate the variance estimate)
    eligible = counts[counts[total_col] >= min_sample].copy()
    if len(eligible) < 5:
        raise RuntimeError(
            f"Too few players with {total_col} >= {min_sample} for K estimation "
            f"(have {len(eligible)} eligible)"
        )
    rates = (eligible[success_col] / eligible[total_col]).clip(1e-6, 1 - 1e-6)
    v = float(rates.var(ddof=1))

    # If observed variance >= max possible Beta variance, fall back to a wide prior.
    max_var = m * (1 - m)
    if v >= max_var or v <= 0:
        K = 50.0
    else:
        K = (m * (1 - m) / v) - 1.0
        K = max(K, 5.0)  # floor to avoid an absurdly diffuse prior

    alpha = m * K
    beta = (1.0 - m) * K
    return BetaPriorFit(
        alpha=alpha, beta=beta, concentration=K, mean=m,
        n_players_used=len(eligible), min_sample=min_sample,
    )


def shrunken(successes: float, trials: float, prior: BetaPriorFit) -> float:
    """Beta-Binomial posterior mean."""
    return (successes + prior.alpha) / (trials + prior.alpha + prior.beta)


# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------

SHOTS_QUERY = """
SELECT
    s.shooter_id,
    s.goalie_id,
    s.distance_ft,
    s.strength_state,
    s.is_goal,
    s.is_sog,
    s.empty_net,
    g.season
FROM shots s
JOIN games g ON g.game_id = s.game_id
WHERE s.is_sog = TRUE
  AND g.season IN %(seasons)s
"""


def load_shots(conn, seasons: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Pull all SOG-eligible shots across the configured seasons.

    If `seasons` is None, uses all keys of SEASON_WEIGHTS (production mode).
    For train-only priors, pass a tuple like ('20222023', '20232024') to
    exclude the test season.
    """
    if seasons is None:
        seasons = tuple(SEASON_WEIGHTS.keys())
    else:
        # Validate that every requested season has a weight defined
        unknown = [s for s in seasons if s not in SEASON_WEIGHTS]
        if unknown:
            raise RuntimeError(
                f"Requested season(s) not in SEASON_WEIGHTS: {unknown}. "
                f"Known seasons: {list(SEASON_WEIGHTS.keys())}"
            )
    log.info("Loading shots for seasons %s ...", seasons)
    df = pd.read_sql(SHOTS_QUERY, conn, params={"seasons": seasons})
    log.info("Loaded %d shots", len(df))

    # Apply recency weight per shot
    df["weight"] = df["season"].astype(str).map(SEASON_WEIGHTS).astype(float)
    if df["weight"].isna().any():
        bad = df[df["weight"].isna()]["season"].unique()
        raise RuntimeError(f"Unmapped season(s) found in shots: {bad}")

    # Bucket strength
    df["shooter_strength"] = df["strength_state"].map(shooter_strength)
    df["goalie_strength"] = df["strength_state"].map(goalie_strength)
    df["zone"] = df["distance_ft"].map(danger_zone)
    return df


# -------------------------------------------------------------------------
# Skater priors
# -------------------------------------------------------------------------

def aggregate_skater_counts(shots: pd.DataFrame, strength_filter: str | None) -> pd.DataFrame:
    """Aggregate weighted shots/goals per shooter, optionally filtered to a strength bucket.

    Returns columns: player_id, weighted_shots, weighted_goals, raw_shots, raw_goals.
    Drops shooter_id NULLs and 'unknown' strength rows.

    Special case: PK shooting % is dominated by empty-net "shots" (when the trailing
    team pulls their goalie, the leading team's PK shifts get credited with empty-net
    goals). Those events represent team tactics, not individual shooting talent, and
    skew the PK shooting prior to ~24% (real-PK rate is ~9%). For the PK bucket only,
    we drop empty-net shots. 5v5 and PP keep them since they're rare there and the
    rates are dominated by real attempts.

    Empty-net shots are kept in the 'all' bucket too -- they're real shots, just not
    representative of PK talent specifically.
    """
    df = shots.dropna(subset=["shooter_id"])
    df = df[df["shooter_strength"] != "unknown"]
    if strength_filter is not None:
        df = df[df["shooter_strength"] == strength_filter]
    if strength_filter == "PK":
        df = df[df["empty_net"] != True]  # noqa: E712 -- pandas bool column

    g = df.groupby("shooter_id", as_index=False).agg(
        weighted_shots=("weight", "sum"),
        weighted_goals=("is_goal", lambda s: float((s.astype(int) * df.loc[s.index, "weight"]).sum())),
        raw_shots=("is_goal", "size"),
        raw_goals=("is_goal", lambda s: int(s.sum())),
    )
    g = g.rename(columns={"shooter_id": "player_id"})
    g["player_id"] = g["player_id"].astype("int64")
    return g


def build_skater_priors(shots: pd.DataFrame) -> pd.DataFrame:
    """Build skater_priors rows for all strength buckets.

    Returns a DataFrame matching the skater_priors schema, ready to upsert.
    """
    rows: list[pd.DataFrame] = []
    # (label, strength_filter, k_fit_min_sample)
    buckets = [
        ("5v5", "5v5", MIN_SHOTS_FOR_SKATER_FIT),
        ("PP",  "PP",  MIN_SHOTS_FOR_SKATER_FIT),
        ("PK",  "PK",  MIN_SHOTS_FOR_SKATER_FIT_PK),
        ("all", None,  MIN_SHOTS_FOR_SKATER_FIT),
    ]

    for label, filt, k_min in buckets:
        agg = aggregate_skater_counts(shots, filt)
        if len(agg) == 0:
            log.warning("No shots in skater bucket %s; skipping", label)
            continue

        prior = fit_beta_prior(
            agg, success_col="weighted_goals", total_col="weighted_shots",
            min_sample=k_min,
        )
        log.info(
            "Skater prior [%s]: K=%.1f mean=%.4f (n_eligible=%d / total=%d)",
            label, prior.concentration, prior.mean, prior.n_players_used, len(agg),
        )

        agg["strength_state"] = label
        agg["prior_alpha"] = prior.alpha
        agg["prior_beta"] = prior.beta
        agg["prior_concentration"] = prior.concentration
        agg["prior_mean"] = prior.mean
        agg["raw_shooting_pct"] = np.where(
            agg["weighted_shots"] > 0,
            agg["weighted_goals"] / agg["weighted_shots"],
            np.nan,
        )
        agg["shrunken_shooting_pct"] = (
            (agg["weighted_goals"] + prior.alpha)
            / (agg["weighted_shots"] + prior.concentration)
        )
        rows.append(agg)

    return pd.concat(rows, ignore_index=True)


# -------------------------------------------------------------------------
# Goalie priors
# -------------------------------------------------------------------------

def aggregate_goalie_counts(
    shots: pd.DataFrame, zone_filter: str | None, strength_filter: str | None,
) -> pd.DataFrame:
    """Aggregate weighted shots-against / goals-against per goalie."""
    df = shots.dropna(subset=["goalie_id"])
    df = df[df["goalie_strength"] != "unknown"]
    df = df[df["zone"] != "unknown"]
    # Exclude empty-net shots from goalie priors (no goalie was actually facing them)
    df = df[df["empty_net"] != True]  # noqa: E712 — pandas bool column

    if zone_filter is not None:
        df = df[df["zone"] == zone_filter]
    if strength_filter is not None:
        df = df[df["goalie_strength"] == strength_filter]

    if len(df) == 0:
        return pd.DataFrame(columns=[
            "player_id", "weighted_shots_against", "weighted_goals_against",
            "raw_shots_against", "raw_goals_against",
        ])

    g = df.groupby("goalie_id", as_index=False).agg(
        weighted_shots_against=("weight", "sum"),
        weighted_goals_against=("is_goal", lambda s: float((s.astype(int) * df.loc[s.index, "weight"]).sum())),
        raw_shots_against=("is_goal", "size"),
        raw_goals_against=("is_goal", lambda s: int(s.sum())),
    )
    g = g.rename(columns={"goalie_id": "player_id"})
    g["player_id"] = g["player_id"].astype("int64")
    return g


def build_goalie_priors(shots: pd.DataFrame) -> pd.DataFrame:
    """Build goalie_priors rows for all (zone, strength) buckets."""
    rows: list[pd.DataFrame] = []
    zones = [("high", "high"), ("mid", "mid"), ("low", "low"), ("all", None)]
    strengths = [("5v5", "5v5"), ("PP_against", "PP_against"),
                 ("PK_against", "PK_against"), ("all", None)]

    for zone_label, zone_filt in zones:
        for str_label, str_filt in strengths:
            agg = aggregate_goalie_counts(shots, zone_filt, str_filt)
            if len(agg) == 0:
                log.warning("No shots in goalie bucket zone=%s strength=%s; skipping",
                            zone_label, str_label)
                continue

            min_n = (MIN_SHOTS_FOR_GOALIE_FIT_OVERALL
                     if (zone_label == "all" and str_label == "all")
                     else MIN_SHOTS_FOR_GOALIE_FIT_BUCKETED)
            try:
                prior = fit_beta_prior(
                    agg, success_col="weighted_goals_against",
                    total_col="weighted_shots_against",
                    min_sample=min_n,
                )
            except RuntimeError as e:
                log.warning("Skipping goalie bucket zone=%s strength=%s: %s",
                            zone_label, str_label, e)
                continue
            log.info(
                "Goalie prior [zone=%s strength=%s]: K=%.1f mean_GA=%.4f (n_eligible=%d / total=%d)",
                zone_label, str_label, prior.concentration, prior.mean,
                prior.n_players_used, len(agg),
            )

            agg["danger_zone"] = zone_label
            agg["strength_state"] = str_label
            agg["prior_alpha"] = prior.alpha
            agg["prior_beta"] = prior.beta
            agg["prior_concentration"] = prior.concentration
            agg["prior_mean"] = prior.mean
            agg["raw_ga_rate"] = np.where(
                agg["weighted_shots_against"] > 0,
                agg["weighted_goals_against"] / agg["weighted_shots_against"],
                np.nan,
            )
            agg["shrunken_ga_rate"] = (
                (agg["weighted_goals_against"] + prior.alpha)
                / (agg["weighted_shots_against"] + prior.concentration)
            )
            rows.append(agg)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# -------------------------------------------------------------------------
# Persistence
# -------------------------------------------------------------------------

def _skater_upsert_sql(table: str) -> str:
    return f"""
INSERT INTO {table} (
    player_id, strength_state,
    weighted_shots, weighted_goals, raw_shots, raw_goals,
    prior_alpha, prior_beta, prior_concentration, prior_mean,
    raw_shooting_pct, shrunken_shooting_pct, computed_at
) VALUES %s
ON CONFLICT (player_id, strength_state) DO UPDATE SET
    weighted_shots = EXCLUDED.weighted_shots,
    weighted_goals = EXCLUDED.weighted_goals,
    raw_shots = EXCLUDED.raw_shots,
    raw_goals = EXCLUDED.raw_goals,
    prior_alpha = EXCLUDED.prior_alpha,
    prior_beta = EXCLUDED.prior_beta,
    prior_concentration = EXCLUDED.prior_concentration,
    prior_mean = EXCLUDED.prior_mean,
    raw_shooting_pct = EXCLUDED.raw_shooting_pct,
    shrunken_shooting_pct = EXCLUDED.shrunken_shooting_pct,
    computed_at = EXCLUDED.computed_at;
"""


def _goalie_upsert_sql(table: str) -> str:
    return f"""
INSERT INTO {table} (
    player_id, danger_zone, strength_state,
    weighted_shots_against, weighted_goals_against,
    raw_shots_against, raw_goals_against,
    prior_alpha, prior_beta, prior_concentration, prior_mean,
    raw_ga_rate, shrunken_ga_rate, computed_at
) VALUES %s
ON CONFLICT (player_id, danger_zone, strength_state) DO UPDATE SET
    weighted_shots_against = EXCLUDED.weighted_shots_against,
    weighted_goals_against = EXCLUDED.weighted_goals_against,
    raw_shots_against = EXCLUDED.raw_shots_against,
    raw_goals_against = EXCLUDED.raw_goals_against,
    prior_alpha = EXCLUDED.prior_alpha,
    prior_beta = EXCLUDED.prior_beta,
    prior_concentration = EXCLUDED.prior_concentration,
    prior_mean = EXCLUDED.prior_mean,
    raw_ga_rate = EXCLUDED.raw_ga_rate,
    shrunken_ga_rate = EXCLUDED.shrunken_ga_rate,
    computed_at = EXCLUDED.computed_at;
"""


# Whitelist of legal table names — guards against SQL injection via --output-suffix
# even though it's a CLI flag, defense-in-depth.
ALLOWED_SKATER_TABLES = {"skater_priors", "skater_priors_train"}
ALLOWED_GOALIE_TABLES = {"goalie_priors", "goalie_priors_train"}


def write_skater_priors(conn, df: pd.DataFrame, table: str = "skater_priors") -> int:
    if table not in ALLOWED_SKATER_TABLES:
        raise ValueError(f"Unknown skater priors table: {table!r}")
    if df.empty:
        log.warning("No skater priors to write")
        return 0
    rows = [
        (
            int(r.player_id), str(r.strength_state),
            float(r.weighted_shots), float(r.weighted_goals),
            int(r.raw_shots), int(r.raw_goals),
            float(r.prior_alpha), float(r.prior_beta),
            float(r.prior_concentration), float(r.prior_mean),
            None if pd.isna(r.raw_shooting_pct) else float(r.raw_shooting_pct),
            float(r.shrunken_shooting_pct),
            pd.Timestamp.utcnow().to_pydatetime(),
        )
        for r in df.itertuples(index=False)
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, _skater_upsert_sql(table), rows, page_size=1000)
    conn.commit()
    return len(rows)


def write_goalie_priors(conn, df: pd.DataFrame, table: str = "goalie_priors") -> int:
    if table not in ALLOWED_GOALIE_TABLES:
        raise ValueError(f"Unknown goalie priors table: {table!r}")
    if df.empty:
        log.warning("No goalie priors to write")
        return 0
    rows = [
        (
            int(r.player_id), str(r.danger_zone), str(r.strength_state),
            float(r.weighted_shots_against), float(r.weighted_goals_against),
            int(r.raw_shots_against), int(r.raw_goals_against),
            float(r.prior_alpha), float(r.prior_beta),
            float(r.prior_concentration), float(r.prior_mean),
            None if pd.isna(r.raw_ga_rate) else float(r.raw_ga_rate),
            float(r.shrunken_ga_rate),
            pd.Timestamp.utcnow().to_pydatetime(),
        )
        for r in df.itertuples(index=False)
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, _goalie_upsert_sql(table), rows, page_size=1000)
    conn.commit()
    return len(rows)


# -------------------------------------------------------------------------
# Validation queries (printed at end)
# -------------------------------------------------------------------------

VALIDATION_QUERIES = [
    ("Skater priors row counts by strength", """
        SELECT strength_state, COUNT(*) AS n_players,
               ROUND(AVG(prior_mean)::numeric, 4) AS prior_mean,
               ROUND(AVG(prior_concentration)::numeric, 1) AS prior_K
        FROM skater_priors GROUP BY strength_state ORDER BY strength_state;
    """),
    ("Top 10 5v5 shooters by sample size (raw vs shrunken)", """
        SELECT sp.player_id,
               COALESCE(p.full_name, '?') AS name,
               sp.raw_shots,
               ROUND(sp.raw_shooting_pct::numeric, 3)        AS raw_pct,
               ROUND(sp.shrunken_shooting_pct::numeric, 3)   AS shrunk_pct
        FROM skater_priors sp
        LEFT JOIN players p ON p.player_id = sp.player_id
        WHERE sp.strength_state = '5v5'
        ORDER BY sp.raw_shots DESC LIMIT 10;
    """),
    ("Small-sample shrinkage demo (5v5, 1-5 shots)", """
        SELECT sp.raw_shots, sp.raw_goals,
               ROUND(sp.raw_shooting_pct::numeric, 3)      AS raw_pct,
               ROUND(sp.shrunken_shooting_pct::numeric, 3) AS shrunk_pct,
               COUNT(*) AS n_players
        FROM skater_priors sp
        WHERE sp.strength_state = '5v5' AND sp.raw_shots <= 5
        GROUP BY sp.raw_shots, sp.raw_goals,
                 sp.raw_shooting_pct, sp.shrunken_shooting_pct
        ORDER BY sp.raw_shots, sp.raw_goals;
    """),
    ("Goalie priors row counts by (zone, strength)", """
        SELECT danger_zone, strength_state, COUNT(*) AS n_goalies,
               ROUND(AVG(prior_mean)::numeric, 4) AS prior_mean_GA,
               ROUND(AVG(prior_concentration)::numeric, 1) AS prior_K
        FROM goalie_priors
        GROUP BY danger_zone, strength_state
        ORDER BY danger_zone, strength_state;
    """),
    ("Top 10 goalies (zone=all, strength=all) by sample", """
        SELECT gp.player_id,
               COALESCE(p.full_name, '?') AS name,
               gp.raw_shots_against,
               ROUND(gp.raw_ga_rate::numeric, 3)      AS raw_GA_rate,
               ROUND(gp.shrunken_ga_rate::numeric, 3) AS shrunk_GA_rate,
               ROUND((1 - gp.shrunken_ga_rate)::numeric, 3) AS shrunk_sv_pct
        FROM goalie_priors gp
        LEFT JOIN players p ON p.player_id = gp.player_id
        WHERE gp.danger_zone = 'all' AND gp.strength_state = 'all'
        ORDER BY gp.raw_shots_against DESC LIMIT 10;
    """),
]


def print_validation(conn) -> None:
    print()
    print("=" * 78)
    print("VALIDATION")
    print("=" * 78)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        for label, sql in VALIDATION_QUERIES:
            print(f"\n--- {label}")
            try:
                cur.execute(sql)
                rows = cur.fetchall()
            except psycopg2.Error as e:
                conn.rollback()
                print(f"  (query failed: {e})")
                continue
            if not rows:
                print("  (no rows)")
                continue
            df = pd.DataFrame(rows)
            print(df.to_string(index=False))
    print()


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

def init_schema(conn, train: bool = False) -> None:
    """Apply schema_stage_b.sql, plus schema_stage_c.sql when --train."""
    schema_files = [SCHEMA_FILE]
    if train:
        train_schema = SCHEMA_FILE.parent / "schema_stage_c.sql"
        if not train_schema.exists():
            raise FileNotFoundError(f"Schema file missing: {train_schema}")
        schema_files.append(train_schema)

    for f in schema_files:
        if not f.exists():
            raise FileNotFoundError(f"Schema file missing: {f}")
        log.info("Applying %s", f)
        with conn.cursor() as cur:
            cur.execute(f.read_text())
        conn.commit()
    log.info("Schema applied")


# Default season splits. Test season is excluded when --train.
DEFAULT_TRAIN_SEASONS = ("20222023", "20232024")
DEFAULT_ALL_SEASONS = ("20222023", "20232024", "20242025")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--init-schema", action="store_true",
                        help="Apply db/schema_stage_b.sql (and schema_stage_c.sql if --train) then exit")
    parser.add_argument("--train", action="store_true",
                        help=("Build leakage-safe priors from train+val seasons only "
                              "(default: 20222023, 20232024) and write to "
                              "skater_priors_train / goalie_priors_train tables. "
                              "Used as features for v2 xG training."))
    parser.add_argument("--seasons", type=str, default=None,
                        help=("Comma-separated season list to override defaults, "
                              "e.g. '20222023,20232024'. Each must have a weight "
                              "defined in SEASON_WEIGHTS."))
    parser.add_argument("--skip-skater", action="store_true")
    parser.add_argument("--skip-goalie", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Resolve seasons + target tables
    if args.seasons:
        seasons = tuple(s.strip() for s in args.seasons.split(",") if s.strip())
    elif args.train:
        seasons = DEFAULT_TRAIN_SEASONS
    else:
        seasons = DEFAULT_ALL_SEASONS

    skater_table = "skater_priors_train" if args.train else "skater_priors"
    goalie_table = "goalie_priors_train" if args.train else "goalie_priors"

    log.info("Mode: %s | seasons=%s | skater_table=%s | goalie_table=%s",
             "TRAIN-ONLY" if args.train else "PRODUCTION",
             seasons, skater_table, goalie_table)

    conn = psycopg2.connect(PG_DSN)
    try:
        if args.init_schema:
            init_schema(conn, train=args.train)
            return 0

        shots = load_shots(conn, seasons=seasons)

        if not args.skip_skater:
            log.info("Building skater priors ...")
            skater_df = build_skater_priors(shots)
            n = write_skater_priors(conn, skater_df, table=skater_table)
            log.info("Wrote %d %s rows", n, skater_table)

        if not args.skip_goalie:
            log.info("Building goalie priors ...")
            goalie_df = build_goalie_priors(shots)
            n = write_goalie_priors(conn, goalie_df, table=goalie_table)
            log.info("Wrote %d %s rows", n, goalie_table)

        # Validation queries hard-code production table names. For --train runs,
        # we skip them since they'd report on the wrong tables. The training
        # tables share schema; the production-mode validation already proved
        # the math is sound.
        if not args.train:
            print_validation(conn)
        else:
            log.info("Skipping validation queries in --train mode (they target production tables)")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())