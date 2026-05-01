"""
Expanding-window skater priors.

For each shot a player took, compute their shooter prior using only data from
the trailing 2 years before that game's date.

Algorithm:
1. Load all shots once. Bucket strength state. Filter empty-net for PK only.
2. Apply season-relative recency weights (5 = same season as target game,
   4 = one season prior, 3 = two seasons prior).
3. Fit K (Beta-Binomial concentration) once per strength bucket on the full
   dataset's per-player aggregates — K is structural, not date-dependent.
4. Compute league mean per bucket on the same eligibility-filtered population
   used to fit K (decoupled from per-player computation, same as Stage B).
5. For each (player, strength_bucket), sort that player's shots by date.
6. For each (player, game_date) where the player shot, use binary search to
   find the [game_date - 2 years, game_date) window in the player's history,
   sum weighted shots/goals in that slice, apply shrinkage:
       shrunken = (alpha + weighted_goals) / (alpha + beta + weighted_shots)
   where alpha = prior_mean * K, beta = (1 - prior_mean) * K.

Usage:
    python -m features.build_priors_expanding --init-schema
    python -m features.build_priors_expanding

Notes:
- Window is half-open: [start, end). game_date itself is NOT included
  in its own prior — that would be leakage.
- Players appearing for the first time get prior_mean as their shrunken_pct
  (zero shots in window → posterior = prior).
- Empty-net shots are excluded from the PK bucket only (matches Stage B).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

from ingest.db import pg_conn

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "db" / "schema_stage_c2.sql"

# Strength bucketing matches Stage B
BUCKET_MAP = {
    "5v5": "5v5", "4v4": "5v5", "3v3": "5v5",
    "5v4": "PP",  "5v3": "PP",  "4v3": "PP",
    "4v5": "PK",  "3v5": "PK",  "3v4": "PK",
}
WINDOW_YEARS = 2

# K-fit eligibility threshold (matches Stage B)
K_FIT_MIN_SHOTS = 200


def init_schema():
    sql = SCHEMA_PATH.read_text()
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()
    print(f"Initialized schema from {SCHEMA_PATH}")


def load_shots() -> pd.DataFrame:
    """Load all shots with player_id, date, bucket, weight inputs."""
    query = """
        SELECT
            s.shooter_id   AS player_id,
            g.game_date    AS game_date,
            g.season       AS season,
            s.strength_state AS raw_strength,
            s.is_goal::int AS is_goal,
            s.empty_net::int AS empty_net
        FROM shots s
        JOIN games g ON g.game_id = s.game_id
        WHERE s.event_type IN ('shot-on-goal', 'goal', 'missed-shot')
          AND s.shooter_id IS NOT NULL
          AND s.strength_state IS NOT NULL
    """
    with pg_conn() as conn:
        df = pd.read_sql(query, conn)

    df["bucket"] = df["raw_strength"].map(BUCKET_MAP)
    df = df[df["bucket"].notna()].copy()

    # PK-only empty-net filter (Stage B parity)
    pk_en_mask = (df["bucket"] == "PK") & (df["empty_net"] == 1)
    n_filtered = pk_en_mask.sum()
    df = df[~pk_en_mask].copy()
    print(f"Filtered {n_filtered:,} empty-net shots from PK bucket")

    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df = df.sort_values(["player_id", "bucket", "game_date"]).reset_index(drop=True)

    print(f"Loaded {len(df):,} shots, "
          f"{df['player_id'].nunique():,} players, "
          f"{df['game_date'].min()} to {df['game_date'].max()}")
    return df


def attach_recency_weight(target_season: int, shot_seasons: np.ndarray) -> np.ndarray:
    """5 if same season, 4 if previous, 3 if two prior, 0 otherwise.

    Seasons stored as integers like 20222023. We compute season distance by
    season-start-year difference.
    """
    target_start = target_season // 10000
    shot_starts = shot_seasons // 10000
    diff = target_start - shot_starts
    weights = np.where(diff == 0, 5.0,
              np.where(diff == 1, 4.0,
              np.where(diff == 2, 3.0, 0.0)))
    return weights


def fit_k_methodofmoments(player_aggs: pd.DataFrame, mean: float) -> float:
    """Beta-Binomial method-of-moments K from per-player (n_shots, n_goals).

    Same approach as Stage B build_priors.py.
    """
    eligible = player_aggs[player_aggs["raw_shots"] >= K_FIT_MIN_SHOTS].copy()
    if len(eligible) < 5:
        print(f"  Warning: only {len(eligible)} eligible players for K fit; using K=100 default")
        return 100.0

    eligible["pct"] = eligible["raw_goals"] / eligible["raw_shots"]
    sample_var = eligible["pct"].var(ddof=1)
    expected_binom_var = (mean * (1 - mean) / eligible["raw_shots"]).mean()
    excess_var = max(sample_var - expected_binom_var, 1e-6)

    # Beta variance: mean*(1-mean)/(K+1) → solve for K
    K = mean * (1 - mean) / excess_var - 1
    K = max(K, 1.0)
    return float(K)


def build_bucket(df_bucket: pd.DataFrame, bucket: str) -> list:
    """Build expanding-window priors for one strength bucket."""
    print(f"\n=== Building bucket: {bucket} ===")
    print(f"  {len(df_bucket):,} shots, {df_bucket['player_id'].nunique():,} players")

    # Per-player aggregates for K and league-mean fitting
    player_aggs = df_bucket.groupby("player_id").agg(
        raw_shots=("is_goal", "size"),
        raw_goals=("is_goal", "sum"),
    ).reset_index()
    eligible = player_aggs[player_aggs["raw_shots"] >= K_FIT_MIN_SHOTS]
    league_mean = (eligible["raw_goals"].sum() / eligible["raw_shots"].sum()
                   if len(eligible) > 0
                   else df_bucket["is_goal"].mean())
    K = fit_k_methodofmoments(player_aggs, league_mean)
    alpha_prior = league_mean * K
    beta_prior = (1 - league_mean) * K
    print(f"  league_mean={league_mean:.4f}  K={K:.1f}  "
          f"(eligible players: {len(eligible)}, threshold {K_FIT_MIN_SHOTS} shots)")

    # Build per-player history arrays once
    rows = []
    grouped = df_bucket.groupby("player_id", sort=False)
    n_players = grouped.ngroups
    for i, (player_id, ph) in enumerate(grouped):
        if i % 200 == 0:
            print(f"    player {i:,}/{n_players:,}", end="\r")

        # Sorted by date already (we sorted globally)
        dates = pd.to_datetime(ph["game_date"]).values  # datetime64[ns]
        seasons = ph["season"].values
        is_goal = ph["is_goal"].values

        # Distinct game-dates this player shot on
        unique_dates = np.unique(dates)

        for gd in unique_dates:
            window_end = gd  # exclusive
            window_start = gd - np.timedelta64(WINDOW_YEARS * 365, "D")

            mask = (dates >= window_start) & (dates < window_end)
            if not mask.any():
                # No prior history: posterior = prior
                rows.append((
                    int(player_id), pd.Timestamp(gd).date(), bucket,
                    0.0, 0.0, 0, 0,
                    alpha_prior, beta_prior, K, league_mean,
                    None, league_mean,
                    pd.Timestamp(window_start).date(),
                    pd.Timestamp(window_end).date(),
                ))
                continue

            window_seasons = seasons[mask]
            window_goals = is_goal[mask]

            # Determine target season (the season of game-date gd)
            target_idx = np.searchsorted(dates, gd, side="left")
            if target_idx < len(seasons):
                target_season = int(seasons[target_idx])
            else:
                target_season = int(seasons[-1])

            weights = attach_recency_weight(target_season, window_seasons)
            weighted_shots = float(weights.sum())
            weighted_goals = float((weights * window_goals).sum())
            raw_shots = int(mask.sum())
            raw_goals = int(window_goals.sum())

            shrunken = (alpha_prior + weighted_goals) / (alpha_prior + beta_prior + weighted_shots)
            raw_pct = (raw_goals / raw_shots) if raw_shots > 0 else None

            rows.append((
                int(player_id), pd.Timestamp(gd).date(), bucket,
                weighted_shots, weighted_goals, raw_shots, raw_goals,
                alpha_prior, beta_prior, K, league_mean,
                raw_pct, float(shrunken),
                pd.Timestamp(window_start).date(),
                pd.Timestamp(window_end).date(),
            ))

    print(f"    player {n_players:,}/{n_players:,}  done.")
    print(f"  Generated {len(rows):,} prior rows for bucket {bucket}")
    return rows


def write_rows(rows: list):
    """Bulk insert with ON CONFLICT replace.

    Coerces numpy scalar types (np.int64, np.float64) to Python natives
    so psycopg2's literal adaptation doesn't emit them as 'np.float64(...)'
    which Postgres parses as schema-qualified function calls and rejects
    with InvalidSchemaName: schema "np" does not exist.
    """
    cols = [
        "player_id", "game_date", "strength_state",
        "weighted_shots", "weighted_goals", "raw_shots", "raw_goals",
        "prior_alpha", "prior_beta", "prior_concentration", "prior_mean",
        "raw_shooting_pct", "shrunken_shooting_pct",
        "window_start_date", "window_end_date",
    ]

    def coerce(v):
        if v is None:
            return None
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        return v

    clean_rows = [tuple(coerce(v) for v in r) for r in rows]

    placeholders = ",".join(["%s"] * len(cols))
    update_cols = [c for c in cols if c not in ("player_id", "game_date", "strength_state")]
    update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
    sql = f"""
        INSERT INTO skater_priors_expanding ({",".join(cols)})
        VALUES %s
        ON CONFLICT (player_id, game_date, strength_state)
        DO UPDATE SET {update_clause}, computed_at = now()
    """
    with pg_conn() as conn, conn.cursor() as cur:
        execute_values(cur, sql, clean_rows, template=f"({placeholders})", page_size=5000)
        conn.commit()
    print(f"Wrote {len(clean_rows):,} rows to skater_priors_expanding")


def sanity_checks():
    """Print key sanity stats matching Stage B validation expectations."""
    with pg_conn() as conn:
        df = pd.read_sql("""
            SELECT strength_state,
                   COUNT(*) AS n_rows,
                   AVG(shrunken_shooting_pct) AS mean_shrunken,
                   AVG(prior_concentration) AS K
            FROM skater_priors_expanding
            GROUP BY strength_state
            ORDER BY strength_state
        """, conn)

        print("\n--- Sanity: per-bucket means ---")
        print(df.to_string(index=False))

        # Ovechkin (8471214) and MacKinnon (8477492) — high-volume shooters
        # should retain rates close to raw
        elite = pd.read_sql("""
            SELECT player_id, strength_state,
                   COUNT(*) AS n_priors,
                   AVG(raw_shooting_pct) AS mean_raw,
                   AVG(shrunken_shooting_pct) AS mean_shrunken
            FROM skater_priors_expanding
            WHERE player_id IN (8471214, 8477492)
              AND strength_state IN ('5v5', 'PP')
            GROUP BY player_id, strength_state
            ORDER BY player_id, strength_state
        """, conn)
        print("\n--- Sanity: elite shooters (Ovechkin 8471214, MacKinnon 8477492) ---")
        if len(elite) > 0:
            print(elite.to_string(index=False))
        else:
            print("  (Neither player_id present — adjust IDs if needed)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init-schema", action="store_true",
                   help="Run schema_stage_c2.sql before building")
    args = p.parse_args()

    if args.init_schema:
        init_schema()

    df = load_shots()

    all_rows = []
    for bucket in ["5v5", "PP", "PK"]:
        df_b = df[df["bucket"] == bucket]
        if len(df_b) == 0:
            print(f"Skipping {bucket}: no shots")
            continue
        rows = build_bucket(df_b, bucket)
        all_rows.extend(rows)

    print(f"\nTotal rows to write: {len(all_rows):,}")
    write_rows(all_rows)
    sanity_checks()


if __name__ == "__main__":
    main()