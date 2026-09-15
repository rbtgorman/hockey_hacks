"""
Expanding-window goalie priors.

The goalie counterpart to build_priors_expanding.py. For each date a goalie
played, compute their goals-against rate using only shots from the trailing
2 years before that date.

This is a PORT, not a new design. The zone/strength taxonomy, the goalie-POV
strength mapping, the empty-net exclusion and the Beta-Binomial shrinkage all
come from build_priors.py (Stage B). The only thing that changes is that the
prior is resolved per game-date instead of pooled across seasons, which is the
same fix that took v2.2 to v2.3.

THREE DECISIONS WORTH KNOWING ABOUT
-----------------------------------
1. Denominator is SHOTS ON GOAL, not unblocked attempts.
   build_priors_expanding.py (skaters) uses Fenwick -- it includes missed
   shots, because a shooter missing the net is information about the shooter.
   A shooter missing the net is NOT information about the goalie: the puck
   never reached them. So the goalie query drops 'missed-shot'.
   This also settles the "GA rate vs save percentage" question -- on a
   shots-on-goal denominator they are the same number, since GA rate = 1 - sv%.

2. Three danger zones, not six or nine.
   high <= 20ft, mid 20-40ft, low > 40ft, matching build_priors.py. It also
   matches the High-Danger / Mid-Range / Long-Range taxonomy the NHL publishes
   through the EDGE endpoints, which means these estimates can be checked
   against a league-published number. Nine zones would give you finer buckets,
   nothing to validate against, and thinner cells in a population that is
   already thin.

3. NO fallback for thin cells. Shrinkage already handles them, correctly.
   The first draft of this module borrowed a goalie's zone-wide record when a
   (zone, strength) cell had fewer than MIN_CELL_SHOTS shots in window. That
   was wrong twice over, and tests/test_goalie_priors.py caught it:

     - It discarded real evidence. A goalie with one shot and one goal in
       window has genuine, if weak, information. Beta-Binomial shrinkage turns
       that into an estimate a hair above the league mean, which is the correct
       Bayesian answer. The fallback threw the observation away and returned
       the league mean exactly -- strictly worse than doing nothing.
     - It substituted a different quantity. Replacing a PP_against estimate
       with a mostly-5v5 zone rate biases it downward, because power-play
       shots are harder. That is not partial pooling, it is contamination.

   Shrinkage already scales with evidence continuously: three shots move the
   estimate barely, three hundred move it a lot. A hard threshold adds a
   discontinuity and buys nothing.

   evidence_level is kept as a DESCRIPTIVE column ('sufficient' / 'sparse' /
   'none'). It records how much history backed each row without altering the
   estimate, so you can check afterwards whether the 9-cell grid is too fine
   for a ~90-goalie population.

   (A proper hierarchical model -- using the zone-level posterior as the prior
   for each cell rather than swapping the data -- is defensible and would be a
   real improvement. It is a two-stage fit and belongs in a later version, not
   bolted on here.)

Usage:
    python -m features.build_goalie_priors_expanding --init-schema
    python -m features.build_goalie_priors_expanding

Notes:
- Window is half-open: [start, end). game_date is NOT in its own prior.
- Empty-net shots are excluded entirely. No goalie faced them, so crediting
  them to a goalie's record is simply wrong (build_priors.py does the same).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

from ingest.db import pg_conn

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "db" / "schema_stage_c3.sql"

# Goalie-POV strength. Shooter-POV '5v4' means the shooter is on the power
# play, so the goalie is FACING a power play -> 'PP_against'. Matches
# goalie_strength() in build_priors.py.
GOALIE_STRENGTH_MAP = {
    "5v5": "5v5", "4v4": "5v5", "3v3": "5v5",
    "5v4": "PP_against", "5v3": "PP_against", "4v3": "PP_against",
    "4v5": "PK_against", "3v5": "PK_against", "3v4": "PK_against",
}

# Danger zones by shot distance, matching danger_zone() in build_priors.py.
ZONE_EDGES = [(0.0, 20.0, "high"), (20.0, 40.0, "mid"), (40.0, np.inf, "low")]

WINDOW_YEARS = 2
K_FIT_MIN_SHOTS = 200      # goalie-level eligibility for the K fit
MIN_CELL_SHOTS = 25        # labelling threshold only; does NOT affect any estimate


def init_schema():
    sql = SCHEMA_PATH.read_text()
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()
    print(f"Initialized schema from {SCHEMA_PATH}")


def classify_zone(distance_ft):
    """Vectorized distance -> zone label. NaN distances become 'unknown'."""
    d = np.asarray(distance_ft, dtype=float)
    out = np.full(d.shape, "unknown", dtype=object)
    for lo, hi, label in ZONE_EDGES:
        out[(d >= lo) & (d < hi)] = label
    return out


def load_shots() -> pd.DataFrame:
    """Load shots on goal faced by an identified goalie.

    Excludes:
      - missed shots and blocked shots (never reached the goalie)
      - empty-net shots (no goalie faced them)
      - rows with no goalie_id, strength_state or distance
    """
    query = """
        SELECT
            s.goalie_id      AS goalie_id,
            g.game_date      AS game_date,
            g.season         AS season,
            s.strength_state AS raw_strength,
            s.distance_ft    AS distance_ft,
            s.is_goal::int   AS is_goal
        FROM shots s
        JOIN games g ON g.game_id = s.game_id
        WHERE s.event_type IN ('shot-on-goal', 'goal')
          AND s.goalie_id IS NOT NULL
          AND s.strength_state IS NOT NULL
          AND s.distance_ft IS NOT NULL
          AND s.empty_net = false
    """
    with pg_conn() as conn:
        df = pd.read_sql(query, conn)

    df["strength"] = df["raw_strength"].map(GOALIE_STRENGTH_MAP)
    df = df[df["strength"].notna()].copy()

    df["zone"] = classify_zone(df["distance_ft"])
    df = df[df["zone"] != "unknown"].copy()

    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df = df.sort_values(["goalie_id", "game_date"]).reset_index(drop=True)

    print(f"Loaded {len(df):,} shots on goal, "
          f"{df['goalie_id'].nunique():,} goalies, "
          f"{df['game_date'].min()} to {df['game_date'].max()}")
    print(f"  League GA rate: {df['is_goal'].mean():.4f} "
          f"(sv% {1 - df['is_goal'].mean():.4f})")
    for z in ("high", "mid", "low"):
        sub = df[df["zone"] == z]
        print(f"  zone {z:<5} {len(sub):>7,} shots  GA rate {sub['is_goal'].mean():.4f}")
    return df


def attach_recency_weight(target_season: int, shot_seasons: np.ndarray) -> np.ndarray:
    """5 same season, 4 one prior, 3 two prior, 0 otherwise. Stage B parity."""
    diff = (target_season // 10000) - (shot_seasons // 10000)
    return np.where(diff == 0, 5.0, np.where(diff == 1, 4.0, np.where(diff == 2, 3.0, 0.0)))


def fit_k_methodofmoments(aggs: pd.DataFrame, mean: float) -> float:
    """Beta-Binomial method-of-moments K. Same as build_priors_expanding.py.

    Note the eligibility filter is applied to the K fit AND the league mean is
    computed on the same filtered population -- keeping those coupled to
    different populations is the Stage B bug that inflated PK means.
    """
    eligible = aggs[aggs["raw_shots"] >= K_FIT_MIN_SHOTS].copy()
    if len(eligible) < 5:
        print(f"    Warning: only {len(eligible)} eligible goalies for K fit; using K=100")
        return 100.0
    eligible["pct"] = eligible["raw_goals"] / eligible["raw_shots"]
    sample_var = eligible["pct"].var(ddof=1)
    expected_binom_var = (mean * (1 - mean) / eligible["raw_shots"]).mean()
    excess_var = max(sample_var - expected_binom_var, 1e-6)
    return float(max(mean * (1 - mean) / excess_var - 1, 1.0))


def fit_cell_prior(df_cell: pd.DataFrame) -> tuple[float, float]:
    """Return (league_mean, K) for one (zone, strength) cell."""
    aggs = df_cell.groupby("goalie_id").agg(
        raw_shots=("is_goal", "size"),
        raw_goals=("is_goal", "sum"),
    ).reset_index()
    eligible = aggs[aggs["raw_shots"] >= K_FIT_MIN_SHOTS]
    league_mean = (eligible["raw_goals"].sum() / eligible["raw_shots"].sum()
                   if len(eligible) > 0 and eligible["raw_shots"].sum() > 0
                   else df_cell["is_goal"].mean())
    K = fit_k_methodofmoments(aggs, league_mean)
    return float(league_mean), float(K)


def _window_slice(dates: np.ndarray, gd: np.datetime64):
    """Half-open [gd - WINDOW_YEARS, gd). gd itself is excluded -- including it
    would let a shot inform its own prior, which is the leak this whole module
    exists to avoid."""
    start = gd - np.timedelta64(WINDOW_YEARS * 365, "D")
    return (dates >= start) & (dates < gd), start


def build_priors(df: pd.DataFrame) -> list:
    """Build one row per (goalie, game_date, zone, strength).

    Pure function over a DataFrame -- no database access -- so the window,
    shrinkage and fallback logic can be tested against synthetic data.
    """
    zones = ["high", "mid", "low"]
    strengths = ["5v5", "PP_against", "PK_against"]

    # Fit priors per cell, plus a zone-level prior used by the fallback.
    cell_prior, zone_prior = {}, {}
    for z in zones:
        dz = df[df["zone"] == z]
        if len(dz) == 0:
            continue
        zone_prior[z] = fit_cell_prior(dz)
        print(f"  zone={z:<5} all-strength  league_mean={zone_prior[z][0]:.4f}  K={zone_prior[z][1]:.1f}")
        for s in strengths:
            dc = dz[dz["strength"] == s]
            if len(dc) == 0:
                continue
            cell_prior[(z, s)] = fit_cell_prior(dc)
            print(f"    strength={s:<11} n={len(dc):>7,}  "
                  f"league_mean={cell_prior[(z, s)][0]:.4f}  K={cell_prior[(z, s)][1]:.1f}")

    rows = []
    grouped = df.groupby("goalie_id", sort=False)
    n = grouped.ngroups
    for i, (goalie_id, gh) in enumerate(grouped):
        if i % 20 == 0:
            print(f"    goalie {i:,}/{n:,}", end="\r")

        dates_all = pd.to_datetime(gh["game_date"]).values
        seasons_all = gh["season"].values
        goals_all = gh["is_goal"].values
        zones_all = gh["zone"].values
        strengths_all = gh["strength"].values

        for gd in np.unique(dates_all):
            in_window, w_start = _window_slice(dates_all, gd)

            tgt_idx = np.searchsorted(dates_all, gd, side="left")
            target_season = int(seasons_all[min(tgt_idx, len(seasons_all) - 1)])

            for z in zones:
                if z not in zone_prior:
                    continue
                zone_mask = in_window & (zones_all == z)

                for s in strengths:
                    prior = cell_prior.get((z, s))
                    if prior is None:
                        continue
                    mean_c, K_c = prior

                    mask = zone_mask & (strengths_all == s)
                    n_cell = int(mask.sum())
                    mean_u, K_u = mean_c, K_c
                    alpha, beta = mean_u * K_u, (1 - mean_u) * K_u

                    # evidence_level is DESCRIPTIVE ONLY. It never changes the
                    # estimate -- see the module docstring for why the earlier
                    # fallback design was wrong.
                    level = "sufficient" if n_cell >= MIN_CELL_SHOTS else "sparse"

                    if not mask.any():
                        rows.append((
                            int(goalie_id), pd.Timestamp(gd).date(), z, s,
                            0.0, 0.0, 0, 0, alpha, beta, K_u, mean_u,
                            None, mean_u, "none",
                            pd.Timestamp(w_start).date(), pd.Timestamp(gd).date(),
                        ))
                        continue

                    w = attach_recency_weight(target_season, seasons_all[mask])
                    g = goals_all[mask]
                    wshots, wgoals = float(w.sum()), float((w * g).sum())
                    rshots, rgoals = int(mask.sum()), int(g.sum())
                    shrunken = (alpha + wgoals) / (alpha + beta + wshots)

                    rows.append((
                        int(goalie_id), pd.Timestamp(gd).date(), z, s,
                        wshots, wgoals, rshots, rgoals,
                        alpha, beta, K_u, mean_u,
                        (rgoals / rshots) if rshots else None, float(shrunken), level,
                        pd.Timestamp(w_start).date(), pd.Timestamp(gd).date(),
                    ))

    print(f"    goalie {n:,}/{n:,}  done.")
    print(f"  Generated {len(rows):,} prior rows")
    return rows


COLS = [
    "goalie_id", "game_date", "danger_zone", "strength_state",
    "weighted_shots", "weighted_goals", "raw_shots", "raw_goals",
    "prior_alpha", "prior_beta", "prior_concentration", "prior_mean",
    "raw_ga_rate", "shrunken_ga_rate", "evidence_level",
    "window_start_date", "window_end_date",
]


def write_rows(rows: list):
    """Bulk upsert. Coerces numpy scalars to Python natives -- psycopg2 renders
    np.float64(x) literally, which Postgres reads as a call into a schema
    named 'np' and rejects. Same guard as build_priors_expanding.py."""
    def coerce(v):
        if v is None:
            return None
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        return v

    clean = [tuple(coerce(v) for v in r) for r in rows]
    key = ("goalie_id", "game_date", "danger_zone", "strength_state")
    update = ", ".join(f"{c} = EXCLUDED.{c}" for c in COLS if c not in key)
    sql = f"""
        INSERT INTO goalie_priors_expanding ({",".join(COLS)})
        VALUES %s
        ON CONFLICT ({",".join(key)})
        DO UPDATE SET {update}, computed_at = now()
    """
    with pg_conn() as conn, conn.cursor() as cur:
        execute_values(cur, sql, clean,
                       template=f"({','.join(['%s'] * len(COLS))})", page_size=5000)
        conn.commit()
    print(f"Wrote {len(clean):,} rows to goalie_priors_expanding")


def sanity_checks():
    """Stage B validation expectations, restated for the expanding-window table."""
    with pg_conn() as conn, conn.cursor() as cur:
        print("\n=== Sanity checks ===")
        cur.execute("""
            SELECT danger_zone, strength_state,
                   COUNT(*), AVG(shrunken_ga_rate), MIN(shrunken_ga_rate), MAX(shrunken_ga_rate)
            FROM goalie_priors_expanding
            GROUP BY danger_zone, strength_state ORDER BY danger_zone, strength_state
        """)
        print(f"{'zone':<6}{'strength':<13}{'rows':>9}{'avg':>9}{'min':>9}{'max':>9}")
        for z, s, n, avg, lo, hi in cur.fetchall():
            print(f"{z:<6}{s:<13}{n:>9,}{avg:>9.4f}{lo:>9.4f}{hi:>9.4f}")
        print("  expect roughly: high ~0.18, mid ~0.06, low ~0.03 (Stage B targets)")

        cur.execute("""
            SELECT evidence_level, COUNT(*),
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)
            FROM goalie_priors_expanding GROUP BY evidence_level ORDER BY 2 DESC
        """)
        print("\nEvidence behind each row:")
        for level, n, pct in cur.fetchall():
            print(f"  {level:<12}{n:>9,}  {pct:>5}%")
        print("  If 'sparse' dominates, the 9-cell grid is too fine for a ~90-goalie")
        print("  population: most rows will sit near the league mean and the feature")
        print("  will carry little signal. Collapse to zone-only before adding it to v2.4.")

        cur.execute("SELECT COUNT(*) FROM goalie_priors_expanding WHERE window_end_date <= window_start_date")
        bad, = cur.fetchone()
        print(f"\nRows with non-positive window: {bad} (must be 0)")


def main():
    ap = argparse.ArgumentParser(description="Build expanding-window goalie priors")
    ap.add_argument("--init-schema", action="store_true")
    ap.add_argument("--checks-only", action="store_true")
    args = ap.parse_args()

    if args.init_schema:
        init_schema()
        return
    if args.checks_only:
        sanity_checks()
        return

    df = load_shots()
    rows = build_priors(df)
    write_rows(rows)
    sanity_checks()


if __name__ == "__main__":
    main()