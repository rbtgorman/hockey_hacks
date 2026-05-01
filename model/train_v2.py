"""Train the v2 xG model — v1 architecture plus Empirical Bayes player priors.

What's different from v1
------------------------
This script is a near-exact mirror of model/train_v1.py:
  - Same LightGBM hyperparameters
  - Same three-way temporal split (2022-23 train, 2023-24 val, 2024-25 test)
  - Same leakage check, same reliability table, same permutation importance
  - Same artifact format

The ONLY differences from v1 are two new numeric features:
  - shooter_prior_pct      : shrunken shooting % from skater_priors_train,
                             matched to the shot's strength state
                             (5v4 -> PP bucket, 4v5 -> PK bucket, etc.)
  - goalie_prior_ga_rate   : shrunken goals-against rate from goalie_priors_train,
                             matched to the shot's danger zone (from distance_ft)
                             and goalie-POV strength state

That isolates the experiment: any AUC delta between v1 and v2 is attributable
to the priors, not to architecture or hyperparameter changes.

Leakage protection
------------------
We read from skater_priors_train / goalie_priors_train, which were built from
2022-23 and 2023-24 only (no 2024-25 data). The test season's shooter and
goalie performance never leaked into the priors used as features.

Production simulator code should read from skater_priors / goalie_priors
(all-seasons pooled). v2 uses the _train tables only because we are evaluating
on the held-out 2024-25 test season.

Targets
-------
v1 baseline:
  Test ROC-AUC      : 0.7705
  Test calibration  : max gap 0.0244 across deciles

v2 success criteria:
  Test ROC-AUC      : > 0.7705 (any improvement attributable to priors is a win)
  Test calibration  : max gap < v1's, ideally < 0.02 (priors are expected to
                       fix v1's underprediction on high-xG shots specifically)

Usage
-----
  python -m model.train_v2
  python -m model.train_v2 --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss,
    precision_recall_curve, auc as sk_auc,
)

from ingest.db import pg_conn


# ---------------------------------------------------------------------------
# Feature list — v1 features plus two new prior columns
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "distance_ft",
    "angle_deg",
    "x_norm",
    "y_norm",
    "score_diff",
    "seconds_since_last_event",
    "distance_from_last_event_ft",
    "period_num",
    "seconds_remaining_in_period",
    # NEW in v2:
    "shooter_prior_pct",
    "goalie_prior_ga_rate",
]
BOOLEAN_FEATURES = [
    "is_rebound",
    "is_rush",
    "empty_net",
]
CATEGORICAL_FEATURES = [
    "shot_type",
    "strength_state",
    "last_event_type",
]
ALL_FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_goal"

MODEL_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = MODEL_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Feature build SQL
#
# We join shot_features (v1's feature table) with the train-only priors.
# Strength-state matching for shooters is done with CASE expressions so each
# shot picks exactly one shooter prior row.
#
# Shooter strength bucket map (matches features/build_priors.py:shooter_strength):
#   5v5, 4v4, 3v3                -> '5v5'
#   5v4, 5v3, 4v3, 6v5, 6v4      -> 'PP'
#   4v5, 3v5, 3v4, 5v6, 4v6      -> 'PK'
#   anything else (penalty shots, 1v0, etc.) -> '5v5' fallback (rarest path)
#
# Goalie strength bucket map (inverse, matches goalie_strength):
#   '5v5' -> '5v5', 'PP' shooter -> 'PP_against', 'PK' shooter -> 'PK_against'
#
# Goalie zone map (matches danger_zone):
#   distance_ft <= 20  -> 'high'
#   distance_ft <= 40  -> 'mid'
#   else               -> 'low'
#   NULL               -> NULL (LEFT JOIN -> NULL prior, handled in pandas below)
# ---------------------------------------------------------------------------

LOAD_QUERY = """
WITH base AS (
    SELECT
        f.*,
        g.season AS season,
        g.game_date AS game_date,
        s.shooter_id,
        s.goalie_id,
        s.distance_ft AS shot_distance_ft,
        s.strength_state AS shot_strength_state,
        s.empty_net AS shot_empty_net,
        -- shooter strength bucket
        CASE
            WHEN s.strength_state IN ('5v5', '4v4', '3v3') THEN '5v5'
            WHEN s.strength_state IN ('5v4', '5v3', '4v3', '6v5', '6v4') THEN 'PP'
            WHEN s.strength_state IN ('4v5', '3v5', '3v4', '5v6', '4v6') THEN 'PK'
            ELSE '5v5'  -- penalty-shot states etc., extremely rare
        END AS shooter_strength_bucket,
        -- goalie strength bucket (inverse of shooter)
        CASE
            WHEN s.strength_state IN ('5v5', '4v4', '3v3') THEN '5v5'
            WHEN s.strength_state IN ('5v4', '5v3', '4v3', '6v5', '6v4') THEN 'PP_against'
            WHEN s.strength_state IN ('4v5', '3v5', '3v4', '5v6', '4v6') THEN 'PK_against'
            ELSE '5v5'
        END AS goalie_strength_bucket,
        -- goalie zone bucket
        CASE
            WHEN s.distance_ft IS NULL THEN NULL
            WHEN s.distance_ft <= 20 THEN 'high'
            WHEN s.distance_ft <= 40 THEN 'mid'
            ELSE 'low'
        END AS goalie_zone_bucket
    FROM shot_features f
    JOIN games g ON g.game_id = f.game_id
    JOIN shots s ON s.game_id = f.game_id AND s.event_idx = f.event_idx
)
SELECT
    base.*,
    sp.shrunken_shooting_pct AS shooter_prior_pct,
    sp.prior_mean             AS shooter_prior_league_mean,
    gp.shrunken_ga_rate       AS goalie_prior_ga_rate,
    gp.prior_mean             AS goalie_prior_league_mean
FROM base
LEFT JOIN skater_priors_train sp
    ON sp.player_id = base.shooter_id
   AND sp.strength_state = base.shooter_strength_bucket
LEFT JOIN goalie_priors_train gp
    ON gp.player_id = base.goalie_id
   AND gp.strength_state = base.goalie_strength_bucket
   AND gp.danger_zone = base.goalie_zone_bucket
ORDER BY base.game_date, base.game_id, base.event_idx
"""


def load_features() -> pd.DataFrame:
    """Load v1 features joined with train-only priors.

    Missing-prior handling: LEFT JOINs leave NULLs for shots whose
    shooter/goalie wasn't seen in training, or whose distance/strength
    didn't bucket cleanly. We fill those with the league-mean prior
    (also returned by the join) — equivalent to "no prior info, fall
    back to league rate." This is what an EB prior would say for a
    player with zero observations.
    """
    with pg_conn() as conn:
        df = pd.read_sql(LOAD_QUERY, conn)

    # Fill missing priors with their respective league means
    # (the join also returned prior_mean for fallback).
    n_missing_shooter = df["shooter_prior_pct"].isna().sum()
    n_missing_goalie = df["goalie_prior_ga_rate"].isna().sum()

    # If shooter_prior_league_mean is itself NaN (no matching strength bucket
    # in priors at all), use the global 5v5 skater mean as the deepest fallback.
    # This shouldn't happen given our SQL coverage but defends against it.
    global_shooter_mean = df["shooter_prior_league_mean"].median()
    global_goalie_mean = df["goalie_prior_league_mean"].median()

    df["shooter_prior_pct"] = df["shooter_prior_pct"].fillna(
        df["shooter_prior_league_mean"]
    ).fillna(global_shooter_mean)
    df["goalie_prior_ga_rate"] = df["goalie_prior_ga_rate"].fillna(
        df["goalie_prior_league_mean"]
    ).fillna(global_goalie_mean)

    # Drop the helper columns; they're not features
    df = df.drop(columns=["shooter_prior_league_mean", "goalie_prior_league_mean",
                          "shooter_id", "goalie_id",
                          "shot_distance_ft", "shot_strength_state", "shot_empty_net",
                          "shooter_strength_bucket", "goalie_strength_bucket",
                          "goalie_zone_bucket"])

    # Match v1 type discipline
    for col in BOOLEAN_FEATURES + [TARGET]:
        df[col] = df[col].astype(int)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Loaded {len(df):,} shots spanning {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"Seasons present: {sorted(df['season'].unique().tolist())}")
    print(f"Missing shooter priors filled with league mean: {n_missing_shooter:,} "
          f"({100*n_missing_shooter/len(df):.2f}%)")
    print(f"Missing goalie priors filled with league mean:  {n_missing_goalie:,} "
          f"({100*n_missing_goalie/len(df):.2f}%)")
    return df


def split_three_way(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Same split as v1: oldest train, middle val, newest test."""
    seasons = sorted(df["season"].unique())
    if len(seasons) < 3:
        raise RuntimeError(
            f"Need 3 seasons for train/val/test split, got {len(seasons)}: {seasons}."
        )
    train_season, val_season, test_season = seasons[0], seasons[1], seasons[2]
    train = df[df["season"] == train_season].copy()
    val = df[df["season"] == val_season].copy()
    test = df[df["season"] == test_season].copy()
    print(f"Train: season {train_season}, {len(train):,} shots, {train[TARGET].mean()*100:.2f}% goals")
    print(f"Val:   season {val_season}, {len(val):,} shots, {val[TARGET].mean()*100:.2f}% goals")
    print(f"Test:  season {test_season}, {len(test):,} shots, {test[TARGET].mean()*100:.2f}% goals")
    return train, val, test


def assert_no_leakage(df: pd.DataFrame) -> None:
    """v1's score_diff sanity check, plus a v2-specific prior-leakage check."""
    sample = df.groupby("game_id")["score_diff"].nunique()
    multi_value_games = (sample > 1).sum()
    pct = 100 * multi_value_games / len(sample)
    print(f"Leakage check (score_diff): {multi_value_games:,}/{len(sample):,} games "
          f"({pct:.1f}%) have varying score_diff — expect >80%.")
    if pct < 50:
        raise RuntimeError("score_diff appears constant within most games. Stopping.")

    # v2-specific: confirm priors don't have test-season info.
    # The priors were built from train+val seasons; if the test set has
    # exact-match priors for shooters who only debuted in 2024-25, that's a
    # smoking gun. We check by spot-comparing prior coverage by season:
    coverage = (
        df.assign(
            has_shooter_prior=df["shooter_prior_pct"].notna()
                              & (df["shooter_prior_pct"] != df["shooter_prior_pct"].median())
        )
        .groupby("season")
        .agg(
            n=("game_id", "size"),
            shooter_prior_present=("has_shooter_prior", "mean"),
        )
    )
    print("\nShooter prior coverage by season (heuristic — not exact):")
    print(coverage.to_string(float_format=lambda v: f"{v:.3f}"))


def train_model(train: pd.DataFrame, val: pd.DataFrame) -> lgb.LGBMClassifier:
    """Identical hyperparameters to train_v1.py."""
    X_train, y_train = train[ALL_FEATURES], train[TARGET]
    X_val, y_val = val[ALL_FEATURES], val[TARGET]

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        reg_alpha=0.1,
        reg_lambda=0.1,
        objective="binary",
        metric="auc",
        importance_type="gain",
        random_state=42,
        verbose=-1,
    )

    print("\nTraining LightGBM (early stopping on VAL, test untouched)...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_names=["val"],
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    print(f"Best iteration: {model.best_iteration_}")
    return model


def evaluate(model: lgb.LGBMClassifier, df: pd.DataFrame, label: str) -> dict:
    X, y = df[ALL_FEATURES], df[TARGET]
    probs = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs)
    ll = log_loss(y, probs)
    brier = brier_score_loss(y, probs)
    pr, rc, _ = precision_recall_curve(y, probs)
    pr_auc = sk_auc(rc, pr)
    print(f"\n--- {label} metrics ---")
    print(f"  ROC-AUC:   {auc:.4f}  (v1 test: 0.7705)")
    print(f"  PR-AUC:    {pr_auc:.4f}  (baseline = {y.mean():.4f})")
    print(f"  Log-loss:  {ll:.4f}")
    print(f"  Brier:     {brier:.4f}")
    return {"auc": auc, "pr_auc": pr_auc, "log_loss": ll, "brier": brier,
            "n": len(df), "base_rate": float(y.mean())}


def reliability_table(model: lgb.LGBMClassifier, df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    X, y = df[ALL_FEATURES], df[TARGET]
    probs = model.predict_proba(X)[:, 1]
    edges = np.quantile(probs, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -1e-9, 1 + 1e-9
    bins = pd.cut(probs, bins=edges, labels=False, include_lowest=True)
    out = pd.DataFrame({"bin": bins, "pred": probs, "actual": y.values})
    table = out.groupby("bin").agg(
        n=("actual", "size"),
        mean_pred=("pred", "mean"),
        actual_rate=("actual", "mean"),
    ).reset_index()
    table["pred_vs_actual_gap"] = table["mean_pred"] - table["actual_rate"]
    return table


def permutation_importance_on_test(model, test_df, n_repeats=3, seed=42):
    """Same as v1 — preserves categorical metadata across shuffle."""
    rng = np.random.default_rng(seed)
    X, y = test_df[ALL_FEATURES].copy(), test_df[TARGET]
    base_probs = model.predict_proba(X)[:, 1]
    base_auc = roc_auc_score(y, base_probs)

    results = []
    for col in ALL_FEATURES:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            shuffled_vals = rng.permutation(Xp[col].values)
            if col in CATEGORICAL_FEATURES:
                Xp[col] = pd.Categorical(shuffled_vals,
                                         categories=X[col].cat.categories)
            else:
                Xp[col] = shuffled_vals
            shuffled_probs = model.predict_proba(Xp)[:, 1]
            drops.append(base_auc - roc_auc_score(y, shuffled_probs))
        results.append({"feature": col,
                        "auc_drop_mean": np.mean(drops),
                        "auc_drop_std": np.std(drops)})
    imp = pd.DataFrame(results).sort_values("auc_drop_mean", ascending=False)
    return imp, base_auc


def compare_to_v1(v2_metrics: dict, v2_max_gap: float) -> None:
    """Print a side-by-side v1/v2 comparison from the v1 meta JSON."""
    v1_meta_path = ARTIFACT_DIR / "xg_v1_meta.json"
    if not v1_meta_path.exists():
        print(f"\n(no v1 meta at {v1_meta_path}, skipping side-by-side)")
        return
    with open(v1_meta_path) as f:
        v1_meta = json.load(f)
    print("\n" + "=" * 60)
    print("v1 vs v2 — TEST set comparison")
    print("=" * 60)
    cols = ["auc", "pr_auc", "log_loss", "brier"]
    for c in cols:
        v1v = v1_meta["metrics"]["test"][c]
        v2v = v2_metrics["test"][c]
        delta = v2v - v1v
        # auc/pr_auc: higher is better; log_loss/brier: lower is better
        better = (delta > 0) if c in ("auc", "pr_auc") else (delta < 0)
        marker = "  ✓" if better else ("  ✗" if abs(delta) > 1e-6 else "  =")
        print(f"  {c:10s}  v1={v1v:.4f}  v2={v2v:.4f}  Δ={delta:+.4f}{marker}")
    print(f"  {'cal_gap':10s}  v1={v1_meta['calibration_max_gap']:.4f}  "
          f"v2={v2_max_gap:.4f}  Δ={v2_max_gap - v1_meta['calibration_max_gap']:+.4f}"
          f"  {'  ✓' if v2_max_gap < v1_meta['calibration_max_gap'] else '  ✗'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Don't save artifacts.")
    args = ap.parse_args()

    df = load_features()
    assert_no_leakage(df)
    train, val, test = split_three_way(df)

    model = train_model(train, val)

    metrics = {
        "train": evaluate(model, train, "Train (sanity only)"),
        "val":   evaluate(model, val,   "Validation (used for early stop)"),
        "test":  evaluate(model, test,  "Test (held out, touch once)"),
    }

    print("\n--- Reliability (calibration) on TEST ---")
    rel = reliability_table(model, test, n_bins=10)
    print(rel.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    max_gap = rel["pred_vs_actual_gap"].abs().max()
    print(f"\nMax |predicted - actual| in any decile: {max_gap:.4f}")
    print("Target: < 0.02 (v1 was 0.0244)")

    print("\n--- LightGBM gain importance ---")
    gain = pd.DataFrame({"feature": ALL_FEATURES, "gain": model.feature_importances_})
    gain = gain.sort_values("gain", ascending=False)
    print(gain.head(20).to_string(index=False))

    if not args.dry_run:
        joblib.dump(model, ARTIFACT_DIR / "xg_v2.pkl")
        meta = {
            "version": "v2",
            "parent_version": "v1",
            "delta_from_v1": "added shooter_prior_pct and goalie_prior_ga_rate "
                             "(from skater_priors_train / goalie_priors_train)",
            "features": {
                "numeric": NUMERIC_FEATURES,
                "boolean": BOOLEAN_FEATURES,
                "categorical": CATEGORICAL_FEATURES,
            },
            "categorical_categories": {
                col: sorted([c for c in df[col].cat.categories.tolist() if c is not None])
                for col in CATEGORICAL_FEATURES
            },
            "train_season": int(sorted(df["season"].unique())[0]),
            "val_season":   int(sorted(df["season"].unique())[1]),
            "test_season":  int(sorted(df["season"].unique())[2]),
            "best_iteration": int(model.best_iteration_ or 0),
            "metrics": metrics,
            "calibration_max_gap": float(max_gap),
        }
        with open(ARTIFACT_DIR / "xg_v2_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        rel.to_csv(ARTIFACT_DIR / "xg_v2_reliability.csv", index=False)
        gain.to_csv(ARTIFACT_DIR / "xg_v2_gain_importance.csv", index=False)
        print(f"\nCore artifacts saved to {ARTIFACT_DIR}/")

    print("\n--- Permutation importance on TEST set ---")
    try:
        perm, base_auc = permutation_importance_on_test(model, test)
        print(perm.head(20).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        if not args.dry_run:
            perm.to_csv(ARTIFACT_DIR / "xg_v2_perm_importance.csv", index=False)
            print("Permutation importance saved.")
    except Exception as e:
        print(f"Permutation importance failed but model is saved. "
              f"Error: {type(e).__name__}: {e}")

    compare_to_v1(metrics, max_gap)
    print("\nDone.")


if __name__ == "__main__":
    main()