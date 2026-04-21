"""Train the v1 xG model.

Design decisions (and why — this matters more than the code):

  1. THREE-WAY TEMPORAL SPLIT, not two-way.
     - Train: 2022-23 season
     - Validation: 2023-24 season (for early stopping only)
     - Test: 2024-25 season (touched once at the end for reporting)
     Using the test set for early stopping leaks the test labels into model
     selection — the reported AUC becomes optimistic. The test set in this
     script is FORBIDDEN from the training loop.

  2. No `random_state`, no stratified split.
     This is time-series prediction. We are predicting the future from the past.
     A random split would put shots from the same game in both train and test,
     inflating AUC by 0.03-0.05.

  3. Season derived from the games table, not parsed from game_id digits.
     Your schema already has a 'season' column. Use it.

  4. Calibration is evaluated with a reliability diagram, not just Brier.
     An xG of 0.10 must mean "10% of such shots score." We bin predictions into
     deciles and check the mean predicted vs actual. If bins drift off the
     diagonal, the model is miscalibrated even when AUC looks fine.

  5. Permutation importance on the TEST set.
     Training-set gain importance says which features the tree used.
     Permutation importance on held-out data says which features the model
     actually relies on. These often disagree, and the second one is the
     one that matters for trustworthy predictions.

Usage:
    python -m model.train_v1
    python -m model.train_v1 --dry-run      # skip save
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


# --- Feature list (explicit, for reproducibility in v2) ---
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


def load_features() -> pd.DataFrame:
    """Load shot_features joined with games to get true season labels."""
    query = """
        SELECT
            f.*,
            g.season AS season,
            g.game_date AS game_date
        FROM shot_features f
        JOIN games g ON g.game_id = f.game_id
        ORDER BY g.game_date, f.game_id, f.event_idx
    """
    with pg_conn() as conn:
        df = pd.read_sql(query, conn)

    # Cast booleans to int (LightGBM is happier with consistent types)
    for col in BOOLEAN_FEATURES + [TARGET]:
        df[col] = df[col].astype(int)

    # Cast categoricals
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")

    # Fill numeric NaNs with a sentinel (LightGBM handles NaN natively but we want reproducibility)
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Loaded {len(df):,} shots spanning {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"Seasons present: {sorted(df['season'].unique().tolist())}")
    return df


def split_three_way(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train on oldest season, validate on middle, test on newest."""
    seasons = sorted(df["season"].unique())
    if len(seasons) < 3:
        raise RuntimeError(
            f"Need 3 seasons for train/val/test split, got {len(seasons)}: {seasons}. "
            "Finish ingest first."
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
    """Sanity check that score_diff varies within games (not leaking final score)."""
    sample = df.groupby("game_id")["score_diff"].nunique()
    multi_value_games = (sample > 1).sum()
    pct = 100 * multi_value_games / len(sample)
    print(f"Leakage check: {multi_value_games:,}/{len(sample):,} games "
          f"({pct:.1f}%) have varying score_diff across shots — "
          f"expect >80% for non-leaky in-game state.")
    if pct < 50:
        raise RuntimeError(
            "score_diff appears constant within most games — possible leak of final score. "
            "Stopping."
        )


def train_model(train: pd.DataFrame, val: pd.DataFrame) -> lgb.LGBMClassifier:
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

    print("\nTraining LightGBM (early stopping on VALIDATION set, test untouched)...")
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
    print(f"  ROC-AUC:   {auc:.4f}  (spec target > 0.75)")
    print(f"  PR-AUC:    {pr_auc:.4f}  (baseline = {y.mean():.4f})")
    print(f"  Log-loss:  {ll:.4f}")
    print(f"  Brier:     {brier:.4f}")
    print(f"  Baseline (predict p=mean): Brier {y.var():.4f}")
    return {"auc": auc, "pr_auc": pr_auc, "log_loss": ll, "brier": brier,
            "n": len(df), "base_rate": float(y.mean())}


def reliability_table(model: lgb.LGBMClassifier, df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Reliability (calibration) table: for each xG decile, compare mean predicted vs actual."""
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
    """Measure AUC drop when each feature is shuffled on the test set."""
    rng = np.random.default_rng(seed)
    X, y = test_df[ALL_FEATURES].copy(), test_df[TARGET]
    base_probs = model.predict_proba(X)[:, 1]
    base_auc = roc_auc_score(y, base_probs)

    results = []
    for col in ALL_FEATURES:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            shuffled_probs = model.predict_proba(Xp)[:, 1]
            drops.append(base_auc - roc_auc_score(y, shuffled_probs))
        results.append({"feature": col,
                        "auc_drop_mean": np.mean(drops),
                        "auc_drop_std": np.std(drops)})
    imp = pd.DataFrame(results).sort_values("auc_drop_mean", ascending=False)
    return imp, base_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Don't save artifacts.")
    args = ap.parse_args()

    df = load_features()
    assert_no_leakage(df)
    train, val, test = split_three_way(df)

    model = train_model(train, val)

    # Evaluate on all three splits so we can see over/underfitting
    metrics = {
        "train": evaluate(model, train, "Train (sanity only)"),
        "val":   evaluate(model, val,   "Validation (used for early stop)"),
        "test":  evaluate(model, test,  "Test (held out, touch once)"),
    }

    # Calibration: the important check for an xG model
    print("\n--- Reliability (calibration) on TEST ---")
    rel = reliability_table(model, test, n_bins=10)
    print(rel.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    max_gap = rel["pred_vs_actual_gap"].abs().max()
    print(f"\nMax |predicted - actual| in any decile: {max_gap:.4f}")
    print("A well-calibrated xG model has all gaps < 0.02.")

    # Importance — two kinds
    print("\n--- LightGBM gain importance (what the TREES used) ---")
    gain = pd.DataFrame({"feature": ALL_FEATURES, "gain": model.feature_importances_})
    gain = gain.sort_values("gain", ascending=False)
    print(gain.head(15).to_string(index=False))

    print("\n--- Permutation importance on TEST set (what the MODEL relies on) ---")
    perm, base_auc = permutation_importance_on_test(model, test)
    print(perm.head(15).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Save artifacts
    if not args.dry_run:
        joblib.dump(model, ARTIFACT_DIR / "xg_v1.pkl")
        meta = {
            "version": "v1",
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
        with open(ARTIFACT_DIR / "xg_v1_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        rel.to_csv(ARTIFACT_DIR / "xg_v1_reliability.csv", index=False)
        gain.to_csv(ARTIFACT_DIR / "xg_v1_gain_importance.csv", index=False)
        perm.to_csv(ARTIFACT_DIR / "xg_v1_perm_importance.csv", index=False)
        print(f"\nArtifacts saved to {ARTIFACT_DIR}/")

    print("\nDone.")


if __name__ == "__main__":
    main()