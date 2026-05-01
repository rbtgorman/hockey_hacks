"""Train v2.1 xG model — regularized v2 plus post-hoc isotonic calibration.

What changed from v2
--------------------
v2 had test AUC 0.7620 (vs v1 0.7705) and a calibration max gap of 0.0270
(vs v1 0.0244). Diagnostics showed:

  - Features (shooter_prior_pct, goalie_prior_ga_rate) ARE useful:
    permutation importance > 0.01 on test set for both.
  - SQL joins are fine: train/val seasons have <0.6% missing priors.
  - Goalie talent is stable across train -> test (mean drift ~0.009).
  - Train AUC 0.863, val AUC 0.792, test AUC 0.762 -> classic overfit.
  - v2 fixed v1's bin-7 underprediction but blew up bin-9 overprediction.

So the features are good. The model is overfit AND miscalibrated. We fix
both:

  1. Stronger regularization (slower learning rate, smaller trees, more
     L1/L2). Reduces overfit by forcing the model to find robust splits
     instead of memorizing train-set quirks.
  2. Post-hoc isotonic calibration on the val set. Squashes systematic
     over/underprediction in any decile without changing AUC ordering.

Hyperparam delta from v1/v2
---------------------------
  v1/v2: learning_rate=0.03, num_leaves=63, min_child_samples=50,
         reg_alpha=0.1, reg_lambda=0.1, n_estimators=1000
  v2.1:  learning_rate=0.02, num_leaves=31, min_child_samples=100,
         reg_alpha=0.3, reg_lambda=0.3, n_estimators=2000

Outputs
-------
  artifacts/xg_v2_1.pkl              -- LightGBM model
  artifacts/xg_v2_1_calibrator.pkl   -- IsotonicRegression fit on val predictions
  artifacts/xg_v2_1_meta.json        -- params, metrics for raw and calibrated
  artifacts/xg_v2_1_reliability_raw.csv
  artifacts/xg_v2_1_reliability_calibrated.csv
  artifacts/xg_v2_1_gain_importance.csv
  artifacts/xg_v2_1_perm_importance.csv

Usage
-----
  python -m model.train_v2_1
  python -m model.train_v2_1 --dry-run

Inference
---------
At predict time:
    raw_probs = model.predict_proba(X)[:, 1]
    calibrated_probs = calibrator.predict(raw_probs)
The calibrated probabilities are what production should use.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss,
    precision_recall_curve, auc as sk_auc,
)
from sklearn.isotonic import IsotonicRegression

from ingest.db import pg_conn


# ---------------------------------------------------------------------------
# Feature list — identical to v2
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
# Feature build SQL — identical to v2
# ---------------------------------------------------------------------------
LOAD_QUERY = """
WITH base AS (
    SELECT
        f.*,
        g.season AS season,
        g.game_date AS game_date,
        s.shooter_id,
        s.goalie_id,
        CASE
            WHEN s.strength_state IN ('5v5', '4v4', '3v3') THEN '5v5'
            WHEN s.strength_state IN ('5v4', '5v3', '4v3', '6v5', '6v4') THEN 'PP'
            WHEN s.strength_state IN ('4v5', '3v5', '3v4', '5v6', '4v6') THEN 'PK'
            ELSE '5v5'
        END AS shooter_strength_bucket,
        CASE
            WHEN s.strength_state IN ('5v5', '4v4', '3v3') THEN '5v5'
            WHEN s.strength_state IN ('5v4', '5v3', '4v3', '6v5', '6v4') THEN 'PP_against'
            WHEN s.strength_state IN ('4v5', '3v5', '3v4', '5v6', '4v6') THEN 'PK_against'
            ELSE '5v5'
        END AS goalie_strength_bucket,
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
    with pg_conn() as conn:
        df = pd.read_sql(LOAD_QUERY, conn)

    n_missing_shooter = df["shooter_prior_pct"].isna().sum()
    n_missing_goalie = df["goalie_prior_ga_rate"].isna().sum()
    global_shooter_mean = df["shooter_prior_league_mean"].median()
    global_goalie_mean = df["goalie_prior_league_mean"].median()

    df["shooter_prior_pct"] = df["shooter_prior_pct"].fillna(
        df["shooter_prior_league_mean"]
    ).fillna(global_shooter_mean)
    df["goalie_prior_ga_rate"] = df["goalie_prior_ga_rate"].fillna(
        df["goalie_prior_league_mean"]
    ).fillna(global_goalie_mean)

    df = df.drop(columns=["shooter_prior_league_mean", "goalie_prior_league_mean",
                          "shooter_id", "goalie_id",
                          "shooter_strength_bucket", "goalie_strength_bucket",
                          "goalie_zone_bucket"])

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
    sample = df.groupby("game_id")["score_diff"].nunique()
    multi_value_games = (sample > 1).sum()
    pct = 100 * multi_value_games / len(sample)
    print(f"Leakage check (score_diff): {multi_value_games:,}/{len(sample):,} games "
          f"({pct:.1f}%) have varying score_diff — expect >80%.")
    if pct < 50:
        raise RuntimeError("score_diff appears constant within most games. Stopping.")


def train_model(train: pd.DataFrame, val: pd.DataFrame) -> lgb.LGBMClassifier:
    """Same loop as v2 but with regularized hyperparameters.

    Changes vs v2:
      learning_rate: 0.03 -> 0.02      (slower fitting, less overshoot)
      num_leaves:    63   -> 31        (smaller trees, less capacity)
      min_child_samples: 50 -> 100     (forces robust leaves)
      reg_alpha:    0.1   -> 0.3       (stronger L1)
      reg_lambda:   0.1   -> 0.3       (stronger L2)
      n_estimators: 1000  -> 2000      (cap raised; the slower LR needs more rounds)
    """
    X_train, y_train = train[ALL_FEATURES], train[TARGET]
    X_val, y_val = val[ALL_FEATURES], val[TARGET]

    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=100,
        reg_alpha=0.3,
        reg_lambda=0.3,
        objective="binary",
        metric="auc",
        importance_type="gain",
        random_state=42,
        verbose=-1,
    )

    print("\nTraining LightGBM v2.1 (regularized; early stopping on VAL)...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_names=["val"],
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[
            lgb.early_stopping(stopping_rounds=75, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    print(f"Best iteration: {model.best_iteration_}")
    return model


def fit_calibrator(model: lgb.LGBMClassifier, val: pd.DataFrame) -> IsotonicRegression:
    """Fit isotonic regression on val predictions -> val labels.

    Why isotonic over Platt: isotonic is non-parametric and handles non-monotone
    miscalibration patterns. Platt assumes the miscalibration is sigmoidal,
    which is wrong for tree models (especially in the tails, which is
    exactly where v2 broke down in bin 9).
    """
    X_val, y_val = val[ALL_FEATURES], val[TARGET]
    val_probs = model.predict_proba(X_val)[:, 1]
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(val_probs, y_val)
    print("Fitted isotonic calibrator on val set.")
    return cal


def evaluate(probs: np.ndarray, y: np.ndarray, label: str, v1_baseline: str = "") -> dict:
    auc = roc_auc_score(y, probs)
    ll = log_loss(y, probs)
    brier = brier_score_loss(y, probs)
    pr, rc, _ = precision_recall_curve(y, probs)
    pr_auc = sk_auc(rc, pr)
    print(f"\n--- {label} metrics ---")
    print(f"  ROC-AUC:   {auc:.4f}  {v1_baseline}")
    print(f"  PR-AUC:    {pr_auc:.4f}  (baseline = {y.mean():.4f})")
    print(f"  Log-loss:  {ll:.4f}")
    print(f"  Brier:     {brier:.4f}")
    return {"auc": auc, "pr_auc": pr_auc, "log_loss": ll, "brier": brier,
            "n": int(len(y)), "base_rate": float(y.mean())}


def reliability_table(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    edges = np.quantile(probs, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -1e-9, 1 + 1e-9
    bins = pd.cut(probs, bins=edges, labels=False, include_lowest=True)
    out = pd.DataFrame({"bin": bins, "pred": probs, "actual": y})
    table = out.groupby("bin").agg(
        n=("actual", "size"),
        mean_pred=("pred", "mean"),
        actual_rate=("actual", "mean"),
    ).reset_index()
    table["pred_vs_actual_gap"] = table["mean_pred"] - table["actual_rate"]
    return table


def permutation_importance_on_test(model, calibrator, test_df, n_repeats=3, seed=42):
    """Same as v2 but reports drop in CALIBRATED AUC.

    AUC ordering is invariant to monotone transforms, so calibration doesn't
    change AUC. We use raw probs for speed but the result applies equally to
    calibrated.
    """
    rng = np.random.default_rng(seed)
    X, y = test_df[ALL_FEATURES].copy(), test_df[TARGET].values
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
                        "auc_drop_mean": float(np.mean(drops)),
                        "auc_drop_std": float(np.std(drops))})
    imp = pd.DataFrame(results).sort_values("auc_drop_mean", ascending=False)
    return imp, base_auc


def compare_versions(v2_1_raw: dict, v2_1_cal: dict,
                     v2_1_max_gap_raw: float, v2_1_max_gap_cal: float) -> None:
    """Side-by-side comparison: v1 vs v2 vs v2.1 (raw) vs v2.1 (calibrated)."""
    print("\n" + "=" * 78)
    print("Model comparison — TEST set")
    print("=" * 78)
    v1_meta_path = ARTIFACT_DIR / "xg_v1_meta.json"
    v2_meta_path = ARTIFACT_DIR / "xg_v2_meta.json"
    v1m = json.load(open(v1_meta_path)) if v1_meta_path.exists() else None
    v2m = json.load(open(v2_meta_path)) if v2_meta_path.exists() else None

    rows = []
    if v1m:
        rows.append(("v1",
                     v1m["metrics"]["test"]["auc"],
                     v1m["metrics"]["test"]["log_loss"],
                     v1m["metrics"]["test"]["brier"],
                     v1m["calibration_max_gap"]))
    if v2m:
        rows.append(("v2",
                     v2m["metrics"]["test"]["auc"],
                     v2m["metrics"]["test"]["log_loss"],
                     v2m["metrics"]["test"]["brier"],
                     v2m["calibration_max_gap"]))
    rows.append(("v2.1 raw", v2_1_raw["auc"], v2_1_raw["log_loss"],
                 v2_1_raw["brier"], v2_1_max_gap_raw))
    rows.append(("v2.1 cal", v2_1_cal["auc"], v2_1_cal["log_loss"],
                 v2_1_cal["brier"], v2_1_max_gap_cal))

    print(f"  {'model':10s}  {'AUC':>7s}  {'log_loss':>9s}  {'brier':>7s}  {'cal_gap':>8s}")
    print(f"  {'-'*10}  {'-'*7}  {'-'*9}  {'-'*7}  {'-'*8}")
    for name, auc, ll, brier, gap in rows:
        print(f"  {name:10s}  {auc:7.4f}  {ll:9.4f}  {brier:7.4f}  {gap:8.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Don't save artifacts.")
    args = ap.parse_args()

    df = load_features()
    assert_no_leakage(df)
    train, val, test = split_three_way(df)

    model = train_model(train, val)

    # Evaluate raw probabilities first
    X_test, y_test = test[ALL_FEATURES], test[TARGET].values
    test_probs_raw = model.predict_proba(X_test)[:, 1]

    metrics_raw = {
        "train": evaluate(model.predict_proba(train[ALL_FEATURES])[:, 1],
                          train[TARGET].values, "Train RAW (sanity)"),
        "val":   evaluate(model.predict_proba(val[ALL_FEATURES])[:, 1],
                          val[TARGET].values, "Val RAW"),
        "test":  evaluate(test_probs_raw, y_test, "Test RAW",
                          "(v1: 0.7705, v2: 0.7620)"),
    }

    print("\n--- Reliability (RAW) on TEST ---")
    rel_raw = reliability_table(test_probs_raw, y_test, n_bins=10)
    print(rel_raw.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    max_gap_raw = float(rel_raw["pred_vs_actual_gap"].abs().max())
    print(f"\nMax |predicted - actual| in any decile (RAW): {max_gap_raw:.4f}")

    # Fit isotonic calibrator on val, then apply to test
    calibrator = fit_calibrator(model, val)
    test_probs_cal = calibrator.predict(test_probs_raw)

    metrics_cal = {
        "test": evaluate(test_probs_cal, y_test, "Test CALIBRATED",
                         "(AUC unchanged by calibration; ll/brier should improve)"),
    }

    print("\n--- Reliability (CALIBRATED) on TEST ---")
    rel_cal = reliability_table(test_probs_cal, y_test, n_bins=10)
    print(rel_cal.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    max_gap_cal = float(rel_cal["pred_vs_actual_gap"].abs().max())
    print(f"\nMax |predicted - actual| in any decile (CAL): {max_gap_cal:.4f}")
    print("Target: < 0.02 (v1: 0.0244, v2: 0.0270)")

    print("\n--- LightGBM gain importance ---")
    gain = pd.DataFrame({"feature": ALL_FEATURES, "gain": model.feature_importances_})
    gain = gain.sort_values("gain", ascending=False)
    print(gain.head(20).to_string(index=False))

    if not args.dry_run:
        joblib.dump(model, ARTIFACT_DIR / "xg_v2_1.pkl")
        joblib.dump(calibrator, ARTIFACT_DIR / "xg_v2_1_calibrator.pkl")
        meta = {
            "version": "v2.1",
            "parent_version": "v2",
            "delta_from_v2": "regularized hyperparams + isotonic calibration on val",
            "hyperparams": {
                "n_estimators": 2000, "learning_rate": 0.02, "num_leaves": 31,
                "min_child_samples": 100, "reg_alpha": 0.3, "reg_lambda": 0.3,
            },
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
            "metrics_raw": metrics_raw,
            "metrics_calibrated": metrics_cal,
            "calibration_max_gap_raw": max_gap_raw,
            "calibration_max_gap_calibrated": max_gap_cal,
        }
        with open(ARTIFACT_DIR / "xg_v2_1_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        rel_raw.to_csv(ARTIFACT_DIR / "xg_v2_1_reliability_raw.csv", index=False)
        rel_cal.to_csv(ARTIFACT_DIR / "xg_v2_1_reliability_calibrated.csv", index=False)
        gain.to_csv(ARTIFACT_DIR / "xg_v2_1_gain_importance.csv", index=False)
        print(f"\nCore artifacts saved to {ARTIFACT_DIR}/")

    print("\n--- Permutation importance on TEST set ---")
    try:
        perm, base_auc = permutation_importance_on_test(model, calibrator, test)
        print(perm.head(20).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        if not args.dry_run:
            perm.to_csv(ARTIFACT_DIR / "xg_v2_1_perm_importance.csv", index=False)
            print("Permutation importance saved.")
    except Exception as e:
        print(f"Permutation importance failed but model is saved. "
              f"Error: {type(e).__name__}: {e}")

    compare_versions(metrics_raw["test"], metrics_cal["test"],
                     max_gap_raw, max_gap_cal)
    print("\nDone.")


if __name__ == "__main__":
    main()