"""
v2.2 xG model: v1 architecture + shooter prior ONLY.

Mirrors train_v1.py exactly:
  - Same FROM shot_features f JOIN games g pattern
  - Same hyperparameters
  - Same temporal split logic (sort seasons, take [0]/[1]/[2])

Adds exactly two things:
  1. JOIN shots to get shooter_id (shot_features doesn't expose it)
  2. LEFT JOIN skater_priors_train to get shrunken_shooting_pct
  3. shooter_prior_pct added to feature list

NO goalie prior, NO calibrator, NO regularization changes.
This isolates the shooter prior's effect on test AUC.

Usage:
    python -m model.train_v2_2
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, average_precision_score
from ingest.db import pg_conn

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

TARGET = "is_goal"

# Mirror v1's feature buckets
BOOLEAN_FEATURES = ["empty_net", "is_rebound", "is_rush"]
CATEGORICAL_FEATURES = ["last_event_type", "shot_type", "strength_state"]
NUMERIC_FEATURES = [
    "distance_ft", "angle_deg", "x_norm", "y_norm",
    "period_num", "game_seconds", "seconds_remaining_in_period",
    "seconds_since_last_event", "distance_from_last_event_ft",
    "score_diff",
    "shooter_prior_pct",  # the only new feature
]
ALL_FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES

# Mirror v1 query, plus join to shots (for shooter_id) and skater_priors_train.
# Strength bucket logic: shot_features.strength_state is raw ('5v5','5v4',...);
# priors are stored bucketed ('5v5','PP','PK'). Bucket on the shots side.
LOAD_QUERY = """
    SELECT
        f.*,
        g.season AS season,
        g.game_date AS game_date,
        COALESCE(sp.shrunken_shooting_pct, lm.league_mean) AS shooter_prior_pct
    FROM shot_features f
    JOIN games g ON g.game_id = f.game_id
    JOIN shots s ON s.shot_id = f.shot_id
    LEFT JOIN skater_priors_train sp
        ON sp.player_id = s.shooter_id
       AND sp.strength_state = CASE
            WHEN f.strength_state IN ('5v5','4v4','3v3') THEN '5v5'
            WHEN f.strength_state IN ('5v4','5v3','4v3') THEN 'PP'
            WHEN f.strength_state IN ('4v5','3v5','3v4') THEN 'PK'
            ELSE '5v5'
       END
    LEFT JOIN (
        SELECT strength_state, AVG(shrunken_shooting_pct) AS league_mean
        FROM skater_priors_train
        GROUP BY strength_state
    ) lm
        ON lm.strength_state = CASE
            WHEN f.strength_state IN ('5v5','4v4','3v3') THEN '5v5'
            WHEN f.strength_state IN ('5v4','5v3','4v3') THEN 'PP'
            WHEN f.strength_state IN ('4v5','3v5','3v4') THEN 'PK'
            ELSE '5v5'
       END
    ORDER BY g.game_date, f.game_id, f.event_idx
"""


def load_features() -> pd.DataFrame:
    with pg_conn() as conn:
        df = pd.read_sql(LOAD_QUERY, conn)

    for col in BOOLEAN_FEATURES + [TARGET]:
        df[col] = df[col].astype(int)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Loaded {len(df):,} shots spanning {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"Seasons present: {sorted(df['season'].unique().tolist())}")

    miss = df["shooter_prior_pct"].isna().sum()
    print(f"Missing shooter priors after league-mean fallback: "
          f"{miss:,} ({miss/len(df)*100:.2f}%)  "
          f"(should be ~0 — only happens if a strength bucket has no priors at all)")
    return df


def split_three_way(df):
    seasons = sorted(df["season"].unique())
    if len(seasons) < 3:
        raise RuntimeError(f"Need 3 seasons, got {len(seasons)}: {seasons}")
    s0, s1, s2 = seasons[0], seasons[1], seasons[2]
    train = df[df["season"] == s0].copy()
    val   = df[df["season"] == s1].copy()
    test  = df[df["season"] == s2].copy()
    print(f"Train: season {s0}, {len(train):,} shots, {train[TARGET].mean()*100:.2f}% goals")
    print(f"Val:   season {s1}, {len(val):,} shots, {val[TARGET].mean()*100:.2f}% goals")
    print(f"Test:  season {s2}, {len(test):,} shots, {test[TARGET].mean()*100:.2f}% goals")
    return train, val, test


def assert_no_leakage(df):
    sample = df.groupby("game_id")["score_diff"].nunique()
    pct = 100 * (sample > 1).sum() / len(sample)
    print(f"Leakage check (score_diff): {(sample>1).sum():,}/{len(sample):,} "
          f"games ({pct:.1f}%) have varying score_diff — expect >80%.")
    if pct < 50:
        raise RuntimeError("score_diff constant in most games — possible leak.")


def reliability_table(y_true, y_pred, n_bins=10):
    df = pd.DataFrame({"y": y_true, "p": y_pred})
    df["bin"] = pd.qcut(df["p"], n_bins, labels=False, duplicates="drop")
    grp = df.groupby("bin").agg(
        n=("y", "size"),
        mean_pred=("p", "mean"),
        actual_rate=("y", "mean"),
    ).reset_index()
    grp["pred_vs_actual_gap"] = grp["mean_pred"] - grp["actual_rate"]
    return grp


def main():
    print("=" * 60)
    print("v2.2: v1 architecture + shooter prior ONLY")
    print("=" * 60)

    df = load_features()
    assert_no_leakage(df)
    train, val, test = split_three_way(df)

    X_train, y_train = train[ALL_FEATURES], train[TARGET]
    X_val,   y_val   = val[ALL_FEATURES],   val[TARGET]
    X_test,  y_test  = test[ALL_FEATURES],  test[TARGET]

    # v1 hyperparameters, exactly
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

    print("\nTraining LightGBM v2.2 (v1 params, +shooter_prior_pct)...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_names=["val"],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=50)],
    )
    print(f"Best iteration: {model.best_iteration_}")

    def eval_split(name, X, y):
        p = model.predict_proba(X)[:, 1]
        print(f"\n--- {name} metrics ---")
        print(f"  ROC-AUC:   {roc_auc_score(y, p):.4f}")
        print(f"  PR-AUC:    {average_precision_score(y, p):.4f}  (baseline = {y.mean():.4f})")
        print(f"  Log-loss:  {log_loss(y, p):.4f}")
        print(f"  Brier:     {brier_score_loss(y, p):.4f}")
        return p

    eval_split("Train", X_train, y_train)
    eval_split("Val",   X_val,   y_val)
    p_test = eval_split("Test",  X_test,  y_test)

    print("\n--- Reliability on TEST ---")
    rel = reliability_table(y_test, p_test)
    print(rel.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    max_gap = rel["pred_vs_actual_gap"].abs().max()
    print(f"\nMax |predicted - actual| in any decile: {max_gap:.4f}")

    print("\n" + "=" * 60)
    print("BASELINES")
    print(f"  v1:        AUC 0.7705, max gap 0.0244")
    print(f"  v2:        AUC 0.7620, max gap 0.0270")
    print(f"  v2.1 raw:  AUC 0.7615, max gap 0.0302")
    print(f"  v2.2:      AUC {roc_auc_score(y_test, p_test):.4f}, max gap {max_gap:.4f}")
    print("=" * 60)

    print("\n--- LightGBM gain importance ---")
    imp = pd.DataFrame({
        "feature": ALL_FEATURES,
        "gain": model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    print(imp.to_string(index=False))

    model_path = ARTIFACT_DIR / "xg_v2_2.txt"
    model.booster_.save_model(str(model_path))
    print(f"\nSaved model to {model_path}")


if __name__ == "__main__":
    main()