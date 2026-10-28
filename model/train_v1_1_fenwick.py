"""v1.1-fenwick: v1-fenwick plus is_post_rebound, on the four-season split.

WHAT CHANGES FROM v1-fenwick
----------------------------
    features  v1's fifteen + is_post_rebound (features/build_features_fenwick.py,
              docstring item 8)
    split     train 2022-23 + 2023-24, val 2024-25, test 2025-26
              (v1-fenwick: train 2022-23, val 2023-24, test 2024-25)

Hyperparameters are v1's, copied here because model.train_v1's helpers are
bound to its own feature list; train_v1.py and train_v1_fenwick.py are
untouched and still reproduce their runs. v1.split_three_way would silently
take the three oldest seasons, so this script names its seasons explicitly.

Validation is 2024-25, the first season of the giveaway/takeaway recording
change (giveaways about 2x, takeaways down a third, faceoffs flat). Early
stopping therefore sees that regime, and the test season is the second year
of it.

COMPARISON ON THE SAME TEST SEASON (2025-26)
--------------------------------------------
Three rows, so the new feature and the new split can be told apart:
    v1-fenwick       the committed .pkl, trained on 2022-23 only, not retrained
    ablation         v1 features on the new split (printed, never saved)
    v1.1-fenwick     v1 features + is_post_rebound on the new split

Test AUC is not comparable with leaderboard rows tested on 2024-25.

Usage (repo root, with the project venv's python):
    .venv/bin/python3 -m model.train_v1_1_fenwick
    .venv/bin/python3 -m model.train_v1_1_fenwick --dry-run    # train and print, write nothing
"""
from __future__ import annotations

import argparse
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc as sk_auc,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

from model import train_v1 as v1
from model import train_v1_fenwick as v1f
from model.reporting import (
    calibration_stats,
    max_abs_gap,
    reliability_table,
    write_results,
)

VERSION = "v1.1-fenwick"
ARTIFACT_STEM = "xg_v1_1_fenwick"

TRAIN_SEASONS = [20222023, 20232024]
VAL_SEASON = 20242025
TEST_SEASON = 20252026

NUMERIC_FEATURES = list(v1.NUMERIC_FEATURES)
BOOLEAN_FEATURES = list(v1.BOOLEAN_FEATURES) + ["is_post_rebound"]
CATEGORICAL_FEATURES = list(v1.CATEGORICAL_FEATURES)
ALL_FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
TARGET = v1.TARGET

# Identical to model.train_v1.train_model.
HYPERPARAMS = dict(
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

DESCRIPTION = (
    "v1-fenwick plus is_post_rebound (a rebound off a post or the crossbar), "
    "trained on 2022-23 + 2023-24, validated on 2024-25, tested on 2025-26."
)


def load_features() -> pd.DataFrame:
    df = v1f.load_features()
    df["is_post_rebound"] = df["is_post_rebound"].astype(int)
    return df


def split(df: pd.DataFrame):
    train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    val = df[df["season"] == VAL_SEASON].copy()
    test = df[df["season"] == TEST_SEASON].copy()
    for name, part in (("Train", train), ("Val", val), ("Test", test)):
        seasons = sorted(part["season"].unique().tolist())
        print(f"{name + ':':6} seasons {seasons}, {len(part):,} attempts, "
              f"{part[TARGET].mean() * 100:.2f}% goals")
    if min(len(train), len(val), len(test)) == 0:
        raise RuntimeError("An empty split: rebuild shot_features_fenwick with 2025-26 first.")
    return train, val, test


def train_model(train, val, features, categorical):
    model = lgb.LGBMClassifier(**HYPERPARAMS)
    print(f"\nTraining LightGBM on {len(features)} features "
          "(early stopping on VALIDATION, test untouched)...")
    model.fit(
        train[features], train[TARGET],
        eval_set=[(val[features], val[TARGET])],
        eval_names=["val"],
        categorical_feature=categorical,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    print(f"Best iteration: {model.best_iteration_}")
    return model


def metrics(y, p) -> dict:
    pr, rc, _ = precision_recall_curve(y, p)
    out = {
        "auc": roc_auc_score(y, p),
        "pr_auc": sk_auc(rc, pr),
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        "n": int(len(y)),
        "goal_rate": float(np.mean(y)),
    }
    out.update(calibration_stats(np.asarray(y), np.asarray(p)))
    return out


def print_metrics(label: str, m: dict) -> None:
    print(f"  {label:26} AUC {m['auc']:.4f}  log loss {m['log_loss']:.4f}  "
          f"Brier {m['brier']:.4f}  O/E {m['o_e']:.3f}  slope {m['cal_slope']:.3f}")


def permutation_importance(model, test, n_repeats=3, seed=42) -> pd.DataFrame:
    """AUC drop on TEST when each feature is shuffled (categories preserved)."""
    rng = np.random.default_rng(seed)
    X, y = test[ALL_FEATURES].copy(), test[TARGET]
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    rows = []
    for col in ALL_FEATURES:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            vals = rng.permutation(Xp[col].values)
            Xp[col] = (pd.Categorical(vals, categories=X[col].cat.categories)
                       if col in CATEGORICAL_FEATURES else vals)
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        rows.append({"feature": col, "auc_drop_mean": np.mean(drops),
                     "auc_drop_std": np.std(drops)})
    return pd.DataFrame(rows).sort_values("auc_drop_mean", ascending=False)


SEGMENTS = dict(v1f.SEGMENTS)
SEGMENTS["last_event"] = lambda d: d["last_event_type"].astype(str)
SEGMENTS["post_rebound"] = lambda d: pd.Series(
    np.where(d["is_post_rebound"] == 1, "post rebound", "other"), index=d.index)


def segment_report(val, p_val, test, p_test) -> pd.DataFrame:
    """O/E per segment, val vs test, with last_event_type and post rebounds added."""
    frames = []
    for dim, fn in SEGMENTS.items():
        sv = v1f.segment_calibration(val[TARGET], p_val, fn(val)).set_index("segment")
        st = v1f.segment_calibration(test[TARGET], p_test, fn(test)).set_index("segment")
        tbl = pd.DataFrame({"val_n": sv["n"], "val_oe": sv["o_e"],
                            "test_n": st["n"], "test_oe": st["o_e"]})
        tbl.index.name = "segment"
        tbl["val_n"] = tbl["val_n"].astype("Int64")
        tbl["test_n"] = tbl["test_n"].astype("Int64")
        tbl = tbl.sort_values("test_n", ascending=False)
        print(f"\n  by {dim}")
        print(tbl.to_string(float_format=lambda v: f"{v:.3f}"))
        frames.append(tbl.reset_index().assign(dimension=dim))
    return pd.concat(frames, ignore_index=True)[
        ["dimension", "segment", "val_n", "val_oe", "test_n", "test_oe"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="v1.1-fenwick: + is_post_rebound, 4-season split")
    ap.add_argument("--dry-run", action="store_true", help="Train and print; write nothing.")
    args = ap.parse_args()

    print("=" * 60)
    print(f"{VERSION}: v1 features + is_post_rebound on {v1f.TABLE}")
    print("=" * 60)

    df = load_features()
    v1.assert_no_leakage(df)
    train, val, test = split(df)

    model = train_model(train, val, ALL_FEATURES, CATEGORICAL_FEATURES)
    preds = {name: model.predict_proba(part[ALL_FEATURES])[:, 1]
             for name, part in (("train", train), ("val", val), ("test", test))}
    splits = {name: metrics(part[TARGET].values, preds[name])
              for name, part in (("train", train), ("val", val), ("test", test))}

    print("\n--- v1.1-fenwick by split ---")
    for name in ("train", "val", "test"):
        print_metrics(f"{name} ({splits[name]['n']:,})", splits[name])

    print("\n--- Same test season (2025-26), three models ---")
    old = joblib.load(v1.ARTIFACT_DIR / f"{v1f.ARTIFACT_STEM}.pkl")
    print_metrics("v1-fenwick (committed)",
                  metrics(test[TARGET].values,
                          old.predict_proba(test[v1.ALL_FEATURES])[:, 1]))
    ablation = train_model(train, val, v1.ALL_FEATURES, v1.CATEGORICAL_FEATURES)
    print_metrics("ablation (no post flag)",
                  metrics(test[TARGET].values,
                          ablation.predict_proba(test[v1.ALL_FEATURES])[:, 1]))
    print_metrics("v1.1-fenwick", splits["test"])

    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    print("\n--- Reliability on VAL ---")
    rel_val = reliability_table(val[TARGET], preds["val"])
    print(rel_val.to_string(index=False, float_format=fmt))
    print(f"Max gap (val): {max_abs_gap(rel_val):.4f}")

    print("\n--- Reliability on TEST ---")
    rel = reliability_table(test[TARGET], preds["test"])
    print(rel.to_string(index=False, float_format=fmt))
    max_gap = max_abs_gap(rel)
    print(f"Max gap (test): {max_gap:.4f}")

    print("\n--- O/E by segment (val = 2024-25, test = 2025-26; 1.000 is right on average) ---")
    seg = segment_report(val, preds["val"], test, preds["test"])

    print("\n--- LightGBM gain importance ---")
    gain = (pd.DataFrame({"feature": ALL_FEATURES, "gain": model.feature_importances_})
              .sort_values("gain", ascending=False))
    print(gain.to_string(index=False))

    print("\n--- Permutation importance on TEST ---")
    perm = permutation_importance(model, test)
    print(perm.to_string(index=False, float_format=fmt))

    if args.dry_run:
        print("\n--dry-run: no files written.")
        return

    best_iter = int(model.best_iteration_ or 0)
    joblib.dump(model, v1.ARTIFACT_DIR / f"{ARTIFACT_STEM}.pkl")
    seasons = [int(s) for s in sorted(df["season"].unique())]
    meta = {
        "version": VERSION,
        "table": v1f.TABLE,
        "features": {"numeric": NUMERIC_FEATURES, "boolean": BOOLEAN_FEATURES,
                     "categorical": CATEGORICAL_FEATURES},
        "categorical_categories": {col: sorted(str(c) for c in df[col].cat.categories)
                                   for col in CATEGORICAL_FEATURES},
        "split": {"train": TRAIN_SEASONS, "val": VAL_SEASON, "test": TEST_SEASON},
        "seasons": seasons,
        "best_iteration": best_iter,
        "splits": splits,
        "calibration_max_gap": max_gap,
    }
    with open(v1.ARTIFACT_DIR / f"{ARTIFACT_STEM}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nModel and meta saved to {v1.ARTIFACT_DIR}/{ARTIFACT_STEM}*")

    results_dir = write_results(
        version=VERSION,
        description=DESCRIPTION,
        splits=splits,
        reliability=rel,
        feature_importance=gain,
        hyperparameters={"best_iteration": best_iter, "settings": "identical to v1"},
        features=ALL_FEATURES,
        data_summary={
            "population": "fenwick",
            "table": v1f.TABLE,
            "n_shots": int(len(df)),
            "seasons": seasons,
            "split": {"train": TRAIN_SEASONS, "val": VAL_SEASON, "test": TEST_SEASON},
        },
        notes=("Tested on 2025-26, not 2024-25: test AUC is not comparable with "
               "earlier leaderboard rows. Compare against the three-model block "
               "in results/v1_1_fenwick_train.log instead."),
    )
    seg.to_csv(results_dir / "segments.csv", index=False)
    perm.to_csv(results_dir / "perm_importance.csv", index=False)
    print(f"Segment and permutation tables saved to {results_dir}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
