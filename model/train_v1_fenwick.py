"""v1-fenwick: the v1 model, retrained on unblocked attempts.

WHAT CHANGES FROM v1
--------------------
Only the data. Same features, same hyperparameters, same training loop, same
three-way temporal split (train 2022-23, val 2023-24, test 2024-25). The
training and evaluation functions are imported from model.train_v1, so the
two runs cannot drift apart, and train_v1.py itself is untouched: it still
reproduces the shots-on-goal leaderboard.

    data   shot_features_fenwick  (features/build_features_fenwick.py)
           instead of shot_features

That table adds missed shots, drops shootout attempts, removes two event
types logged in only part of the window, and merges shot-on-goal and
missed-shot in last_event_type. The builder's docstring has the evidence.

WHAT TO LOOK FOR
----------------
  - Test O/E near 1.00. The shots-on-goal v2.3 ran at 1.070 on 2024-25,
    because the NHL started recording goalie-touched wide pucks as misses.
  - Calibration slope below 1. v1's settings overfit (v2.3's log shows
    train AUC 0.859 against val 0.778); on the shots-on-goal test, that
    over-confidence was hiding behind the drift in the top decile.
  - AUC and log loss are not comparable with the shots-on-goal leaderboard.

Usage:
    python3 -m model.train_v1_fenwick
    python3 -m model.train_v1_fenwick --dry-run    # train and print, write nothing
"""
from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

from ingest.db import pg_conn
from model import train_v1 as v1
from model.reporting import (
    calibration_stats,
    max_abs_gap,
    reliability_table,
    write_results,
)

VERSION = "v1-fenwick"
TABLE = "shot_features_fenwick"
ARTIFACT_STEM = "xg_v1_fenwick"

DESCRIPTION = (
    "v1 features and hyperparameters retrained on unblocked attempts "
    "(shot_features_fenwick). Removes the 2023-24 on-goal relabel from the "
    "target population."
)


def load_features() -> pd.DataFrame:
    query = f"""
        SELECT f.*, g.season AS season, g.game_date AS game_date
        FROM {TABLE} f
        JOIN games g ON g.game_id = f.game_id
        ORDER BY g.game_date, f.game_id, f.event_idx
    """
    with pg_conn() as conn:
        df = pd.read_sql(query, conn)

    for col in v1.BOOLEAN_FEATURES + [v1.TARGET]:
        df[col] = df[col].astype(int)
    for col in v1.CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    for col in v1.NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Loaded {len(df):,} unblocked attempts from {TABLE}, "
          f"{df['game_date'].min()} to {df['game_date'].max()}")
    print(f"Seasons present: {sorted(df['season'].unique().tolist())}")
    mix = (df.groupby("season")["event_type"]
             .value_counts(normalize=True)
             .unstack()
             .round(3))
    print("\nEvent mix by season (share of rows):")
    print(mix.to_string())
    return df


def split_metrics(model, df: pd.DataFrame, label: str):
    """v1's metrics plus O/E and calibration slope, in the reporting schema."""
    m = v1.evaluate(model, df, label)
    p = model.predict_proba(df[v1.ALL_FEATURES])[:, 1]
    cal = calibration_stats(df[v1.TARGET].values, p)
    print(f"  O/E:       {cal['o_e']:.4f}  (observed {cal['observed_rate']:.4f}, "
          f"predicted {cal['mean_pred']:.4f})")
    print(f"  Cal slope: {cal['cal_slope']:.4f}  (intercept {cal['cal_intercept']:.4f})")
    out = {k: v for k, v in m.items() if k != "base_rate"}
    out["goal_rate"] = m["base_rate"]
    out.update(cal)
    return out, p


def main() -> None:
    ap = argparse.ArgumentParser(description="v1 model on unblocked attempts")
    ap.add_argument("--dry-run", action="store_true",
                    help="Train and print; write nothing.")
    args = ap.parse_args()

    print("=" * 60)
    print(f"{VERSION}: v1 features + hyperparameters on {TABLE}")
    print("=" * 60)

    df = load_features()
    v1.assert_no_leakage(df)
    train, val, test = v1.split_three_way(df)
    model = v1.train_model(train, val)

    splits, preds = {}, {}
    for name, part, label in (
        ("train", train, "Train (sanity only)"),
        ("val", val, "Validation (used for early stop)"),
        ("test", test, "Test (held out, touch once)"),
    ):
        splits[name], preds[name] = split_metrics(model, part, label)

    fmt = lambda v: f"{v:.4f}"  # noqa: E731

    print("\n--- Reliability on VAL ---")
    rel_val = reliability_table(val[v1.TARGET], preds["val"])
    print(rel_val.to_string(index=False, float_format=fmt))
    print(f"Max gap (val): {max_abs_gap(rel_val):.4f}")

    print("\n--- Reliability on TEST ---")
    rel = reliability_table(test[v1.TARGET], preds["test"])
    print(rel.to_string(index=False, float_format=fmt))
    max_gap = max_abs_gap(rel)
    print(f"Max gap (test): {max_gap:.4f}")

    print("\n--- LightGBM gain importance ---")
    gain = (pd.DataFrame({"feature": v1.ALL_FEATURES,
                          "gain": model.feature_importances_})
              .sort_values("gain", ascending=False))
    print(gain.to_string(index=False))

    print("\n" + "=" * 60)
    print("Shots-on-goal reference (different population, not comparable):")
    print("  v1    test AUC 0.7705  max gap 0.0244")
    print("  v2.3  test AUC 0.7707  max gap 0.0167  O/E 1.070")
    print("=" * 60)

    results_dir = None
    if args.dry_run:
        print("\n--dry-run: no files written.")
    else:
        best_iter = int(model.best_iteration_ or 0)
        joblib.dump(model, v1.ARTIFACT_DIR / f"{ARTIFACT_STEM}.pkl")
        meta = {
            "version": VERSION,
            "table": TABLE,
            "features": {
                "numeric": v1.NUMERIC_FEATURES,
                "boolean": v1.BOOLEAN_FEATURES,
                "categorical": v1.CATEGORICAL_FEATURES,
            },
            "categorical_categories": {
                col: sorted(str(c) for c in df[col].cat.categories)
                for col in v1.CATEGORICAL_FEATURES
            },
            "seasons": [int(s) for s in sorted(df["season"].unique())],
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
            hyperparameters={"best_iteration": best_iter,
                             "settings": "identical to v1"},
            features=v1.ALL_FEATURES,
            data_summary={
                "population": "fenwick",
                "table": TABLE,
                "n_shots": int(len(df)),
                "seasons": [int(s) for s in sorted(df["season"].unique())],
            },
            notes=(
                "Reference point for the Fenwick population. Compare later "
                "Fenwick versions against this, not against the shots-on-goal v1."
            ),
        )

    print("\n--- Permutation importance on TEST (what the model relies on) ---")
    try:
        perm, _ = v1.permutation_importance_on_test(model, test)
        print(perm.to_string(index=False, float_format=fmt))
        if results_dir is not None:
            perm.to_csv(results_dir / "perm_importance.csv", index=False)
            print(f"Saved to {results_dir}/perm_importance.csv")
    except Exception as e:
        print(f"Permutation importance failed (model already saved): "
              f"{type(e).__name__}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()