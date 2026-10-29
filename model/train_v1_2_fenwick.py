"""v1.2-fenwick: v1.1 plus last_event_by_shooting_team, refit to include 2024-25.

WHY THIS EXISTS
---------------
v1.1 over-predicts shots that follow a giveaway: observed/expected 0.907 on
2024-25 and 0.908 on 2025-26. Splitting those shots by whose giveaway it was
(features/build_features_fenwick.py check 10) shows two causes:

    recording  Giveaways per game about doubled in 2024-25 (17.3 / 15.4 ->
               30.0 / 30.8) while the own/opponent mix barely moved (75/25
               -> 72/28). After an opponent's giveaway the goal rate fell from
               10.1% / 9.6% to 6.6% / 7.4%: the label now covers many routine
               turnovers. v1.1 never trained on that regime, so it expects the
               old danger: O/E 0.85 (2024-25) and 0.88 (2025-26) on those shots.
    team       last_event_type does not say whose event it was. Even in v1.1's
               training seasons, shots after the shooting team's own giveaway
               ran 0.80-0.95 of expected, against 1.05-1.07 after the opponent's.

WHAT CHANGES FROM v1.1-fenwick
------------------------------
    features  v1.1's sixteen + last_event_by_shooting_team (build docstring item 9)
    training  two steps, so the model learns from 2024-25 without touching the test:
                1. select  train 2022-23 + 2023-24, early-stop on 2024-25 -> N trees
                2. refit   train 2022-23 + 2023-24 + 2024-25 with exactly N trees
              The saved model is the refit. Test stays 2025-26, and 2026-27 to
              date is a second, forward test (small: about a month of games).
    results   "val" in metrics.json is the select step on 2024-25, since the
              refit trains on that season.

Hyperparameters are v1's (model.train_v1_1_fenwick.HYPERPARAMS); only the
number of trees comes from step 1.

COMPARISON ON THE SAME TEST SEASONS
-----------------------------------
    v1.1-fenwick   the saved .pkl, not retrained
    ablation       v1.1 features, refit on 2022-25 with v1.1's tree count (printed, never saved)
    v1.2-fenwick   v1.1 features + last_event_by_shooting_team, refit on 2022-25
v1.1 -> ablation is the effect of training on 2024-25; ablation -> v1.2 is the
effect of the new feature.

Usage (repo root, with the project venv's python):
    .venv/bin/python3 -m model.train_v1_2_fenwick
    .venv/bin/python3 -m model.train_v1_2_fenwick --dry-run    # train and print, write nothing
"""
from __future__ import annotations

import argparse
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from model import train_v1 as v1
from model import train_v1_fenwick as v1f
from model import train_v1_1_fenwick as v11
from model.reporting import max_abs_gap, reliability_table, write_results

VERSION = "v1.2-fenwick"
ARTIFACT_STEM = "xg_v1_2_fenwick"

SELECT_TRAIN = [20222023, 20232024]
SELECT_VAL = 20242025
REFIT_TRAIN = SELECT_TRAIN + [SELECT_VAL]
TEST_SEASON = 20252026
FORWARD_SEASON = 20262027

NUMERIC_FEATURES = list(v11.NUMERIC_FEATURES)
BOOLEAN_FEATURES = list(v11.BOOLEAN_FEATURES) + ["last_event_by_shooting_team"]
CATEGORICAL_FEATURES = list(v11.CATEGORICAL_FEATURES)
ALL_FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
TARGET = v11.TARGET

DESCRIPTION = (
    "v1.1-fenwick plus last_event_by_shooting_team (whose event preceded the shot); "
    "trees chosen on 2024-25, then refit on 2022-23 to 2024-25; tested on 2025-26."
)


def load_features() -> pd.DataFrame:
    df = v11.load_features()
    df["last_event_by_shooting_team"] = df["last_event_by_shooting_team"].astype(int)
    return df


def fit(train, features, categorical, n_estimators, val=None):
    """v1's hyperparameters. Early stopping only when val is given."""
    model = lgb.LGBMClassifier(**dict(v11.HYPERPARAMS, n_estimators=n_estimators))
    kwargs = {"categorical_feature": categorical}
    if val is not None:
        kwargs.update(
            eval_set=[(val[features], val[TARGET])],
            eval_names=["val"],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
                       lgb.log_evaluation(period=100)],
        )
    model.fit(train[features], train[TARGET], **kwargs)
    return model


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


SEGMENTS = dict(v11.SEGMENTS)
SEGMENTS["turnover"] = lambda d: pd.Series(
    np.where(d["last_event_type"].astype(str).isin(["giveaway", "takeaway"]),
             d["last_event_type"].astype(str)
             + np.where(d["last_event_by_shooting_team"] == 1,
                        " by shooting team", " by opponent"),
             "other"),
    index=d.index)


def segment_report(test, test_preds: dict, fwd, p_fwd) -> pd.DataFrame:
    """O/E per segment: every model on 2025-26, and v1.2 on 2026-27 to date."""
    frames = []
    for dim, fn in SEGMENTS.items():
        seg_test = fn(test)
        cols = {}
        for label, p in test_preds.items():
            s = v1f.segment_calibration(test[TARGET], p, seg_test).set_index("segment")
            cols.setdefault("test_n", s["n"].astype("Int64"))
            cols[f"oe_{label}"] = s["o_e"]
        s = v1f.segment_calibration(fwd[TARGET], p_fwd, fn(fwd)).set_index("segment")
        cols["fwd_n"] = s["n"].astype("Int64")
        cols["fwd_oe_v1.2"] = s["o_e"]
        tbl = pd.DataFrame(cols)
        tbl.index.name = "segment"
        tbl = tbl.sort_values("test_n", ascending=False)
        print(f"\n  by {dim}")
        print(tbl.to_string(float_format=lambda v: f"{v:.3f}"))
        frames.append(tbl.reset_index().assign(dimension=dim))
    out = pd.concat(frames, ignore_index=True)
    return out[["dimension"] + [c for c in out.columns if c != "dimension"]]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="v1.2-fenwick: + last_event_by_shooting_team, refit on 2022-25")
    ap.add_argument("--dry-run", action="store_true", help="Train and print; write nothing.")
    args = ap.parse_args()

    print("=" * 60)
    print(f"{VERSION}: v1.1 features + last_event_by_shooting_team on {v1f.TABLE}")
    print("=" * 60)

    df = load_features()
    v1.assert_no_leakage(df)
    parts = {
        "select_train": df[df["season"].isin(SELECT_TRAIN)],
        "val": df[df["season"] == SELECT_VAL],
        "refit_train": df[df["season"].isin(REFIT_TRAIN)],
        "test": df[df["season"] == TEST_SEASON],
        "forward": df[df["season"] == FORWARD_SEASON],
    }
    for name, part in parts.items():
        print(f"{name + ':':14} seasons {sorted(part['season'].unique().tolist())}, "
              f"{len(part):,} attempts, {part[TARGET].mean() * 100:.2f}% goals")
    if min(len(part) for part in parts.values()) == 0:
        raise RuntimeError("An empty split: ingest 2026-27 and rebuild shot_features_fenwick first.")
    val, test, fwd = parts["val"], parts["test"], parts["forward"]

    print("\n--- Step 1: choose the tree count (train 2022-24, early stop on 2024-25) ---")
    select = fit(parts["select_train"], ALL_FEATURES, CATEGORICAL_FEATURES,
                 v11.HYPERPARAMS["n_estimators"], val=val)
    n_trees = int(select.best_iteration_)
    print(f"Best iteration: {n_trees}")
    p_select_val = select.predict_proba(val[ALL_FEATURES])[:, 1]
    v11.print_metrics("select, val 2024-25", v11.metrics(val[TARGET].values, p_select_val))
    v11.print_metrics("select, test 2025-26",
                      v11.metrics(test[TARGET].values,
                                  select.predict_proba(test[ALL_FEATURES])[:, 1]))

    print(f"\n--- Step 2: refit on 2022-25 with {n_trees} trees ---")
    model = fit(parts["refit_train"], ALL_FEATURES, CATEGORICAL_FEATURES, n_trees)
    preds = {name: model.predict_proba(parts[name][ALL_FEATURES])[:, 1]
             for name in ("refit_train", "test", "forward")}
    splits = {
        "train": v11.metrics(parts["refit_train"][TARGET].values, preds["refit_train"]),
        "val": v11.metrics(val[TARGET].values, p_select_val),
        "test": v11.metrics(test[TARGET].values, preds["test"]),
        "forward": v11.metrics(fwd[TARGET].values, preds["forward"]),
    }
    for name, label in (("train", "train 2022-25 (in-sample)"),
                        ("test", "test 2025-26"), ("forward", "forward 2026-27")):
        v11.print_metrics(f"{label} ({splits[name]['n']:,})", splits[name])

    print("\n--- Same test seasons, three models ---")
    old = joblib.load(v1.ARTIFACT_DIR / f"{v11.ARTIFACT_STEM}.pkl")
    with open(v1.ARTIFACT_DIR / f"{v11.ARTIFACT_STEM}_meta.json") as f:
        old_trees = int(json.load(f)["best_iteration"])
    ablation = fit(parts["refit_train"], v11.ALL_FEATURES, v11.CATEGORICAL_FEATURES, old_trees)
    compare = {}
    for name, d in (("test", test), ("forward", fwd)):
        compare[name] = {
            "v1.1": old.predict_proba(d[v11.ALL_FEATURES])[:, 1],
            "ablation": ablation.predict_proba(d[v11.ALL_FEATURES])[:, 1],
            "v1.2": preds[name],
        }
        print(f"\n  {name}: {'2025-26' if name == 'test' else '2026-27 to date'}, "
              f"{len(d):,} attempts")
        for label, p in compare[name].items():
            title = {"v1.1": "v1.1-fenwick (saved)",
                     "ablation": f"ablation (refit, {old_trees} trees)",
                     "v1.2": VERSION}[label]
            v11.print_metrics(title, v11.metrics(d[TARGET].values, p))

    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    print("\n--- Reliability on TEST (2025-26) ---")
    rel = reliability_table(test[TARGET], preds["test"])
    print(rel.to_string(index=False, float_format=fmt))
    max_gap = max_abs_gap(rel)
    print(f"Max gap (test): {max_gap:.4f}")

    print("\n--- O/E by segment (test = 2025-26 for each model; fwd = 2026-27 to date, v1.2) ---")
    seg = segment_report(test, compare["test"], fwd, preds["forward"])

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

    joblib.dump(model, v1.ARTIFACT_DIR / f"{ARTIFACT_STEM}.pkl")
    seasons = [int(s) for s in sorted(df["season"].unique())]
    split_desc = {"select_train": SELECT_TRAIN, "select_val": SELECT_VAL,
                  "refit_train": REFIT_TRAIN, "test": TEST_SEASON, "forward": FORWARD_SEASON}
    meta = {
        "version": VERSION,
        "table": v1f.TABLE,
        "features": {"numeric": NUMERIC_FEATURES, "boolean": BOOLEAN_FEATURES,
                     "categorical": CATEGORICAL_FEATURES},
        "categorical_categories": {col: sorted(str(c) for c in df[col].cat.categories)
                                   for col in CATEGORICAL_FEATURES},
        "split": split_desc,
        "seasons": seasons,
        "n_trees": n_trees,
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
        hyperparameters={"n_trees": n_trees,
                         "settings": "v1's; trees by early stopping on 2024-25, then refit"},
        features=ALL_FEATURES,
        data_summary={
            "population": "fenwick",
            "table": v1f.TABLE,
            "n_shots": int(len(df)),
            "seasons": seasons,
            "split": split_desc,
        },
        notes=("Tested on 2025-26 like v1.1-fenwick, so test AUC is comparable with "
               "that row only. 'val' is the select step on 2024-25 (the refit trains "
               "on it); 'forward' is 2026-27 to date."),
    )
    seg.to_csv(results_dir / "segments.csv", index=False)
    perm.to_csv(results_dir / "perm_importance.csv", index=False)
    print(f"Segment and permutation tables saved to {results_dir}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
