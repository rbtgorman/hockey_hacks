"""v1.3-fenwick: handle the 2025-26 tip-in labelling change, choose the fix without 2026-27.

WHY THIS EXISTS
---------------
From 2025-26 the NHL labels many more shots as tip-ins (results/checks/tipins.txt:
30 of 32 rinks log more close tips per game; tip-in goals / xG 1.03 in 2022-23,
0.79 in 2025-26, 0.78 in 2026-27 to date). v1.2 never trained on 2025-26.

Two ways to handle it, and the owner asked to try both and keep the better one:

    recency   sample weights 0.5 ** (seasons_old / half_life), newest season = 1.
              Half-lives 1 and 2 are both tried; the better one represents this option.
    era       a boolean feature, new_label_era = season >= 2025-26.

A plain retrain (v1.2's recipe on the same data, no weights, no flag) runs too, as a
reference: if neither fix beats it, that matters.

HOW THE WINNER IS CHOSEN (2026-27 NEVER LOOKED AT)
--------------------------------------------------
    select train  2022-23, 2023-24, 2024-25, and 2025-26 games before the season's
                  median game date. The era flag needs new-era rows to learn from.
    select val    2025-26 games on or after that date.
Each candidate early-stops on select val (that fixes its tree count). The winner has
the lowest log loss on select val; AUC, O/E and tip-in O/E are shown alongside.

FINAL MODEL
-----------
The winner is refit on all of 2022-23 to 2025-26 with its tree count and tested on
2026-27 to date. Only the winner is scored on 2026-27. The saved v1.2 model is scored
there too, as a reference: it was already chosen, so this does not pick anything.

Usage (repo root, with the project venv's python):
    .venv/bin/python3 -m model.train_v1_3_fenwick
    .venv/bin/python3 -m model.train_v1_3_fenwick --dry-run    # train and print, write nothing
"""
from __future__ import annotations

import argparse
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from model import train_v1 as v1
from model import train_v1_fenwick as v1f
from model import train_v1_1_fenwick as v11
from model import train_v1_2_fenwick as v12
from model.reporting import max_abs_gap, reliability_table, write_results

VERSION = "v1.3-fenwick"
ARTIFACT_STEM = "xg_v1_3_fenwick"

TRAIN_SEASONS = [20222023, 20232024, 20242025, 20252026]
SPLIT_SEASON = 20252026          # halved by game date for selection
TEST_SEASON = 20262027
ERA_START = 20252026
TARGET = v12.TARGET
HALF_LIVES = (1.0, 2.0)

BASE_NUMERIC = list(v12.NUMERIC_FEATURES)
BASE_BOOLEAN = list(v12.BOOLEAN_FEATURES)
CATEGORICAL_FEATURES = list(v12.CATEGORICAL_FEATURES)
BASE_FEATURES = BASE_NUMERIC + BASE_BOOLEAN + CATEGORICAL_FEATURES
ERA_FEATURES = BASE_NUMERIC + BASE_BOOLEAN + ["new_label_era"] + CATEGORICAL_FEATURES

# Set by main() once the winner is known, so model.score_fenwick can import
# load_features / ALL_FEATURES from here. Rewritten from the saved meta file.
ALL_FEATURES = BASE_FEATURES


def _features_from_meta() -> list[str]:
    try:
        with open(v1.ARTIFACT_DIR / f"{ARTIFACT_STEM}_meta.json") as f:
            meta = json.load(f)["features"]
        return meta["numeric"] + meta["boolean"] + meta["categorical"]
    except FileNotFoundError:
        return BASE_FEATURES


ALL_FEATURES = _features_from_meta()


def load_features() -> pd.DataFrame:
    df = v12.load_features()
    df["new_label_era"] = (df["season"] >= ERA_START).astype(int)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def recency_weights(seasons: pd.Series, half_life: float) -> np.ndarray:
    order = {s: i for i, s in enumerate(TRAIN_SEASONS)}
    age = len(TRAIN_SEASONS) - 1 - seasons.map(order).to_numpy()
    return 0.5 ** (age / half_life)


CANDIDATES = {
    "plain retrain": dict(features=BASE_FEATURES, half_life=None),
    "recency h=1": dict(features=BASE_FEATURES, half_life=1.0),
    "recency h=2": dict(features=BASE_FEATURES, half_life=2.0),
    "era flag": dict(features=ERA_FEATURES, half_life=None),
}
FAMILY = {"plain retrain": "reference", "recency h=1": "recency",
          "recency h=2": "recency", "era flag": "era"}


def fit(train, spec, n_estimators, val=None):
    model = lgb.LGBMClassifier(**dict(v11.HYPERPARAMS, n_estimators=n_estimators))
    kwargs = {"categorical_feature": CATEGORICAL_FEATURES}
    if spec["half_life"] is not None:
        kwargs["sample_weight"] = recency_weights(train["season"], spec["half_life"])
    if val is not None:
        kwargs.update(
            eval_set=[(val[spec["features"]], val[TARGET])],
            eval_names=["val"],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
    model.fit(train[spec["features"]], train[TARGET], **kwargs)
    return model


def tipin_oe(d, p) -> float:
    m = d["shot_type"].astype(str).str.lower().str.contains("tip")
    return float(d.loc[m, TARGET].sum() / p[m.to_numpy()].sum())


def main() -> None:
    ap = argparse.ArgumentParser(description="v1.3-fenwick: recency weights vs era flag")
    ap.add_argument("--dry-run", action="store_true", help="Train and print; write nothing.")
    args = ap.parse_args()

    print("=" * 60)
    print(f"{VERSION}: recency weighting vs an era flag, chosen without 2026-27")
    print("=" * 60)

    df = load_features()
    v1.assert_no_leakage(df)
    split_season = df[df["season"] == SPLIT_SEASON]
    cutoff = split_season["game_date"].median().normalize()
    before = df["season"].isin(TRAIN_SEASONS[:-1]) | (
        (df["season"] == SPLIT_SEASON) & (df["game_date"] < cutoff))
    parts = {
        "select_train": df[before],
        "select_val": df[(df["season"] == SPLIT_SEASON) & (df["game_date"] >= cutoff)],
        "refit_train": df[df["season"].isin(TRAIN_SEASONS)],
        "test": df[df["season"] == TEST_SEASON],
    }
    print(f"2025-26 split at {cutoff}")
    for name, part in parts.items():
        print(f"{name + ':':14} {part['game_date'].min()} to {part['game_date'].max()}, "
              f"{len(part):,} attempts, {part[TARGET].mean() * 100:.2f}% goals")
    if min(len(p) for p in parts.values()) == 0:
        raise RuntimeError("An empty split.")
    sel_train, sel_val, test = parts["select_train"], parts["select_val"], parts["test"]

    print("\n--- Step 1: every candidate on select val (late 2025-26). 2026-27 untouched. ---")
    rows, trees = [], {}
    for name, spec in CANDIDATES.items():
        m = fit(sel_train, spec, v11.HYPERPARAMS["n_estimators"], val=sel_val)
        trees[name] = int(m.best_iteration_)
        p = m.predict_proba(sel_val[spec["features"]])[:, 1]
        met = v11.metrics(sel_val[TARGET].values, p)
        rows.append({"candidate": name, "family": FAMILY[name], "n_trees": trees[name],
                     "log_loss": met["log_loss"], "auc": met["auc"], "brier": met["brier"],
                     "o_e": met["o_e"], "cal_slope": met["cal_slope"],
                     "tipin_o_e": tipin_oe(sel_val, p)})
    sel = pd.DataFrame(rows)
    print(sel.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    finalists = sel[sel["family"] != "reference"].sort_values("log_loss")
    winner = finalists.iloc[0]["candidate"]
    ref = sel.set_index("candidate").loc["plain retrain", "log_loss"]
    print(f"\nWinner: {winner} (log loss {finalists.iloc[0]['log_loss']:.5f}; "
          f"plain retrain {ref:.5f}; runner-up {finalists.iloc[1]['candidate']} "
          f"{finalists.iloc[1]['log_loss']:.5f})")
    spec, n_trees = CANDIDATES[winner], trees[winner]

    print(f"\n--- Step 2: refit {winner} on 2022-26 with {n_trees} trees, test on 2026-27 ---")
    feats = spec["features"]
    model = fit(parts["refit_train"], spec, n_trees)
    p_train = model.predict_proba(parts["refit_train"][feats])[:, 1]
    p_test = model.predict_proba(test[feats])[:, 1]
    old = joblib.load(v1.ARTIFACT_DIR / f"{v12.ARTIFACT_STEM}.pkl")
    p_old = old.predict_proba(test[v12.ALL_FEATURES])[:, 1]
    splits = {
        "train": v11.metrics(parts["refit_train"][TARGET].values, p_train),
        "val": v11.metrics(sel_val[TARGET].values,
                           fit(sel_train, spec, n_trees).predict_proba(sel_val[feats])[:, 1]),
        "test": v11.metrics(test[TARGET].values, p_test),
    }
    v11.print_metrics(f"{VERSION} test", splits["test"])
    v11.print_metrics("v1.2-fenwick (saved) test", v11.metrics(test[TARGET].values, p_old))
    print(f"  tip-in O/E on 2026-27: {VERSION} {tipin_oe(test, p_test):.3f}, "
          f"v1.2 {tipin_oe(test, p_old):.3f}")

    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    rel = reliability_table(test[TARGET], p_test)
    print("\n--- Reliability on 2026-27 ---")
    print(rel.to_string(index=False, float_format=fmt))
    max_gap = max_abs_gap(rel)

    print("\n--- O/E by segment on 2026-27 ---")
    frames = []
    for dim, fn in v12.SEGMENTS.items():
        seg = fn(test)
        a = v1f.segment_calibration(test[TARGET], p_test, seg).set_index("segment")
        b = v1f.segment_calibration(test[TARGET], p_old, seg).set_index("segment")
        tbl = pd.DataFrame({"n": a["n"], "oe_v1.3": a["o_e"], "oe_v1.2": b["o_e"]})
        tbl.index.name = "segment"
        tbl = tbl.sort_values("n", ascending=False)
        print(f"\n  by {dim}")
        print(tbl.to_string(float_format=lambda v: f"{v:.3f}"))
        frames.append(tbl.reset_index().assign(dimension=dim))
    segs = pd.concat(frames, ignore_index=True)

    gain = (pd.DataFrame({"feature": feats, "gain": model.feature_importances_})
              .sort_values("gain", ascending=False))
    print("\n--- LightGBM gain importance ---")
    print(gain.to_string(index=False))

    if args.dry_run:
        print("\n--dry-run: no files written.")
        return

    joblib.dump(model, v1.ARTIFACT_DIR / f"{ARTIFACT_STEM}.pkl")
    num = BASE_NUMERIC
    boolean = BASE_BOOLEAN + (["new_label_era"] if "new_label_era" in feats else [])
    split_desc = {"select_train": f"2022-23 to 2024-25 + 2025-26 before {cutoff}",
                  "select_val": f"2025-26 from {cutoff}",
                  "refit_train": TRAIN_SEASONS, "test": TEST_SEASON}
    meta = {
        "version": VERSION, "table": v1f.TABLE, "winner": winner,
        "half_life": spec["half_life"],
        "features": {"numeric": num, "boolean": boolean, "categorical": CATEGORICAL_FEATURES},
        "split": split_desc, "n_trees": n_trees, "splits": splits,
        "calibration_max_gap": max_gap,
    }
    with open(v1.ARTIFACT_DIR / f"{ARTIFACT_STEM}_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    how = (f"recency weights, half-life {spec['half_life']:g} seasons"
           if spec["half_life"] else "an era flag (new_label_era, 2025-26 on)"
           if "new_label_era" in feats else "no weights or flag")
    results_dir = write_results(
        version=VERSION,
        description=(f"v1.2-fenwick features, trained on 2022-23 to 2025-26 with {how}; "
                     "chosen over the other tip-in fix on late 2025-26; tested on 2026-27."),
        splits=splits, reliability=rel, feature_importance=gain,
        hyperparameters={"n_trees": n_trees, "half_life": spec["half_life"],
                         "settings": "v1's; trees by early stopping on late 2025-26, then refit"},
        features=feats,
        data_summary={"population": "fenwick", "table": v1f.TABLE, "n_shots": int(len(df)),
                      "seasons": [int(s) for s in sorted(df["season"].unique())],
                      "split": split_desc},
        notes=("Test is 2026-27 to date, not 2025-26, so test numbers are not comparable "
               "with earlier rows. 'val' is late 2025-26 with the selection fit."),
    )
    sel.to_csv(results_dir / "selection.csv", index=False)
    segs.to_csv(results_dir / "segments.csv", index=False)
    print(f"\nSaved model to {v1.ARTIFACT_DIR}/{ARTIFACT_STEM}* and results to {results_dir}/")


if __name__ == "__main__":
    main()
