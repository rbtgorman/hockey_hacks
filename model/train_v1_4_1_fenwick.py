"""v1.4.1-fenwick: v1.4's recipe refit on the table with the stoppage leak closed.

WHAT IT IS
----------
The "baseline" from model/train_v1_5_fenwick.py, made into the live model. Same 17
features, frozen settings (results/v1_4-fenwick/selection.json), recency weights
(half-life 1 season) and distance constraint as v1.4, no calibration. The tree count
(1,692) is the one the baseline early-stopped at on late 2025-26 in
results/v1_5-fenwick/selection.json. Refit on 2022-23 to 2025-26.

WHY THERE'S NO SELECTION STAGE
------------------------------
Replacing v1.4 was decided on principle (the owner, 2026-11-04): v1.4 learned from 170
shots whose previous event was a stoppage the shot itself caused (docs/FINDINGS.md). So
this script searches nothing and picks nothing. Its 2026-27 numbers are reported next to
v1.4's and never used to choose.

It rebuilds the selection fit first and stops if that doesn't reproduce the baseline's
late 2025-26 log loss, which would mean the table or the recipe has moved.

Saves the model and results; scores nothing into shot_xg (model.score_fenwick does).

Usage (repo root, with the project venv's python):
    .venv/bin/python3 -m model.train_v1_4_1_fenwick
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from model import train_v1 as v1
from model import train_v1_fenwick as v1f
from model import train_v1_1_fenwick as v11
from model import train_v1_2_fenwick as v12
from model import train_v1_3_fenwick as v13
from model import train_v1_4_fenwick as v14
from model import train_v1_5_fenwick as v15
from model.reporting import max_abs_gap, reliability_table, write_results

VERSION = "v1.4.1-fenwick"
ARTIFACT_STEM = "xg_v1_4_1_fenwick"
TARGET = v14.TARGET
ALL_FEATURES = list(v14.ALL_FEATURES)
HALF_LIFE = v14.HALF_LIFE
SETTINGS = v15.SETTINGS

with open(v15.SELECTION_JSON) as _f:
    _BASELINE = next(c for c in json.load(_f)["candidates"] if c["candidate"] == "baseline")
N_TREES = int(_BASELINE["n_trees"])


def load_features() -> pd.DataFrame:
    """Same rows and columns as v1.4; model.score_fenwick imports this."""
    return v14.load_features()


def main() -> None:
    print("=" * 60)
    print(f"{VERSION}: {N_TREES} trees, no search, no selection")
    print("=" * 60)
    df = load_features()
    v1.assert_no_leakage(df)
    left = int((df["last_event_type"].astype(str) == "stoppage").sum())
    if left:
        raise RuntimeError(f"{left} shots still have a stoppage as their previous event.")
    cutoff, parts = v14.splits(df)
    for name, part in parts.items():
        print(f"{name + ':':14} {part['game_date'].min()} to {part['game_date'].max()}, "
              f"{len(part):,} attempts, {part[TARGET].mean() * 100:.2f}% goals")

    print("\n--- Reproducing the baseline's selection fit ---")
    va = parts["select_val"]
    sel_fit = v14.fit(parts["select_train"], SETTINGS, monotone=True, n_trees=N_TREES)
    p_val = sel_fit.predict_proba(va[ALL_FEATURES])[:, 1]
    val_metrics = v11.metrics(va[TARGET].values, p_val)
    print(f"  late 2025-26 log loss {val_metrics['log_loss']:.6f}, "
          f"baseline on file {_BASELINE['log_loss']:.6f}")
    if abs(val_metrics["log_loss"] - _BASELINE["log_loss"]) > 5e-6:
        raise RuntimeError("The selection fit doesn't match the v1.5 baseline. Stopping.")

    print(f"\n--- Refit on 2022-23 to 2025-26: {N_TREES} trees ---")
    model = v14.fit(parts["refit_train"], SETTINGS, monotone=True, n_trees=N_TREES)
    chk = v14.monotone_check(lambda X: model.predict_proba(X)[:, 1],
                             va.sample(2000, random_state=v14.SEED))
    print(f"  distance check: {chk}")
    if chk["rises"]:
        raise RuntimeError("The refit model's xG rises with distance somewhere.")

    print("\n--- 2026-27 to date, reported only (chooses nothing) ---")
    test = parts["test"]
    y = test[TARGET].to_numpy()
    p_new = model.predict_proba(test[ALL_FEATURES])[:, 1]
    v14_model = joblib.load(v1.ARTIFACT_DIR / f"{v14.ARTIFACT_STEM}.pkl")
    p_old = v14_model.predict_proba(test[ALL_FEATURES])[:, 1]
    m_new, m_old = v14.summary(y, p_new, test), v14.summary(y, p_old, test)
    v14.show(VERSION, m_new)
    v14.show("v1.4-fenwick (saved, on the fixed table)", m_old)
    g = v15.game_bootstrap(y, p_old, p_new, test["game_id"].to_numpy())
    print(f"  log loss change vs v1.4: {g['diff']:+.5f} ({g['lo']:+.5f} to {g['hi']:+.5f})")
    print(f"  goals {int(y.sum()):,}, xG v1.4.1 {p_new.sum():,.1f}, xG v1.4 {p_old.sum():,.1f}")

    rel = reliability_table(test[TARGET], p_new)
    print("\n--- Reliability on 2026-27 ---")
    print(rel.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    frames = []
    for dim, fn in v12.SEGMENTS.items():
        seg = fn(test)
        a = v1f.segment_calibration(test[TARGET], p_new, seg).set_index("segment")
        b = v1f.segment_calibration(test[TARGET], p_old, seg).set_index("segment")
        tbl = pd.DataFrame({"n": a["n"], "oe_v1.4.1": a["o_e"], "oe_v1.4": b["o_e"]})
        tbl.index.name = "segment"
        frames.append(tbl.sort_values("n", ascending=False).reset_index().assign(dimension=dim))
    segs = pd.concat(frames, ignore_index=True)
    print("\n--- O/E by segment on 2026-27 ---")
    print(segs.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    train = parts["refit_train"]
    split_metrics = {
        "train": v11.metrics(train[TARGET].values, model.predict_proba(train[ALL_FEATURES])[:, 1]),
        "val": val_metrics,
        "test": v11.metrics(y, p_new),
    }
    gain = (pd.DataFrame({"feature": ALL_FEATURES,
                          "gain": model.booster_.feature_importance("gain")})
              .sort_values("gain", ascending=False))
    split_desc = {"select_train": f"2022-23 to 2024-25 + 2025-26 before {cutoff}",
                  "select_val": f"2025-26 from {cutoff}",
                  "refit_train": v13.TRAIN_SEASONS, "test": v13.TEST_SEASON}
    joblib.dump(model, v1.ARTIFACT_DIR / f"{ARTIFACT_STEM}.pkl")
    meta = {"version": VERSION, "table": v1f.TABLE, "half_life": HALF_LIFE,
            "features": ALL_FEATURES, "monotone": {"distance_ft": -1}, "settings": SETTINGS,
            "n_trees": N_TREES, "calibration": None, "split": split_desc,
            "splits": split_metrics, "calibration_max_gap": max_abs_gap(rel),
            "distance_check": chk}
    with open(v1.ARTIFACT_DIR / f"{ARTIFACT_STEM}_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    results_dir = write_results(
        version=VERSION,
        description=("v1.4's features, settings, recency weights and distance constraint, "
                     "refit on the table with the stoppage leak closed (170 shots). "
                     "Replaces v1.4 on principle; 2026-27 reported only."),
        splits=split_metrics, reliability=rel, feature_importance=gain,
        hyperparameters={**SETTINGS, "n_trees": N_TREES, "half_life": HALF_LIFE,
                         "monotone": "distance_ft -1", "calibration": None},
        features=ALL_FEATURES,
        data_summary={"population": "fenwick", "table": v1f.TABLE, "n_shots": int(len(df)),
                      "seasons": [int(s) for s in sorted(df["season"].unique())],
                      "split": split_desc},
        notes=("Test is 2026-27 to date, reported only. 'val' is late 2025-26, which set "
               "the tree count, so it's optimistic."),
    )
    segs.to_csv(results_dir / "segments.csv", index=False)
    pd.DataFrame([{"model": VERSION, **m_new}, {"model": "v1.4-fenwick", **m_old},
                  {"model": "log loss change", "log_loss": g["diff"]}]).to_csv(
        results_dir / "test_vs_v1_4.csv", index=False)
    print(f"\nSaved {ARTIFACT_STEM}* and results to {results_dir}/. Not scored into shot_xg.")


if __name__ == "__main__":
    main()
