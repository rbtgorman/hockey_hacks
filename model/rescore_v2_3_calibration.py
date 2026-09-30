"""Add O/E and calibration slope to the committed v2.3 run, without retraining.

WHY
---
v2.3 was trained before model.reporting.calibration_stats() existed, so
results/v2_3/metrics.json has AUC, PR-AUC, log loss and Brier per split but
no O/E. The README and dashboard fill that gap with a number computed by hand
from reliability.csv. This script puts the real value into the run record.

HOW
---
It loads the saved booster (model/artifacts/xg_v2_3.txt, written by
train_v2_3.py), rebuilds the same feature frame with train_v2_3's own
load_features() and split_three_way(), predicts every split, and merges
calibration_stats() into each split's metrics.

A GUARD, BECAUSE THE ARTIFACT IS NOT IN GIT
-------------------------------------------
model/artifacts/ is gitignored, and skater_priors_expanding can be rebuilt,
so there is no guarantee the booster and table on disk still produce the
committed run. Before writing anything, the script recomputes AUC, log loss
and Brier for every split and the test reliability table's max gap, and
compares them with metrics.json. Any difference above 1e-6 aborts with
nothing written. If it passes, the new O/E belongs to the committed run and
not to some later state of the database.

WHAT IT WRITES
--------------
results/v2_3/metrics.json only: o_e, citl, mean_pred, observed_rate,
cal_intercept and cal_slope inside each split, plus a "rescored" block
recording when and at which commit. git_commit and generated_at_utc keep
their original values, because the model itself is unchanged. Then it
rebuilds results/leaderboard.md.

Usage (repo root, project venv):
    .venv/bin/python3 -m model.rescore_v2_3_calibration --dry-run
    .venv/bin/python3 -m model.rescore_v2_3_calibration
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import lightgbm as lgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from model import train_v2_3 as v23
from model.reporting import (
    RESULTS_DIR,
    _git_commit,
    _git_dirty,
    build_leaderboard,
    calibration_stats,
    max_abs_gap,
    reliability_table,
)

METRICS_PATH = RESULTS_DIR / "v2_3" / "metrics.json"
MODEL_PATH = v23.ARTIFACT_DIR / "xg_v2_3.txt"
TOL = 1e-6


def main() -> int:
    ap = argparse.ArgumentParser(description="Add O/E to results/v2_3/metrics.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="Score and compare, write nothing.")
    args = ap.parse_args()

    rec = json.loads(METRICS_PATH.read_text())
    if rec.get("version") != "v2.3":
        raise SystemExit(f"{METRICS_PATH} is version {rec.get('version')!r}, not v2.3")
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model file not found: {MODEL_PATH}")

    booster = lgb.Booster(model_file=str(MODEL_PATH))
    df = v23.load_features()
    train, val, test = v23.split_three_way(df)

    mismatches = []
    new = {}
    p_test = None
    for name, part in (("train", train), ("val", val), ("test", test)):
        y = part[v23.TARGET].values
        # predict() on a Booster returns probabilities for a binary objective.
        # The saved model keeps the pandas category levels from training, so
        # the categorical columns map to the same codes as in the original run.
        p = booster.predict(part[v23.ALL_FEATURES])
        if name == "test":
            p_test = p
        got = {
            "n": int(len(y)),
            "auc": float(roc_auc_score(y, p)),
            "log_loss": float(log_loss(y, p)),
            "brier": float(brier_score_loss(y, p)),
        }
        stored = rec["splits"][name]
        for k, v in got.items():
            if abs(v - float(stored[k])) > TOL:
                mismatches.append(f"{name}.{k}: committed {stored[k]}, now {v}")
        new[name] = calibration_stats(y, p)

    gap = max_abs_gap(reliability_table(test[v23.TARGET].values, p_test))
    if abs(gap - rec["calibration"]["max_abs_gap"]) > TOL:
        mismatches.append(f"test max gap: committed "
                          f"{rec['calibration']['max_abs_gap']}, now {gap}")

    print("\nComparison with the committed run (tolerance 1e-6):")
    if mismatches:
        for m in mismatches:
            print(f"  MISMATCH  {m}")
        print("\nThe booster or the feature table no longer reproduces the "
              "committed v2.3 run. Nothing written.")
        return 1
    print("  n, AUC, log loss, Brier on all three splits and the test max gap "
          "all match.")

    print("\nCalibration by split:")
    for name, c in new.items():
        print(f"  {name:5s}  O/E {c['o_e']:.4f}  slope {c['cal_slope']:.4f}  "
              f"intercept {c['cal_intercept']:.4f}  "
              f"(observed {c['observed_rate']:.4f}, predicted {c['mean_pred']:.4f})")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    for name, c in new.items():
        rec["splits"][name].update(c)
    rec["rescored"] = {
        "what": "calibration_stats() added to each split; model unchanged",
        "at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
    }
    METRICS_PATH.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"\nUpdated {METRICS_PATH}")
    build_leaderboard()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())