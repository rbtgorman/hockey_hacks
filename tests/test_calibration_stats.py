"""Synthetic check for model.reporting.calibration_stats.

Outcomes are drawn from known probabilities. The function is then given
predictions distorted in known ways, so the right answers are known:

  1. the true probabilities         slope 1.0, intercept 0.0, O/E 1.0
  2. logits stretched by 1.25       slope 0.8  (over-confident model)
  3. logits lowered by 0.1          slope 1.0, intercept 0.1, O/E > 1
                                    (a model that under-predicts, like the
                                    shots-on-goal models on 2024-25)

Run from the repo root:
    python3 -m tests.test_calibration_stats
Exit code is non-zero on any failure.
"""
from __future__ import annotations

import sys

import numpy as np

from model.reporting import calibration_stats


def main() -> int:
    rng = np.random.default_rng(7)
    n = 200_000
    true_logit = rng.normal(-2.6, 1.0, n)  # ~10% base rate, xG-like spread
    p_true = 1.0 / (1.0 + np.exp(-true_logit))
    y = (rng.random(n) < p_true).astype(int)

    failures = 0

    def check(label: str, got: float, want: float, tol: float) -> None:
        nonlocal failures
        ok = abs(got - want) <= tol
        failures += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: "
              f"got {got:.4f}, want {want:.4f} +/- {tol}")

    print("1. Calibrated predictions")
    s = calibration_stats(y, p_true)
    check("slope", s["cal_slope"], 1.0, 0.03)
    check("intercept", s["cal_intercept"], 0.0, 0.08)
    check("O/E", s["o_e"], 1.0, 0.02)

    print("2. Over-confident predictions (logit x 1.25)")
    s = calibration_stats(y, 1.0 / (1.0 + np.exp(-1.25 * true_logit)))
    check("slope", s["cal_slope"], 0.8, 0.03)

    print("3. Under-predicting (logit - 0.1)")
    s = calibration_stats(y, 1.0 / (1.0 + np.exp(-(true_logit - 0.1))))
    check("slope", s["cal_slope"], 1.0, 0.03)
    check("intercept", s["cal_intercept"], 0.1, 0.08)
    sign_ok = s["o_e"] > 1.0 and s["citl"] < 0.0
    failures += not sign_ok
    print(f"  [{'PASS' if sign_ok else 'FAIL'}] O/E {s['o_e']:.4f} > 1 "
          f"and citl {s['citl']:.4f} < 0")

    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())