"""Test export/period_math against hand-worked cases."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from export.period_math import at_least, convolve, count_distribution, regression_k


def assert_close(name, got, want, tol=1e-9):
    ok = got is not None and want is not None and abs(got - want) <= tol
    ok = ok or (got is None and want is None)
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    return ok


def main():
    failures = 0

    # Two shots at 0.5: P(0) = .25, P(1) = .5, P(2) = .25
    d = count_distribution([0.5, 0.5])
    for k, want in enumerate((0.25, 0.5, 0.25)):
        if not assert_close(f"two coin flips P({k})", d[k], want): failures += 1
    if not assert_close("distribution sums to 1", sum(d), 1.0): failures += 1

    # No shots: certainly zero goals
    if not assert_close("no shots P(0)", count_distribution([])[0], 1.0): failures += 1

    # Cap: three sure goals with max_goals=2 all land in the top bucket
    capped = count_distribution([1.0, 1.0, 1.0], max_goals=2)
    if not assert_close("cap bucket holds P(>=2)", capped[2], 1.0): failures += 1
    if not assert_close("cap leaves P(1) empty", capped[1], 0.0): failures += 1

    # at_least: 1 - P(0) - P(1)
    d3 = count_distribution([0.1, 0.2, 0.3])
    p0 = 0.9 * 0.8 * 0.7
    p1 = 0.1 * 0.8 * 0.7 + 0.9 * 0.2 * 0.7 + 0.9 * 0.8 * 0.3
    if not assert_close("P(>=2) of three shots", at_least(d3, 2), 1 - p0 - p1): failures += 1

    # Convolving two teams equals one pooled Poisson-binomial
    a, b = count_distribution([0.1, 0.4]), count_distribution([0.25, 0.3, 0.05])
    pooled = count_distribution([0.1, 0.4, 0.25, 0.3, 0.05])
    conv = convolve(a, b)
    if not all(abs(x - y) < 1e-12 for x, y in zip(conv, pooled)):
        print("  [FAIL] convolve matches pooled shots"); failures += 1
    else:
        print("  [OK] convolve matches pooled shots")

    # Tango k: rates .6 and .4 over 100 games around a .5 league rate.
    # observed var .01, chance .0025, true .0075 -> k = .25 / .0075
    if not assert_close("regression_k", regression_k([(60, 100, 0.5), (40, 100, 0.5)]),
                        0.25 / 0.0075, tol=1e-9): failures += 1
    # Spread no wider than chance -> None (shrink fully to the league rate)
    if not assert_close("regression_k at chance", regression_k([(50, 100, 0.5), (51, 100, 0.5)]),
                        None): failures += 1

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    return failures


if __name__ == "__main__":
    sys.exit(main())
