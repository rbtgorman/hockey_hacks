"""Math for the period tables, standard library only.

export/period_tables.py does the database work with pandas; the parts worth
testing on their own live here so CI can run tests/test_period_math.py with
no dependencies, like the parser tests.
"""
from __future__ import annotations

from typing import Iterable, Sequence

MAX_GOALS = 10  # distributions stop here; the last entry holds P(>= MAX_GOALS)


def count_distribution(probs: Iterable[float], max_goals: int = MAX_GOALS) -> list[float]:
    """P(k goals), k = 0..max_goals, from independent per-shot goal chances.

    This is the Poisson-binomial: each shot either scores (p) or not (1 - p).
    The last entry absorbs every count >= max_goals.
    """
    dist = [1.0] + [0.0] * max_goals
    for p in probs:
        q = 1.0 - p
        top = dist[max_goals] + dist[max_goals - 1] * p
        for k in range(max_goals - 1, 0, -1):
            dist[k] = dist[k] * q + dist[k - 1] * p
        dist[0] *= q
        dist[max_goals] = top
    return dist


def convolve(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """Distribution of the sum of two independent counts, capped like count_distribution."""
    top = len(a) - 1
    out = [0.0] * (top + 1)
    for i, pa in enumerate(a):
        if pa == 0.0:
            continue
        for j, pb in enumerate(b):
            out[min(i + j, top)] += pa * pb
    return out


def at_least(dist: Sequence[float], n: int) -> float:
    """P(count >= n) from a distribution."""
    return max(0.0, 1.0 - sum(dist[:n]))


def regression_k(rows: Iterable[tuple[int, int, float]]) -> float | None:
    """Tango's regression constant for a yes/no team metric.

    rows holds one (hits, games, league rate that season) per team-season.
    The spread of team rates around the league rate is split into chance
    (rate * (1 - rate) / games) and true talent; k is the number of games at
    which a team's own record and the league rate deserve equal weight, so a
    shrunk rate is (hits + k * league) / (games + k).

    Returns None when the observed spread is no wider than chance alone, which
    means every team is best estimated at the league rate.
    """
    rows = list(rows)
    if not rows:
        return None
    observed = sum((h / n - p) ** 2 for h, n, p in rows) / len(rows)
    chance = sum(p * (1 - p) / n for _, n, p in rows) / len(rows)
    true_var = observed - chance
    if true_var <= 0:
        return None
    p_bar = sum(p for _, _, p in rows) / len(rows)
    return p_bar * (1 - p_bar) / true_var
