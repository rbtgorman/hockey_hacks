"""Leak test for sim/ratings.py and sim/elo.py: a game's numbers use only earlier dates.

Made-up games, no database. Everything on or after a cut-off date is scrambled; every
game on or before that date must get exactly the same league numbers, team totals and
Elo chance as before (same-day games must not see each other). Needs numpy; runs in
CI's reporting-tests job.

Run: .venv/bin/python tests/test_sim_ratings.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim import elo as EL
from sim import ratings as R
from sim.data import STATES


def fake_games(seed):
    rng = np.random.default_rng(seed)
    teams = np.arange(1, 9)
    rows = []
    day = np.datetime64("2022-10-01")
    for season, start in ((20222023, 0), (20232024, 365), (20242025, 730)):
        for k in range(120):
            date = day + start + k // 4                 # four games a day
            h, a = rng.choice(teams, 2, replace=False)
            rows.append((season, date, h, a))
    n = len(rows)
    S = len(STATES)
    d = dict(
        game_id=np.arange(n), season=np.array([r[0] for r in rows]),
        date=np.array([r[1] for r in rows], "datetime64[D]"),
        team=np.array([(r[2], r[3]) for r in rows]),
        toi=rng.uniform(0, 3000, (n, 2, S)), xgf=rng.uniform(0, 3, (n, 2, S)),
        gf=rng.integers(0, 3, (n, 2, S)).astype(float), ppd=rng.integers(0, 4, (n, 2, 3)).astype(float),
        length=np.full(n, 3600.0), goals_period=np.zeros((n, 2, 4)), en_goals=np.zeros((n, 2)),
        final=np.zeros((n, 2)), went_ot=rng.random(n) < 0.2, went_so=rng.random(n) < 0.07,
        home_won=rng.random(n) < 0.55, has_shifts=np.ones(n, bool),
    )
    return d


def scramble(d, cut, seed):
    rng = np.random.default_rng(seed)
    e = {k: v.copy() for k, v in d.items()}
    m = e["date"] >= cut
    for k in ("toi", "xgf", "gf", "ppd"):
        e[k][m] = rng.permutation(e[k][m].ravel()).reshape(e[k][m].shape) * 1.7
    e["home_won"][m] = ~e["home_won"][m]
    e["went_so"][m] = ~e["went_so"][m]
    return e


def check(name, ok):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    return ok


def main():
    failures = 0
    d = fake_games(1)
    cut = d["date"][d["season"] == 20232024][50]        # mid-season, mid-day block
    e = scramble(d, cut, 2)
    upto = d["date"] <= cut
    print(f"Scrambling every game from {cut} on; checking the {upto.sum()} games up to that date")
    lg_d, lg_e = R.league(d), R.league(e)
    for k in ("rate", "pp_rate", "mix", "hxg", "hpp", "e1"):
        failures += not check(f"league {k} unchanged", np.allclose(lg_d[k][upto], lg_e[k][upto]))
    failures += not check("shootout home rate unchanged (earlier seasons only)",
                          np.allclose(lg_d["so"][upto], lg_e["so"][upto], equal_nan=True))
    for H, keep in ((5, 0.5), (40, 1.0)):
        a = R.team_totals(d, lg_d, H, keep)
        b = R.team_totals(e, lg_e, H, keep)
        failures += not check(f"team totals unchanged (H={H}, keep={keep})",
                              np.allclose(a[0][upto], b[0][upto]) and np.allclose(a[1][upto], b[1][upto]))
        ra = R.ratings(*a, lg_d["e1"], 10, 10)
        rb = R.ratings(*b, lg_e["e1"], 10, 10)
        failures += not check(f"ratings unchanged (H={H}, keep={keep})", np.allclose(ra[upto], rb[upto]))
    pa, pb = EL.elo(d, 8, 30, 0.7), EL.elo(e, 8, 30, 0.7)
    failures += not check("Elo chances unchanged", np.allclose(pa[upto], pb[upto]))
    after = d["date"] > cut
    failures += not check("and the scramble does change later games (the test can fail)",
                          not np.allclose(pa[after], pb[after]))
    print(f"\n{'All passed' if failures == 0 else f'{failures} FAILED'}")
    return failures


if __name__ == "__main__":
    sys.exit(main())
