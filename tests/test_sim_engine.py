"""Test sim/engine.py follows the rules of a game. Needs numpy; runs in CI's reporting-tests job.

Run: .venv/bin/python tests/test_sim_engine.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim import engine as E
from sim.data import S_IDX


def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    return ok


def rates(n=1, xh=None, xa=None, draw=(0.0, 0.0), ot=(0.0, 0.0), mix=(1.0, 0.0, 0.0), so=0.5):
    """One game's rates; xh / xa are 3x3 tables by (home in box, away in box), per second."""
    return dict(
        xg_home=np.tile(np.zeros((3, 3)) if xh is None else np.asarray(xh, float), (n, 1, 1)),
        xg_away=np.tile(np.zeros((3, 3)) if xa is None else np.asarray(xa, float), (n, 1, 1)),
        draw=np.tile(np.asarray(draw, float), (n, 1)), ot=np.tile(np.asarray(ot, float), (n, 1)),
        mix=np.tile(np.asarray(mix, float), (n, 1)), so=np.full(n, so),
    )


def main():
    failures = 0
    print("Power-play goals and the box")
    rem = np.array([[[0, 0], [100.0, 40.0]],      # away: two minors, 100 s and 40 s left
                    [[0, 0], [200.0, 0]],         # away: double minor in its first half
                    [[0, 0], [90.0, 0]],          # away: double minor in its second half
                    [[0, 0], [250.0, 0]]])        # away: major
    kind = np.array([[[0, 0], [E.MINOR, E.MINOR]], [[0, 0], [E.DOUBLE, 0]],
                     [[0, 0], [E.DOUBLE, 0]], [[0, 0], [E.MAJOR, 0]]], np.int8)
    E._end_one_minor(rem, kind, np.arange(4), 1)
    failures += not check("ends the minor with the least time left", list(rem[0, 1]) == [100.0, 0.0])
    failures += not check("ends the first half of a double minor", rem[1, 1, 0] == 120.0)
    failures += not check("ends the second half of a double minor", rem[2, 1, 0] == 0.0)
    failures += not check("leaves a major running", rem[3, 1, 0] == 250.0)

    n = 4000
    print("Regulation")
    r = rates()
    o = E.simulate(r, [0], n, seed=1, track_time=True)
    failures += not check("no goals and no calls: always tied after 60", o["p_ot"][0] == 1.0)
    failures += not check("no calls: every second is 5-on-5",
                          abs(o["toi"][0, S_IDX["5v5"]] - 3600) < 1e-6)
    sums = o["p_home_reg"] + o["p_away_reg"] + o["p_ot"]
    failures += not check("regulation home win + away win + tie = 1", abs(sums[0] - 1) < 1e-12)

    print("Overtime and shootout")
    oh, oa = 2 / 3600, 1 / 3600                          # per second at 3-on-3
    o = E.simulate(rates(ot=(oh, oa), so=0.6), [0], n, seed=2)
    p_goal = 1 - np.exp(-(oh + oa) * 300)
    failures += not check("overtime: 5 minutes, sudden death, home share = its rate share",
                          abs(o["p_home_ot"][0] - p_goal * 2 / 3) < 1e-12)
    failures += not check("shootout only if overtime is scoreless",
                          abs(o["p_so"][0] - (1 - p_goal)) < 1e-12)
    failures += not check("home win = overtime + shootout parts here",
                          abs(o["p_home"][0] - (p_goal * 2 / 3 + (1 - p_goal) * 0.6)) < 1e-12)
    o = E.simulate(rates(ot=(1.0, 1.0)), [0], n, seed=3)
    failures += not check("a sure overtime goal leaves no shootout", o["p_so"][0] < 1e-12)
    o = E.simulate(rates(xh=np.full((3, 3), 5 / 3600)), [0], n, seed=4)
    failures += not check("a lead after 60 means no overtime", o["p_ot"][0] < 0.01,
                          f"overtime share {o['p_ot'][0]:.4f}")

    print("Power plays")
    draw = (3 / 3600, 0.0)                               # home draws 3 an hour
    o = E.simulate(rates(draw=draw), [0], n, seed=5, track_time=True)
    # player-seconds in the box: a 5-on-3 second is two power plays running
    box = o["toi"][0, S_IDX["5v4"]] + 2 * o["toi"][0, S_IDX["5v3"]]
    failures += not check("no scoring: each minor is a full 2 minutes in the box "
                          "(less a little cut off at the final buzzer)",
                          2.85 < box / 120 < 3.0, f"{box / 120:.2f} minors' worth")
    fast = np.zeros((3, 3))
    fast[0, 1] = fast[0, 2] = 1.0                        # home scores at once with more skaters
    o = E.simulate(rates(xh=fast, draw=draw), [0], n, seed=6, track_time=True)
    pp = o["toi"][0, S_IDX["5v4"]] + o["toi"][0, S_IDX["5v3"]]
    goals = o["goals_period"][0, 0].sum()
    failures += not check("a power-play goal ends the minor (power plays last seconds)",
                          pp < 30, f"{pp:.1f} s on the power play")
    failures += not check("one goal per minor", abs(goals - 3) < 0.15, f"{goals:.2f} goals")
    o = E.simulate(rates(xh=fast, draw=draw, mix=(0, 1, 0)), [0], n, seed=7)
    goals = o["goals_period"][0, 0].sum()
    failures += not check("two goals per double minor (one per half)", abs(goals - 6) < 0.3,
                          f"{goals:.2f} goals")
    o = E.simulate(rates(xh=fast, draw=draw, mix=(0, 0, 1)), [0], n, seed=8, track_time=True)
    pp = o["toi"][0, S_IDX["5v4"]] + o["toi"][0, S_IDX["5v3"]]
    failures += not check("a major runs its full 5 minutes despite goals",
                          pp > 3 * 300 * 0.8, f"{pp:.0f} s on the power play")
    even = np.zeros((3, 3))
    even[1, 1] = 1.0                                     # home scores at once at 4-on-4 only
    o = E.simulate(rates(xh=even, draw=(3 / 3600, 3 / 3600)), [0], n, seed=9, track_time=True)
    failures += not check("a goal at 4-on-4 ends nothing (4-on-4 time still builds up)",
                          o["toi"][0, S_IDX["4v4"]] > 1.0, f"{o['toi'][0, S_IDX['4v4']]:.1f} s")

    print(f"\n{'All passed' if failures == 0 else f'{failures} FAILED'}")
    return failures


if __name__ == "__main__":
    sys.exit(main())
