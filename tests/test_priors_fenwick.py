"""Test features/build_priors_fenwick on synthetic data (no database).

    python tests/test_priors_fenwick.py

1. A later game's result never moves an earlier prior: flip every goal on one
   date, or delete every game from that date on, and every prior on or before
   that date must come out exactly the same (shooters and goalies).
2. Goalie standardization: a league-average goalie who faces mostly power plays
   lets in more goals per shot but still comes out near 1.0.
3. Good and bad goalies separate in the right direction.
4. Shrinkage: three shots, three goals moves a new goalie only a little.
5. No history means exactly 1.0; shots older than the 2-year window don't count.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from features.build_priors_fenwick import compute_priors

RATE = {("high", "5v5"): 0.16, ("high", "5v4"): 0.22, ("mid", "5v5"): 0.06,
        ("mid", "5v4"): 0.09, ("low", "5v5"): 0.02, ("low", "5v4"): 0.03}


def season_days(start_year: int, n=90):
    first = np.datetime64(f"{start_year}-10-10")
    return [first + np.timedelta64(2 * i, "D") for i in range(n)], start_year * 10001 + 1


def synthetic(seed=7):
    """30 goalies over 3 seasons. Goalie 0 is average but faces 70% power plays."""
    rng = np.random.default_rng(seed)
    talent = np.r_[1.0, rng.choice([0.8, 1.0, 1.2], size=29)]
    rows = []
    for year in (2022, 2023, 2024):
        days, season = season_days(year)
        for day in days:
            for gk in rng.choice(30, size=12, replace=False):
                pp_share = 0.7 if gk == 0 else 0.15
                n = 25
                zones = rng.choice(["high", "mid", "low"], size=n, p=[.3, .4, .3])
                strength = np.where(rng.random(n) < pp_share, "5v4", "5v5")
                p = np.array([RATE[z, s] for z, s in zip(zones, strength)]) * talent[gk]
                goals = (rng.random(n) < p).astype(int)
                for z, s, g in zip(zones, strength, goals):
                    rows.append((int(gk), day, season, z, s, g))
    hist = pd.DataFrame(rows, columns=["entity", "game_date", "season", "cell", "stratum",
                                       "is_goal"])
    hist["game_date"] = pd.to_datetime(hist["game_date"])
    return hist, talent


def targets_for(hist):
    return hist[["entity", "game_date", "season", "cell"]].drop_duplicates().reset_index(drop=True)


def same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    cols = ["prior", "k", "league_mean", "weighted_shots", "window_shots"]
    return all(np.array_equal(a[c].to_numpy(), b[c].to_numpy(), equal_nan=True) for c in cols)


def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    return ok


def main():
    failures = 0
    hist, talent = synthetic()
    goalie_targets = targets_for(hist)
    base = compute_priors(hist, goalie_targets)

    # Shooter-style input: one stratum per cell, so no standardization.
    shooter_hist = hist.assign(stratum=hist["cell"])
    shooter_base = compute_priors(shooter_hist, goalie_targets)

    dates = np.sort(hist["game_date"].unique())
    for label, h, b in (("goalie", hist, base), ("shooter", shooter_hist, shooter_base)):
        for cut in (dates[40], dates[150], dates[200]):
            on_or_before = (goalie_targets["game_date"] <= cut).to_numpy()
            after = ~on_or_before

            flipped = h.copy()
            m = flipped["game_date"] == cut
            flipped.loc[m, "is_goal"] = 1 - flipped.loc[m, "is_goal"]
            f = compute_priors(flipped, goalie_targets)
            failures += not check(f"{label}: flipping {str(cut)[:10]}'s results leaves every "
                                  "prior up to that date alone",
                                  same(f[on_or_before], b[on_or_before]))
            failures += not check(f"{label}: ...and does move some later ones",
                                  not same(f[after], b[after]))

            trimmed = h[h["game_date"] < cut]
            t = compute_priors(trimmed, goalie_targets[on_or_before].reset_index(drop=True))
            failures += not check(f"{label}: deleting everything from {str(cut)[:10]} on "
                                  "changes nothing up to that date",
                                  same(t, b[on_or_before].reset_index(drop=True)))

    # Late in season 3: each goalie's mean prior across zones
    late = base[base["game_date"] >= dates[-30]]
    by_goalie = late.groupby("entity")["prior"].mean()
    raw = hist[hist["game_date"] < dates[-30]].groupby("entity")["is_goal"].mean()
    league_raw = hist[hist["game_date"] < dates[-30]]["is_goal"].mean()
    failures += not check("PP-heavy average goalie lets in more per shot than the league",
                          raw[0] > league_raw * 1.15, f"{raw[0]:.4f} vs {league_raw:.4f}")
    failures += not check("...but standardized, sits near 1.0",
                          abs(by_goalie[0] - 1) < 0.08, f"prior {by_goalie[0]:.3f}")
    good = by_goalie[[g for g in by_goalie.index if talent[g] == 0.8]].mean()
    avg = by_goalie[[g for g in by_goalie.index if g and talent[g] == 1.0]].mean()
    bad = by_goalie[[g for g in by_goalie.index if talent[g] == 1.2]].mean()
    failures += not check("talent 0.8 < 1.0 < 1.2 comes through in order",
                          good < avg < bad, f"{good:.3f} / {avg:.3f} / {bad:.3f}")
    failures += not check("shrinkage pulls them toward 1.0", 0.8 < good and bad < 1.2)

    # A brand-new goalie: 3 high-danger shots, 3 goals, then priced the next day.
    d0 = pd.Timestamp(dates[-5])
    newbie = pd.DataFrame({"entity": [99] * 3, "game_date": [d0] * 3, "season": [20242025] * 3,
                           "cell": ["high"] * 3, "stratum": ["5v5"] * 3, "is_goal": [1] * 3})
    tg = pd.DataFrame({"entity": [99, 99], "game_date": [d0, d0 + pd.Timedelta(days=1)],
                       "season": [20242025] * 2, "cell": ["high"] * 2})
    nb = compute_priors(pd.concat([hist, newbie], ignore_index=True), tg)
    failures += not check("a goalie's first game is priced at exactly 1.0", nb["prior"][0] == 1.0)
    weight = 3 / (nb["k"][1] + 3)
    failures += not check("3 for 3 the next day: above 1.0, with the shots weighted 3 / (K + 3)",
                          1.0 < nb["prior"][1] < 1.2 and weight < 0.05,
                          f"prior {nb['prior'][1]:.4f}, K {nb['k'][1]:.0f}, weight {weight:.3f}")

    # Window: shots only in 2022-23, priced in late 2024-25 (> 730 days later).
    old = hist[(hist["entity"] == 1) & (hist["season"] == 20222023)].assign(entity=77)
    tg = pd.DataFrame({"entity": [77], "game_date": [pd.Timestamp(dates[-1])],
                       "season": [20242025], "cell": ["mid"]})
    w = compute_priors(pd.concat([hist, old], ignore_index=True), tg)
    failures += not check("shots older than the 2-year window don't count",
                          w["window_shots"][0] == 0 and w["prior"][0] == 1.0)

    print(f"\n{'All passed' if not failures else f'{failures} FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
