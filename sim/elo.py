"""Plain Elo, the baseline the simulator has to beat. Wins and losses only.

Every team starts 2022-23 at 1500. The home team's chance:
    1 / (1 + 10 ** (-(home rating + home bonus - away rating) / 400))
After the game both ratings move by K x (result - chance): the winner gains what the
loser drops. An overtime or shootout win counts as a win. At a team's first game of a new
season its rating keeps `keep` of its distance from 1500 (the off-season pull). Games on
the same date are all predicted before any of them updates the ratings.
"""
from __future__ import annotations

import numpy as np


def elo(d, K, home_bonus, keep):
    n = len(d["game_id"])
    p = np.zeros(n)
    rating, season_of = {}, {}
    dates = d["date"]
    i = 0
    while i < n:
        j = i
        while j < n and dates[j] == dates[i]:
            j += 1
        for k in range(i, j):
            h, a = d["team"][k]
            for t in (h, a):
                if t not in rating:
                    rating[t], season_of[t] = 1500.0, d["season"][k]
                elif season_of[t] != d["season"][k]:
                    rating[t] = 1500.0 + keep * (rating[t] - 1500.0)
                    season_of[t] = d["season"][k]
            p[k] = 1 / (1 + 10 ** (-(rating[h] + home_bonus - rating[a]) / 400))
        for k in range(i, j):
            h, a = d["team"][k]
            delta = K * (float(d["home_won"][k]) - p[k])
            rating[h] += delta
            rating[a] -= delta
        i = j
    return p


def home_rate(d):
    """Every game gets the league's home-win rate from the seasons before it."""
    p = np.full(len(d["game_id"]), np.nan)
    for s in np.unique(d["season"]):
        earlier = d["season"] < s
        if earlier.any():
            p[d["season"] == s] = d["home_won"][earlier].mean()
    return p
