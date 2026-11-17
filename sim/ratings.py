"""Game simulator v0, the ratings: what each team is expected to do, from earlier games only.

Every rating is a ratio to the league: 1.10 means 10% more than an average team.

League numbers (`league`): from the last LEAGUE_WINDOW regular-season games played before
the game's date (one season's worth, rolling). They give the league's xG per second at
each strength, power plays drawn per second of game, the mix of 2/4/5-minute power
plays, and the home multipliers. Home ice: hxg = sqrt(home xG rate / away xG rate) over
every strength second, the same for power plays drawn (hpp). The home team's rates get
x h and the away team's / h, so the league average stays where it is. The shootout home
win rate comes from every shootout in earlier seasons.

Team ratings, six of them, each = what the team did / what an average team would have
done in the same time:
    att5  xG for at 5-on-5            def5  xG against at 5-on-5
    pp    xG for on the power play    pk    xG against shorthanded
    draw  power plays drawn           give  power plays given up
- Recent games count more: before each new game, a team's totals are multiplied by
  0.5 ** (1 / H), so a game's weight halves every H of the team's games.
- Off-season pull: at a team's first game of a new season, its "what it did" totals move
  a share (1 - keep) of the way to "what an average team would have done", so the
  rating keeps `keep` of its distance from average.
- Pulled toward average: K games of exactly-average play are added before dividing
  (K5 for att5/def5, Kst for the other four), so a team with little history sits near
  1.00. K counts games, not minutes, so one K fits the special-teams xG rates and the
  power-play counts alike.

Only games on earlier dates feed a game's ratings and league numbers (games on the same
day don't see each other); tests/test_sim_ratings.py checks it.
"""
from __future__ import annotations

import numpy as np

from sim.data import MIRROR, PP_STATES, S_IDX, SH_STATES, STATES

LEAGUE_WINDOW = 1312
CATS = ["att5", "def5", "pp", "pk", "draw", "give"]
I5 = S_IDX["5v5"]


def league(d):
    """League numbers for every game, from the LEAGUE_WINDOW games on earlier dates."""
    n = len(d["game_id"])
    dates = d["date"]
    first_same = np.searchsorted(dates, dates, side="left")       # games before this date
    lo = np.maximum(0, first_same - LEAGUE_WINDOW)

    def window_sum(x):
        c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
        return c[first_same] - c[lo]

    toi = d["toi"].sum(axis=1)                  # both teams, own view, [n, S]
    xgf = d["xgf"].sum(axis=1)
    rate = window_sum(xgf) / np.maximum(window_sum(toi), 1)          # xG per second, [n, S]
    team_secs = window_sum(2 * d["length"])
    pp_rate = window_sum(d["ppd"].sum(axis=(1, 2))) / np.maximum(team_secs, 1)
    mix = window_sum(d["ppd"].sum(axis=1))
    mix = mix / np.maximum(mix.sum(axis=1, keepdims=True), 1)
    hx = window_sum(d["xgf"][:, 0].sum(axis=1)) / np.maximum(window_sum(d["toi"][:, 0].sum(axis=1)), 1)
    ax = window_sum(d["xgf"][:, 1].sum(axis=1)) / np.maximum(window_sum(d["toi"][:, 1].sum(axis=1)), 1)
    hxg = np.where(ax > 0, np.sqrt(hx / np.maximum(ax, 1e-12)), 1.0)    # 1.0 before any games
    hp = window_sum(d["ppd"][:, 0].sum(axis=1)) / np.maximum(window_sum(d["ppd"][:, 1].sum(axis=1)), 1)
    hpp = np.where(hp > 0, np.sqrt(hp), 1.0)
    # What one average team does in one game, per rating: the size of one "K game".
    ng = np.maximum(first_same - lo, 1)
    xg5 = window_sum(toi[:, I5]) / 2 / ng * rate[:, I5]
    xg_pp = (window_sum(toi[:, PP_STATES]) / 2 / ng[:, None] * rate[:, PP_STATES]).sum(axis=1)
    pps = pp_rate * window_sum(d["length"]) / ng
    e1 = np.stack([xg5, xg5, xg_pp, xg_pp, pps, pps], axis=1)
    # shootout home-win rate from earlier seasons
    so_rate = np.full(n, np.nan)
    for s in np.unique(d["season"]):
        m = (d["season"] < s) & d["went_so"]
        if m.any():
            so_rate[d["season"] == s] = d["home_won"][m].mean()
    return dict(rate=rate, pp_rate=pp_rate, mix=mix, hxg=hxg, hpp=hpp, e1=e1, so=so_rate,
                n_window=first_same - lo)


def contributions(d, lg):
    """Each team-game's (did, average team would have) for the six ratings, [n, 2, 6]."""
    n = len(d["game_id"])
    did = np.zeros((n, 2, 6))
    avg = np.zeros((n, 2, 6))
    rate = lg["rate"]
    for side in (0, 1):
        other = 1 - side
        toi = d["toi"][:, side]
        xgf, xga = d["xgf"][:, side], d["xgf"][:, other][:, MIRROR]   # xga in own-view states
        did[:, side, 0] = xgf[:, I5]
        avg[:, side, 0] = toi[:, I5] * rate[:, I5]
        did[:, side, 1] = xga[:, I5]
        avg[:, side, 1] = toi[:, I5] * rate[:, I5]
        did[:, side, 2] = xgf[:, PP_STATES].sum(axis=1)
        avg[:, side, 2] = (toi[:, PP_STATES] * rate[:, PP_STATES]).sum(axis=1)
        did[:, side, 3] = xga[:, SH_STATES].sum(axis=1)
        mir_rate = rate[:, [MIRROR[s] for s in SH_STATES]]                # opponent's PP rate
        avg[:, side, 3] = (toi[:, SH_STATES] * mir_rate).sum(axis=1)
        did[:, side, 4] = d["ppd"][:, side].sum(axis=1)
        avg[:, side, 4] = d["length"] * lg["pp_rate"]
        did[:, side, 5] = d["ppd"][:, other].sum(axis=1)
        avg[:, side, 5] = d["length"] * lg["pp_rate"]
    return did, avg


def team_totals(d, lg, H, keep):
    """Each team's decayed (did, avg) totals going into every game, [n, 2, 6] each.

    Games on the same date are all read before any of them is added.
    """
    did, avg = contributions(d, lg)
    n = len(d["game_id"])
    decay = 0.5 ** (1.0 / H)
    tot_did, tot_avg, season_of = {}, {}, {}
    pre_did = np.zeros((n, 2, 6))
    pre_avg = np.zeros((n, 2, 6))
    dates = d["date"]
    i = 0
    while i < n:
        j = i
        while j < n and dates[j] == dates[i]:
            j += 1
        for k in range(i, j):                        # read
            for side in (0, 1):
                t = d["team"][k, side]
                if t in tot_did:
                    if season_of[t] != d["season"][k]:     # off-season pull, once
                        tot_did[t] = keep * tot_did[t] + (1 - keep) * tot_avg[t]
                        season_of[t] = d["season"][k]
                    pre_did[k, side], pre_avg[k, side] = tot_did[t], tot_avg[t]
        for k in range(i, j):                        # then add
            if not d["has_shifts"][k]:
                continue
            for side in (0, 1):
                t = d["team"][k, side]
                if t not in tot_did:
                    tot_did[t], tot_avg[t] = np.zeros(6), np.zeros(6)
                    season_of[t] = d["season"][k]
                tot_did[t] = decay * tot_did[t] + did[k, side]
                tot_avg[t] = decay * tot_avg[t] + avg[k, side]
        i = j
    return pre_did, pre_avg


def ratings(pre_did, pre_avg, e1, K5, Kst):
    """The six ratings going into every game, [n, 2, 6]."""
    K = np.array([K5, K5, Kst, Kst, Kst, Kst], float)
    prior = K[None, None, :] * e1[:, None, :]
    return (pre_did + prior) / np.maximum(pre_avg + prior, 1e-12)


def game_rates(r, lg):
    """Per-second rates for the engine, from home's point of view.

    Returns xg_home[n, 3, 3], xg_away[n, 3, 3] indexed by (home penalties, away
    penalties) in the box (0-2), the power-play draw rates [n, 2], the 3-on-3 overtime
    rates [n, 2], the power-play length mix [n, 3] and the shootout home-win rate [n].
    """
    n = r.shape[0]
    rate, hxg, hpp = lg["rate"], lg["hxg"], lg["hpp"]
    h, a = r[:, 0], r[:, 1]
    xg_home = np.zeros((n, 3, 3))
    xg_away = np.zeros((n, 3, 3))
    for nh in range(3):
        for na in range(3):
            hs, as_ = 5 - nh, 5 - na
            s_h = S_IDX[f"{hs}v{as_}"]
            s_a = S_IDX[f"{as_}v{hs}"]
            if hs == as_:                             # even: 5v5 ratings
                xh = rate[:, s_h] * h[:, 0] * a[:, 1]
                xa = rate[:, s_a] * a[:, 0] * h[:, 1]
            elif hs > as_:                            # home power play
                xh = rate[:, s_h] * h[:, 2] * a[:, 3]
                xa = rate[:, s_a] * a[:, 0] * h[:, 1]  # shorthanded attack: 5v5 ratings
            else:
                xh = rate[:, s_h] * h[:, 0] * a[:, 1]
                xa = rate[:, s_a] * a[:, 2] * h[:, 3]
            xg_home[:, nh, na] = xh * hxg
            xg_away[:, nh, na] = xa / hxg
    draw = np.stack([lg["pp_rate"] * h[:, 4] * a[:, 5] * hpp,
                     lg["pp_rate"] * a[:, 4] * h[:, 5] / hpp], axis=1)
    i3 = S_IDX["3v3"]
    ot = np.stack([rate[:, i3] * h[:, 0] * a[:, 1] * hxg,
                   rate[:, i3] * a[:, 0] * h[:, 1] / hxg], axis=1)
    return dict(xg_home=xg_home, xg_away=xg_away, draw=draw, ot=ot, mix=lg["mix"], so=lg["so"])
