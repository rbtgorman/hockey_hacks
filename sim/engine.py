"""Game simulator v0, the engine: play a game out many times from its rates.

Regulation, 3 x 20 minutes, one simulated game at a time but thousands side by side:
- Four things can happen at any moment, each at its own steady rate: a home goal, an away
  goal, the home team draws a power play, the away team draws one. The rates depend on
  who's in the box (sim.ratings.game_rates). The wait until the next one is random
  (exponential), the standard way to simulate things that happen at a steady rate.
- A power play is 2, 4 or 5 minutes, drawn from the league mix. A team can have at most
  two players in the box (a third call in the simulation is dropped; it's rare).
- A power-play goal ends the shorthanded team's minor with the least time left. A double
  minor is two 2-minute halves, and a goal ends only the half it's in. A major runs its
  full 5 minutes. A goal at 4-on-4 or 3-on-3 ends nothing.
- Penalties carry over into the next period, as in the NHL.

Tied after 60 minutes: 5 minutes of 3-on-3, first goal wins (no penalties in overtime in
v0). The chance someone scores in 5 minutes, and who, is worked out exactly from the two
3-on-3 rates rather than simulated. Still tied: a shootout, which the home team wins at
the league's shootout home-win rate from earlier seasons.
"""
from __future__ import annotations

import numpy as np

from sim.data import S_IDX

REG = 3600.0
OT = 300.0
MINOR, DOUBLE, MAJOR = 0, 1, 2
LENGTH = np.array([120.0, 240.0, 300.0])
MAX_GOALS = 20
# home-view strength for (home in box, away in box)
STATE_OF = np.array([S_IDX[f"{5 - nh}v{5 - na}"] for nh in range(3) for na in range(3)])


def _end_one_minor(rem, kind, rows, team):
    """A power-play goal against `team` in these rows: end (or halve) its shortest minor."""
    r = rem[rows, team]                                   # [m, 2]
    k = kind[rows, team]
    eligible = (r > 0) & (k != MAJOR)
    rr = np.where(eligible, r, np.inf)
    slot = rr.argmin(axis=1)
    has = np.isfinite(rr.min(axis=1))
    rows, slot = rows[has], slot[has]
    cur = rem[rows, team, slot]
    dbl = kind[rows, team, slot] == DOUBLE
    rem[rows, team, slot] = np.where(dbl & (cur > 120.0), 120.0, 0.0)


def _add_penalty(rem, kind, rows, team, kinds):
    """Put a player from `team` in the box in these rows (dropped if two are already in)."""
    free0 = rem[rows, team, 0] <= 0
    free1 = ~free0 & (rem[rows, team, 1] <= 0)
    for slot, m in ((0, free0), (1, free1)):
        rr = rows[m]
        rem[rr, team, slot] = LENGTH[kinds[m]]
        kind[rr, team, slot] = kinds[m]


def simulate(rates, games, n_sims, seed, track_time=False, chunk_rows=4_000_000):
    """Simulate each game in `games` (indices into the rate arrays) n_sims times.

    Returns per game: home win chance (and its regulation / overtime / shootout parts),
    the chance of overtime and of a shootout, mean goals by team and period (regulation),
    the spread of combined goals (overtime goal included, shootout not), and, with
    track_time, mean seconds at each home-view strength.
    """
    games = np.asarray(games)
    per_chunk = max(1, chunk_rows // n_sims)
    parts = [_simulate_chunk(rates, games[i:i + per_chunk], n_sims, seed + i, track_time)
             for i in range(0, len(games), per_chunk)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _simulate_chunk(rates, games, n_sims, seed, track_time):
    rng = np.random.default_rng(seed)
    G = len(games)
    M = G * n_sims
    gi = np.repeat(np.arange(G), n_sims)
    xh = rates["xg_home"][games].reshape(G, 9)
    xa = rates["xg_away"][games].reshape(G, 9)
    dh, da = rates["draw"][games, 0], rates["draw"][games, 1]
    mix_cum = np.cumsum(rates["mix"][games], axis=1)

    t = np.zeros(M)
    goals = np.zeros((M, 2, 3), np.int16)            # team, period
    rem = np.zeros((M, 2, 2))                        # team, box slot: seconds left
    kind = np.zeros((M, 2, 2), np.int8)
    toi = np.zeros(G * 9)
    live = np.arange(M)
    while live.size:
        n_box = (rem[live] > 0).sum(axis=2)          # [m, 2]
        k = n_box[:, 0] * 3 + n_box[:, 1]
        g = gi[live]
        rh, ra, ph, pa = xh[g, k], xa[g, k], dh[g], da[g]
        total = rh + ra + ph + pa
        with np.errstate(divide="ignore"):
            wait = rng.exponential(1.0, live.size) / total     # inf if nothing can happen
        r_live = rem[live].reshape(-1, 4)
        expire = np.where(r_live > 0, r_live, np.inf).min(axis=1)
        to_end = REG - t[live]
        step = np.minimum(wait, np.minimum(expire, to_end))
        if track_time:
            toi += np.bincount(g * 9 + STATE_OF[k], weights=step, minlength=G * 9)
        t[live] += step
        r_live = np.where(r_live > 0, r_live - step[:, None], 0.0)
        r_live[r_live <= 1e-9] = 0.0
        rem[live] = r_live.reshape(-1, 2, 2)

        happen = (wait <= expire) & (wait < to_end)
        rows = live[happen]
        if rows.size:
            u = rng.random(rows.size) * total[happen]
            c1 = rh[happen]
            c2 = c1 + ra[happen]
            c3 = c2 + ph[happen]
            period = np.minimum((t[rows] // 1200).astype(int), 2)
            nb = n_box[happen]
            hg = u < c1
            ag = (u >= c1) & (u < c2)
            hd = (u >= c2) & (u < c3)
            ad = u >= c3
            np.add.at(goals, (rows[hg], 0, period[hg]), 1)
            np.add.at(goals, (rows[ag], 1, period[ag]), 1)
            pp_h = hg & (nb[:, 0] < nb[:, 1])          # home scored with more skaters
            pp_a = ag & (nb[:, 1] < nb[:, 0])
            if pp_h.any():
                _end_one_minor(rem, kind, rows[pp_h], 1)
            if pp_a.any():
                _end_one_minor(rem, kind, rows[pp_a], 0)
            for drew, team in ((hd, 1), (ad, 0)):      # home draws -> away to the box
                if drew.any():
                    rr = rows[drew]
                    kinds = (rng.random(rr.size)[:, None] > mix_cum[gi[rr]]).sum(axis=1)
                    _add_penalty(rem, kind, rr, team, np.minimum(kinds, 2))
        live = live[t[live] < REG - 1e-9]

    hg = goals[:, 0].sum(axis=1).reshape(G, n_sims)
    ag = goals[:, 1].sum(axis=1).reshape(G, n_sims)
    tie = hg == ag
    oh, oa = rates["ot"][games, 0], rates["ot"][games, 1]
    p_ot_goal = 1 - np.exp(-(oh + oa) * OT)
    p_home_ot = np.divide(oh, oh + oa, out=np.full(G, 0.5), where=(oh + oa) > 0) * p_ot_goal
    p_tie = tie.mean(axis=1)
    so = rates["so"][games]
    reg_home = (hg > ag).mean(axis=1)
    reg_away = (ag > hg).mean(axis=1)
    tot = (hg + ag).clip(0, MAX_GOALS - 1)
    hist = np.zeros((G, MAX_GOALS + 1))
    for j in range(G):
        b = np.bincount(tot[j], minlength=MAX_GOALS)[:MAX_GOALS].astype(float)
        bt = np.bincount(tot[j][tie[j]], minlength=MAX_GOALS)[:MAX_GOALS].astype(float)
        hist[j, :MAX_GOALS] = b - bt * p_ot_goal[j]
        hist[j, 1:] += bt * p_ot_goal[j]
    hist /= n_sims
    out = dict(
        p_home=reg_home + p_tie * (p_home_ot + (1 - p_ot_goal) * so),
        p_home_reg=reg_home, p_away_reg=reg_away,
        p_home_ot=p_tie * p_home_ot, p_away_ot=p_tie * (p_ot_goal - p_home_ot),
        p_home_so=p_tie * (1 - p_ot_goal) * so, p_away_so=p_tie * (1 - p_ot_goal) * (1 - so),
        p_ot=p_tie, p_so=p_tie * (1 - p_ot_goal),
        goals_period=goals.reshape(G, n_sims, 2, 3).mean(axis=1),
        total_hist=hist,
    )
    if track_time:
        out["toi"] = toi.reshape(G, 9) / n_sims
    return out
