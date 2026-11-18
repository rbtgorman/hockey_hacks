"""Read-only check: why does simulator v0 play fewer power-play minutes than real games?

The smoke test had 4.30 simulated power-play minutes per team-game against 4.66 on the
shift charts. This breaks the gap down. Build seasons only (2023-24 to 2025-26), regular
season; 2026-27 is never loaded.

METHOD
------
1. Real seconds, from the shift charts exactly as sim/data.py reads them (a shift covers
   second t when start < t <= end; a strength needs both goalies in and 3-5 skaters a
   side). A team is on a power play in a second when it has more skaters (5v4, 5v3,
   4v3). Regulation is seconds 1-3600; the rest is overtime.
2. Counted power plays: the calls sim.data.power_plays() counts (after netting and the
   shift-chart confirmation), with their second and length. Each one's window is the
   seconds t+1 .. t+60*minutes after the call. A power play that ends early on a goal
   still gets its full window, so "outside every window" is a strict test.
3. Real length of a counted power play: the team's power-play seconds inside the union
   of its windows, divided by the number of counted power plays (regulation calls,
   regulation seconds).
4. Power-play seconds outside every counted window, split into runs of consecutive
   seconds: short runs (under 10 s) and long ones. Long runs are sorted by what's
   nearby: a call that power_plays() didn't count (netted away or not confirmed) with
   the run inside its window, or nothing.
5. Line-change timing: a slow change can read as 4 skaters for a second (a fake power
   play for the other side) or as 6 (thrown out, not any strength). Seconds with both
   goalies in and a side on 6 skaters are counted for scale.
6. Penalty-call rates by the called team's strength: counted regulation calls against
   a team, by its strength at the last readable second at or before the call (so a
   delayed penalty with the goalie pulled is read from before the extra skater), per 60
   minutes the team spent at that strength in regulation. Even = 5v5, 4v4, 3v3.
7. Simulated side: one middle setting from step 2's grid (H 40, K5 10, Kst 15, keep 0.5),
   1,000 simulations a game, regulation only (the engine doesn't simulate overtime time).
   Power-play minutes and power plays drawn per team-game; average simulated length =
   minutes / power plays drawn.

Writes results/checks/sim_pp_time.txt.
Run: .venv/bin/python -m diagnostics.check_sim_pp_time
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

from sim import data as SD

BUILD = [20232024, 20242025, 20252026]
REG = 3600
SHORT_RUN = 10
OUT = Path(__file__).resolve().parent.parent / "results" / "checks" / "sim_pp_time.txt"


def runs(mask):
    """(start, length) of each run of True in a 1-d bool array."""
    if not mask.any():
        return []
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    s = np.flatnonzero(d == 1)
    e = np.flatnonzero(d == -1)
    return list(zip(s, e - s))


def real_side(seasons):
    games, shifts, goalies, _shots, _goals, pens = SD._query(seasons)
    starts = np.searchsorted(shifts[:, 0], [g[0] for g in games])
    ends = np.searchsorted(shifts[:, 0], [g[0] for g in games], side="right")
    pp_idx = set(SD.PP_STATES)
    A = Counter()
    by_len = Counter()             # lone counted PPs by length: n, real seconds
    long_runs = Counter()
    long_secs = Counter()
    run_hist = Counter()
    n_team_games = 0
    call_n, call_secs = Counter(), Counter()
    for k, (gid, season, date, home, away, hs, as_) in enumerate(games):
        sh = shifts[starts[k]:ends[k]]
        if len(sh) == 0:
            continue
        n_team_games += 2
        h_sk, a_sk, ok, T = SD.on_ice_counts(sh, gid, home, away, goalies)
        ev = []
        SD.power_plays(pens.get(gid, []), h_sk, a_sk, ok, home, away, T, events=ev)
        sec = np.arange(T)
        reg = (sec >= 1) & (sec <= REG)
        six = ok & ((h_sk == 6) | (a_sk == 6))
        A["six_reg"] += int((six & reg).sum())
        A["six_ot"] += int((six & ~reg & (sec > REG)).sum())
        for team, own, opp in ((home, h_sk, a_sk), (away, a_sk, h_sk)):
            st = np.where(ok, SD.state_code(own, opp), -1)
            pp = np.isin(st, list(pp_idx))
            A["pp_reg"] += int((pp & reg).sum())
            A["pp_ot"] += int((pp & (sec > REG)).sum())
            cls = np.full(T, "", dtype="<U4")
            m = ok & (own >= 3) & (own <= 5) & (opp >= 3) & (opp <= 5)
            cls[m & (own == opp)], cls[m & (own > opp)], cls[m & (own < opp)] = "even", "pp", "pk"
            for c in ("even", "pp", "pk"):
                call_secs[c] += int(((cls == c) & reg).sum())
            for t, tm, L, c in ev:                 # calls against `team`: drawn by the other
                if tm != team and c and t < REG:
                    prev = np.flatnonzero(cls[:t + 1] != "")
                    call_n[cls[prev[-1]] if prev.size else "none"] += 1
            counted = [(t, L) for t, tm, L, c in ev if tm == team and c]
            other = [(t, L) for t, tm, L, c in ev if tm == team and not c]
            A["n_counted_reg"] += sum(t < REG for t, _ in counted)
            A["n_counted_ot"] += sum(t >= REG for t, _ in counted)
            A["n_uncounted"] += len(other)
            win = np.zeros(T, bool)
            for t, L in counted:
                win[t + 1:min(T, t + 1 + 60 * L)] = True
            A["pp_in_win_reg"] += int((pp & win & reg).sum())
            # lone regulation power plays: no other call for either side in the window
            for t, L in counted:
                if t + 60 * L > REG:
                    continue
                if any(t2 != t and t <= t2 <= t + 60 * L for t2, _, _, _ in ev) or \
                        sum(t2 == t for t2, tm, _, c in ev if tm == team and c) > 1:
                    continue
                w = slice(t + 1, t + 1 + 60 * L)
                by_len[(L, "n")] += 1
                by_len[(L, "secs")] += int(pp[w].sum())
                first_off = np.flatnonzero(~pp[w])
                # seconds from the call to the first non-power-play second after it began
                on = np.flatnonzero(pp[w])
                if on.size:
                    after = first_off[first_off > on[0]]
                    by_len[(L, "first_run")] += int((after[0] if after.size else 60 * L) - on[0])
                    by_len[(L, "start_delay")] += int(on[0])
                    by_len[(L, "n_on")] += 1
            near = np.zeros(T, bool)
            for t, L in other:
                near[t + 1:min(T, t + 1 + 60 * L)] = True
            outside = pp & ~win
            for s, n in runs(outside):
                where = "reg" if s <= REG else "ot"
                if n < SHORT_RUN:
                    A[f"short_runs_{where}"] += 1
                    A[f"short_secs_{where}"] += n
                    run_hist[n] += 1
                else:
                    kind = "uncounted call" if near[s:s + n].mean() > 0.5 else "no call"
                    long_runs[(where, kind)] += 1
                    long_secs[(where, kind)] += n
    return A, by_len, long_runs, long_secs, run_hist, n_team_games, call_n, call_secs


def sim_side():
    from sim import pick as P
    P.setup()
    o = P.run_setting(40, 0.5, 10, 15, 1_000, track_time=True)
    toi = o["toi"]
    pp = toi[:, SD.PP_STATES].sum() + toi[:, SD.MIRROR][:, SD.PP_STATES].sum()
    n_tg = 2 * toi.shape[0]
    return pp / n_tg / 60, o["pp_drawn"].sum() / n_tg, P.D, P.BUILD_IDX


def main():
    A, by_len, long_runs, long_secs, run_hist, n_tg, call_n, call_secs = real_side(BUILD)
    sim_min, sim_n, D, BI = sim_side()
    real_ppd_cache = D["ppd"][BI].sum(axis=2).mean()
    L = []
    w = L.append
    w("SIMULATOR v0: POWER-PLAY TIME, SIMULATED VS. REAL (read-only check)")
    w("=" * 88)
    w(f"Regular season 2023-24 to 2025-26, {n_tg // 2:,} games with shift charts "
      f"({n_tg:,} team-games). 2026-27 not loaded.")
    w("Simulated: H 40, K5 10, Kst 15, keep 0.5, 1,000 simulations a game, regulation only.")
    w("")
    m = lambda secs: secs / n_tg / 60
    w("1. POWER-PLAY MINUTES PER TEAM-GAME")
    w(f"  Real, regulation + overtime     {m(A['pp_reg'] + A['pp_ot']):.3f}  (what step 2's report compared against)")
    w(f"  Real, regulation only           {m(A['pp_reg']):.3f}")
    w(f"  Real, overtime only             {m(A['pp_ot']):.3f}  (mostly 4-on-3)")
    w(f"  Simulated, regulation           {sim_min:.3f}")
    w("")
    w("2. POWER PLAYS PER TEAM-GAME")
    w(f"  Counted calls, all              {(A['n_counted_reg'] + A['n_counted_ot']) / n_tg:.3f}  "
      f"(sim cache: {real_ppd_cache:.3f})")
    w(f"  Counted calls, regulation       {A['n_counted_reg'] / n_tg:.3f}")
    w(f"  Counted calls, overtime         {A['n_counted_ot'] / n_tg:.3f}")
    w(f"  Simulated, regulation           {sim_n:.3f}  (includes any dropped as a third penalty)")
    w(f"  Calls left after netting that weren't counted: {A['n_uncounted'] / n_tg:.3f} per team-game")
    w("")
    w("3. AVERAGE LENGTH OF A POWER PLAY (minutes)")
    real_len = A["pp_in_win_reg"] / max(A["n_counted_reg"], 1) / 60
    w(f"  Real, regulation: power-play seconds inside counted windows / counted calls  {real_len:.3f}")
    w(f"  Simulated: power-play minutes / power plays drawn                          {sim_min / sim_n:.3f}")
    w("  Lone regulation power plays (no other call in the window), real:")
    for Ln in SD.PP_LENGTHS:
        n = by_len[(Ln, "n")]
        if not n:
            continue
        non = max(by_len[(Ln, "n_on")], 1)
        w(f"    {Ln}-minute: {n:6,} calls, {by_len[(Ln, 'secs')] / n / 60:.3f} min of power play each; "
          f"first second on the power play {by_len[(Ln, 'start_delay')] / non:.1f} s after the call, "
          f"first unbroken run {by_len[(Ln, 'first_run')] / non / 60:.3f} min")
    w("")
    w("4. REAL POWER-PLAY SECONDS OUTSIDE EVERY COUNTED POWER PLAY (minutes per team-game)")
    w(f"  Short runs (under {SHORT_RUN} s), regulation: {A['short_runs_reg'] / n_tg:.3f} runs, "
      f"{m(A['short_secs_reg']):.3f} min")
    w(f"  Short runs (under {SHORT_RUN} s), overtime:   {A['short_runs_ot'] / n_tg:.3f} runs, "
      f"{m(A['short_secs_ot']):.3f} min")
    for (where, kind), n in sorted(long_runs.items()):
        w(f"  Long runs, {where:3s}, {kind:15s} {n:6,} runs ({n / n_tg:.3f} per team-game), "
          f"{m(long_secs[(where, kind)]):.3f} min, {long_secs[(where, kind)] / n:.0f} s each")
    w("  Short runs by length (s): " + ", ".join(f"{s}: {run_hist[s]:,}" for s in range(1, SHORT_RUN)))
    w("")
    w("5. LINE-CHANGE TIMING SCALE")
    w(f"  Seconds with both goalies in and a side on 6 skaters (thrown out): "
      f"{A['six_reg'] / (n_tg / 2) / 60:.3f} min a game in regulation, "
      f"{A['six_ot'] / (n_tg / 2) / 60:.3f} in overtime")
    w("")
    w("6. PENALTY CALLS AGAINST A TEAM, BY ITS STRENGTH (counted calls, regulation)")
    tot_n = sum(call_n[c] for c in ("even", "pp", "pk"))
    tot_s = sum(call_secs.values())
    for c, lab in (("even", "even strength"), ("pp", "on the power play"), ("pk", "shorthanded")):
        w(f"  {lab:18s} {call_n[c]:6,} calls in {call_secs[c] / 3600:6,.0f} team-hours = "
          f"{call_n[c] / call_secs[c] * 3600:.2f} per 60  ({call_n[c] / call_secs[c] / (tot_n / tot_s):.2f} x the overall rate)")
    w(f"  overall            {tot_n / tot_s * 3600:.2f} per 60; calls with no readable second before them: {call_n['none']}")
    w("  The engine uses one rate whatever the strength.")
    w("")
    w("7. WHERE THE GAP GOES")
    w(f"  Real (reg + OT) minus simulated: {m(A['pp_reg'] + A['pp_ot']) - sim_min:+.3f} min per team-game")
    w(f"    overtime power play                 {m(A['pp_ot']):+.3f}")
    w(f"    regulation short runs (<{SHORT_RUN} s)       {m(A['short_secs_reg']):+.3f}")
    reg_long = sum(v for (wh, _), v in long_secs.items() if wh == "reg")
    w(f"    regulation long runs outside        {m(reg_long):+.3f}")
    w(f"    rest (regulation, inside windows)   "
      f"{m(A['pp_in_win_reg']) - sim_min:+.3f}  (= real counted-PP time minus simulated)")
    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
