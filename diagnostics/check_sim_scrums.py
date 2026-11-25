"""Read-only check: scrum penalties and missed power plays in sim.data.power_plays().

Simulator review 1b. Build seasons only (2023-24 to 2025-26), regular season; 2026-27 is
never loaded. Uses the same penalties and shift charts as sim/data.py.

METHOD
------
1. Case A: one player gets two 2-minute minors at a second, and the other team has at
   least one 2-minute minor at that same second. Case B: one player gets a 5-minute
   major and a 2-minute minor at the same second; split by whether the other team has a
   major at that second too (B5, usually a fight). For each: how often, what
   power_plays() makes of that second's calls, and what the ice shows: the called team's
   skater edge (its skaters minus the other side's) in each of the 9 minutes after the
   call, against the edge at the last second with both goalies in at or before the call.
2. The long uncounted-call runs from results/checks/sim_pp_time.txt (power-play seconds,
   10+ in a row, regulation, outside every counted window, mostly inside the window of a
   call power_plays() didn't count), tagged by the call they sit in: case A, B, or
   neither.
3. Missed power plays: calls left after netting that power_plays() didn't count, alone
   at their second, in regulation, with 10+ power-play seconds for the drawing team in
   the call's window. Sorted by kind using the play-by-play (first that applies wins):
     1 a goal by the drawing team at the call's second (a power-play goal ends the
       earlier power play just as the new one starts, so the edge never moves)
     2 a goal by the drawing team 1-10 s after the call (the 5-on-3 shows, then ends
       before the 10-second check)
     3 the called team was already 2 skaters down or at 3 (the new penalty waits for a
       box spot)
     4 another call 1-30 s later
     5 the edge was already down at the last second before the call with no earlier
       counted power play to explain it (the shift chart has the player off a few
       seconds before the call's recorded second)
     6 the drawing team was already on a counted power play with 10 s or less left
       (it runs out during the check)
     7 no second with both goalies in 10-30 s after the call (goalie pulled)
     8 the drawing team's power play starts more than 30 s after the call
     9 other
   Two examples of each kind with the ice (home v away skaters, seconds in a row; noG
   = a goalie out) and the goals and penalties around the call.

Writes results/checks/sim_scrums.txt.
Run: .venv/bin/python -m diagnostics.check_sim_scrums
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from sim import data as SD

BUILD = [20232024, 20242025, 20252026]
REG = 3600
OUT = Path(__file__).resolve().parent.parent / "results" / "checks" / "sim_scrums.txt"
KINDS = {
    1: "power-play goal at the call's second",
    2: "power-play goal 1-10 s after the call",
    3: "called team already 2 down or at 3 (penalty waits for a box spot)",
    4: "another call 1-30 s later",
    5: "edge already down before the call, nothing counted explains it (shift chart early)",
    6: "earlier power play runs out within 10 s of the call",
    7: "no second with both goalies in 10-30 s after the call",
    8: "drawing team's power play starts more than 30 s after the call",
    9: "other",
}


def runs(mask):
    if not mask.any():
        return []
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    s = np.flatnonzero(d == 1)
    return list(zip(s, np.flatnonzero(d == -1) - s))


def ice(h, a, ok, t0, t1):
    out = []
    for t in range(max(t0, 0), min(t1, len(h))):
        s = f"{h[t]}v{a[t]}" if ok[t] else "noG"
        if out and out[-1][0] == s:
            out[-1][1] += 1
        else:
            out.append([s, 1])
    return " ".join(f"{s}:{n}" for s, n in out)


def goals_and_calls():
    """Goals (second, team) and all penalty calls (second, team, code, minutes) per game."""
    from ingest.db import pg_conn
    with pg_conn() as conn:
        cur = conn.cursor()
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute("""SELECT game_id, game_seconds, shooter_team_id FROM shots s
                       JOIN games g USING (game_id)
                       WHERE s.event_type = 'goal' AND g.game_type = 2 AND g.season = ANY(%s)""", (BUILD,))
        G = defaultdict(list)
        for g, t, team in cur.fetchall():
            G[g].append((t, team))
        cur.execute("""SELECT p.game_id, p.game_seconds, p.team_id, p.raw_details_json->>'typeCode',
                              p.raw_details_json->>'duration'
                       FROM plays_context p JOIN games g USING (game_id)
                       WHERE p.event_type = 'penalty' AND g.game_type = 2 AND g.season = ANY(%s)""", (BUILD,))
        C = defaultdict(list)
        for g, t, team, code, dur in cur.fetchall():
            C[g].append((t, team, code, dur))
    return G, C


def main():
    games, shifts, goalies, _s, _g, pens = SD._query(BUILD)
    G, CALLS = goals_and_calls()
    st = np.searchsorted(shifts[:, 0], [g[0] for g in games])
    en = np.searchsorted(shifts[:, 0], [g[0] for g in games], side="right")
    C = Counter()
    ex_case = defaultdict(list)
    prof = defaultdict(lambda: np.zeros((3, 9 * 60)))
    run_tag, run_secs = Counter(), Counter()
    kind_n = defaultdict(Counter)              # already short-handed? -> kind -> calls
    kind_ex = defaultdict(list)
    n_tg = 0
    for k, (gid, season, date, home, away, *_r) in enumerate(games):
        sh = shifts[st[k]:en[k]]
        if len(sh) == 0:
            continue
        n_tg += 2
        h, a, ok, T = SD.on_ice_counts(sh, gid, home, away, goalies)
        P = pens.get(gid, [])
        side = lambda x: "home" if x == home else "away"

        def context(t):
            lines = [f"{s2 - t:+5d}s goal {side(tm)}" for s2, tm in G[gid] if abs(s2 - t) <= 130]
            lines += [f"{s2 - t:+5d}s penalty {side(tm)} {code} {dur}" for s2, tm, code, dur in CALLS[gid]
                      if abs(s2 - t) <= 130]
            return sorted(lines, key=lambda x: int(x.split("s")[0]))

        # 1. the scrum cases
        by = defaultdict(lambda: defaultdict(list))
        for t, team, L, pl in P:
            if pl is not None:
                by[t][(team, pl)].append(L)
        tag = {}
        for t, d in by.items():
            for (team, pl), ls in d.items():
                other = away if team == home else home
                other_twos = sum(x.count(2) for (tm2, _), x in d.items() if tm2 == other)
                other_twos += sum(1 for t2, tm2, L2, p2 in P if t2 == t and tm2 == other and p2 is None and L2 == 2)
                other_fives = sum(x.count(5) for (tm2, _), x in d.items() if tm2 == other)
                if ls.count(2) >= 2 and other_twos >= 1:
                    case = "A"
                elif 5 in ls and 2 in ls:
                    case = "B5" if other_fives else "B"
                else:
                    continue
                tag[(t, team)] = case
                C[case] += 1
                ev = []
                SD.power_plays([p for p in P if p[0] == t], h, a, ok, home, away, T, events=ev)
                got = tuple(sorted((f"{L2}-min PP for {'called' if tm == team else 'other'} team, "
                                    f"{'counted' if c else 'not counted'}") for _, tm, L2, c in ev))
                C[(case, got)] += 1
                edge = (h.astype(int) - a) if team == home else (a.astype(int) - h)
                okb = np.flatnonzero(ok[:t + 1])
                if okb.size == 0 or t + 9 * 60 > T:
                    continue
                e0 = edge[okb[-1]]
                w = np.arange(t + 1, t + 1 + 9 * 60)
                ew, okw = edge[w], ok[w]
                prof[case][0] += okw & (ew < e0)
                prof[case][1] += okw & (ew == e0)
                prof[case][2] += okw & (ew > e0)
                C[(case, "n_prof")] += 1
                C[(case, "short")] += int((okw & (ew < e0)).sum())
                C[(case, "4v4")] += int((okw & (ew == e0) & (h[w] < 5) & (a[w] < 5)).sum())
                if len(ex_case[case]) < 3:
                    ex_case[case].append((gid, t, side(team), list(got), ice(h, a, ok, t - 3, t + 8 * 60)))

        ev = []
        SD.power_plays(P, h, a, ok, home, away, T, events=ev)
        # 2. the long uncounted-call runs
        for team, own, opp in ((home, h, a), (away, a, h)):
            pp = ok & np.isin(np.where(ok, SD.state_code(own, opp), -1), SD.PP_STATES)
            win = np.zeros(T, bool)
            owner = np.full(T, -1)
            for t, tm, L, c in ev:
                if tm == team and c:
                    win[t + 1:min(T, t + 1 + 60 * L)] = True
                elif tm == team:
                    owner[t + 1:min(T, t + 1 + 60 * L)] = t
            for s, n in runs(pp & ~win):
                if s > REG or n < 10 or (owner[s:s + n] >= 0).mean() <= 0.5:
                    continue
                o = owner[s:s + n]
                t = Counter(o[o >= 0]).most_common(1)[0][0]
                case = tag.get((t, away if team == home else home), "neither")
                run_tag[case] += 1
                run_secs[case] += n
        # 3. missed power plays, by kind
        for t, tm, L, c in ev:
            if c or t >= REG or sum(p[0] == t for p in P) > 1:
                continue
            own, opp = (h, a) if tm == home else (a, h)
            pp = ok & np.isin(np.where(ok, SD.state_code(own, opp), -1), SD.PP_STATES)
            wpp = pp[t + 1:t + 1 + 60 * L]
            if wpp.sum() < 10:
                continue
            okb = np.flatnonzero(ok[:t + 1])
            pre = (int(own[okb[-1]]), int(opp[okb[-1]])) if okb.size else (5, 5)
            already = pre[0] > pre[1]
            prev = [t2 + 60 * L2 - t for t2, tm2, L2, c2 in ev if tm2 == tm and c2 and t2 < t <= t2 + 60 * L2]
            left = min(prev) if prev else -1
            g_dr = [s2 for s2, tm2 in G[gid] if tm2 == tm]
            later = [s2 for s2, _, code, _ in CALLS[gid] if 0 < s2 - t <= 30 and code in ("MIN", "BEN", "MAJ", "MAT")]
            first_pp = int(np.flatnonzero(wpp)[0]) + 1
            if t in g_dr:
                kd = 1
            elif any(0 < g - t <= 10 for g in g_dr):
                kd = 2
            elif already and (pre[0] - pre[1] >= 2 or pre[1] <= 3):
                kd = 3
            elif later:
                kd = 4
            elif already and left < 0:
                kd = 5
            elif already and left <= 10:
                kd = 6
            elif not ok[t + SD.CONFIRM_AFTER:t + 31].any():
                kd = 7
            elif first_pp > 30:
                kd = 8
            else:
                kd = 9
            grp = "already on a power play before the call" if already else "not on a power play before the call"
            kind_n[grp][kd] += 1
            if len(kind_ex[(grp, kd)]) < 2:
                kind_ex[(grp, kd)].append((gid, t, side(tm), L, left, ice(h, a, ok, t - 20, t + 160), context(t)))

    L = []
    w = L.append
    w("SIMULATOR v0: SCRUM PENALTIES AND MISSED POWER PLAYS IN power_plays() (read-only check)")
    w("=" * 88)
    w(f"Regular season 2023-24 to 2025-26, {n_tg // 2:,} games with shift charts ({n_tg:,} team-games). "
      "2026-27 not loaded.")
    w("")
    w("1. THE SCRUM CASES")
    names = {"A": "Case A: 2+2 on one player vs a 2 on the other team, same second",
             "B": "Case B: 5 + 2 on one player, the other team has no major at that second",
             "B5": "Case B5: 5 + 2 on one player, the other team has a major too (fights)"}
    for case, name in names.items():
        w(name)
        w(f"  {C[case]} times ({C[case] / (n_tg / 2):.4f} a game)")
        outs = [(k2[1], v) for k2, v in C.items() if isinstance(k2, tuple) and k2[0] == case and isinstance(k2[1], tuple)]
        for got, v in sorted(outs, key=lambda x: -x[1]):
            w(f"  power_plays() gives {'; '.join(got) or 'no power play'}: {v}")
        n = C[(case, "n_prof")]
        if n:
            w(f"  On the ice, {n} with 9 minutes of game left: the called team is below its skater edge from")
            w(f"  just before the call for {C[(case, 'short')] / n / 60:.2f} of the next 9 minutes on average; "
              f"even with both sides under 5 skaters {C[(case, '4v4')] / n / 60:.2f} min")
            p = prof[case] / n
            w("  Share of each minute the called team is short / at the same edge / up:")
            for m in range(9):
                sl = slice(m * 60, (m + 1) * 60)
                w(f"    minute {m + 1}: {p[0, sl].mean():.2f} / {p[1, sl].mean():.2f} / {p[2, sl].mean():.2f}")
        for gid, t, team, got, s in ex_case[case]:
            w(f"  e.g. game {gid} second {t}, called team {team}: power_plays() gives {'; '.join(got) or 'nothing'}")
            w(f"       ice from 3 s before (home v away: seconds): {s}")
        w("")
    w("2. THE LONG UNCOUNTED-CALL RUNS (188 in results/checks/sim_pp_time.txt)")
    w(f"  {sum(run_tag.values())} runs here, by the call they sit in:")
    for case in ("A", "B", "B5", "neither"):
        w(f"    {case:8s} {run_tag[case]:4d} runs, {run_secs[case] / 60:6.1f} min in all")
    w("")
    w("3. MISSED POWER PLAYS: uncounted calls, alone at their second, regulation, with 10+ power-play")
    w("   seconds for the drawing team in the call's window")
    for grp in ("already on a power play before the call", "not on a power play before the call"):
        w(f"  Drawing team {grp}: {sum(kind_n[grp].values())} calls")
        for kd, lab in KINDS.items():
            if kind_n[grp][kd]:
                w(f"    {kd} {lab}: {kind_n[grp][kd]}")
    w("")
    w("  Examples (ice from 20 s before the call, home v away: seconds; then goals and penalties, "
      "seconds from the call)")
    for (grp, kd), exs in sorted(kind_ex.items(), key=lambda x: (x[0][1], x[0][0])):
        for gid, t, tm, Ln, left, s, ctx in exs:
            lt = f", {left} s left on the earlier power play" if left >= 0 else ""
            w(f"  [kind {kd}] game {gid} second {t}, {Ln}-min call drawn by {tm} ({grp}{lt})")
            w(f"     ice: {s}")
            for line in ctx:
                w(f"     {line}")
    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
