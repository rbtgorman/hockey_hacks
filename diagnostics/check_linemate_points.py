"""Read-only check: do teammates who share ice time get points together?

Question: for two teammates in the same game, how does the time they spend on the ice
together relate to the chance that one, the other, or both get a point?

METHOD, step by step
--------------------
1. Games: regular season, 2022-23 to 2025-26 for the main results; 2026-27 so far is
   reported on its own, only as a check that the pattern holds.
2. Points: every goal event in `shots`, except shootout goals (period_type 'SO').
   Empty-net and penalty-shot goals count. The scorer and the first and second assists
   come from raw_details_json (scoringPlayerId, assist1PlayerId, assist2PlayerId). A
   player gets a point in a game if they're any of those three on any goal.
3. Who's on the ice: `shifts`. A shift covers second t when start < t <= end (the
   project's rule). Goalies (position G in game_rosters) aren't part of any pair, but
   they're used to tell whether a net is empty.
4. Strength, second by second, from the shift charts' skater counts:
   - 5-on-5: both teams have 5 skaters and a goalie on the ice.
   - Power play (for a team): both goalies in, the team has more skaters than the other
     side, and the other side has at least 3 (5-on-4, 5-on-3, 4-on-3).
   - All strengths: every second.
   A goal's strength is read the same way at the second it was scored.
5. Pairs: every two skaters on the same team in the same game who both played (any
   ice time). For each strength:
   - time together = seconds both were on the ice at once, at that strength;
   - time either = seconds at least one of them was on (A's time + B's time - together);
   - link = together / either, 0 to 100%. A pair-game is used for a strength only if
     time either > 0 there;
   - A and B each got a point at that strength (yes/no), and whether both were in on
     the same goal at that strength.
6. Bins: pair-games are grouped by link into 5 bins with fixed edges (BIN_EDGES), the
   same for every split, picked so each has plenty of pair-games in all strengths.
7. Per bin: how often A gets a point, B gets a point, both, either. A is the pair's
   player with the higher point rate (over the pair's games together that season, at
   that strength), B the other one.
8. Lift: each pair-season has its own point rates pA and pB, worked out over the games
   it's in (at that strength). If the two were unrelated, the chance both get a point
   in a game would be pA x pB. Expected "both" games in a bin = the sum of pA x pB over
   its pair-games. Lift = actual "both" games / expected. 1.0 means no link; 1.5 means
   50% more games with both than their own scoring would explain. Because each pair is
   compared with its own rates, a top line scoring a lot can't fake a link: only games
   where they score *together more than their rates say* can.
   Pairs with few games together pull lift toward 1 a little (their rates come from
   the same few games), so the lifts here, if anything, understate the link.
9. Phi: the plain correlation between "A got a point" and "B got a point" over the
   bin's pair-games, from -1 to 1. Unlike lift, it isn't adjusted for each pair's
   scoring, so it mixes the link with talent.
10. Same-goal share: of the pair-games where both got a point, the share where at least
   one goal had both of them on it (scorer + assist, or both assists).
11. Splits: forward-forward, forward-defense, defense-defense (position from
   game_rosters; C/L/R = F), and 5-on-5 vs power play.
12. Examples: the 10 pair-seasons with the most time together (all strengths), 40+
   games together, with their lift over the season.
13. The main tables are repeated with only pair-seasons that have 20+ games together.
14. Ranges: 95% intervals from resampling games 1,000 times (whole games, so both
   players' outcomes move together); pair rates are worked out again in every resample.

Checks first (the script stops if one fails): season points match the official totals
for a few stars, and no pair's time together exceeds either player's time on ice.

Run: .venv/bin/python -m diagnostics.check_linemate_points > results/checks/linemate_points.txt
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict

import numpy as np

from ingest.db import pg_conn

MAIN_SEASONS = [20222023, 20232024, 20242025, 20252026]
CHECK_SEASON = 20262027
STRENGTHS = ["all", "5v5", "pp"]
STRENGTH_NAME = {"all": "All strengths", "5v5": "5-on-5", "pp": "Power play"}
BIN_EDGES = [0.0, 0.10, 0.25, 0.40, 0.55, 1.0001]   # link, as a share
BIN_LABELS = ["0-10%", "10-25%", "25-40%", "40-55%", "55-100%"]
GROUPS = ["FF", "FD", "DD"]
GROUP_NAME = {"FF": "Forward-forward", "FD": "Forward-defense", "DD": "Defense-defense"}
N_BOOT = 1000
SEED = 20261117      # fixed, so a rerun prints exactly the same numbers
MIN_GAMES = 20       # the second set of tables: pair-seasons with this many games together

# Official regular-season totals (goals, assists) from NHL.com.
OFFICIAL = [
    ("C. McDavid", 8478402, 20222023, 64, 89),
    ("D. Pastrnak", 8477956, 20222023, 61, 52),
    ("N. Kucherov", 8476453, 20232024, 44, 100),
    ("N. MacKinnon", 8477492, 20232024, 51, 89),
    ("L. Draisaitl", 8477934, 20242025, 52, 54),
]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- loading
def load(seasons):
    with pg_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT game_id, season, home_team_id, away_team_id FROM games
            WHERE game_type = 2 AND game_state = 'OFF' AND season = ANY(%s)
            ORDER BY game_id""", (seasons,))
        games = cur.fetchall()
        ids = [g[0] for g in games]
        cur.execute("""
            SELECT game_id, player_id, team_id, start_seconds, end_seconds FROM shifts
            WHERE game_id = ANY(%s) ORDER BY game_id""", (ids,))
        shifts = np.array(cur.fetchall(), dtype=np.int64)
        cur.execute("""
            SELECT game_id, player_id, position FROM game_rosters
            WHERE game_id = ANY(%s)""", (ids,))
        pos = {(g, p): ('G' if x == 'G' else 'D' if x == 'D' else 'F') for g, p, x in cur.fetchall()}
        cur.execute("""
            SELECT game_id, game_seconds, shooter_team_id,
                   (raw_details_json->>'scoringPlayerId')::bigint,
                   (raw_details_json->>'assist1PlayerId')::bigint,
                   (raw_details_json->>'assist2PlayerId')::bigint
            FROM shots
            WHERE event_type = 'goal' AND period_type <> 'SO' AND game_id = ANY(%s)""", (ids,))
        goals = defaultdict(list)
        for g, t, team, s, a1, a2 in cur.fetchall():
            goals[g].append((t, team, {p for p in (s, a1, a2) if p is not None}))
        cur.execute("SELECT player_id, full_name FROM players")
        names = dict(cur.fetchall())
    return games, shifts, pos, goals, names


# ---------------------------------------------------------------- per game
def process_game(gid, season, home, away, sh, pos, goals, out, issues):
    pids = np.unique(sh[:, 1])
    idx = {p: i for i, p in enumerate(pids)}
    T = int(sh[:, 4].max()) + 2
    d = np.zeros((len(pids), T + 1), dtype=np.int16)
    rows = np.array([idx[p] for p in sh[:, 1]])
    keep = sh[:, 4] > sh[:, 3]
    np.add.at(d, (rows[keep], sh[keep, 3] + 1), 1)       # on from start+1 ...
    np.add.at(d, (rows[keep], sh[keep, 4] + 1), -1)      # ... through end
    on = np.cumsum(d, axis=1)[:, :T] > 0
    team_of = {}
    for p, tm in zip(sh[:, 1], sh[:, 2]):
        team_of[p] = tm
    is_g = np.array([pos.get((gid, p)) == 'G' for p in pids])
    if any(pos.get((gid, p)) is None for p in pids):
        issues["shift player not on roster"] += 1
    team_arr = np.array([team_of[p] for p in pids])

    sk = {}
    gin = {}
    for tm in (home, away):
        sk[tm] = on[(team_arr == tm) & ~is_g].sum(axis=0)
        gin[tm] = on[(team_arr == tm) & is_g].any(axis=0)
    both_g = gin[home] & gin[away]
    masks = {
        "5v5": both_g & (sk[home] == 5) & (sk[away] == 5),
        "pp": {home: both_g & (sk[home] > sk[away]) & (sk[away] >= 3),
               away: both_g & (sk[away] > sk[home]) & (sk[home] >= 3)},
    }

    # points per player per strength, and goals per strength (for the same-goal flag)
    gl = goals.get(gid, [])
    for t, team, ps in gl:
        if team not in (home, away):
            issues["goal team not in game"] += 1
            continue
        for p in ps:
            if p not in idx:
                issues["point-getter with no shifts"] += 1
    for tm, opp in ((home, away), (away, home)):
        tsk = np.where((team_arr == tm) & ~is_g)[0]
        n = len(tsk)
        if n < 2:
            continue
        S = on[tsk].astype(np.float32)
        ii, jj = np.triu_indices(n, 1)
        tpids = pids[tsk]
        rec = {"game": np.full(len(ii), gid), "season": np.full(len(ii), season),
               "team": np.full(len(ii), tm), "p1": tpids[ii], "p2": tpids[jj]}
        posv = np.array([pos.get((gid, p), 'F') for p in tpids])
        pi, pj = posv[ii], posv[jj]
        grp = np.where((pi == 'F') & (pj == 'F'), 0, np.where((pi == 'D') & (pj == 'D'), 2, 1))
        rec["grp"] = grp
        for s in STRENGTHS:
            if s == "all":
                m = np.ones(T, bool)
            elif s == "5v5":
                m = masks["5v5"]
            else:
                m = masks["pp"][tm]
            X = S * m
            tog = X @ X.T
            toi = np.diag(tog)
            together = tog[ii, jj]
            either = toi[ii] + toi[jj] - together
            if (together > np.minimum(toi[ii], toi[jj]) + 1e-6).any():
                issues["together > a player's TOI"] += 1
            # points at this strength
            pt = np.zeros(n, bool)
            same = set()
            loc = {p: k for k, p in enumerate(tpids)}
            for t, team, ps in gl:
                if team != tm:
                    continue
                tt = min(int(t), T - 1)
                if s != "all" and not m[tt]:
                    continue
                ks = sorted(loc[p] for p in ps if p in loc)
                for k in ks:
                    pt[k] = True
                for a in range(len(ks)):
                    for b in range(a + 1, len(ks)):
                        same.add((ks[a], ks[b]))
            sg = np.array([(a, b) in same for a, b in zip(ii, jj)], bool) if same else np.zeros(len(ii), bool)
            rec[f"tog_{s}"] = together.astype(np.float32)
            rec[f"eit_{s}"] = either.astype(np.float32)
            rec[f"pa_{s}"] = pt[ii]
            rec[f"pb_{s}"] = pt[jj]
            rec[f"sg_{s}"] = sg
        out.append(rec)


def build(seasons):
    t0 = time.time()
    games, shifts, pos, goals, names = load(seasons)
    log(f"loaded {len(games)} games, {len(shifts)} shift rows in {time.time()-t0:.0f}s")
    starts = np.searchsorted(shifts[:, 0], [g[0] for g in games])
    ends = np.searchsorted(shifts[:, 0], [g[0] for g in games], side="right")
    out = []
    issues = defaultdict(int)
    for k, (gid, season, home, away) in enumerate(games):
        sh = shifts[starts[k]:ends[k]]
        if len(sh) == 0:
            issues["game with no shifts"] += 1
            continue
        process_game(gid, season, home, away, sh, pos, goals, out, issues)
        if k % 1000 == 0:
            log(f"  {k} games, {time.time()-t0:.0f}s")
    cols = list(out[0])
    data = {c: np.concatenate([r[c] for r in out]) for c in cols}
    log(f"built {len(data['game'])} pair-games in {time.time()-t0:.0f}s")
    return data, goals, names, games, dict(issues)


# ---------------------------------------------------------------- checks
def check_points(goals, games):
    season_of = {g[0]: g[1] for g in games}
    pts = defaultdict(lambda: [0, 0])
    # goals and assists need the role, so re-read from the DB
    with pg_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT game_id, (raw_details_json->>'scoringPlayerId')::bigint,
                   (raw_details_json->>'assist1PlayerId')::bigint,
                   (raw_details_json->>'assist2PlayerId')::bigint
            FROM shots WHERE event_type='goal' AND period_type <> 'SO' AND game_id = ANY(%s)""",
                    (list(season_of),))
        for gid, s, a1, a2 in cur.fetchall():
            pts[(s, season_of[gid])][0] += 1
            for a in (a1, a2):
                if a is not None:
                    pts[(a, season_of[gid])][1] += 1
    ok = True
    print("CHECK 1: season points against official totals (regular season, no shootout goals)")
    for name, pid, season, g, a in OFFICIAL:
        mg, ma = pts[(pid, season)]
        good = (mg, ma) == (g, a)
        ok &= good
        print(f"  {name:14s} {season}: ours {mg} G {ma} A {mg+ma} P   official {g} G {a} A {g+a} P"
              f"   {'match' if good else 'MISMATCH'}")
    return ok


# ---------------------------------------------------------------- analysis
def prep(data, s):
    """Rows usable at strength s, with pair-season ids, bins, and A/B order."""
    use = data[f"eit_{s}"] > 0
    d = {k: v[use] for k, v in data.items()}
    link = d[f"tog_{s}"] / d[f"eit_{s}"]
    b = np.digitize(link, BIN_EDGES) - 1
    key = np.stack([d["season"], np.minimum(d["p1"], d["p2"]), np.maximum(d["p1"], d["p2"])], 1)
    _, ps = np.unique(key, axis=0, return_inverse=True)
    ps = ps.ravel()
    return dict(d=d, link=link, bin=b, ps=ps, nps=ps.max() + 1,
                a=d[f"pa_{s}"].astype(np.float64), b=d[f"pb_{s}"].astype(np.float64),
                sg=d[f"sg_{s}"].astype(np.float64), grp=d["grp"])


def stats(P, w):
    """Per (group-or-all, bin) sums with row weights w (resample counts)."""
    ps, nps = P["ps"], P["nps"]
    n = np.bincount(ps, w, nps)
    ra = np.bincount(ps, w * P["a"], nps)
    rb = np.bincount(ps, w * P["b"], nps)
    with np.errstate(invalid="ignore", divide="ignore"):
        pa = np.where(n > 0, ra / n, 0)[ps]
        pb = np.where(n > 0, rb / n, 0)[ps]
    ab = P["a"] * P["b"]
    # A = higher-rate player: swap per row where pb > pa
    swap = pb > pa
    A = np.where(swap, P["b"], P["a"])
    B = np.where(swap, P["a"], P["b"])
    nb = len(BIN_LABELS)
    # one pass per quantity over (group, bin) cells; "all" and "all bins" are sums of cells
    key = P["grp"] * nb + P["bin"]
    sums = {k: np.bincount(key, w * v, 3 * nb).reshape(3, nb)
            for k, v in dict(n=np.ones_like(ab), A=A, B=B, both=ab, exp=pa * pb,
                             sg=ab * P["sg"]).items()}
    out = {}
    for gname, gi in [("all", slice(None))] + [(g, slice(i, i + 1)) for i, g in enumerate(GROUPS)]:
        out[gname] = {}
        for k, m in sums.items():
            cells = m[gi].sum(axis=0)
            out[gname][k] = np.append(cells, cells.sum())
    return out


def min_games(data, k):
    """Keep only pair-seasons with k+ games together (pair-games at any strength)."""
    key = np.stack([data["season"], np.minimum(data["p1"], data["p2"]),
                    np.maximum(data["p1"], data["p2"])], 1)
    _, ps = np.unique(key, axis=0, return_inverse=True)
    ps = ps.ravel()
    keep = np.bincount(ps)[ps] >= k
    return {c: v[keep] for c, v in data.items()}


def summarize(o):
    n = o["n"]
    with np.errstate(invalid="ignore", divide="ignore"):
        pA, pB, p11 = o["A"] / n, o["B"] / n, o["both"] / n
        either = pA + pB - p11
        lift = o["both"] / o["exp"]
        phi = (p11 - pA * pB) / np.sqrt(pA * (1 - pA) * pB * (1 - pB))
        sg = o["sg"] / o["both"]
    return dict(n=n, pA=pA, pB=pB, both=p11, either=either, lift=lift, phi=phi, sg=sg)


def analyse(data, label, rng, n_boot=N_BOOT):
    games_u, gidx = np.unique(data["game"], return_inverse=True)
    G = len(games_u)
    res = {}
    for s in STRENGTHS:
        P = prep(data, s)
        rows_g = gidx[data[f"eit_{s}"] > 0]
        point = summarize_all(stats(P, np.ones(len(rows_g))))
        boots = {g: {"lift": [], "phi": [], "sg": []} for g in point}
        for _ in range(n_boot):
            wg = np.bincount(rng.integers(0, G, G), minlength=G).astype(np.float64)
            sm = summarize_all(stats(P, wg[rows_g]))
            for g in sm:
                for k in boots[g]:
                    boots[g][k].append(sm[g][k])
        rng_ = {g: {k: np.nanpercentile(np.array(v), [2.5, 97.5], axis=0) for k, v in b.items()}
                for g, b in boots.items()}
        res[s] = (point, rng_)
        log(f"  {label} {s}: done")
    return res


def summarize_all(st):
    return {g: summarize(o) for g, o in st.items()}


def fmt_range(lo, hi):
    return f"({lo:.2f}-{hi:.2f})"


def noise_note(lo, hi):
    return "inside the noise" if lo <= 1.0 <= hi else ("above 1" if lo > 1 else "below 1")


def print_table(point, rng_, group, title):
    p, r = point[group], rng_[group]
    print(f"\n  {title}")
    print(f"  {'link bin':9s} {'pair-games':>11s} {'A pt':>6s} {'B pt':>6s} {'both':>6s} {'either':>7s}"
          f" {'lift':>5s} {'95% range':>12s} {'phi':>6s} {'same goal':>10s}")
    for k, lab in enumerate(BIN_LABELS + ["all bins"]):
        lo, hi = r["lift"][0][k], r["lift"][1][k]
        print(f"  {lab:9s} {int(p['n'][k]):>11,d} {p['pA'][k]*100:5.1f}% {p['pB'][k]*100:5.1f}%"
              f" {p['both'][k]*100:5.1f}% {p['either'][k]*100:6.1f}% {p['lift'][k]:5.2f}"
              f" {fmt_range(lo, hi):>12s} {p['phi'][k]:6.3f} {p['sg'][k]*100:8.0f}%"
              f"   {noise_note(lo, hi) if lo <= 1 <= hi else ''}")


def top_pairs(data, names, rng, n_boot=N_BOOT):
    """10 pair-seasons with the most time together (all strengths), 40+ games."""
    key = np.stack([data["season"], np.minimum(data["p1"], data["p2"]),
                    np.maximum(data["p1"], data["p2"])], 1)
    uk, ps = np.unique(key, axis=0, return_inverse=True)
    ps = ps.ravel()
    ng = np.bincount(ps)
    tog = np.bincount(ps, data["tog_all"])
    cand = np.where(ng >= 40)[0]
    top = cand[np.argsort(-tog[cand])][:10]
    games_u, gidx = np.unique(data["game"], return_inverse=True)
    print("\nEXAMPLES: the 10 pair-seasons with the most time together (all strengths, 40+ games together)")
    print(f"  {'season':8s} {'pair':34s} {'games':>5s} {'min together':>12s} {'avg link':>8s}"
          f" {'A pt':>5s} {'B pt':>5s} {'both':>5s} {'lift':>5s} {'95% range':>12s} {'same goal':>9s}")
    for t in top:
        rows = np.where(ps == t)[0]
        a = data["pa_all"][rows].astype(float)
        b = data["pb_all"][rows].astype(float)
        sg = data["sg_all"][rows].astype(float)
        n = len(rows)
        lift = (a * b).sum() / (n * a.mean() * b.mean())
        # resample this pair's games (equivalent to resampling the season's games)
        lifts = []
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            aa, bb = a[i], b[i]
            e = n * aa.mean() * bb.mean()
            lifts.append((aa * bb).sum() / e if e > 0 else np.nan)
        lo, hi = np.nanpercentile(lifts, [2.5, 97.5])
        season, p1, p2 = uk[t]
        if b.mean() > a.mean():
            p1, p2, a, b = p2, p1, b, a
        link = (data["tog_all"][rows] / data["eit_all"][rows]).mean()
        nm = f"{names.get(p1, p1)} + {names.get(p2, p2)}"
        both = (a * b).sum()
        print(f"  {season:8d} {nm:34s} {n:5d} {tog[t]/60:12.0f} {link*100:7.0f}%"
              f" {a.mean()*100:4.0f}% {b.mean()*100:4.0f}% {both/n*100:4.0f}% {lift:5.2f}"
              f" {fmt_range(lo, hi):>12s} {(sg.sum()/both*100 if both else float('nan')):8.0f}%")


def main():
    rng = np.random.default_rng(SEED)
    print(__doc__.split("Run:")[0])
    print("=" * 100)
    data, goals, names, games, issues = build(MAIN_SEASONS)
    print(f"Main seasons {MAIN_SEASONS[0]}-{MAIN_SEASONS[-1]}: {len(games):,} games, "
          f"{len(data['game']):,} pair-games (all strengths)")
    print(f"Issues while building: {issues or 'none'}")

    ok = check_points(goals, games)
    over = issues.get("together > a player's TOI", 0)
    print(f"CHECK 2: pair time together above either player's time on ice: {over} team-games "
          f"{'(pass)' if over == 0 else '(FAIL)'}")
    if not ok or over:
        print("STOPPING: a check failed.")
        sys.exit(1)

    for s in STRENGTHS:
        P = prep(data, s)
        print(f"  {STRENGTH_NAME[s]}: {len(P['link']):,} pair-games with time either > 0; "
              f"link median {np.median(P['link'])*100:.0f}%")

    res = analyse(data, "main", rng)
    print("\n" + "=" * 100)
    print("RESULTS, 2022-23 to 2025-26")
    print("A = the pair's player with the higher point rate; lift = 'both' games / expected if "
          "unrelated;\nrange = 95% from 1,000 game resamples; phi = plain correlation of the two "
          "yes/no outcomes;\nsame goal = share of 'both' games where both were on one goal.")
    for s in STRENGTHS:
        point, r = res[s]
        print(f"\n--- {STRENGTH_NAME[s]} (time together and points both at this strength) ---")
        for g in ["all"] + GROUPS:
            print_table(point, r, g, "All pairs" if g == "all" else GROUP_NAME[g])

    data20 = min_games(data, MIN_GAMES)
    print("\n" + "=" * 100)
    print(f"SAME TABLES, ONLY PAIRS WITH {MIN_GAMES}+ GAMES TOGETHER IN THE SEASON "
          f"({len(data20['game']):,} of {len(data['game']):,} pair-games)")
    print("Pairs with few games together have rates worked out from those same few games, "
          "which pulls\ntheir lift toward 1. This drops them.")
    res20 = analyse(data20, f"main {MIN_GAMES}+", rng)
    for s in STRENGTHS:
        point, r = res20[s]
        print(f"\n--- {STRENGTH_NAME[s]}, {MIN_GAMES}+ games together ---")
        for g in ["all"] + GROUPS:
            print_table(point, r, g, "All pairs" if g == "all" else GROUP_NAME[g])

    top_pairs(data, names, rng)

    print("\n" + "=" * 100)
    print("CHECK SEASON 2026-27 SO FAR (only to see whether the pattern holds)")
    data2, goals2, _n, games2, issues2 = build([CHECK_SEASON])
    print(f"{len(games2):,} games, {len(data2['game']):,} pair-games. Issues: {issues2 or 'none'}")
    over2 = issues2.get("together > a player's TOI", 0)
    print(f"Time together above a player's TOI: {over2} team-games")
    res2 = analyse(data2, "2026-27", rng)
    for s in STRENGTHS:
        point, r = res2[s]
        print(f"\n--- {STRENGTH_NAME[s]}, 2026-27 so far ---")
        print_table(point, r, "all", "All pairs")


if __name__ == "__main__":
    main()
