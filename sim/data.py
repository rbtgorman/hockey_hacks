"""Game simulator v0, the data: what happened in every regular-season game, per team.

Read-only. For each game and each team (home, away) this collects:

- seconds at each strength, from the shift charts (a shift covers second t when
  start < t <= end). A strength is (own skaters, other side's skaters), only while both
  goalies are on the ice, with 3 to 5 skaters a side: 5v5, 4v4, 3v3, 5v4, 5v3, 4v3, 4v5,
  3v5, 3v4. Seconds with a net empty or odd counts aren't any strength (v0 has no
  empty nets);
- xG (v1.4.1, from shot_xg joined through shot_features_fenwick) and goals at each
  strength, with the strength read off the shift charts at the shot's second;
- power plays drawn, by length (2, 4 or 5 minutes). See power_plays() for the rule;
- the result: goals per period without empty-net goals, and whether the game went to
  overtime or a shootout.

2026-27 is the test season: load() refuses it unless allow_test=True, and nothing in
steps 1-2 passes that.
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from ingest.db import pg_conn

TEST_SEASON = 20262027
MODEL = "v1.4.1-fenwick"
STATES = ["5v5", "4v4", "3v3", "5v4", "5v3", "4v3", "4v5", "3v5", "3v4"]
S_IDX = {s: i for i, s in enumerate(STATES)}
MIRROR = [S_IDX[f"{s[2]}v{s[0]}"] for s in STATES]          # the other side's view
PP_STATES = [S_IDX[s] for s in ("5v4", "5v3", "4v3")]      # own side has more skaters
SH_STATES = [S_IDX[s] for s in ("4v5", "3v5", "3v4")]
EVEN_STATES = [S_IDX[s] for s in ("5v5", "4v4", "3v3")]
PP_LENGTHS = [2, 4, 5]                                      # minutes: minor, double, major
CONFIRM_AFTER = 10                                          # seconds after the call
CACHE = Path(__file__).resolve().parent / "cache"


def state_code(own: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """Index into STATES for each second, -1 if it isn't one of them."""
    out = np.full(own.shape, -1, dtype=np.int8)
    for s, i in S_IDX.items():
        out[(own == int(s[0])) & (opp == int(s[2]))] = i
    return out


def power_plays(pens, h_sk, a_sk, ok, home, away, T, events=None):
    """Power plays drawn by (home, away), by length, and the calls that didn't count.

    If `events` is a list, each call left after netting is appended to it as
    (second, team that drew it, minutes, counted?), for diagnostics.

    A power play is a penalty that leaves one side short-handed. Steps:
    1. Two 2-minute minors on one player at the same second are served back to back:
       one 4-minute power play, like a double minor.
    2. Calls at the same game second are netted: each home call cancels an away call of
       the same length (fighting majors, matching minors).
    3. What's left counts only if the shift charts show it: the penalized team's skater
       edge (its skaters minus the other side's) is lower CONFIRM_AFTER to 30 seconds
       after the call than just before it, both times with both goalies in. "Just
       before" is the last such second at or before the call, so a delayed penalty
       with the goalie pulled is read from before the extra skater came on. A call
       that drops the edge by 1 counts once even if two players went in.
    Penalty shots, misconducts and game misconducts never count; they don't change
    the strength.
    """
    drawn = {home: np.zeros(3), away: np.zeros(3)}
    unconfirmed = 0
    by_sec = defaultdict(lambda: {home: defaultdict(list), away: defaultdict(list)})
    for t, team, length, player in pens:
        if team in (home, away):
            by_sec[t][team][player if player is not None else ("bench", len(by_sec[t][team]))].append(length)
    edge = {home: h_sk.astype(int) - a_sk, away: a_sk.astype(int) - h_sk}
    ok_idx = np.flatnonzero(ok)
    for t, calls in by_sec.items():
        left = {}
        for team in (home, away):
            lens = []
            for player, ls in calls[team].items():
                ls = sorted(ls)
                while ls.count(2) >= 2:                  # two minors at once: one 4-minute
                    ls.remove(2)
                    ls.remove(2)
                    ls.append(4)
                lens += ls
            left[team] = lens
        for L in PP_LENGTHS:
            n = min(left[home].count(L), left[away].count(L))
            for _ in range(n):
                left[home].remove(L)
                left[away].remove(L)
        for team, other in ((home, away), (away, home)):
            if not left[team]:
                continue
            i = np.searchsorted(ok_idx, min(t, T - 1), side="right") - 1
            j = np.searchsorted(ok_idx, t + CONFIRM_AFTER, side="left")
            if i < 0 or j >= len(ok_idx) or ok_idx[j] > t + 30:
                n_pp = 0
            else:
                drop = edge[team][ok_idx[i]] - edge[team][ok_idx[j]]
                n_pp = min(len(left[team]), max(int(drop), 0))
            unconfirmed += len(left[team]) - n_pp
            for r, L in enumerate(sorted(left[team], reverse=True)):
                if r < n_pp:
                    drawn[other][PP_LENGTHS.index(L)] += 1
                if events is not None:
                    events.append((t, other, L, r < n_pp))
    return drawn, unconfirmed


def on_ice_counts(sh, gid, home, away, goalies):
    """Skaters on the ice each second for home and away, both goalies in, and T.

    sh: the game's shift rows (game_id, player_id, team_id, start, end). Index t is
    second t (start < t <= end), for t = 0 .. T-1.
    """
    T = int(sh[:, 4].max()) + 2
    cnt = {}
    for team in (home, away):
        part = sh[sh[:, 2] == team]
        is_g = np.array([(gid, p) in goalies for p in part[:, 1]], bool)
        for kind, m in (("sk", ~is_g), ("g", is_g)):
            d = np.zeros(T + 1, np.int32)
            keep = part[m, 4] > part[m, 3]
            np.add.at(d, part[m][keep, 3] + 1, 1)
            np.add.at(d, part[m][keep, 4] + 1, -1)
            cnt[(team, kind)] = np.cumsum(d)[:T]
    ok = (cnt[(home, "g")] > 0) & (cnt[(away, "g")] > 0)
    return cnt[(home, "sk")], cnt[(away, "sk")], ok, T


def _query(seasons):
    with pg_conn() as conn:
        cur = conn.cursor()
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute("""
            SELECT game_id, season, game_date, home_team_id, away_team_id, home_score, away_score
            FROM games WHERE game_type = 2 AND game_state = 'OFF' AND season = ANY(%s)
            ORDER BY game_date, game_id""", (seasons,))
        games = cur.fetchall()
        ids = [g[0] for g in games]
        cur.execute("""SELECT game_id, player_id, team_id, start_seconds, end_seconds
                       FROM shifts WHERE game_id = ANY(%s) ORDER BY game_id""", (ids,))
        shifts = np.array(cur.fetchall(), dtype=np.int64)
        cur.execute("SELECT game_id, player_id FROM game_rosters WHERE position = 'G' AND game_id = ANY(%s)",
                    (ids,))
        goalies = set(cur.fetchall())
        cur.execute("""
            SELECT f.game_id, f.game_seconds, s.shooter_team_id, x.xg, f.is_goal
            FROM shot_features_fenwick f
            JOIN shots s ON s.shot_id = f.shot_id
            JOIN shot_xg x ON x.shot_id = f.shot_id AND x.model_version = %s
            WHERE f.game_id = ANY(%s)""", (MODEL, ids))
        shots = defaultdict(list)
        for g, t, team, xg, goal in cur.fetchall():
            shots[g].append((t, team, xg, goal))
        cur.execute("""
            SELECT game_id, period, period_type, shooter_team_id, empty_net
            FROM shots WHERE event_type = 'goal' AND game_id = ANY(%s)""", (ids,))
        goals = defaultdict(list)
        for g, per, ptype, team, en in cur.fetchall():
            goals[g].append((per, ptype, team, bool(en)))
        cur.execute("""
            SELECT game_id, game_seconds, team_id, (raw_details_json->>'duration')::int,
                   raw_details_json->>'typeCode', actor_1_id
            FROM plays_context WHERE event_type = 'penalty' AND game_id = ANY(%s)""", (ids,))
        pens = defaultdict(list)
        for g, t, team, dur, code, player in cur.fetchall():
            if code in ("MIN", "BEN", "MAJ", "MAT") and dur in PP_LENGTHS:
                pens[g].append((t, team, dur, player))
    return games, shifts, goalies, shots, goals, pens


def build(seasons):
    games, shifts, goalies, shots, goals, pens = _query(seasons)
    n = len(games)
    out = dict(
        game_id=np.zeros(n, np.int64), season=np.zeros(n, np.int64),
        date=np.zeros(n, "datetime64[D]"), team=np.zeros((n, 2), np.int64),
        toi=np.zeros((n, 2, len(STATES))), xgf=np.zeros((n, 2, len(STATES))),
        gf=np.zeros((n, 2, len(STATES))), ppd=np.zeros((n, 2, 3)),
        length=np.zeros(n), goals_period=np.zeros((n, 2, 4)), en_goals=np.zeros((n, 2)),
        final=np.zeros((n, 2)), went_ot=np.zeros(n, bool), went_so=np.zeros(n, bool),
        home_won=np.zeros(n, bool), has_shifts=np.ones(n, bool),
    )
    issues = defaultdict(int)
    starts = np.searchsorted(shifts[:, 0], [g[0] for g in games])
    ends = np.searchsorted(shifts[:, 0], [g[0] for g in games], side="right")
    for k, (gid, season, date, home, away, hs, as_) in enumerate(games):
        out["game_id"][k], out["season"][k], out["date"][k] = gid, season, np.datetime64(date)
        out["team"][k] = (home, away)
        out["final"][k] = (hs, as_)
        out["home_won"][k] = hs > as_
        gl = goals.get(gid, [])
        out["went_so"][k] = any(x[1] == "SO" for x in gl)
        out["went_ot"][k] = out["went_so"][k] or any(x[1] == "OT" for x in gl)
        for per, ptype, team, en in gl:
            if ptype == "SO":
                continue
            side = 0 if team == home else 1
            if en:
                out["en_goals"][k, side] += 1
            else:
                out["goals_period"][k, side, min(per, 4) - 1] += 1
        sh = shifts[starts[k]:ends[k]]
        if len(sh) == 0:
            out["has_shifts"][k] = False
            issues["game with no shifts"] += 1
            continue
        out["length"][k] = int(sh[:, 4].max())
        h_sk, a_sk, ok, T = on_ice_counts(sh, gid, home, away, goalies)
        st_home = np.where(ok, state_code(h_sk, a_sk), -1)
        st_away = np.where(ok, state_code(a_sk, h_sk), -1)
        for side, st in ((0, st_home), (1, st_away)):
            v = st[st >= 0]
            out["toi"][k, side] = np.bincount(v, minlength=len(STATES))
        for t, team, xg, goal in shots.get(gid, []):
            if team not in (home, away):
                issues["shot team not in game"] += 1
                continue
            side = 0 if team == home else 1
            s = (st_home if side == 0 else st_away)[min(int(t), T - 1)]
            if s < 0:
                continue
            out["xgf"][k, side, s] += xg
            out["gf"][k, side, s] += bool(goal)
        drawn, unconf = power_plays(pens.get(gid, []), h_sk, a_sk, ok, home, away, T)
        out["ppd"][k, 0], out["ppd"][k, 1] = drawn[home], drawn[away]
        issues["calls not confirmed by the shift chart"] += unconf
    return out, dict(issues)


def load(seasons, allow_test=False, refresh=False):
    """Per-game data for these seasons (cached in sim/cache/, which git ignores)."""
    seasons = sorted(seasons)
    if TEST_SEASON in seasons and not allow_test:
        raise ValueError("2026-27 is the test season; steps 1-2 never load it.")
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"games_{'_'.join(map(str, seasons))}.pkl"
    if path.exists() and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)
    data = build(seasons)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data
