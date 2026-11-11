"""Read-only data audit: the PC checks from docs/DATA_PIPELINE.md, section 7.

Changes nothing: SELECTs only, reads the cached raw files, and check 5 downloads
30 play-by-play files into a temp folder that is deleted afterwards (never
raw_data/). Each check prints its counts, shares, worst examples and a verdict
line with the threshold it used:
    OK    nothing beyond the stated threshold
    LOOK  something to read and decide on together; nothing is fixed here

Checks
  1  completeness: games per season, games not OFF, ingest_log failures,
     missing regular-season game numbers
  2  goals add up: shots' goals (plus the shootout winner) vs games' final score
  3  strength agrees: situationCode skaters vs shift-chart skaters at each
     unblocked shot (shift rule start < t <= end, as export/profiles.py)
  4  blocked shots' team: is eventOwnerTeamId the shooter's or the blocker's team
  5  late corrections: 30 random older games re-downloaded and compared with
     the cached raw file and the database
  6  starting goalies: zero or two starters, the "first goalie listed"
     fallback, and starter vs the goalie in net for the first shot against
  7  shift coverage: per team-game time covered by shifts, average skaters,
     goalie time, by season and API vs htmlreport rows
  8  shot-table checks (python -m features.build_features_fenwick --verify-only):
     run separately; its output is summarised in results/checks/data_audit.txt

Usage (repo root):
    .venv/bin/python3 -m diagnostics.check_data_audit > results/checks/data_audit.txt
    .venv/bin/python3 -m diagnostics.check_data_audit --skip-download   # no check 5
"""
from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ingest.config import PBP_URL, RAW_DATA_DIR, USER_AGENT, HTTP_TIMEOUT
from ingest.db import pg_conn
from ingest.parser import parse_pbp

SEASON_GAMES = 1312
SEED = 20261111


def q(sql: str, params=None) -> pd.DataFrame:
    with pg_conn() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def head(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def verdict(ok: bool, text: str) -> None:
    print(f"\nVERDICT: {'OK  ' if ok else 'LOOK'}  {text}")


def show(df: pd.DataFrame, n: int | None = None) -> None:
    if df.empty:
        print("  (none)")
        return
    print(df.head(n).to_string(index=False) if n else df.to_string(index=False))


def pct(a, b) -> str:
    return f"{100.0 * a / b:.3f}%" if b else "n/a"


# ---------------------------------------------------------------------------
def check1() -> None:
    head("1. Completeness")
    g = q("""SELECT season, game_type, game_state, COUNT(*) AS games
             FROM games GROUP BY 1,2,3 ORDER BY 1,2,3""")
    show(g)
    not_off = g[g.game_state != "OFF"].games.sum()
    fails = q("""SELECT kind, status, COUNT(*) AS n FROM ingest_log
                 GROUP BY 1,2 ORDER BY 1,2""")
    print("\ningest_log by kind and status:")
    show(fails)
    n_fail = fails[fails.status != "ok"].n.sum()
    if n_fail:
        show(q("SELECT * FROM ingest_log WHERE status <> 'ok' ORDER BY completed_at"), 30)

    reg = q("""SELECT season, game_id FROM games WHERE game_type = 2""")
    reg["num"] = reg.game_id % 10000
    print("\nRegular-season game numbers (each season runs 0001..1312):")
    short_full = 0
    for s, grp in reg.groupby("season"):
        have = set(grp.num)
        top = SEASON_GAMES if s < 20262027 else max(have)
        missing = sorted(set(range(1, top + 1)) - have)
        extra = sorted(n for n in have if n > SEASON_GAMES)
        tag = "complete" if not missing and len(have) == SEASON_GAMES else ""
        print(f"  {s}: {len(have)} games, numbers 1..{top} missing {len(missing)}"
              f" {missing[:15]}{' ...' if len(missing) > 15 else ''}"
              f"{', above 1312: ' + str(extra) if extra else ''} {tag}")
        if s < 20262027:
            short_full += len(missing) + len(extra)
    verdict(not_off == 0 and n_fail == 0 and short_full == 0,
            f"games not OFF: {not_off}; ingest_log non-ok rows: {n_fail}; "
            f"full seasons with missing/extra game numbers: {short_full} "
            "(2026-27 checked only up to its highest game number)")


# ---------------------------------------------------------------------------
def check2() -> None:
    head("2. Goals add up to the final score")
    df = q("""
        SELECT g.game_id, g.season, g.game_type, g.home_score, g.away_score,
          COUNT(*) FILTER (WHERE s.is_goal AND s.period_type <> 'SO'
                                 AND s.shooter_team_id = g.home_team_id) AS h_goals,
          COUNT(*) FILTER (WHERE s.is_goal AND s.period_type <> 'SO'
                                 AND s.shooter_team_id = g.away_team_id) AS a_goals,
          COUNT(*) FILTER (WHERE s.is_goal AND s.period_type = 'SO'
                                 AND s.shooter_team_id = g.home_team_id) AS h_so,
          COUNT(*) FILTER (WHERE s.is_goal AND s.period_type = 'SO'
                                 AND s.shooter_team_id = g.away_team_id) AS a_so,
          COUNT(*) FILTER (WHERE s.period_type = 'SO') AS so_attempts
        FROM games g LEFT JOIN shots s ON s.game_id = g.game_id
        WHERE g.game_state = 'OFF'
        GROUP BY 1,2,3,4,5""")
    so = df.so_attempts > 0
    df["exp_h"] = df.h_goals + (so & (df.h_so > df.a_so)).astype(int)
    df["exp_a"] = df.a_goals + (so & (df.a_so > df.h_so)).astype(int)
    df["bad"] = (df.exp_h != df.home_score) | (df.exp_a != df.away_score)
    df["so_tied_or_untied_reg"] = so & ((df.h_goals != df.a_goals) | (df.h_so == df.a_so))
    n = len(df)
    bad = df[df.bad | df.so_tied_or_untied_reg]
    print(f"Games: {n:,}; shootout games: {int(so.sum()):,}")
    print(f"Mismatches with the final score: {int(df.bad.sum())} ({pct(df.bad.sum(), n)})")
    print(f"Shootout games where regulation+OT wasn't tied or the shootout "
          f"goals were level: {int(df.so_tied_or_untied_reg.sum())}")
    print("\nBy season (mismatches):")
    show(df.groupby(["season", "game_type"]).bad.agg(["sum", "count"]).reset_index())
    print("\nEvery mismatch:")
    show(bad[["game_id", "home_score", "away_score", "h_goals", "a_goals",
              "h_so", "a_so"]].sort_values("game_id"))
    verdict(len(bad) == 0, f"{len(bad)} games where the goals don't add up (threshold 0)")


# ---------------------------------------------------------------------------
def load_shift_arrays():
    """Per game: per-second skater and goalie counts for each team, plus the
    covered-seconds info check 7 needs. Shift rule: on at t if start < t <= end."""
    print("Loading shifts...", flush=True)
    sh = q("""SELECT sh.game_id, sh.team_id, sh.player_id, sh.period,
                     sh.start_seconds, sh.end_seconds,
                     (sh.raw_json->>'source') = 'htmlreport' AS html,
                     (r.position = 'G') AS is_g
              FROM shifts sh
              LEFT JOIN game_rosters r
                ON r.game_id = sh.game_id AND r.player_id = sh.player_id
              WHERE sh.end_seconds > sh.start_seconds""")
    sh["is_g"] = sh.is_g.fillna(False).astype(bool)
    sh["html"] = sh.html.fillna(False).astype(bool)
    return sh


def per_second(part: pd.DataFrame, T: int) -> np.ndarray:
    d = np.zeros(T + 2, dtype=np.int32)
    np.add.at(d, part.start_seconds.to_numpy() + 1, 1)
    np.add.at(d, part.end_seconds.to_numpy() + 1, -1)
    return np.cumsum(d)[: T + 1]  # index t = count on ice at second t


def per_second_alt(part: pd.DataFrame, T: int) -> np.ndarray:
    """The other edge rule, start <= t < end."""
    d = np.zeros(T + 2, dtype=np.int32)
    np.add.at(d, part.start_seconds.to_numpy(), 1)
    np.add.at(d, part.end_seconds.to_numpy(), -1)
    return np.cumsum(d)[: T + 1]


def check3_and_7(sh: pd.DataFrame) -> None:
    games = q("""SELECT game_id, season, game_type, home_team_id, away_team_id
                 FROM games WHERE game_state = 'OFF'""").set_index("game_id")
    shots = q("""SELECT s.game_id, s.game_seconds, s.situation_code, s.is_home_shot,
                        s.event_type, s.period
                 FROM shot_features_fenwick f JOIN shots s ON s.shot_id = f.shot_id""")
    ends = q("""SELECT p.game_id, MAX(p.game_seconds) AS len_s, MAX(p.period) AS last_period
                FROM plays_context p JOIN games g ON g.game_id = p.game_id
                WHERE p.event_type = 'period-end'
                  AND NOT (g.game_type = 2 AND p.period >= 5)
                GROUP BY 1""").set_index("game_id")

    shots_by_game = dict(tuple(shots.groupby("game_id")))
    sh_by_game = dict(tuple(sh.groupby("game_id")))

    rows3 = []
    rows7 = []
    for gid, g in games.iterrows():
        s_g = sh_by_game.get(gid)
        L = int(ends.len_s.get(gid, 0)) if gid in ends.index else 0
        for side, tid in (("home", g.home_team_id), ("away", g.away_team_id)):
            part = s_g[s_g.team_id == tid] if s_g is not None else None
            if part is None or part.empty:
                rows7.append(dict(game_id=gid, season=g.season, game_type=g.game_type,
                                  side=side, source="none", L=L, covered=0,
                                  avg_sk=np.nan, goalie_s=0, past_end=0, n_shifts=0))
                continue
            T = max(L, int(part.end_seconds.max())) + 1
            sk = per_second(part[~part.is_g], T)
            gl = per_second(part[part.is_g], T)
            on = per_second(part, T)
            covered = int((on[1:L + 1] > 0).sum()) if L else 0
            rows7.append(dict(
                game_id=gid, season=g.season, game_type=g.game_type, side=side,
                source="htmlreport" if part.html.any() else "api", L=L, covered=covered,
                avg_sk=float(sk[1:L + 1].mean()) if L else np.nan,
                goalie_s=int((gl[1:L + 1] > 0).sum()) if L else 0,
                past_end=int(max(0, part.end_seconds.max() - L)),
                n_shifts=len(part)))

        sg = shots_by_game.get(gid)
        if sg is None or s_g is None:
            continue
        T = int(max(s_g.end_seconds.max(), sg.game_seconds.max())) + 1
        arr = {}
        for side, tid in (("home", g.home_team_id), ("away", g.away_team_id)):
            part = s_g[s_g.team_id == tid]
            arr[side] = (per_second(part[~part.is_g], T), per_second(part[part.is_g], T),
                         per_second_alt(part[~part.is_g], T))
        t = sg.game_seconds.to_numpy()
        for i, (code, tt, per) in enumerate(zip(sg.situation_code, t, sg.period)):
            if not isinstance(code, str) or len(code) != 4 or not code.isdigit():
                rows3.append(dict(game_id=gid, season=g.season, game_type=g.game_type,
                                  t=tt, period=per, code=code, bad_code=True))
                continue
            ag, ask, hsk, hg = (int(c) for c in code)
            rows3.append(dict(
                game_id=gid, season=g.season, game_type=g.game_type, t=tt, period=per,
                code=code, bad_code=False,
                sc_h=hsk, sc_a=ask, sh_h=int(arr["home"][0][tt]), sh_a=int(arr["away"][0][tt]),
                alt_h=int(arr["home"][2][tt]), alt_a=int(arr["away"][2][tt]),
                sc_hg=hg, sc_ag=ag,
                sh_hg=int(arr["home"][1][tt] > 0), sh_ag=int(arr["away"][1][tt] > 0)))

    r3 = pd.DataFrame(rows3)
    r7 = pd.DataFrame(rows7)
    report3(r3)
    report7(r7, ends, sh)


def strength_class(h, a):
    return np.where(h == a, "even", np.where(h > a, "home up", "away up"))


def report3(r: pd.DataFrame) -> None:
    head("3. Strength agrees: situationCode vs shift charts at each unblocked shot")
    print("Shift rule here is start < t <= end (export/profiles.py). 'alt' is the other")
    print("edge, start <= t < end, shown only to see if the edge rule explains a gap.")
    n_bad_code = int(r.bad_code.sum())
    r = r[~r.bad_code].copy()
    r["sk_match"] = (r.sc_h == r.sh_h) & (r.sc_a == r.sh_a)
    r["alt_match"] = (r.sc_h == r.alt_h) & (r.sc_a == r.alt_a)
    r["class_match"] = strength_class(r.sc_h, r.sc_a) == strength_class(r.sh_h, r.sh_a)
    r["g_match"] = (r.sc_hg == r.sh_hg) & (r.sc_ag == r.sh_ag)
    r["is55"] = (r.sc_h == 5) & (r.sc_a == 5) & (r.sc_hg == 1) & (r.sc_ag == 1)
    r["sh55"] = (r.sh_h == 5) & (r.sh_a == 5) & (r.sh_hg == 1) & (r.sh_ag == 1)
    n = len(r)
    print(f"\nShots compared: {n:,}; shots with a missing/odd situationCode: {n_bad_code}")
    agg = r.groupby(["season", "game_type"]).agg(
        shots=("sk_match", "size"),
        skaters_differ=("sk_match", lambda x: (~x).sum()),
        alt_rule_differs=("alt_match", lambda x: (~x).sum()),
        strength_class_differs=("class_match", lambda x: (~x).sum()),
        goalie_flag_differs=("g_match", lambda x: (~x).sum()),
        code_5v5_shift_not=("is55", lambda x: (x & ~r.loc[x.index, "sh55"]).sum()),
        shift_5v5_code_not=("sh55", lambda x: (x & ~r.loc[x.index, "is55"]).sum()),
    ).reset_index()
    for c in ["skaters_differ", "strength_class_differs", "goalie_flag_differs"]:
        agg[c + "_pct"] = (100 * agg[c] / agg.shots).round(2)
    show(agg)
    diff = r[~r.sk_match]
    print("\nHow the skater counts differ (code home/away vs shifts home/away), top 15:")
    combo = (diff.sc_h.astype(str) + "v" + diff.sc_a.astype(str) + " code, "
             + diff.sh_h.astype(str) + "v" + diff.sh_a.astype(str) + " shifts")
    show(combo.value_counts().head(15).rename_axis("combo").reset_index(name="shots"))
    print("\nBy period (share of shots where skaters differ):")
    show(r.groupby("period").sk_match.agg(lambda x: round(100 * (~x).mean(), 2))
         .rename("pct_differ").reset_index())
    print("\nGames with the most differing shots:")
    worst = (r.groupby("game_id").agg(shots=("sk_match", "size"),
                                      differ=("sk_match", lambda x: (~x).sum()))
             .sort_values("differ", ascending=False).reset_index())
    show(worst.head(12))
    nopo = r[~((r.game_type == 3) & (r.period >= 5))]
    print(f"\nLeaving out playoff periods 5+ (no shift data there, see check 7): "
          f"{len(nopo):,} shots, skaters differ on {pct((~nopo.sk_match).sum(), len(nopo))}, "
          f"strength class on {pct((~nopo.class_match).sum(), len(nopo))}")
    odd = r[(r.sc_h <= 1) | (r.sc_a <= 1)]
    print(f"Shots with 0 or 1 skater on a side in the code (penalty shots): {len(odd)}; "
          "codes: " + ", ".join(f"{k} x{v}" for k, v in odd.code.value_counts().items()))
    share = (~r.class_match).mean()
    verdict(share < 0.01,
            f"strength class (even / home up / away up) differs on {pct((~r.class_match).sum(), n)}"
            f" of shots; skater counts on {pct((~r.sk_match).sum(), n)} (threshold: class < 1%)")


def report7(r: pd.DataFrame, ends: pd.DataFrame, sh: pd.DataFrame) -> None:
    head("7. Shift coverage per team-game")
    r = r.copy()
    r["gap_s"] = r.L - r.covered
    r["no_shifts"] = r.source == "none"
    r["gap_flag"] = r.gap_s > 60
    r["sk_flag"] = (r.avg_sk < 4.5) | (r.avg_sk > 5.5)
    r["g_flag"] = (r.goalie_s < r.L - 300) | (r.goalie_s > r.L)
    r["past_flag"] = r.past_end > 5
    agg = r.groupby(["season", "game_type", "source"]).agg(
        team_games=("L", "size"),
        no_shifts=("no_shifts", "sum"),
        gap_over_60s=("gap_flag", "sum"),
        avg_skaters_off=("sk_flag", "sum"),
        goalie_time_off=("g_flag", "sum"),
        shift_past_end_5s=("past_flag", "sum"),
        median_avg_skaters=("avg_sk", "median"),
        median_gap_s=("gap_s", "median"),
    ).reset_index()
    show(agg)
    print("\nFlags: no shifts at all; more than 60 s of the game with nobody from the team")
    print("on the ice; average skaters outside 4.5-5.5; goalie on the ice more than the")
    print("game or more than 5 minutes less; a shift ending more than 5 s after the game.")
    flagged = r[r.no_shifts | r.gap_flag | r.sk_flag | r.g_flag | r.past_flag]
    print(f"\nFlagged team-games: {len(flagged)} of {len(r):,} ({pct(len(flagged), len(r))})")
    show(flagged.sort_values("gap_s", ascending=False)
         [["game_id", "side", "source", "L", "gap_s", "avg_sk", "goalie_s", "past_end",
           "n_shifts"]].round(2), 25)

    reg = flagged[flagged.game_type == 2]
    print(f"\nRegular-season flagged team-games ({len(reg)}):")
    show(reg.sort_values("game_id")[["game_id", "side", "source", "L", "gap_s", "avg_sk",
                                     "goalie_s", "past_end", "n_shifts"]].round(2))

    # Playoff periods 5+: the API shift parser skips every period >= 5
    po = ends[ends.last_period >= 5]
    has5 = set(sh.loc[sh.period >= 5, "game_id"].unique())
    po_games = q("SELECT game_id, season FROM games WHERE game_type = 3")
    po = po[po.index.isin(po_games.game_id)]
    print(f"\nPlayoff games that reached a 2nd overtime (period 5+): {len(po)}; "
          f"with any shift in period 5+: {len(set(po.index) & has5)}")
    show(po.reset_index().merge(po_games, on="game_id")[["game_id", "season",
                                                          "last_period", "len_s"]])
    verdict(len(flagged) == 0 and len(set(po.index) - has5) == 0,
            f"{len(flagged)} flagged team-games; {len(set(po.index) - has5)} playoff games "
            "with no shifts after the first overtime")


# ---------------------------------------------------------------------------
def check4() -> None:
    head("4. Blocked shots: whose team is eventOwnerTeamId?")
    df = q("""
        SELECT g.season, s.event_type,
               s.shooter_team_id AS owner,
               rs.team_id AS shooter_roster_team,
               rb.team_id AS blocker_roster_team
        FROM shots s
        JOIN games g ON g.game_id = s.game_id
        LEFT JOIN game_rosters rs ON rs.game_id = s.game_id AND rs.player_id = s.shooter_id
        LEFT JOIN game_rosters rb ON rb.game_id = s.game_id
              AND rb.player_id = (s.raw_details_json->>'blockingPlayerId')::bigint
        WHERE s.period_type <> 'SO'""")
    df["blocked"] = df.event_type == "blocked-shot"
    df["who"] = np.select(
        [df.shooter_roster_team.isna(),
         df.owner == df.shooter_roster_team,
         df.owner == df.blocker_roster_team],
        ["shooter not on roster", "shooter's team", "blocker's team"], "neither")
    t = (df.groupby(["season", "blocked", "who"]).size().rename("n").reset_index())
    t["share_pct"] = (100 * t.n / t.groupby(["season", "blocked"]).n.transform("sum")).round(3)
    show(t)
    b = df[df.blocked]
    share_blocker = (b.who == "blocker's team").mean()
    share_shooter = (b.who == "shooter's team").mean()
    print(f"\nBlocked shots: shooter's team {100*share_shooter:.2f}%, "
          f"blocker's team {100*share_blocker:.2f}%")
    verdict(share_shooter > 0.99,
            "eventOwnerTeamId on blocked shots is the "
            + ("shooter's team" if share_shooter > 0.5 else "BLOCKER'S team")
            + " (threshold: shooter's team on > 99%)")


# ---------------------------------------------------------------------------
def _norm_play(p: dict) -> tuple:
    d = p.get("details") or {}
    return (p.get("typeDescKey"), (p.get("periodDescriptor") or {}).get("number"),
            p.get("timeInPeriod"), p.get("situationCode"), d.get("xCoord"),
            d.get("yCoord"), d.get("eventOwnerTeamId"), d.get("shotType"),
            d.get("shootingPlayerId") or d.get("scoringPlayerId"), d.get("goalieInNetId"),
            d.get("reason"), d.get("typeCode"), d.get("descKey"), d.get("duration"))


def check5(n_games: int = 30) -> None:
    head(f"5. Late corrections: {n_games} random older games re-downloaded")
    pool = q("""SELECT game_id, season FROM games
                WHERE game_state = 'OFF' AND season <= 20252026 ORDER BY game_id""")
    rng = random.Random(SEED)
    pick = sorted(rng.sample(list(pool.game_id), n_games))
    sess = requests.Session()
    sess.headers["User-Agent"] = USER_AGENT
    rows = []
    examples = []
    with tempfile.TemporaryDirectory() as tmp:
        for gid in pick:
            season = str(pool.loc[pool.game_id == gid, "season"].iloc[0])
            cached = RAW_DATA_DIR / "pbp" / season / f"{gid}.json"
            resp = sess.get(PBP_URL.format(game_id=gid), timeout=HTTP_TIMEOUT)
            time.sleep(0.5)
            if resp.status_code != 200:
                rows.append(dict(game_id=gid, error=f"HTTP {resp.status_code}"))
                continue
            fresh_path = Path(tmp) / f"{gid}.json"
            fresh_path.write_bytes(resp.content)
            fresh = json.loads(resp.content)
            old = json.loads(cached.read_text()) if cached.exists() else None

            # fresh vs cached raw file, matched on eventId
            fp = {p["eventId"]: p for p in fresh.get("plays", [])}
            op = {p["eventId"]: p for p in (old or {}).get("plays", [])}
            added = set(fp) - set(op)
            removed = set(op) - set(fp)
            changed = [e for e in set(fp) & set(op) if _norm_play(fp[e]) != _norm_play(op[e])]
            for e in sorted(changed)[:3]:
                a, b = _norm_play(op[e]), _norm_play(fp[e])
                examples.append((gid, e, [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]))

            # fresh parse vs database
            res = parse_pbp(fresh)
            db = q("""SELECT event_idx, event_type, is_goal, shooter_team_id, shooter_id,
                             x_raw, y_raw, shot_type, situation_code, period_type
                      FROM shots WHERE game_id = %s""", (gid,))
            fs = pd.DataFrame([dict(event_idx=s.event_idx, event_type=s.event_type,
                                    is_goal=s.is_goal, shooter_team_id=s.shooter_team_id,
                                    shooter_id=s.shooter_id, x_raw=s.x_raw, y_raw=s.y_raw,
                                    shot_type=s.shot_type, situation_code=s.situation_code,
                                    period_type=s.period_type) for s in res.shots])
            m = fs.merge(db, on="event_idx", how="outer", suffixes=("_new", "_db"),
                         indicator=True)
            both = m[m._merge == "both"]
            cols = ["event_type", "is_goal", "shooter_team_id", "shooter_id", "x_raw",
                    "y_raw", "shot_type", "situation_code"]
            row_diff = sum(  # how many of the columns differ anywhere in the game
                bool((both[c + "_new"].astype(str) != both[c + "_db"].astype(str)).any())
                for c in cols)
            rows_differ = int((np.column_stack([
                both[c + "_new"].astype(str).to_numpy() != both[c + "_db"].astype(str).to_numpy()
                for c in cols]).any(axis=1)).sum()) if len(both) else 0
            g = q("SELECT home_score, away_score FROM games WHERE game_id = %s", (gid,))
            rows.append(dict(
                game_id=gid,
                plays_new=len(fp), plays_cached=len(op),
                added=len(added), removed=len(removed), changed=len(changed),
                shots_new=len(fs), shots_db=len(db),
                only_new=int((m._merge == "left_only").sum()),
                only_db=int((m._merge == "right_only").sum()),
                shot_rows_differ=rows_differ, cols_differing=int(row_diff),
                goals_new=int(fs.is_goal.sum()) if len(fs) else 0,
                goals_db=int(db.is_goal.sum()),
                score_new=f"{fresh['homeTeam'].get('score')}-{fresh['awayTeam'].get('score')}",
                score_db=f"{g.home_score.iloc[0]}-{g.away_score.iloc[0]}"))
    r = pd.DataFrame(rows)
    show(r)
    if examples:
        print("\nExamples of changed plays (field index, cached value, new value).")
        print("Fields: 0 type, 1 period, 2 time, 3 situationCode, 4 x, 5 y, 6 team,")
        print("7 shotType, 8 shooter, 9 goalie, 10 reason, 11 typeCode, 12 descKey, 13 duration")
        for gid, e, d in examples[:20]:
            print(f"  {gid} event {e}: {d}")
    any_change = r[(r.added > 0) | (r.removed > 0) | (r.changed > 0) | (r.only_new > 0)
                   | (r.only_db > 0) | (r.shot_rows_differ > 0) | (r.goals_new != r.goals_db)
                   | (r.score_new != r.score_db)] if "added" in r else r
    verdict(len(any_change) == 0,
            f"{len(any_change)} of {len(r)} games differ in any way from what we stored "
            "(threshold 0); goal or score changes: "
            f"{int(((r.goals_new != r.goals_db) | (r.score_new != r.score_db)).sum()) if 'added' in r else 'n/a'}")


# ---------------------------------------------------------------------------
def check6() -> None:
    head("6. Starting goalies")
    st = q("""
        SELECT g.game_id, g.season, g.game_type, t.team_id,
               COUNT(r.player_id) FILTER (WHERE r.position = 'G' AND r.is_starter) AS starters,
               COUNT(r.player_id) FILTER (WHERE r.position = 'G') AS goalies
        FROM games g
        CROSS JOIN LATERAL (VALUES (g.home_team_id), (g.away_team_id)) t(team_id)
        LEFT JOIN game_rosters r ON r.game_id = g.game_id AND r.team_id = t.team_id
        WHERE g.game_state = 'OFF'
        GROUP BY 1,2,3,4""")
    print("Team-games by number of goalies marked starter:")
    show(st.groupby(["season", "starters"]).size().rename("team_games").reset_index())
    odd = st[st.starters != 1]
    show(odd.sort_values("game_id"), 20)

    # The fallback: cached box scores with no goalie flagged starter
    fallback = []
    for season_dir in sorted((RAW_DATA_DIR / "boxscore").iterdir()):
        for f in season_dir.glob("*.json"):
            try:
                d = json.loads(f.read_text())
            except Exception:
                fallback.append((f.stem, "unreadable"))
                continue
            for side in ("homeTeam", "awayTeam"):
                gl = ((d.get("playerByGameStats") or {}).get(side) or {}).get("goalies") or []
                if gl and not any(x.get("starter") is True for x in gl):
                    fallback.append((f.stem, side))
    print(f"\nTeam-games whose cached box score flags no starter (the 'first goalie "
          f"listed' fallback): {len(fallback)}")
    for x in fallback[:20]:
        print("  ", x)

    # Starter vs the goalie in net for the team's first shot faced
    chk = q("""
        WITH first_shot AS (
          SELECT DISTINCT ON (s.game_id, s.defending_team_id)
                 s.game_id, s.defending_team_id AS team_id, s.goalie_id
          FROM shots s
          WHERE s.period_type <> 'SO' AND s.goalie_id IS NOT NULL
            AND s.event_type IN ('shot-on-goal', 'goal')
          ORDER BY s.game_id, s.defending_team_id, s.game_seconds, s.event_idx)
        SELECT g.season, f.game_id, f.team_id, f.goalie_id, r.player_id AS starter_id
        FROM first_shot f JOIN games g ON g.game_id = f.game_id
        LEFT JOIN game_rosters r ON r.game_id = f.game_id AND r.team_id = f.team_id
             AND r.position = 'G' AND r.is_starter
        WHERE g.game_state = 'OFF'""")
    chk["agree"] = chk.goalie_id == chk.starter_id
    print("\nStarter vs goalie in net for the first shot on goal faced:")
    show(chk.groupby("season").agree.agg(["sum", "count"]).reset_index()
         .rename(columns={"sum": "agree", "count": "team_games"}))
    dis = chk[~chk.agree]
    show(dis.sort_values("game_id"), 20)
    verdict(len(odd) == 0 and len(dis) <= 5,
            f"{len(odd)} team-games without exactly one starter; {len(fallback)} from the "
            f"fallback; {len(dis)} where the starter wasn't in net for the first shot "
            "(threshold: 0, and a handful of the last)")


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-download", action="store_true")
    a = p.parse_args()
    print("Read-only data audit (diagnostics/check_data_audit.py), "
          f"run {time.strftime('%Y-%m-%d %H:%M')}")
    check1()
    check2()
    check4()
    check6()
    sh = load_shift_arrays()
    check3_and_7(sh)
    if not a.skip_download:
        check5()


if __name__ == "__main__":
    main()
