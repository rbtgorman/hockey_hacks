"""Export team and player profiles for site/team.html and site/player.html.

WHAT IT WRITES (site/data/)
---------------------------
    profiles_index.json     schema and model version, seasons, the metric
                            registry (export/profile_metrics.py), league figures
                            per season, and the team and player lists for search
    teams/<ABBR>.json       one team, every season: counts, metrics, shot zones
                            for and against, and its skaters and goalies, with
                            only the games each one played for this team
    players/<id>.json       one player, every season: counts, metrics, shot zones

Every season row also carries "ranks": {metric: [rank, pool size, tied, tier]}
from profile_metrics.rank_rows, for metrics with a better direction. The pages
only display them.

A file is only rewritten when its contents change, so an in-season update
touches the teams and players who played since the last run. Only the index
carries a timestamp. Arizona and Utah are separate teams with separate files.

WHAT GOES IN
------------
Regular-season games that are final (game_state OFF). Shots are the rows of
shot_features_fenwick (unblocked, no shootouts, partial-window events such as
'short' misses already removed) with their xG from shot_xg. Always join
through shot_features_fenwick: shot_xg can still hold rows for shots the
feature table has since dropped.

Time on ice comes from shift charts. For each second of a game the exporter
counts each side's skaters and whether its goalie is on: 5-on-5 is five
skaters and a goalie each; a power play is more skaters than the other side
(3 to 5 each, both goalies in), so a pulled goalie is not a power play.
Shots use the NHL's own strength state. A player is on the ice for a shot at
game second t if a shift has start < t <= end, the usual rule for a shot on a
line change.

Usage (repo root, with the project venv's python):
    .venv/bin/python3 -m export.profiles
    .venv/bin/python3 -m export.profiles --model v1.1-fenwick
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ingest.db import pg_conn

from . import profile_metrics as pm

OUT_DIR = Path(__file__).resolve().parent.parent / "site" / "data"
DEFAULT_MODEL = "v1.4.1-fenwick"
INDEX_BUDGET_KB = 150
PROFILE_BUDGET_KB = 40
SKATER_ROSTER = ["id", "name", "pos", "gp", "toi", "goals", "ixg", "on_xgf_pct_5v5"]
GOALIE_ROSTER = ["id", "name", "gp", "toi", "sa", "ga", "xga", "gsax"]
GAMES_PER_TEAM = 82

TEAM_NAMES = {
    "ANA": "Anaheim Ducks", "ARI": "Arizona Coyotes", "BOS": "Boston Bruins",
    "BUF": "Buffalo Sabres", "CAR": "Carolina Hurricanes", "CBJ": "Columbus Blue Jackets",
    "CGY": "Calgary Flames", "CHI": "Chicago Blackhawks", "COL": "Colorado Avalanche",
    "DAL": "Dallas Stars", "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers", "LAK": "Los Angeles Kings", "MIN": "Minnesota Wild",
    "MTL": "Montreal Canadiens", "NJD": "New Jersey Devils", "NSH": "Nashville Predators",
    "NYI": "New York Islanders", "NYR": "New York Rangers", "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers", "PIT": "Pittsburgh Penguins", "SEA": "Seattle Kraken",
    "SJS": "San Jose Sharks", "STL": "St. Louis Blues", "TBL": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs", "UTA": "Utah Mammoth", "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights", "WPG": "Winnipeg Jets", "WSH": "Washington Capitals",
}

GAMES_SQL = """
SELECT game_id, season, home_team_id, away_team_id, home_team_abbr, away_team_abbr
FROM games
WHERE game_type = 2 AND game_state = 'OFF' AND season = %(season)s
"""
PLAYERS_SQL = "SELECT player_id, full_name, position, shoots_catches FROM players"
ROSTERS_SQL = """
SELECT r.game_id, r.player_id, r.team_id, r.position
FROM game_rosters r JOIN games g USING (game_id)
WHERE g.game_type = 2 AND g.game_state = 'OFF' AND g.season = %(season)s
"""
SHIFTS_SQL = """
SELECT sh.game_id, sh.player_id, sh.team_id, sh.start_seconds, sh.end_seconds
FROM shifts sh JOIN games g USING (game_id)
WHERE g.game_type = 2 AND g.game_state = 'OFF' AND g.season = %(season)s
  AND sh.end_seconds > sh.start_seconds
"""
SHOTS_SQL = """
SELECT f.game_id, f.game_seconds, f.is_goal, f.strength_state, f.empty_net,
       f.x_norm, f.distance_ft, f.angle_deg,
       s.shooter_id, s.goalie_id, s.shooter_team_id, s.defending_team_id, x.xg
FROM shot_features_fenwick f
JOIN shots s ON s.shot_id = f.shot_id
JOIN shot_xg x ON x.shot_id = f.shot_id AND x.model_version = %(model)s
JOIN games g ON g.game_id = f.game_id
WHERE g.game_type = 2 AND g.game_state = 'OFF' AND g.season = %(season)s
"""
SEASONS_SQL = """
SELECT season, COUNT(*) AS games, MIN(game_date) AS first_date, MAX(game_date) AS last_date
FROM games WHERE game_type = 2 AND game_state = 'OFF'
GROUP BY season ORDER BY season
"""


def season_label(season: int) -> str:
    s = str(season)
    return f"{s[:4]}-{s[6:]}"


def read_only(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")


# ------------------------------------------------------- shift arithmetic
def game_strength(shifts: pd.DataFrame, goalies: set, home: int, away: int):
    """Per-second state for one game.

    Returns (five, pp_home, pp_away), boolean arrays where index s covers
    game second [s, s + 1).
    """
    T = int(shifts["end_seconds"].max()) + 1
    count = {}
    for team in (home, away):
        part = shifts[shifts["team_id"] == team]
        is_g = part["player_id"].isin(goalies).to_numpy()
        for kind, mask in (("sk", ~is_g), ("g", is_g)):
            d = np.zeros(T + 1, dtype=np.int32)
            np.add.at(d, part["start_seconds"].to_numpy()[mask], 1)
            np.add.at(d, part["end_seconds"].to_numpy()[mask], -1)
            count[(team, kind)] = np.cumsum(d)[:T]
    sh, sa = count[(home, "sk")], count[(away, "sk")]
    both_g = (count[(home, "g")] > 0) & (count[(away, "g")] > 0)
    five = both_g & (sh == 5) & (sa == 5)
    special = both_g & (sh >= 3) & (sh <= 5) & (sa >= 3) & (sa <= 5)
    return five, special & (sh > sa), special & (sa > sh)


def seconds_in(state: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Seconds of each [start, end) interval spent in state."""
    cum = np.concatenate([[0], np.cumsum(state, dtype=np.int64)])
    return cum[end] - cum[start]


# ------------------------------------------------------------- one season
def zero_counts(entity: str) -> dict:
    return {k: 0.0 for k in pm.COUNTS[entity]}


def zero_zones() -> dict:
    return {z: [0, 0, 0.0] for z in pm.ZONE_KEYS}  # attempts, goals, xG


def build_season(season: int, model: str):
    """Counts, zones and team links for every team, skater and goalie in a season."""
    with pg_conn() as conn:
        read_only(conn)
        games = pd.read_sql(GAMES_SQL, conn, params={"season": season})
        rosters = pd.read_sql(ROSTERS_SQL, conn, params={"season": season})
        shifts = pd.read_sql(SHIFTS_SQL, conn, params={"season": season})
        shots = pd.read_sql(SHOTS_SQL, conn, params={"season": season, "model": model})
    if games.empty:
        return None
    if shots.empty:
        raise RuntimeError(f"No {model} xG for {season}: run model.score_fenwick first.")

    abbr = {}
    for g in games.itertuples():
        abbr[(g.game_id, g.home_team_id)] = g.home_team_abbr
        abbr[(g.game_id, g.away_team_id)] = g.away_team_abbr
    goalie_ids = set(rosters.loc[rosters["position"] == "G", "player_id"])

    team = defaultdict(lambda: zero_counts("team"))
    team_zf, team_za = defaultdict(zero_zones), defaultdict(zero_zones)
    skater = defaultdict(lambda: zero_counts("skater"))
    goalie = defaultdict(lambda: zero_counts("goalie"))
    p_zones = defaultdict(zero_zones)
    p_games = defaultdict(lambda: defaultdict(int))  # player -> team abbr -> games
    # (player, team abbr) -> counts for games with that team only, for the team
    # rosters. The player files keep whole-season totals across every team.
    split = defaultdict(lambda: defaultdict(float))

    shots = shots.copy()
    shots["zone"] = [pm.shot_zone(x, d, a) for x, d, a in
                     zip(shots["x_norm"], shots["distance_ft"], shots["angle_deg"])]
    shots["bucket"] = [pm.strength_bucket(s) for s in shots["strength_state"]]
    shots["is_goal"] = shots["is_goal"].astype(bool)
    shots["empty_net"] = shots["empty_net"].astype(bool)

    # ---- team and player shot counts, straight from the shots
    for r in shots.itertuples(index=False):
        tf = abbr.get((r.game_id, r.shooter_team_id))
        ta = abbr.get((r.game_id, r.defending_team_id))
        if tf is None or ta is None:
            continue
        g, x = int(r.is_goal), float(r.xg)
        cf, ca = team[tf], team[ta]
        cf["gf"] += g; cf["xgf"] += x; cf["af"] += 1
        ca["ga"] += g; ca["xga"] += x; ca["aa"] += 1
        if r.bucket == "5v5":
            cf["gf_5v5"] += g; cf["xgf_5v5"] += x
            ca["ga_5v5"] += g; ca["xga_5v5"] += x
        elif r.bucket == "pp":
            cf["gf_pp"] += g; cf["xgf_pp"] += x
            ca["ga_pk"] += g; ca["xga_pk"] += x
        if r.zone:
            for zz in (team_zf[tf][r.zone], p_zones[int(r.shooter_id)][r.zone]):
                zz[0] += 1; zz[1] += g; zz[2] += x
            zz = team_za[ta][r.zone]
            zz[0] += 1; zz[1] += g; zz[2] += x
        s = skater[int(r.shooter_id)]
        s["att"] += 1; s["goals"] += g; s["ixg"] += x
        sp = split[(int(r.shooter_id), tf)]
        sp["goals"] += g; sp["ixg"] += x
        if not r.empty_net and not pd.isna(r.goalie_id):
            gk = int(r.goalie_id)
            c = goalie[gk]
            c["sa"] += 1; c["ga"] += g; c["xga"] += x
            sp = split[(gk, ta)]
            sp["sa"] += 1; sp["ga"] += g; sp["xga"] += x
            if r.zone:
                zz = p_zones[gk][r.zone]
                zz[0] += 1; zz[1] += g; zz[2] += x

    # ---- per game: time on ice by strength, and who was on for each 5v5 shot
    team_gp = defaultdict(int)
    shifts_by_game = dict(tuple(shifts.groupby("game_id")))
    shots5 = shots[(shots["bucket"] == "5v5") & ~shots["empty_net"]]
    shots5_by_game = dict(tuple(shots5.groupby("game_id")))
    for g in games.itertuples():
        team_gp[g.home_team_abbr] += 1
        team_gp[g.away_team_abbr] += 1
        sh = shifts_by_game.get(g.game_id)
        if sh is None or sh.empty:
            continue
        five, pp_home, pp_away = game_strength(sh, goalie_ids, g.home_team_id, g.away_team_id)
        for tid, t_abbr, pp, pk in ((g.home_team_id, g.home_team_abbr, pp_home, pp_away),
                                    (g.away_team_id, g.away_team_abbr, pp_away, pp_home)):
            c = team[t_abbr]
            c["toi_5v5"] += five.sum() / 60.0
            c["toi_pp"] += pp.sum() / 60.0
            c["toi_pk"] += pk.sum() / 60.0

        st = sh["start_seconds"].to_numpy()
        en = sh["end_seconds"].to_numpy()
        pid = sh["player_id"].to_numpy()
        tid = sh["team_id"].to_numpy()
        dur = en - st
        t5 = seconds_in(five, st, en)
        per_player = defaultdict(lambda: [0, 0])
        for p, d, f, t in zip(pid, dur, t5, tid):
            rec = per_player[(int(p), int(t))]
            rec[0] += int(d); rec[1] += int(f)
        for (p, t), (d, f) in per_player.items():
            t_abbr = abbr.get((g.game_id, t))
            if t_abbr is None:
                continue
            c = goalie[p] if p in goalie_ids else skater[p]
            c["gp"] += 1
            c["toi"] += d / 60.0
            if p not in goalie_ids:
                c["toi_5v5"] += f / 60.0
            p_games[p][t_abbr] += 1
            sp = split[(p, t_abbr)]
            sp["gp"] += 1; sp["toi"] += d / 60.0

        s5 = shots5_by_game.get(g.game_id)
        if s5 is None or s5.empty:
            continue
        sk = ~np.isin(pid, list(goalie_ids))
        t = s5["game_seconds"].to_numpy()
        on = (st[None, :] < t[:, None]) & (en[None, :] >= t[:, None]) & sk[None, :]
        i, j = np.nonzero(on)
        same = tid[j] == s5["shooter_team_id"].to_numpy()[i]
        xg = s5["xg"].to_numpy()
        goal = s5["is_goal"].to_numpy().astype(int)
        for ii, jj, sm in zip(i, j, same):
            p = int(pid[jj])
            c = skater[p]
            sp = split[(p, abbr.get((g.game_id, int(tid[jj]))))]
            if sm:
                c["on_xgf_5v5"] += xg[ii]; c["on_gf_5v5"] += goal[ii]
                sp["on_xgf_5v5"] += xg[ii]
            else:
                c["on_xga_5v5"] += xg[ii]; c["on_ga_5v5"] += goal[ii]
                sp["on_xga_5v5"] += xg[ii]

    for t_abbr, n in team_gp.items():
        team[t_abbr]["gp"] = n

    return {"team": dict(team), "team_zf": dict(team_zf), "team_za": dict(team_za),
            "skater": {p: c for p, c in skater.items() if p not in goalie_ids and c["gp"] > 0},
            "goalie": {p: c for p, c in goalie.items() if c["gp"] > 0},
            "zones": dict(p_zones), "p_games": p_games, "split": split}


# --------------------------------------------------------------- output
def r4(v):
    if v is None:
        return None
    return round(float(v), 4)


def counts_out(c: dict) -> dict:
    return {k: (int(v) if float(v).is_integer() else round(float(v), 3)) for k, v in c.items()}


def zones_out(z: dict) -> dict:
    return {k: [int(a), int(g), round(float(x), 3)] for k, (a, g, x) in z.items()}


def pos_group(pos: str | None) -> str:
    if pos == "G":
        return "G"
    return "D" if pos == "D" else "F"


def write_if_changed(path: Path, payload: dict) -> bool:
    text = json.dumps(payload, separators=(",", ":"))
    kb = len(text.encode()) / 1024
    if kb > PROFILE_BUDGET_KB:
        raise RuntimeError(f"{path.name} is {kb:.1f} KB, over the {PROFILE_BUDGET_KB} KB budget")
    if path.exists() and path.read_text() == text:
        return False
    path.write_text(text)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Export team and player profiles to site/data/.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="shot_xg model_version to use")
    args = ap.parse_args()

    with pg_conn() as conn:
        read_only(conn)
        players = pd.read_sql(PLAYERS_SQL, conn).set_index("player_id")
        seasons = pd.read_sql(SEASONS_SQL, conn)
    positions = players["position"].to_dict()
    name = players["full_name"].to_dict()

    teams_out = defaultdict(list)
    players_out = defaultdict(list)
    league = {}
    season_meta = []
    for srow in seasons.itertuples():
        season = int(srow.season)
        d = build_season(season, args.model)
        if d is None:
            continue
        label = season_label(season)
        # A full season is every team's 82 games, each game counted once.
        full = len(d["team"]) * GAMES_PER_TEAM // 2
        season_meta.append({"season": season, "label": label, "games": int(srow.games),
                            "in_progress": season == int(seasons["season"].max())
                                           and int(srow.games) < full,
                            "first_date": str(srow.first_date), "last_date": str(srow.last_date)})

        sk_by_group = defaultdict(list)
        for p, c in d["skater"].items():
            sk_by_group[pos_group(positions.get(p))].append(c)
        league[str(season)] = {
            "team": {k: r4(v) for k, v in pm.league_values("team", list(d["team"].values())).items()},
            "F": {k: r4(v) for k, v in pm.league_values("skater", sk_by_group["F"]).items()},
            "D": {k: r4(v) for k, v in pm.league_values("skater", sk_by_group["D"]).items()},
            "G": {k: r4(v) for k, v in pm.league_values("goalie", list(d["goalie"].values())).items()},
            "zones": zones_out({z: [sum(tz[z][i] for tz in d["team_zf"].values()) for i in range(3)]
                                for z in pm.ZONE_KEYS}),
        }

        # Ranks: teams among the season's teams, skaters among their group
        # (F or D), goalies among goalies, each only against qualifiers.
        ranks = {"team": pm.rank_rows("team", d["team"]), "skater": {}, "goalie": {}}
        for grp in ("F", "D"):
            ranks["skater"].update(pm.rank_rows(
                "skater", {p: c for p, c in d["skater"].items()
                           if pos_group(positions.get(p)) == grp}))
        ranks["goalie"] = pm.rank_rows("goalie", d["goalie"])

        for kind in ("skater", "goalie"):
            for p, c in d[kind].items():
                vals = pm.compute(kind, c)
                teams = sorted(d["p_games"][p].items(), key=lambda kv: -kv[1])
                players_out[p].append({
                    "season": season, "label": label,
                    "teams": [[t, n] for t, n in teams],
                    "counts": counts_out(c),
                    "values": {k: r4(v) for k, v in vals.items()},
                    "ranks": ranks[kind].get(p, {}),
                    "zones": zones_out(d["zones"].get(p, zero_zones())),
                })

        # Rosters count only the games each player played for that team.
        skaters_on, goalies_on = defaultdict(list), defaultdict(list)
        for (p, t), c in d["split"].items():
            if t is None or c["gp"] <= 0:
                continue
            nm = name.get(p, f"Player {p}")
            if p in d["goalie"]:
                goalies_on[t].append([int(p), nm, int(c["gp"]), round(c["toi"], 1),
                                      int(c["sa"]), int(c["ga"]), round(c["xga"], 2),
                                      round(c["xga"] - c["ga"], 2)])
            elif p in d["skater"]:
                skaters_on[t].append([int(p), nm, positions.get(p), int(c["gp"]),
                                      round(c["toi"], 1), int(c["goals"]), round(c["ixg"], 2),
                                      r4(pm.share(c["on_xgf_5v5"], c["on_xga_5v5"]))])

        for t, c in d["team"].items():
            teams_out[t].append({
                "season": season, "label": label,
                "counts": counts_out(c),
                "values": {k: r4(v) for k, v in pm.compute("team", c).items()},
                "ranks": ranks["team"].get(t, {}),
                "zones_for": zones_out(d["team_zf"].get(t, zero_zones())),
                "zones_against": zones_out(d["team_za"].get(t, zero_zones())),
                "skater_columns": SKATER_ROSTER,
                "skaters": sorted(skaters_on[t], key=lambda r: -r[4]),
                "goalie_columns": GOALIE_ROSTER,
                "goalies": sorted(goalies_on[t], key=lambda r: -r[3]),
            })
        print(f"{label}: {len(d['team'])} teams, {len(d['skater'])} skaters, "
              f"{len(d['goalie'])} goalies")

    (OUT_DIR / "teams").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "players").mkdir(parents=True, exist_ok=True)

    written = 0
    team_list = []
    for t, rows in sorted(teams_out.items()):
        rows.sort(key=lambda r: r["season"], reverse=True)
        payload = {"schema_version": pm.SCHEMA_VERSION, "abbr": t,
                   "name": TEAM_NAMES.get(t, t), "seasons": rows}
        written += write_if_changed(OUT_DIR / "teams" / f"{t}.json", payload)
        team_list.append([t, TEAM_NAMES.get(t, t), rows[-1]["season"], rows[0]["season"]])

    player_list = []
    for p, rows in players_out.items():
        rows.sort(key=lambda r: r["season"], reverse=True)
        pos = positions.get(p)
        payload = {"schema_version": pm.SCHEMA_VERSION, "id": int(p),
                   "name": name.get(p, f"Player {p}"), "pos": pos,
                   "group": pos_group(pos),
                   "kind": "goalie" if pos == "G" else "skater",
                   "shoots": players["shoots_catches"].get(p),
                   "seasons": rows}
        written += write_if_changed(OUT_DIR / "players" / f"{int(p)}.json", payload)
        player_list.append([int(p), payload["name"], pos, rows[0]["teams"][0][0],
                            rows[-1]["season"], rows[0]["season"]])
    player_list.sort(key=lambda r: r[1])

    index = {
        "schema_version": pm.SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": args.model,
        "seasons": sorted(season_meta, key=lambda s: s["season"], reverse=True),
        "zones": [{"key": k, "label": lab, "define": dfn} for k, lab, dfn in pm.ZONES],
        "metrics": pm.registry_json(),
        "league": league,
        "team_columns": ["abbr", "name", "first_season", "last_season"],
        "teams": team_list,
        "player_columns": ["id", "name", "pos", "team", "first_season", "last_season"],
        "players": player_list,
    }
    text = json.dumps(index, separators=(",", ":"))
    kb = len(text.encode()) / 1024
    if kb > INDEX_BUDGET_KB:
        raise RuntimeError(f"profiles_index.json is {kb:.1f} KB, over the {INDEX_BUDGET_KB} KB budget")
    (OUT_DIR / "profiles_index.json").write_text(text)

    sizes = [f.stat().st_size / 1024 for f in (OUT_DIR / "players").glob("*.json")]
    tsizes = [f.stat().st_size / 1024 for f in (OUT_DIR / "teams").glob("*.json")]
    print(f"\nprofiles_index.json: {kb:.0f} KB, {len(team_list)} teams, {len(player_list):,} players")
    print(f"teams/: {len(tsizes)} files, largest {max(tsizes):.1f} KB")
    print(f"players/: {len(sizes):,} files, largest {max(sizes):.1f} KB")
    print(f"{written:,} files written or changed")


if __name__ == "__main__":
    main()
