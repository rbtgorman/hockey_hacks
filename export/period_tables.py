"""Export period scoring tables for site/periods.html as static JSON.

WHAT IT WRITES (site/data/)
---------------------------
    periods_index.json      seasons, metric definitions, league baselines,
                            shrinkage constants, xG scaling factors, model version
    periods_<season>.json   one row per team per regular-season game

Rows carry counts and 0/1 outcomes, not percentages, so the page can rebuild
any window (season, last 10) with its sample size and nothing is rounded twice.

DEFINITIONS
-----------
These follow the period charts bettors use (reports/Period scoring xG website
plan.md checks them against NHL data):
    regulation only   periods 1-3; overtime and shootout goals never count
    empty-net goals   count, and are kept apart so the 3rd period can also be
                      shown without them
    first 10 minutes  a goal by either team before 10:00 elapsed in the 1st
    over 1.5          both teams together score 2+ in the period

BASELINES AND SHRINKAGE
-----------------------
The league rate is pooled over HISTORY_SEASONS. k is Tango's regression
constant from the same team-seasons (export/period_math.regression_k): a team's
shrunk rate is (hits + k * league) / (games + k). For most period metrics k is
in the hundreds of games, so early-season team rates shrink almost all the way
to the league rate. That is the finding, not a bug.

xG
--
From shot_xg (model/score_fenwick.py), unblocked attempts, periods 1-3. An
expected outcome treats each game's actual shots as independent chances
(Poisson-binomial). That runs low on "2+ goals" and "both teams score" and high
on scoreless periods, so each expected metric is scaled by its observed /
expected ratio over HISTORY_SEASONS (xg_factor in the index). Ties get no xG
expectation at all: shot independence misses them by about 7 points.

Usage (repo root, with the project venv's python):
    .venv/bin/python3 -m export.period_tables
    .venv/bin/python3 -m export.period_tables --model v1.1-fenwick

export/refresh.sh runs the whole chain for the season in progress: ingest,
shifts, feature rebuild, scoring, then this export.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ingest.db import pg_conn

from .period_math import at_least, convolve, count_distribution, regression_k

OUT_DIR = Path(__file__).resolve().parent.parent / "site" / "data"
HISTORY_SEASONS = [20222023, 20232024, 20242025, 20252026]
DEFAULT_MODEL = "v1.2-fenwick"
FIRST_TEN = 600  # seconds elapsed in the 1st period

# key, label, group, game_level (same outcome for both teams), has_xg
METRICS = [
    ("p1_ov15", "1st period: 2+ goals", "Period totals (both teams)", True, True),
    ("p2_ov15", "2nd period: 2+ goals", "Period totals (both teams)", True, True),
    ("p3_ov15", "3rd period: 2+ goals", "Period totals (both teams)", True, True),
    ("p3_ov15_noen", "3rd period: 2+ goals, empty-net goals removed",
     "Period totals (both teams)", True, True),
    ("f10", "Goal in the first 10 minutes", "Period totals (both teams)", True, True),
    ("p1_tt_for", "1st period: team scores 2+", "1st period", False, True),
    ("p1_tt_against", "1st period: opponent scores 2+", "1st period", False, True),
    ("p1_shut_for", "1st period: team scores 0", "1st period", False, True),
    ("p1_shut_against", "1st period: opponent scores 0", "1st period", False, True),
    ("p1_00", "1st period ends 0-0", "1st period", True, True),
    ("p1_btts", "1st period: both teams score", "1st period", True, True),
    ("p1_tied", "1st period: tied", "1st period", True, False),
    ("p2_btts", "2nd period: both teams score", "2nd period and regulation", True, True),
    ("reg_tied", "Tied after regulation", "2nd period and regulation", True, False),
    ("reg_ov55", "Regulation: 6+ goals", "2nd period and regulation", True, True),
]
KEYS = [m[0] for m in METRICS]
XG_KEYS = [m[0] for m in METRICS if m[4]]

ROW_COLUMNS = (["team", "date", "opp", "home",
                "gf1", "gf2", "gf3", "ga1", "ga2", "ga3", "enf3", "ena3",
                "xgf1", "xgf2", "xgf3", "xga1", "xga2", "xga3"]
               + [f"y_{k}" for k in KEYS] + [f"e_{k}" for k in XG_KEYS])

GAMES_SQL = """
SELECT game_id, season, game_date, home_team_id, away_team_id,
       home_team_abbr, away_team_abbr
FROM games
WHERE game_type = 2 AND game_state = 'OFF'
ORDER BY game_date, game_id
"""
GOALS_SQL = """
SELECT s.game_id, s.period, s.period_seconds, s.shooter_team_id, s.empty_net
FROM shots s JOIN games g USING (game_id)
WHERE g.game_type = 2 AND s.is_goal AND s.period BETWEEN 1 AND 3
"""
XG_SQL = """
SELECT s.game_id, s.period, s.period_seconds, s.shooter_team_id, s.empty_net, x.xg
FROM shot_xg x
JOIN shots s USING (shot_id)
JOIN games g ON g.game_id = s.game_id
WHERE x.model_version = %(model)s AND g.game_type = 2 AND s.period BETWEEN 1 AND 3
"""


def season_label(season: int) -> str:
    s = str(season)
    return f"{s[:4]}-{s[6:]}"


def expected(shots: dict) -> dict:
    """Unscaled expected outcomes for one game, from the home side's view.

    shots[(side, period)] is a list of (xg, period_seconds, empty_net) with side
    'home' or 'away'. Returns {metric: probability} for the home team; the
    away row swaps for/against.
    """
    def dist(side, period, no_en=False):
        return count_distribution(p for p, _, en in shots.get((side, period), [])
                                  if not (no_en and en))

    h = {per: dist("home", per) for per in (1, 2, 3)}
    a = {per: dist("away", per) for per in (1, 2, 3)}
    tot = {per: convolve(h[per], a[per]) for per in (1, 2, 3)}
    tot3_noen = convolve(dist("home", 3, no_en=True), dist("away", 3, no_en=True))
    reg = convolve(convolve(tot[1], tot[2]), tot[3])
    no_goal_first_ten = 1.0
    for side in ("home", "away"):
        for p, secs, _ in shots.get((side, 1), []):
            if secs < FIRST_TEN:
                no_goal_first_ten *= 1.0 - p
    return {
        "p1_ov15": at_least(tot[1], 2),
        "p2_ov15": at_least(tot[2], 2),
        "p3_ov15": at_least(tot[3], 2),
        "p3_ov15_noen": at_least(tot3_noen, 2),
        "f10": 1.0 - no_goal_first_ten,
        "p1_tt_for": at_least(h[1], 2),
        "p1_tt_against": at_least(a[1], 2),
        "p1_shut_for": h[1][0],
        "p1_shut_against": a[1][0],
        "p1_00": tot[1][0],
        "p1_btts": (1.0 - h[1][0]) * (1.0 - a[1][0]),
        "p2_btts": (1.0 - h[2][0]) * (1.0 - a[2][0]),
        "reg_ov55": at_least(reg, 6),
    }


def outcomes(gf: list[int], ga: list[int], enf3: int, ena3: int, f10: bool) -> dict:
    """0/1 outcomes for one team-game. gf/ga are goals in periods 1-3."""
    return {
        "p1_ov15": gf[0] + ga[0] >= 2,
        "p2_ov15": gf[1] + ga[1] >= 2,
        "p3_ov15": gf[2] + ga[2] >= 2,
        "p3_ov15_noen": (gf[2] - enf3) + (ga[2] - ena3) >= 2,
        "f10": f10,
        "p1_tt_for": gf[0] >= 2,
        "p1_tt_against": ga[0] >= 2,
        "p1_shut_for": gf[0] == 0,
        "p1_shut_against": ga[0] == 0,
        "p1_00": gf[0] + ga[0] == 0,
        "p1_btts": gf[0] >= 1 and ga[0] >= 1,
        "p1_tied": gf[0] == ga[0],
        "p2_btts": gf[1] >= 1 and ga[1] >= 1,
        "reg_tied": sum(gf) == sum(ga),
        "reg_ov55": sum(gf) + sum(ga) >= 6,
    }


def swap(exp: dict) -> dict:
    """Home-view expectations as seen by the away team."""
    out = dict(exp)
    out["p1_tt_for"], out["p1_tt_against"] = exp["p1_tt_against"], exp["p1_tt_for"]
    out["p1_shut_for"], out["p1_shut_against"] = exp["p1_shut_against"], exp["p1_shut_for"]
    return out


def build_rows(model: str) -> pd.DataFrame:
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
        games = pd.read_sql(GAMES_SQL, conn)
        goals = pd.read_sql(GOALS_SQL, conn)
        xg = pd.read_sql(XG_SQL, conn, params={"model": model})
    if xg.empty:
        raise RuntimeError(f"No shot_xg rows for {model}: run model.score_fenwick first.")

    side_of = {}
    for g in games.itertuples():
        side_of[(g.game_id, g.home_team_id)] = "home"
        side_of[(g.game_id, g.away_team_id)] = "away"

    gl = defaultdict(lambda: {"home": [0, 0, 0], "away": [0, 0, 0],
                              "en_home": 0, "en_away": 0, "f10": False})
    for r in goals.itertuples():
        side = side_of.get((r.game_id, r.shooter_team_id))
        if side is None:
            continue
        rec = gl[r.game_id]
        rec[side][r.period - 1] += 1
        if r.empty_net and r.period == 3:
            rec[f"en_{side}"] += 1
        if r.period == 1 and r.period_seconds < FIRST_TEN:
            rec["f10"] = True

    shots = defaultdict(lambda: defaultdict(list))
    xg_sum = defaultdict(lambda: {"home": [0.0, 0.0, 0.0], "away": [0.0, 0.0, 0.0]})
    for r in xg.itertuples():
        side = side_of.get((r.game_id, r.shooter_team_id))
        if side is None:
            continue
        shots[r.game_id][(side, r.period)].append((r.xg, r.period_seconds, bool(r.empty_net)))
        xg_sum[r.game_id][side][r.period - 1] += r.xg

    rows = []
    for g in games.itertuples():
        rec = gl[g.game_id]
        exp_home = expected(shots[g.game_id])
        for side, team, opp, home in (("home", g.home_team_abbr, g.away_team_abbr, 1),
                                      ("away", g.away_team_abbr, g.home_team_abbr, 0)):
            other = "away" if side == "home" else "home"
            gf, ga = rec[side], rec[other]
            y = outcomes(gf, ga, rec[f"en_{side}"], rec[f"en_{other}"], rec["f10"])
            e = exp_home if side == "home" else swap(exp_home)
            xs = xg_sum[g.game_id]
            rows.append({
                "season": g.season, "game_id": g.game_id,
                "team": team, "date": str(g.game_date), "opp": opp, "home": home,
                "gf1": gf[0], "gf2": gf[1], "gf3": gf[2],
                "ga1": ga[0], "ga2": ga[1], "ga3": ga[2],
                "enf3": rec[f"en_{side}"], "ena3": rec[f"en_{other}"],
                **{f"xgf{i + 1}": round(xs[side][i], 3) for i in range(3)},
                **{f"xga{i + 1}": round(xs[other][i], 3) for i in range(3)},
                **{f"y_{k}": int(y[k]) for k in KEYS},
                **{f"e_{k}": round(e[k], 4) for k in XG_KEYS},
            })
    return pd.DataFrame(rows)


def metric_summaries(df: pd.DataFrame) -> list[dict]:
    hist = df[df["season"].isin(HISTORY_SEASONS)]
    out = []
    for key, label, group, game_level, has_xg in METRICS:
        y = hist[f"y_{key}"]
        per_team = (hist.groupby(["season", "team"])[f"y_{key}"].agg(["sum", "size"])
                        .reset_index())
        league_by_season = hist.groupby("season")[f"y_{key}"].mean()
        k = regression_k((int(r["sum"]), int(r["size"]), float(league_by_season[r["season"]]))
                         for _, r in per_team.iterrows())
        factor = None
        if has_xg:
            factor = float(y.sum() / hist[f"e_{key}"].sum())
        out.append({
            "key": key, "label": label, "group": group,
            "game_level": game_level, "has_xg": has_xg,
            "baseline": round(float(y.mean()), 4),
            "baseline_by_season": {str(s): round(float(v), 4)
                                   for s, v in league_by_season.items()},
            "k": None if k is None else round(k, 1),
            "xg_factor": None if factor is None else round(factor, 4),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Export period tables to site/data/.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="shot_xg model_version to use")
    args = ap.parse_args()

    df = build_rows(args.model)
    metrics = metric_summaries(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seasons = []
    for season, part in df.groupby("season"):
        fname = f"periods_{season}.json"
        payload = {"season": int(season), "label": season_label(season),
                   "columns": ROW_COLUMNS,
                   "rows": json.loads(part[ROW_COLUMNS].to_json(orient="values",
                                                                double_precision=4))}
        (OUT_DIR / fname).write_text(json.dumps(payload, separators=(",", ":")))
        seasons.append({"season": int(season), "label": season_label(season), "file": fname,
                        "games": int(part["game_id"].nunique()),
                        "first_date": part["date"].min(), "last_date": part["date"].max()})
        print(f"{fname}: {part['game_id'].nunique():,} games, {len(part):,} team rows, "
              f"{(OUT_DIR / fname).stat().st_size / 1024:.0f} KB")

    index = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": args.model,
        "history_seasons": HISTORY_SEASONS,
        "seasons": sorted(seasons, key=lambda s: s["season"], reverse=True),
        "metrics": metrics,
    }
    (OUT_DIR / "periods_index.json").write_text(json.dumps(index, indent=1))

    print("\nmetric            baseline      k  xg_factor")
    for m in metrics:
        k = "-" if m["k"] is None else f"{m['k']:.0f}"
        f = "-" if m["xg_factor"] is None else f"{m['xg_factor']:.3f}"
        print(f"{m['key']:16} {m['baseline']:8.4f} {k:>6} {f:>10}")


if __name__ == "__main__":
    main()
