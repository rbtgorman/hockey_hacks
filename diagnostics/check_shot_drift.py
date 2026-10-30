"""Step 1 of the model tune-up: is the data drifting, or is the model wrong?

v1.2-fenwick's 2025-26 test misses in three places (results/v1_2-fenwick/
segments.csv): it expects 2% too many goals overall, tip-ins score 78 for
every 100 it expects, and distance runs high under 20 ft and past 60 ft but
low at 20-40 ft. Each could be the model or the way the NHL records shots.
These checks split the two, one season per column.

CHECKS
------
1. 'short' misses. Since 2023-24 the API tags some missed shots with the
   reason 'short', which HockeyStats.com describes as closer to fanned
   attempts or flubs. None can be a goal, but each one carries xG, so the
   report shows goals / xG with and without them.
2. Shot types. Each raw label's share of attempts, goal rate and goals / xG
   by season. A label whose share or goal rate moves while the others hold
   points at labelling, not shooting.
3. Distance. Goals / xG by distance band and season, then each rink: the
   average recorded distance in a team's home games minus its road games
   (both teams' attempts each time). Same team in both, different scorers,
   so a large gap points at the rink. z is that gap over its standard error.
   A biased rink also nudges every other team's road average, by about 1/31
   of its own gap, so the extremes are what to read.

Goals / xG on 2022-23 to 2024-25 is in-sample for v1.2-fenwick (it was
refit on them), so drift shows up as a season breaking from the rest.
The first section checks that the rows match the model's own test set.

2026-27 is left out unless --include-forward. It is the next clean test,
so nothing gets tuned while looking at it.

Read-only: one SELECT inside a READ ONLY transaction.

Usage (repo root, with the project venv's python):
    .venv/bin/python3 -m diagnostics.check_shot_drift
    .venv/bin/python3 -m diagnostics.check_shot_drift --out results/checks/shot_drift.txt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import pandas as pd

from ingest.db import pg_conn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "v1.2-fenwick"
TEST_SEASON = 20252026
HISTORY_SEASONS = [20222023, 20232024, 20242025, 20252026]
FORWARD_SEASON = 20262027
BANDS = [0, 20, 40, 60, math.inf]
BAND_LABELS = ["<20 ft", "20-40 ft", "40-60 ft", "60+ ft"]

ROWS_SQL = """
SELECT g.season, g.game_id, g.home_team_abbr, g.away_team_abbr,
       sf.event_type, sf.is_goal, sf.distance_ft,
       COALESCE(s.shot_type, '(none)') AS raw_shot_type,
       s.raw_details_json::jsonb ->> 'reason' AS reason,
       x.xg
FROM shot_features_fenwick sf
JOIN shots s ON s.shot_id = sf.shot_id
JOIN games g ON g.game_id = sf.game_id
LEFT JOIN shot_xg x ON x.shot_id = sf.shot_id AND x.model_version = %(model)s
WHERE g.season = ANY(%(seasons)s)
"""


def season_label(season: int) -> str:
    s = str(season)
    return f"{s[:4]}-{s[6:]}"


def load_rows(model: str, seasons: list[int]) -> pd.DataFrame:
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
        with warnings.catch_warnings():  # pandas prefers SQLAlchemy; a psycopg2 connection works fine
            warnings.simplefilter("ignore", UserWarning)
            df = pd.read_sql(ROWS_SQL, conn, params={"model": model, "seasons": seasons})
    df["season"] = df["season"].map(season_label)
    df["is_short"] = (df["event_type"] == "missed-shot") & (df["reason"] == "short")
    return df


def ratio(num: float, den: float) -> float:
    return num / den if den else float("nan")


# ------------------------------------------------------------ 0. row check
def row_check(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Rows, missing xG and the test season against the model's metrics.json."""
    out = df.groupby("season").agg(attempts=("is_goal", "size"),
                                   goals=("is_goal", "sum"),
                                   missing_xg=("xg", lambda s: int(s.isna().sum())))
    test = season_label(TEST_SEASON)
    metrics = PROJECT_ROOT / "results" / model.replace(".", "_") / "metrics.json"
    out["metrics_json"] = ""
    if metrics.exists() and test in out.index:
        m = json.loads(metrics.read_text())["splits"]["test"]
        t = df[df["season"] == test]
        same = len(t) == m["n"] and abs(ratio(t["is_goal"].sum(), t["xg"].sum()) - m["o_e"]) < 0.0005
        out.loc[test, "metrics_json"] = (f"n {m['n']:,}, O/E {m['o_e']:.3f}: "
                                         + ("match" if same else "DIFFERENT"))
    return out


# --------------------------------------------------------- 1. short misses
def short_misses(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, g in df.groupby("season"):
        sh = g[g["is_short"]]
        goals, xg, xg_short = g["is_goal"].sum(), g["xg"].sum(), sh["xg"].sum()
        rows.append({
            "season": season,
            "games": g["game_id"].nunique(),
            "attempts": len(g),
            "short": len(sh),
            "short_per_game": ratio(len(sh), g["game_id"].nunique()),
            "short_share_%": 100 * ratio(len(sh), len(g)),
            "short_goals": int(sh["is_goal"].sum()),
            "xg_on_short": xg_short,
            "goals_over_xg": ratio(goals, xg),
            "without_short": ratio(goals, xg - xg_short),
        })
    return pd.DataFrame(rows).set_index("season")


# ------------------------------------------------------------ 2. shot types
def shot_types(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    g = df.groupby(["raw_shot_type", "season"]).agg(n=("is_goal", "size"),
                                                   goals=("is_goal", "sum"),
                                                   xg=("xg", "sum"))
    g["share_%"] = 100 * g["n"] / g.groupby("season")["n"].transform("sum")
    g["goal_rate_%"] = 100 * g["goals"] / g["n"]
    g["goals_over_xg"] = g["goals"] / g["xg"]
    order = g.groupby("raw_shot_type")["n"].sum().sort_values(ascending=False).index
    return {col: g[col].unstack("season").reindex(order)
            for col in ["n", "share_%", "goal_rate_%", "goals_over_xg"]}


# --------------------------------------------------------------- 3. distance
def distance_bands(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["distance_ft"]).copy()
    d["band"] = pd.cut(d["distance_ft"], BANDS, labels=BAND_LABELS, right=False)
    g = d.groupby(["band", "season"], observed=True).agg(goals=("is_goal", "sum"), xg=("xg", "sum"))
    return (g["goals"] / g["xg"]).unstack("season")


def rink_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Mean recorded distance in each team's home games minus its road games."""
    d = df.dropna(subset=["distance_ft"])
    rows = []
    for team in sorted(d["home_team_abbr"].unique()):
        home = d[d["home_team_abbr"] == team]
        road = d[d["away_team_abbr"] == team]
        if len(home) < 2 or len(road) < 2:
            continue
        gap = home["distance_ft"].mean() - road["distance_ft"].mean()
        se = math.sqrt(home["distance_ft"].var() / len(home) + road["distance_ft"].var() / len(road))
        row = {"rink": team, "home_games": home["game_id"].nunique(),
               "home_ft": home["distance_ft"].mean(), "road_ft": road["distance_ft"].mean(),
               "gap_ft": gap, "z": ratio(gap, se),
               "home_goals_over_xg": ratio(home["is_goal"].sum(), home["xg"].sum()),
               "road_goals_over_xg": ratio(road["is_goal"].sum(), road["xg"].sum())}
        for season in sorted(d["season"].unique()):
            h, r = home[home["season"] == season], road[road["season"] == season]
            row[f"gap {season}"] = (h["distance_ft"].mean() - r["distance_ft"].mean()
                                   if len(h) and len(r) else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows).set_index("rink").sort_values("z")


# ------------------------------------------------------------------ report
def report(df: pd.DataFrame, model: str) -> str:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    parts = []

    def section(title: str, note: str, table: pd.DataFrame, fmt: str = "{:,.3f}") -> None:
        body = table.to_string(float_format=lambda v: fmt.format(v))
        parts.append(f"\n{'=' * 78}\n{title}\n{'=' * 78}\n{note}\n\n{body}\n")

    section("0. Row check",
            f"Rows per season in shot_features_fenwick, and rows with no {model} xG.\n"
            "The test season should match the model's metrics.json exactly.",
            row_check(df, model), "{:,.0f}")
    section("1. 'short' misses",
            "goals_over_xg is goals / xG for every attempt; without_short drops the\n"
            "xG on 'short' misses. short_goals should be 0.",
            short_misses(df))
    t = shot_types(df)
    section("2a. Shot types: share of attempts (%)", "Raw labels, before wrist and snap are merged.",
            t["share_%"], "{:.1f}")
    section("2b. Shot types: goal rate (%)", "Goals / attempts.", t["goal_rate_%"], "{:.2f}")
    section("2c. Shot types: goals / xG", "1.00 is right. 2022-23 to 2024-25 are in-sample.",
            t["goals_over_xg"])
    section("2d. Shot types: attempts", "How many rows each cell above stands on.", t["n"], "{:,.0f}")
    section("3a. Goals / xG by distance band", "1.00 is right. 2022-23 to 2024-25 are in-sample.",
            distance_bands(df))
    section("3b. Recorded distance by rink",
            "gap_ft: mean distance in the team's home games minus its road games, both\n"
            "teams' attempts. Negative means shots get recorded closer in that rink.\n"
            "|z| > 3 is worth a look; with 32 rinks, one or two past 2 is just chance.\n"
            "A biased rink shifts other teams' road averages by about 1/31 of its gap.",
            rink_gaps(df), "{:.2f}")
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only data drift checks for the xG model.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="shot_xg model_version to use")
    ap.add_argument("--include-forward", action="store_true",
                    help=f"also include {season_label(FORWARD_SEASON)} (the next clean test)")
    ap.add_argument("--out", type=Path, help="also write the report to this file")
    args = ap.parse_args()

    seasons = HISTORY_SEASONS + ([FORWARD_SEASON] if args.include_forward else [])
    df = load_rows(args.model, seasons)
    if df.empty:
        sys.exit("No rows: build shot_features_fenwick and run model.score_fenwick first.")
    text = (f"Shot drift checks, model {args.model}, seasons "
            f"{', '.join(season_label(s) for s in seasons)}\n" + report(df, args.model))
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
