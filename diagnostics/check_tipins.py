"""Read-only check: why tip-ins go in less than v1.2-fenwick expects.

The drift check (check_shot_drift.py) found tip-ins at 0.79 goals per xG in
2025-26 while their recorded distance stayed flat. This splits tip-ins by
season, on-net vs missed, distance band and home rink to tell a change in
how plays get labelled apart from a change in how the model sees them.

Run: .venv/bin/python -m diagnostics.check_tipins > results/checks/tipins.txt
"""
from __future__ import annotations

import warnings

import pandas as pd

from ingest.db import pg_conn

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

MODEL = "v1.2-fenwick"
SQL = """
SELECT g.season, g.game_id, g.home_team_abbr AS home, f.event_type,
       f.is_goal::int AS g, f.distance_ft AS d, f.is_rebound::int AS reb, x.xg
FROM shot_features_fenwick f
JOIN shots s ON s.shot_id = f.shot_id
JOIN shot_xg x ON x.shot_id = f.shot_id AND x.model_version = %(model)s
JOIN games g ON g.game_id = f.game_id
WHERE g.game_type = 2 AND s.raw_details_json->>'shotType' = 'tip-in'
"""


def section(title: str, note: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78 + f"\n{note}\n")


def main() -> None:
    with pg_conn() as conn:
        df = pd.read_sql(SQL, conn, params={"model": MODEL})
    s = df.season.astype(str)
    df["season"] = s.str[:4] + "-" + s.str[6:]
    print(f"Tip-in checks, model {MODEL}. Latest season is in progress.")

    section("1. Tip-ins by season",
            "gr = goal rate, oe = goals / xG, miss = share that missed the net,\n"
            "close = share recorded within 10 ft.")
    t = df.groupby("season").agg(n=("g", "size"), goals=("g", "sum"), gr=("g", "mean"),
                                 xg=("xg", "mean"), d=("d", "mean"), reb=("reb", "mean"))
    t["oe"] = df.groupby("season").g.sum() / df.groupby("season").xg.sum()
    t["miss"] = df.assign(m=df.event_type.eq("missed-shot")).groupby("season").m.mean()
    t["close"] = df.assign(c=df.d.le(10)).groupby("season").c.mean()
    print(t.round(4).to_string())

    section("2. On-net tip-ins only",
            "Goals / on-net tip-ins. A fall here is not explained by more misses.")
    on = df[df.event_type != "missed-shot"]
    print(on.groupby("season").agg(n=("g", "size"), gr=("g", "mean"), xg=("xg", "mean"))
          .round(4).to_string())

    section("3. Goals / xG and attempts by distance band", "")
    df["band"] = pd.cut(df.d, [0, 10, 20, 30, 200])
    b = df.groupby(["band", "season"], observed=True).agg(n=("g", "size"), G=("g", "sum"),
                                                          X=("xg", "sum"))
    print((b.G / b.X).unstack().round(3).to_string())
    print()
    print(b.n.unstack().to_string())

    section("4. Tip-ins within 10 ft per game, by home rink",
            "League-wide rises point at a scoring change; a few rinks at a rink effect.")
    df["close"] = df.d.le(10).astype(int)
    per = (df.groupby(["home", "season"]).close.sum()
           / df.groupby(["home", "season"]).game_id.nunique()).unstack()
    seasons = sorted(df.season.unique())
    full = [x for x in seasons[:-1]][-2:]
    per["diff"] = per[full[1]] - per[full[0]]
    both = per.dropna(subset=["diff"])
    print(per.sort_values("diff").round(2).to_string())
    print(f"\nRinks up between {full[0]} and {full[1]}: {(both['diff'] > 0).sum()} of {len(both)}; "
          f"median change {per['diff'].median():+.2f} per game")


if __name__ == "__main__":
    main()
