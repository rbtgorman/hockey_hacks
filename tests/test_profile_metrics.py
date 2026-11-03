"""Test export/profile_metrics against hand-worked cases."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from export import profile_metrics as pm


def check(name, got, want, tol=1e-9):
    if isinstance(want, float) and got is not None:
        ok = abs(got - want) <= tol
    else:
        ok = got == want
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    return ok


def main():
    failures = 0

    # Strength buckets, from the shooter's side
    for state, want in (("5v5", "5v5"), ("5v4", "pp"), ("4v5", "pk"), ("5v3", "pp"),
                        ("4v3", "pp"), ("3v5", "pk"), ("4v4", None), ("3v3", None),
                        ("6v5", None), ("5v6", None), ("1v0", None), (None, None), ("x", None)):
        if not check(f"bucket {state}", pm.strength_bucket(state), want): failures += 1

    # Zones: rules apply in order
    for args, want in (((95.0, 8.0, 80.0), "behind"), ((82.0, 7.0, 10.0), "crease"),
                       ((70.0, 20.0, 15.0), "slot"), ((75.0, 25.0, 50.0), "low"),
                       ((50.0, 40.0, 10.0), "high_slot"), ((60.0, 40.0, 45.0), "circles"),
                       ((30.0, 60.0, 5.0), "point"), ((10.0, 80.0, 3.0), "long"),
                       ((None, 10.0, 0.0), None), ((50.0, float("nan"), 0.0), None)):
        if not check(f"zone {args}", pm.shot_zone(*args), want): failures += 1

    # Every metric's sample is a count, every count is listed, keys are unique
    for entity, counts in pm.COUNTS.items():
        ms = pm.metrics_for(entity)
        keys = [m["key"] for m in ms]
        if not check(f"{entity} keys unique", len(keys), len(set(keys))): failures += 1
        for m in ms:
            if m["sample"] not in counts:
                print(f"  [FAIL] {entity}.{m['key']} samples unknown count {m['sample']}")
                failures += 1
        zero = {k: 0 for k in counts}
        try:
            pm.compute(entity, zero)
            print(f"  [OK] {entity} metrics survive all-zero counts")
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {entity} metrics on zero counts: {e!r}")
            failures += 1

    # Hand-worked team-season
    team = {k: 0.0 for k in pm.TEAM_COUNTS}
    team.update(gp=10, gf=30, ga=20, xgf=27.0, xga=23.0, xgf_5v5=12.0, xga_5v5=8.0,
                gf_5v5=9, ga_5v5=6, toi_5v5=480.0, gf_pp=4, xgf_pp=3.0, toi_pp=45.0)
    v = pm.compute("team", team)
    if not check("gf_pg", v["gf_pg"], 3.0): failures += 1
    if not check("xgf_pct", v["xgf_pct"], 0.54): failures += 1
    if not check("g_minus_xg", v["g_minus_xg"], 3.0): failures += 1
    if not check("xgf60_5v5", v["xgf60_5v5"], 1.5): failures += 1
    if not check("gf_pct_5v5", v["gf_pct_5v5"], 0.6): failures += 1
    if not check("xgf60_pp", v["xgf60_pp"], 4.0): failures += 1
    if not check("xga60_pk with no PK time", v["xga60_pk"], None): failures += 1

    # Goalie: 1,000 shots, 70 goals on 75 xG
    g = {"gp": 20, "toi": 1200.0, "sa": 1000, "ga": 70, "xga": 75.0}
    v = pm.compute("goalie", g)
    if not check("gsax", v["gsax"], 5.0): failures += 1
    if not check("gsax100", v["gsax100"], 0.5): failures += 1
    if not check("fsv", v["fsv"], 0.93): failures += 1

    # League: pooled is the metric on summed counts, mean averages the values
    a = dict(team, gf=30, xgf=27.0, gp=10)
    b = dict(team, gf=10, xgf=13.0, gp=10)
    lg = pm.league_values("team", [a, b])
    if not check("league gf_pg pooled", lg["gf_pg"], 2.0): failures += 1
    if not check("league g_minus_xg mean", lg["g_minus_xg"], 0.0): failures += 1
    if not check("no league figure for games", "gp" in lg, False): failures += 1

    # The registry the pages read has no functions in it
    reg = pm.registry_json()
    if not check("registry has no calc", any("calc" in m for ms in reg.values() for m in ms), False):
        failures += 1

    # ------------------------------------------------------------ ranks
    def team_with(**kw):
        c = {k: 0.0 for k in pm.TEAM_COUNTS}
        c.update(gp=20, toi_5v5=900.0, toi_pp=100.0, toi_pk=100.0)
        c.update(kw)
        return c

    # Direction: most goals is 1st for goals per game, fewest is 1st against
    teams = {i: team_with(gf=20 + i, ga=60 - i) for i in range(32)}
    r = pm.rank_rows("team", teams)
    if not check("high is better: most goals 1st", r[31]["gf_pg"][0], 1): failures += 1
    if not check("high is better: fewest goals 32nd", r[0]["gf_pg"][0], 32): failures += 1
    if not check("low is better: fewest against 1st", r[31]["ga_pg"][0], 1): failures += 1
    if not check("low is better: most against 32nd", r[0]["ga_pg"][0], 32): failures += 1
    if not check("pool size", r[0]["gf_pg"][1], 32): failures += 1
    if not check("no rank for games", "gp" in r[0], False): failures += 1
    if not check("no rank for PP minutes", "toi_pp" in r[0], False): failures += 1

    # 11 / 10 / 11 at 32
    tiers = [pm.tier(k, 32) for k in range(1, 33)]
    if not check("32: top third is 11", tiers.count("top"), 11): failures += 1
    if not check("32: middle third is 10", tiers.count("mid"), 10): failures += 1
    if not check("32: bottom third is 11", tiers.count("bot"), 11): failures += 1
    if not check("32: 11th top, 12th mid", (pm.tier(11, 32), pm.tier(12, 32)), ("top", "mid")):
        failures += 1
    if not check("32: 21st mid, 22nd bot", (pm.tier(21, 32), pm.tier(22, 32)), ("mid", "bot")):
        failures += 1
    if not check("tier stored with rank", r[31]["gf_pg"][3], "top"): failures += 1

    # Ties at the precision shown share the better rank. 2.004 and 2.001 both
    # show as 2.00; two teams sit above them.
    tied = {i: team_with(gf=20 * (1 + i / 1000)) for i in range(10)}
    tied.update({"a": team_with(gf=50.0), "b": team_with(gf=48.0),
                 "x": team_with(gf=40.08), "y": team_with(gf=40.02)})
    r = pm.rank_rows("team", tied)
    if not check("tie shares the better rank", (r["x"]["gf_pg"][0], r["y"]["gf_pg"][0]), (3, 3)):
        failures += 1
    if not check("tie is flagged", (r["x"]["gf_pg"][2], r["y"]["gf_pg"][2]), (1, 1)): failures += 1
    if not check("next rank skips past the tie", r[9]["gf_pg"][0], 5): failures += 1
    if not check("untied not flagged", r["a"]["gf_pg"][2], 0): failures += 1
    if not check("pct ties at 0.1%", pm.shown(0.5003, "pct") == pm.shown(0.5001, "pct"), True):
        failures += 1
    if not check("pct differs at 0.1%", pm.shown(0.5008, "pct") == pm.shown(0.5001, "pct"), False):
        failures += 1
    if not check("dec2 like toFixed: 2.675 is 2.67 in binary", str(pm.shown(2.675, "dec2")), "2.67"):
        failures += 1
    if not check("signed shows half up", pm.shown(1.25, "signed"), 1.3): failures += 1

    # Players below a metric's minimum aren't in the pool and get no rank
    def skater(toi, att, goals):
        c = {k: 0.0 for k in pm.SKATER_COUNTS}
        c.update(gp=20, toi=toi, toi_5v5=toi * 0.8, att=att, goals=goals, ixg=att * 0.07)
        return c
    pool = {i: skater(300.0, 60, i) for i in range(12)}
    pool["short"] = skater(50.0, 60, 30)      # under 100 minutes: out of the per-60 pools
    r = pm.rank_rows("skater", pool)
    if not check("below minimum gets no rank", "goals60" in r["short"], False): failures += 1
    if not check("below minimum not in the pool", r[0]["goals60"][1], 12): failures += 1
    if not check("best qualifier is 1st, not beaten by the short sample",
                 r[11]["goals60"][0], 1): failures += 1
    # Same player still ranked where it qualifies (goals minus xG needs 50 shots)
    if not check("ranked where it qualifies", "g_minus_ixg" in r["short"], True): failures += 1

    # Fewer than 10 qualifiers: nobody gets a rank for that metric
    small = {i: skater(300.0, 60, i) for i in range(9)}
    small.update({f"s{i}": skater(50.0, 60, i) for i in range(5)})
    r = pm.rank_rows("skater", small)
    if not check("9 qualifiers: no ranks", any("goals60" in v for v in r.values()), False):
        failures += 1
    if not check("14 qualify for another metric: ranked", r[0]["g_minus_ixg"][1], 14):
        failures += 1

    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
