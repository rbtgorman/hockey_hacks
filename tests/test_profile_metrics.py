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

    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
