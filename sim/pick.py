"""Simulator v0, step 2: pick v0's settings and Elo's on 2023-24 to 2025-26, freeze them.

2022-23 is history only, so the first 2023-24 games have something to go on. 2026-27 is
never loaded.

v0's grid: H (games until a game's weight halves), K5 and Kst (games of average play
added to the 5-on-5 and the special-teams ratings) and keep (share of a rating's distance
from average kept over the summer). Stage 1 runs every setting at STAGE1_SIMS simulations
per game, all with the same random numbers so settings are compared on equal luck. Stage
2 reruns the best FINALISTS at FINAL_SIMS (the number the plan uses), and the lowest log
loss there is the pick. Elo's grid (K, home bonus, keep) is picked on the same games the
same way (it's exact, no simulation).

Writes results/sim_v0/: selection.json (frozen), grid.csv, finalists.csv, elo_grid.csv,
build_predictions.csv, build_report.txt.

Run: .venv/bin/python -m sim.pick
"""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from sim import elo as EL
from sim import engine as E
from sim import evaluate as EV
from sim import ratings as R
from sim.data import MIRROR, PP_STATES, S_IDX, SH_STATES, load

HISTORY = [20222023]
BUILD = [20232024, 20242025, 20252026]
GRID = dict(H=[10, 20, 40, 80], K5=[5, 10, 20, 40], Kst=[5, 15, 40], keep=[0.25, 0.5, 0.75, 1.0])
ELO_GRID = dict(K=[2, 4, 6, 8, 10, 12, 15, 20, 25], home_bonus=list(range(0, 90, 10)),
                keep=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
STAGE1_SIMS, FINAL_SIMS, FINALISTS = 1_000, 10_000, 5
SEED = 20261117
WORKERS = 4
OUT = Path(__file__).resolve().parent.parent / "results" / "sim_v0"

D, LG, BUILD_IDX, Y = None, None, None, None


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def setup():
    global D, LG, BUILD_IDX, Y
    D, issues = load(HISTORY + BUILD)
    LG = R.league(D)
    BUILD_IDX = np.flatnonzero(np.isin(D["season"], BUILD))
    Y = D["home_won"][BUILD_IDX].astype(float)
    return issues


def run_setting(H, keep, K5, Kst, n_sims, track_time=False):
    totals = R.team_totals(D, LG, H, keep)
    rates = R.game_rates(R.ratings(*totals, LG["e1"], K5, Kst), LG)
    return E.simulate(rates, BUILD_IDX, n_sims, SEED, track_time=track_time, chunk_rows=1_000_000)


def stage1_worker(args):
    H, keep = args
    totals = R.team_totals(D, LG, H, keep)
    rows = []
    for K5, Kst in itertools.product(GRID["K5"], GRID["Kst"]):
        rates = R.game_rates(R.ratings(*totals, LG["e1"], K5, Kst), LG)
        o = E.simulate(rates, BUILD_IDX, STAGE1_SIMS, SEED, chunk_rows=1_000_000)
        rows.append((H, K5, Kst, keep, EV.log_loss(o["p_home"], Y)))
    log(f"  stage 1: H={H} keep={keep} done")
    return rows


def finalist_worker(s):
    o = run_setting(s["H"], s["keep"], s["K5"], s["Kst"], FINAL_SIMS, track_time=True)
    log(f"  stage 2: {s} done")
    return s, o


def elo_pick():
    rows = []
    for K, hb, keep in itertools.product(ELO_GRID["K"], ELO_GRID["home_bonus"], ELO_GRID["keep"]):
        p = EL.elo(D, K, hb, keep)[BUILD_IDX]
        rows.append((K, hb, keep, EV.log_loss(p, Y)))
    rows.sort(key=lambda r: r[3])
    return rows


def on_edge(value, grid):
    return value in (grid[0], grid[-1])


def fmt_range(r, nd=4):
    return f"({r[0]:.{nd}f} to {r[1]:.{nd}f})"


def main():
    t0 = time.time()
    issues = setup()
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"{len(BUILD_IDX)} build games; stage 1: {np.prod([len(v) for v in GRID.values()])} settings")

    with Pool(WORKERS) as pool:
        parts = pool.map(stage1_worker, list(itertools.product(GRID["H"], GRID["keep"])))
    grid = sorted([r for p in parts for r in p], key=lambda r: r[4])
    with open(OUT / "grid.csv", "w") as f:
        f.write("H,K5,Kst,keep,log_loss_1000_sims\n")
        for r in grid:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.6f}\n")
    finalists = [dict(H=r[0], K5=r[1], Kst=r[2], keep=r[3]) for r in grid[:FINALISTS]]
    with Pool(min(WORKERS, FINALISTS)) as pool:
        results = pool.map(finalist_worker, finalists)
    fin = sorted([(s, EV.log_loss(o["p_home"], Y), o) for s, o in results], key=lambda r: r[1])
    with open(OUT / "finalists.csv", "w") as f:
        f.write("H,K5,Kst,keep,log_loss_1000_sims,log_loss_10000_sims\n")
        ll1 = {(r[0], r[1], r[2], r[3]): r[4] for r in grid}
        for s, ll, _ in fin:
            f.write(f"{s['H']},{s['K5']},{s['Kst']},{s['keep']},"
                    f"{ll1[(s['H'], s['K5'], s['Kst'], s['keep'])]:.6f},{ll:.6f}\n")
    pick, pick_ll, o = fin[0]
    log(f"v0 pick {pick} log loss {pick_ll:.5f}  ({time.time() - t0:.0f}s)")

    elo_rows = elo_pick()
    with open(OUT / "elo_grid.csv", "w") as f:
        f.write("K,home_bonus,keep,log_loss\n")
        for r in elo_rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.6f}\n")
    eK, ehb, ekeep, _ = elo_rows[0]
    p_elo = EL.elo(D, eK, ehb, ekeep)[BUILD_IDX]
    p_home = EL.home_rate(D)[BUILD_IDX]
    p_v0 = o["p_home"]

    # per-game predictions
    with open(OUT / "build_predictions.csv", "w") as f:
        f.write("game_id,season,date,home_won,went_ot,went_so,v0_home_win,v0_reg_home,v0_reg_away,"
                "v0_ot,v0_so,v0_mean_goals,elo_home_win,home_rate\n")
        mg = (o["total_hist"] * np.arange(o["total_hist"].shape[1])).sum(axis=1)
        for j, k in enumerate(BUILD_IDX):
            f.write(f"{D['game_id'][k]},{D['season'][k]},{D['date'][k]},{int(D['home_won'][k])},"
                    f"{int(D['went_ot'][k])},{int(D['went_so'][k])},{p_v0[j]:.5f},"
                    f"{o['p_home_reg'][j]:.5f},{o['p_away_reg'][j]:.5f},{o['p_ot'][j]:.5f},"
                    f"{o['p_so'][j]:.5f},{mg[j]:.3f},{p_elo[j]:.5f},{p_home[j]:.5f}\n")

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, cwd=OUT.parent.parent).stdout.strip()
    except OSError:
        commit = ""
    selection = dict(
        frozen="2026-11-17", code_commit=commit, picked_on="2023-24 to 2025-26 regular season",
        history="2022-23", test="2026-27 at 800 games, once (not touched)",
        v0=dict(**pick, n_sims=FINAL_SIMS, seed=SEED, league_window_games=R.LEAGUE_WINDOW,
                xg_model="v1.4.1-fenwick", log_loss_build=round(pick_ll, 6)),
        elo=dict(K=eK, home_bonus=ehb, keep=ekeep, log_loss_build=round(elo_rows[0][3], 6)),
        home_rate="league home-win rate from the seasons before the game's",
        grids=dict(v0=GRID, elo=ELO_GRID, stage1_sims=STAGE1_SIMS, finalists=FINALISTS),
    )
    with open(OUT / "selection.json", "w") as f:
        json.dump(selection, f, indent=2)

    report(pick, o, p_v0, p_elo, p_home, elo_rows, grid, fin, issues, (eK, ehb, ekeep))
    log(f"done in {time.time() - t0:.0f}s")


def report(pick, o, p_v0, p_elo, p_home, elo_rows, grid, fin, issues, elo_pick_):
    y = Y
    seasons = D["season"][BUILD_IDX]
    lines = []
    w = lines.append
    w("SIMULATOR v0, STEP 2: SETTINGS PICKED ON 2023-24 TO 2025-26 (report)")
    w("=" * 92)
    w(f"Games: {len(BUILD_IDX):,} regular-season games (2022-23 used only as history). "
      "2026-27 not loaded.")
    w(f"Power-play calls the shift charts didn't confirm (left out), 2022-23 to 2025-26: "
      f"{issues.get('calls not confirmed by the shift chart', 0):,}")
    w("Caveat: v1.4.1 learned from 2022-23 to 2025-26 shots, so xG is in-sample here and "
      "flatters v0 a bit.\nThe real test is 2026-27.")
    w("")
    w("1. THE PICKS")
    w(f"  v0: H = {pick['H']} games, K5 = {pick['K5']} games, Kst = {pick['Kst']} games, "
      f"keep = {pick['keep']}")
    for k in ("H", "K5", "Kst", "keep"):
        if on_edge(pick[k], GRID[k]):
            w(f"      note: {k} = {pick[k]} is on the edge of its grid {GRID[k]}")
    w(f"  Elo: K = {elo_pick_[0]}, home bonus = {elo_pick_[1]}, keep = {elo_pick_[2]}")
    for k, v in zip(("K", "home_bonus", "keep"), elo_pick_):
        if on_edge(v, ELO_GRID[k]):
            w(f"      note: {k} = {v} is on the edge of its grid {ELO_GRID[k]}")
    w("  v0's ten best settings at 1,000 simulations (log loss):")
    for r in grid[:10]:
        w(f"    H={r[0]:<3} K5={r[1]:<3} Kst={r[2]:<3} keep={r[3]:<5} {r[4]:.5f}")
    w(f"  Spread over all {len(grid)} settings: {grid[0][4]:.5f} to {grid[-1][4]:.5f}")
    w("  Finalists at 10,000 simulations:")
    for s, ll, _ in fin:
        w(f"    H={s['H']:<3} K5={s['K5']:<3} Kst={s['Kst']:<3} keep={s['keep']:<5} {ll:.5f}")
    w("")
    preds = {"v0": p_v0, "elo": p_elo, "home_rate": p_home}
    rr = EV.resample(preds, y, seed=SEED)
    w("2. LOG LOSS (lower is better; ranges are 95% from 1,000 game resamples)")
    for k in preds:
        w(f"  {k:10s} {EV.log_loss(preds[k], y):.5f} {fmt_range(rr[f'll:{k}'])}")
    w(f"  v0 minus home rate  {EV.log_loss(p_v0, y) - EV.log_loss(p_home, y):+.5f} "
      f"{fmt_range(tuple(-x for x in rr['gap:home_rate-v0'][::-1]))}")
    w(f"  v0 minus Elo        {EV.log_loss(p_v0, y) - EV.log_loss(p_elo, y):+.5f} "
      f"{fmt_range(tuple(-x for x in rr['gap:elo-v0'][::-1]))}")
    w(f"  Elo minus home rate {EV.log_loss(p_elo, y) - EV.log_loss(p_home, y):+.5f} "
      f"{fmt_range(rr['gap:elo-home_rate'])}")
    w("  By season:")
    for s in BUILD:
        m = seasons == s
        w(f"    {s}: v0 {EV.log_loss(p_v0[m], y[m]):.5f}  Elo {EV.log_loss(p_elo[m], y[m]):.5f}  "
          f"home rate {EV.log_loss(p_home[m], y[m]):.5f}  (home teams won {y[m].mean():.3f})")
    w("")
    w("3. CALIBRATION LINE (slope near 1 and intercept near 0 is right)")
    for k in ("v0", "elo"):
        a, b = EV.calibration_line(preds[k], y)
        w(f"  {k:4s} slope {b:.3f} {fmt_range(rr[f'cal_slope:{k}'], 3)}   intercept {a:+.3f} "
          f"{fmt_range(rr[f'cal_int:{k}'], 3)}")
    w("  v0's ten groups (predicted home-win share vs. actual, range on actual):")
    groups, _ = EV.ten_groups(p_v0, y)
    gr = EV.group_ranges(p_v0, y, seed=SEED)
    for (pm, am, n), r in zip(groups, gr):
        flag = "" if r[0] <= pm <= r[1] else "   predicted outside the range"
        w(f"    {n:4d} games  predicted {pm:.3f}  actual {am:.3f} {fmt_range(r, 3)}{flag}")
    w(f"  Spread of v0's chances: 10th to 90th percentile {np.percentile(p_v0, 10):.3f} to "
      f"{np.percentile(p_v0, 90):.3f}; Elo's {np.percentile(p_elo, 10):.3f} to {np.percentile(p_elo, 90):.3f}")
    w("")
    w("4. OVERTIME AND SHOOTOUTS (share of games)")
    ot, so = D["went_ot"][BUILD_IDX], D["went_so"][BUILD_IDX]
    for s in BUILD + [None]:
        m = np.ones(len(y), bool) if s is None else seasons == s
        lab = "all" if s is None else str(s)
        w(f"  {lab:9s} overtime: predicted {o['p_ot'][m].mean():.3f}, actual {ot[m].mean():.3f}   "
          f"shootout: predicted {o['p_so'][m].mean():.3f}, actual {so[m].mean():.3f}")
    w("")
    w("5. GOALS PER GAME (real goals without empty-net goals; overtime goal counted, shootout not)")
    hist = o["total_hist"]
    gp = D["goals_period"][BUILD_IDX]
    real_tot = gp.sum(axis=(1, 2))
    w(f"  Combined: predicted {(hist * np.arange(hist.shape[1])).sum(axis=1).mean():.3f}, "
      f"actual {real_tot.mean():.3f}; empty-net goals left out: "
      f"{D['en_goals'][BUILD_IDX].sum(axis=1).mean():.3f} a game")
    sim_p = o["goals_period"].sum(axis=1).mean(axis=0)
    real_p = gp.sum(axis=1).mean(axis=0)
    w("  By period (both teams): " + ", ".join(
        f"P{i + 1} predicted {sim_p[i]:.3f} actual {real_p[i]:.3f}" for i in range(3)))
    w(f"  Home / away (regulation): predicted {o['goals_period'][:, 0].sum(axis=1).mean():.3f} / "
      f"{o['goals_period'][:, 1].sum(axis=1).mean():.3f}, actual {gp[:, 0, :3].sum(axis=1).mean():.3f} / "
      f"{gp[:, 1, :3].sum(axis=1).mean():.3f}")
    w("  Spread of combined goals (share of games):")
    pred_h = hist.mean(axis=0)
    real_h = np.bincount(real_tot.astype(int).clip(0, 12), minlength=13) / len(real_tot)
    pred_c = np.r_[pred_h[:12], pred_h[12:].sum()]
    w("    goals     " + " ".join(f"{k:>5}" for k in list(range(12)) + ["12+"]))
    w("    predicted " + " ".join(f"{x:5.3f}" for x in pred_c))
    w("    actual    " + " ".join(f"{x:5.3f}" for x in real_h))
    w("")
    w("6. MINUTES AT EACH STRENGTH, PER TEAM PER GAME (simulated regulation vs. the shift charts)")
    toi_sim = o["toi"]                             # home view [G, 9], regulation
    real_home = D["toi"][BUILD_IDX, 0]
    real_away = D["toi"][BUILD_IDX, 1]
    sim_away = toi_sim[:, MIRROR]
    groups_ = dict(even5=[S_IDX["5v5"]], pp=PP_STATES, pk=SH_STATES, four=[S_IDX["4v4"]])
    names = dict(even5="5-on-5", pp="power play", pk="penalty kill", four="4-on-4")
    for g, idx in groups_.items():
        sim_m = (toi_sim[:, idx].sum(axis=1).mean() + sim_away[:, idx].sum(axis=1).mean()) / 2 / 60
        real_m = (real_home[:, idx].sum(axis=1).mean() + real_away[:, idx].sum(axis=1).mean()) / 2 / 60
        w(f"  {names[g]:13s} simulated {sim_m:6.2f}   shift charts {real_m:6.2f}")
    sim_ppd = o["pp_drawn"].mean() / 2
    real_ppd = D["ppd"][BUILD_IDX].sum(axis=2).mean()
    w(f"  power plays drawn per team-game: simulated {sim_ppd:.2f}, counted {real_ppd:.2f}; "
      f"calls the shift charts didn't confirm still play out in real games but aren't counted")
    w(f"  (The shift charts also have {real_home[:, S_IDX['3v3']].mean() / 60:.2f} min of 3-on-3, "
      "nearly all overtime, which the engine works out without simulating time.)")
    teams = D["team"][BUILD_IDX]
    w("  By team-season, power-play and penalty-kill minutes per game (simulated vs. shift charts):")
    for g in ("pp", "pk"):
        idx = groups_[g]
        sims, reals = [], []
        for s in BUILD:
            for t in np.unique(teams[seasons == s]):
                mh = (seasons == s) & (teams[:, 0] == t)
                ma = (seasons == s) & (teams[:, 1] == t)
                n = mh.sum() + ma.sum()
                sims.append((toi_sim[mh][:, idx].sum() + sim_away[ma][:, idx].sum()) / n / 60)
                reals.append((real_home[mh][:, idx].sum() + real_away[ma][:, idx].sum()) / n / 60)
        sims, reals = np.array(sims), np.array(reals)
        w(f"    {names[g]:13s} correlation {np.corrcoef(sims, reals)[0, 1]:.2f}, mean gap "
          f"{(sims - reals).mean():+.2f} min, typical miss {np.abs(sims - reals).mean():.2f} min "
          f"({len(sims)} team-seasons)")
    (OUT / "build_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
