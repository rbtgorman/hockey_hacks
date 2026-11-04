"""Shooter and goalie priors for the Fenwick model (v1.5 shooter, v1.6 goalie).

Writes shot_priors_fenwick: one row per shot_features_fenwick row, with
    shooter_prior   the shooter's shrunk goal rate / the league average (1 = average)
    goalie_prior    the goalie's shrunk, strength-standardized goals-against rate
                    in the shot's distance zone / the league average (1 = average)
plus the counts behind each, for checking.

WHAT COMES FROM THE EXISTING BUILDERS
-------------------------------------
build_priors_expanding.py (shooters) and build_goalie_priors_expanding.py stay
frozen, like build_features.py, so v2.3 still reproduces. This module imports
their pieces instead of copying them:
  - the trailing 2-year window, half-open [date - 730 days, date): a shot's own
    game date is never in its prior;
  - season weights 5/4/3 divided by W_MAX (1, 0.8, 0.6), so weighted shots are
    in the same units as K (the old units bug, see docs/FINDINGS.md);
  - Beta-Binomial empirical Bayes: posterior = (K * mean + n_w * own rate) /
    (K + n_w), i.e. w * own + (1 - w) * mean with w = n_w / (n_w + K);
  - K by method of moments on RAW shot counts, from entities with >= 200 shots,
    and the league mean on that same eligible group (K = 100 if fewer than 5);
  - shooter strength buckets 5v5 / PP / PK, empty-net shots out of PK only;
  - goalie distance zones high < 20 ft, mid 20-40, low 40+.

WHAT'S DIFFERENT
----------------
1. K and the league mean are refit for every game date from shots BEFORE that
   date. The old builders fit them once on every season, the test season
   included, so a later game's result moved every earlier prior through them.
   tests/test_priors_fenwick.py checks that changing or deleting a later game
   leaves every earlier prior exactly the same.
2. History is shot_features_fenwick, the model's own population (no 'short'
   misses, failed bank attempts, teammate blocks or shootout attempts).
3. Goalies follow the plan in docs/FINDINGS.md: zone only (3 cells, not 9),
   on unblocked attempts (a save relabelled as a miss is still in the
   denominator, so the 2023-24 recording change doesn't move it), and
   standardized for strength mix: each shot a goalie faced counts the league's
   goal rate for its zone and strength state (as of the target date), and the
   goalie's rate is  zone rate * goals / expected  before shrinking. A goalie
   who faces lots of power plays isn't marked down for it.
4. Both are stored as a ratio to the league mean, so a player with no history
   gets exactly 1.0 (the league average), even on the first night of the data.
5. Lookups for strength states outside the three buckets: 6v5, 5v6, 1v0
   (penalty shots) and 0v1 use the 5v5 prior; 6v4 and 6v3 use PP; 4v6 uses PK.
   Their own shots stay out of the history, as before. Empty-net shots and
   shots with no distance get goalie_prior 1.0.

RUN
---
    python3 -m features.build_priors_fenwick      # about a minute or two
"""
from __future__ import annotations

import contextlib
import io
import time

import numpy as np
import pandas as pd

from features.build_goalie_priors_expanding import classify_zone
from features.build_priors_expanding import (BUCKET_MAP, K_FIT_MIN_SHOTS, WINDOW_YEARS,
                                             attach_recency_weight, fit_k_methodofmoments)

WINDOW = np.timedelta64(WINDOW_YEARS * 365, "D")
SHOOTER_LOOKUP = dict(BUCKET_MAP, **{"6v5": "5v5", "5v6": "5v5", "1v0": "5v5", "0v1": "5v5",
                                     "6v4": "PP", "6v3": "PP", "4v6": "PK"})
TABLE = "shot_priors_fenwick"


# ---------------------------------------------------------------- the engine (no database)

def _fit_k(n: np.ndarray, x: np.ndarray, mean: float) -> tuple[float, bool]:
    """The existing builders' K, quietly. Returns (K, used_the_default)."""
    aggs = pd.DataFrame({"raw_shots": n, "raw_goals": x * n})
    with contextlib.redirect_stdout(io.StringIO()) as out:
        k = fit_k_methodofmoments(aggs, mean)
    return k, bool(out.getvalue())


def asof_params(hist: pd.DataFrame, axis: np.ndarray, n_strata: int):
    """League rates, league mean and K for one cell at every date on ``axis``,
    each from history strictly before that date.

    hist needs integer columns ent, day (index into axis), stratum, is_goal.
    Returns L (days x strata, league goal rate by stratum), Lc (league rate in
    the cell), mean (on the eligible group) and K, all arrays over days.
    """
    D, S = len(axis), n_strata
    n_ent = int(hist["ent"].max()) + 1 if len(hist) else 0
    shots = np.zeros((n_ent, D, S))
    goals = np.zeros((n_ent, D, S))
    np.add.at(shots, (hist["ent"].to_numpy(), hist["day"].to_numpy(), hist["stratum"].to_numpy()), 1)
    np.add.at(goals, (hist["ent"].to_numpy(), hist["day"].to_numpy(), hist["stratum"].to_numpy()),
              hist["is_goal"].to_numpy())
    # exclusive cumulative: totals from days strictly before each day
    cs = np.cumsum(shots, axis=1) - shots
    cg = np.cumsum(goals, axis=1) - goals
    lg_s, lg_g = cs.sum(axis=0), cg.sum(axis=0)                     # days x strata
    with np.errstate(invalid="ignore", divide="ignore"):
        Lc = lg_g.sum(axis=1) / lg_s.sum(axis=1)                     # nan before any shot
        L = np.where(lg_s > 0, lg_g / np.where(lg_s > 0, lg_s, 1), Lc[:, None])

    mean = np.full(D, np.nan)
    K = np.full(D, 100.0)
    defaults = 0
    for d in range(D):
        if not np.isfinite(Lc[d]):
            continue
        n = cs[:, d, :].sum(axis=1)
        ga = cg[:, d, :].sum(axis=1)
        e = (cs[:, d, :] * L[d]).sum(axis=1)
        has = n > 0
        n, ga, e = n[has], ga[has], e[has]
        x = np.where(e > 0, Lc[d] * ga / np.where(e > 0, e, 1), Lc[d])
        elig = n >= K_FIT_MIN_SHOTS
        mean[d] = (Lc[d] * ga[elig].sum() / e[elig].sum()
                   if elig.any() and e[elig].sum() > 0 else Lc[d])
        K[d], used_default = _fit_k(n, x, mean[d])
        defaults += used_default
    return L, Lc, mean, K, defaults


def compute_priors(hist: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Shrunk ratio-to-league priors for every target.

    hist:    entity, game_date, season, cell, stratum, is_goal   (the evidence)
    targets: entity, game_date, season, cell                     (what to price)
    Returns targets with window_shots, weighted_shots, weighted_goals,
    weighted_expected, k, league_mean, shrunk_rate, prior (1.0 = league average).
    For shooters pass stratum == cell: the standardization then does nothing and
    this is exactly the existing shooter formula.
    """
    hist = hist.copy()
    hist["game_date"] = hist["game_date"].to_numpy().astype("datetime64[D]")
    out = targets.copy().reset_index(drop=True)
    out["game_date"] = out["game_date"].to_numpy().astype("datetime64[D]")
    for c in ("window_shots", "weighted_shots", "weighted_goals", "weighted_expected",
              "k", "league_mean", "shrunk_rate"):
        out[c] = np.nan
    out["prior"] = 1.0

    for cell in sorted(set(out["cell"].dropna()) | set(hist["cell"])):
        h = hist[hist["cell"] == cell]
        t_idx = np.flatnonzero((out["cell"] == cell).to_numpy())
        if len(t_idx) == 0:
            continue
        axis = np.unique(np.concatenate([h["game_date"].to_numpy(),
                                         out.loc[t_idx, "game_date"].to_numpy()]))
        ents = {e: i for i, e in enumerate(pd.unique(h["entity"]))}
        strata = {s: i for i, s in enumerate(sorted(pd.unique(h["stratum"])))}
        hh = pd.DataFrame({"ent": h["entity"].map(ents).to_numpy(),
                           "day": np.searchsorted(axis, h["game_date"].to_numpy()),
                           "stratum": h["stratum"].map(strata).to_numpy(),
                           "is_goal": h["is_goal"].to_numpy().astype(float),
                           "season": h["season"].to_numpy()})
        L, Lc, mean, K, _ = asof_params(hh, axis, max(len(strata), 1))

        hh = hh.sort_values(["ent", "day"], kind="stable")
        by_ent = {e: g for e, g in hh.groupby("ent", sort=False)}
        tt = out.loc[t_idx]
        t_day = np.searchsorted(axis, tt["game_date"].to_numpy())
        t_start = np.searchsorted(axis, tt["game_date"].to_numpy() - WINDOW)  # first day in window
        rows = {k: np.full(len(tt), np.nan) for k in
                ("window_shots", "weighted_shots", "weighted_goals", "weighted_expected",
                 "k", "league_mean", "shrunk_rate")}
        prior = np.ones(len(tt))
        for j, (ent, season) in enumerate(zip(tt["entity"].to_numpy(), tt["season"].to_numpy())):
            d = t_day[j]
            rows["k"][j], rows["league_mean"][j] = K[d], mean[d]
            g = by_ent.get(ents.get(ent, -1))
            if g is None or not mean[d] > 0:   # no league history yet (or no goals in it)
                rows["window_shots"][j] = 0
                continue
            days = g["day"].to_numpy()
            lo, hi = np.searchsorted(days, t_start[j], "left"), np.searchsorted(days, d, "left")
            rows["window_shots"][j] = hi - lo
            if hi == lo:
                continue
            w = attach_recency_weight(int(season), g["season"].to_numpy()[lo:hi])
            nw = w.sum()
            if nw <= 0:
                continue
            gw = (w * g["is_goal"].to_numpy()[lo:hi]).sum()
            ew = (w * L[d, g["stratum"].to_numpy()[lo:hi]]).sum()
            x = Lc[d] * gw / ew if ew > 0 else Lc[d]
            post = (K[d] * mean[d] + nw * x) / (K[d] + nw)
            rows["weighted_shots"][j], rows["weighted_goals"][j] = nw, gw
            rows["weighted_expected"][j], rows["shrunk_rate"][j] = ew, post
            prior[j] = post / mean[d]
        for k, v in rows.items():
            out.loc[t_idx, k] = v
        out.loc[t_idx, "prior"] = prior
    return out


# ---------------------------------------------------------------- inputs from the table

def shooter_frames(shots: pd.DataFrame):
    s = shots[shots["shooter_id"].notna()]
    hist = s.assign(cell=s["strength_state"].map(BUCKET_MAP))
    hist = hist[hist["cell"].notna() & ~((hist["cell"] == "PK") & hist["empty_net"])]
    hist = pd.DataFrame({"entity": hist["shooter_id"], "game_date": hist["game_date"],
                         "season": hist["season"], "cell": hist["cell"],
                         "stratum": hist["cell"], "is_goal": hist["is_goal"]})
    targets = pd.DataFrame({"shot_id": shots["shot_id"], "entity": shots["shooter_id"],
                            "game_date": shots["game_date"], "season": shots["season"],
                            "cell": shots["strength_state"].map(SHOOTER_LOOKUP)})
    return hist, targets


def goalie_frames(shots: pd.DataFrame):
    zone = pd.Series(classify_zone(shots["distance_ft"].to_numpy()), index=shots.index)
    zone = zone.where(zone != "unknown")
    ok = shots["goalie_id"].notna() & ~shots["empty_net"] & zone.notna()
    s = shots[ok]
    hist = pd.DataFrame({"entity": s["goalie_id"], "game_date": s["game_date"],
                         "season": s["season"], "cell": zone[ok],
                         "stratum": s["strength_state"], "is_goal": s["is_goal"]})
    targets = pd.DataFrame({"shot_id": shots["shot_id"], "entity": shots["goalie_id"],
                            "game_date": shots["game_date"], "season": shots["season"],
                            "cell": zone.where(ok)})
    return hist, targets


def load_shots() -> pd.DataFrame:
    from ingest.db import pg_conn
    q = """
        SELECT f.shot_id, s.shooter_id, s.goalie_id, g.game_date, g.season,
               f.strength_state, COALESCE(f.empty_net, FALSE) AS empty_net,
               f.distance_ft, f.is_goal::int AS is_goal
        FROM shot_features_fenwick f
        JOIN shots s ON s.shot_id = f.shot_id
        JOIN games g ON g.game_id = f.game_id
    """
    with pg_conn() as conn:
        df = pd.read_sql(q, conn)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["empty_net"] = df["empty_net"].astype(bool)
    return df


def build() -> pd.DataFrame:
    from ingest.db import pg_conn
    t0 = time.time()
    shots = load_shots()
    print(f"Loaded {len(shots):,} attempts from shot_features_fenwick")
    sh = compute_priors(*shooter_frames(shots))
    print(f"Shooter priors done ({time.time() - t0:.0f} s)")
    go = compute_priors(*goalie_frames(shots))
    print(f"Goalie priors done ({time.time() - t0:.0f} s)")
    out = pd.DataFrame({
        "shot_id": shots["shot_id"].to_numpy(),
        "shooter_bucket": sh["cell"].to_numpy(), "shooter_prior": sh["prior"].to_numpy(),
        "shooter_window_shots": sh["window_shots"].to_numpy(),
        "shooter_weighted_shots": sh["weighted_shots"].to_numpy(),
        "shooter_k": sh["k"].to_numpy(), "shooter_league_mean": sh["league_mean"].to_numpy(),
        "goalie_zone": go["cell"].to_numpy(), "goalie_prior": go["prior"].to_numpy(),
        "goalie_window_shots": go["window_shots"].to_numpy(),
        "goalie_weighted_shots": go["weighted_shots"].to_numpy(),
        "goalie_k": go["k"].to_numpy(), "goalie_league_mean": go["league_mean"].to_numpy(),
    })
    buf = io.StringIO()
    out.to_csv(buf, index=False, header=False)
    buf.seek(0)
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(f"""
            DROP TABLE IF EXISTS {TABLE};
            CREATE TABLE {TABLE} (
                shot_id BIGINT PRIMARY KEY,
                shooter_bucket TEXT, shooter_prior DOUBLE PRECISION,
                shooter_window_shots DOUBLE PRECISION, shooter_weighted_shots DOUBLE PRECISION,
                shooter_k DOUBLE PRECISION, shooter_league_mean DOUBLE PRECISION,
                goalie_zone TEXT, goalie_prior DOUBLE PRECISION,
                goalie_window_shots DOUBLE PRECISION, goalie_weighted_shots DOUBLE PRECISION,
                goalie_k DOUBLE PRECISION, goalie_league_mean DOUBLE PRECISION)""")
        cur.copy_expert(f"COPY {TABLE} FROM STDIN WITH (FORMAT csv)", buf)
        conn.commit()
    print(f"Wrote {len(out):,} rows to {TABLE} ({time.time() - t0:.0f} s)")
    return out.assign(season=shots["season"].to_numpy(), game_date=shots["game_date"].to_numpy())


def report(out: pd.DataFrame) -> None:
    """What the priors look like, by season."""
    pd.set_option("display.width", 200)
    for who in ("shooter", "goalie"):
        p, n = out[f"{who}_prior"], out[f"{who}_window_shots"]
        g = pd.DataFrame({"season": out["season"], "prior": p, "none": n.fillna(0) == 0,
                          "k": out[f"{who}_k"], "mean": out[f"{who}_league_mean"]})
        tab = g.groupby("season").agg(
            shots=("prior", "size"), no_history=("none", "mean"),
            p10=("prior", lambda v: v.quantile(.1)), p50=("prior", "median"),
            p90=("prior", lambda v: v.quantile(.9)), sd=("prior", "std"))
        print(f"\n--- {who}_prior by season (1.0 = league average) ---")
        print(tab.round(3).to_string())
        cell = out[f"{who}_bucket" if who == "shooter" else f"{who}_zone"]
        g = g.assign(cell=cell, date=out["game_date"]).dropna(subset=["cell"])
        last = g[g["date"] == g["date"].max()].groupby("cell")[["k", "mean"]].first()
        print(f"K and league mean on the latest date, by {'bucket' if who == 'shooter' else 'zone'}:")
        print(last.round(4).to_string())


if __name__ == "__main__":
    report(build())
