"""v1.4-fenwick: v1.3 plus a settings search, a distance constraint and a calibration check.

WHAT STAYS FROM v1.3
--------------------
Same 17 features, same recency weights (half-life 1 season), same seasons and the same
way of choosing: train on 2022-23 to 2024-25 plus 2025-26 before the season's median game
date ("select train"), score on 2025-26 from that date ("select val"), judge on log loss.

WHAT'S NEW
----------
1. Distance constraint. LightGBM's monotone_constraints = -1 on distance_ft: with every
   other feature held the same, xG never goes up as distance goes up. It's on for every
   try in the search. The check (``monotone_check``) sweeps distance from 0 to 200 ft on
   real shots and counts any step where xG rises.
2. Settings search. A fixed budget of 50 tries, each early-stopped on select val log loss
   (up to 5,000 trees). Try 1 is v1.3's settings with the constraint; tries 2-50 are drawn
   at random (seed 1414) from SPACE. Every try is appended to
   results/v1_4-fenwick/search.csv as it finishes, so a cut-off run picks up where it
   stopped. The lowest select-val log loss wins.
3. What the constraint costs: v1.3's settings and the winner's settings, each trained with
   and without it on the same split. Diagnostics only; they choose nothing.
4. Calibration. Platt (logistic on logit xG) and isotonic, scored by 5-fold
   cross-fitting on select val, folds split by game. One is kept only if its
   cross-fitted log loss beats the raw model's; it's then fit on all of select val. Both
   are increasing in xG, so they can't break the distance constraint.

STAGES (2026-27 only ever touched by --stage final)
---------------------------------------------------
    --stage select   search, costs, calibration; freezes the choice in selection.json
    --stage final    refits the frozen choice on 2022-23 to 2025-26 with the selection
                     tree count, tests on 2026-27 once, next to the saved v1.3 model on
                     the same shots. Saves the model, but scores nothing into shot_xg.

Usage (repo root, with the project venv's python):
    .venv/bin/python3 -m model.train_v1_4_fenwick --stage select
    .venv/bin/python3 -m model.train_v1_4_fenwick --stage final
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from model import train_v1 as v1
from model import train_v1_fenwick as v1f
from model import train_v1_1_fenwick as v11
from model import train_v1_2_fenwick as v12
from model import train_v1_3_fenwick as v13
from model.reporting import RESULTS_DIR, max_abs_gap, reliability_table, write_results

VERSION = "v1.4-fenwick"
ARTIFACT_STEM = "xg_v1_4_fenwick"
OUT = RESULTS_DIR / "v1_4-fenwick"
SEARCH_CSV = OUT / "search.csv"
SELECTION_JSON = OUT / "selection.json"

TARGET = v13.TARGET
NUMERIC = list(v13.BASE_NUMERIC)
BOOLEAN = list(v13.BASE_BOOLEAN)
CATEGORICAL = list(v13.CATEGORICAL_FEATURES)
ALL_FEATURES = NUMERIC + BOOLEAN + CATEGORICAL
HALF_LIFE = 1.0
MONOTONE = [-1 if f == "distance_ft" else 0 for f in ALL_FEATURES]

BUDGET = 50
SEED = 1414
MAX_TREES = 5000
STOP_ROUNDS = 100

FIXED = dict(objective="binary", metric="binary_logloss", importance_type="gain",
             random_state=42, verbose=-1, n_jobs=-1)
V13_SETTINGS = {k: v11.HYPERPARAMS[k] for k in
                ("learning_rate", "num_leaves", "max_depth", "min_child_samples",
                 "reg_alpha", "reg_lambda")}
V13_SETTINGS.update(subsample=1.0, subsample_freq=0, colsample_bytree=1.0, max_bin=255,
                    monotone_constraints_method="basic")


def draw(rng: np.random.Generator) -> dict:
    """One random point from the search space."""
    logu = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))  # noqa: E731
    return {
        "learning_rate": logu(0.01, 0.1),
        "num_leaves": int(round(logu(8, 255))),
        "max_depth": int(rng.choice([-1, -1, 6, 8, 10, 12])),
        "min_child_samples": int(round(logu(10, 2000))),
        "reg_alpha": logu(1e-3, 10),
        "reg_lambda": logu(1e-3, 30),
        "subsample": float(rng.uniform(0.5, 1.0)),
        "subsample_freq": 1,
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "max_bin": int(rng.choice([63, 127, 255, 511])),
        "monotone_constraints_method": str(rng.choice(["basic", "intermediate", "advanced"])),
    }


def tries() -> list[dict]:
    rng = np.random.default_rng(SEED)
    return [dict(V13_SETTINGS)] + [draw(rng) for _ in range(BUDGET - 1)]


def make(settings: dict, n_trees: int, monotone: bool) -> lgb.LGBMClassifier:
    params = dict(FIXED, **settings, n_estimators=n_trees)
    if monotone:
        params["monotone_constraints"] = MONOTONE
    else:
        params.pop("monotone_constraints_method", None)
    return lgb.LGBMClassifier(**params)


def fit(train, settings, monotone=True, n_trees=MAX_TREES, val=None, stop_metric="logloss"):
    model = make(settings, n_trees, monotone)
    kwargs = {"categorical_feature": CATEGORICAL,
              "sample_weight": v13.recency_weights(train["season"], HALF_LIFE)}
    if val is not None:
        if stop_metric == "auc":   # how v1.3 picked its tree count
            model.set_params(metric="auc")
        kwargs.update(eval_set=[(val[ALL_FEATURES], val[TARGET])],
                      callbacks=[lgb.early_stopping(STOP_ROUNDS if stop_metric == "logloss"
                                                    else 50, verbose=False)])
    model.fit(train[ALL_FEATURES], train[TARGET], **kwargs)
    return model


# ---------------------------------------------------------------- calibration

class Calibrated:
    """A fitted LightGBM model with an increasing map applied to its xG."""

    def __init__(self, model, method: str | None, calibrator=None):
        self.model, self.method, self.calibrator = model, method, calibrator

    def predict_proba(self, X):
        p = self.model.predict_proba(X)[:, 1]
        if self.method is not None:
            p = apply_cal(self.method, self.calibrator, p)
        return np.column_stack([1 - p, p])

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_cal(method: str, p, y):
    if method == "platt":
        return LogisticRegression(C=1e6).fit(_logit(p), y)
    return IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip").fit(p, y)


def apply_cal(method: str, cal, p):
    if method == "platt":
        return cal.predict_proba(_logit(p))[:, 1]
    return cal.predict(p)


def crossfit_cal(p, y, games, k=5, seed=SEED) -> dict:
    """Out-of-fold calibrated xG on select val, folds split by game."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(games)
    fold_of = dict(zip(uniq, rng.permutation(len(uniq)) % k))
    fold = np.array([fold_of[g] for g in games])
    out = {}
    for method in ("platt", "isotonic"):
        q = np.empty_like(p)
        for f in range(k):
            tr, te = fold != f, fold == f
            q[te] = apply_cal(method, fit_cal(method, p[tr], y[tr]), p[te])
        out[method] = q
    return out


# ---------------------------------------------------------------- distance check

def monotone_check(predict, shots: pd.DataFrame, grid=np.arange(0, 201, 1.0)) -> dict:
    """Sweep distance_ft over ``grid`` with everything else fixed; count xG rises."""
    reps = shots.loc[shots.index.repeat(len(grid))].copy()
    reps["distance_ft"] = np.tile(grid, len(shots))
    p = predict(reps[ALL_FEATURES]).reshape(len(shots), len(grid))
    step = np.diff(p, axis=1)
    rises = step > 1e-12
    return {"shots": int(len(shots)), "steps": int(step.size),
            "rises": int(rises.sum()), "shots_with_a_rise": int(rises.any(axis=1).sum()),
            "largest_rise": float(step.max()) if step.size else 0.0}


# ---------------------------------------------------------------- data

def splits(df: pd.DataFrame):
    split_season = df[df["season"] == v13.SPLIT_SEASON]
    cutoff = split_season["game_date"].median().normalize()
    before = df["season"].isin(v13.TRAIN_SEASONS[:-1]) | (
        (df["season"] == v13.SPLIT_SEASON) & (df["game_date"] < cutoff))
    return cutoff, {
        "select_train": df[before],
        "select_val": df[(df["season"] == v13.SPLIT_SEASON) & (df["game_date"] >= cutoff)],
        "refit_train": df[df["season"].isin(v13.TRAIN_SEASONS)],
        "test": df[df["season"] == v13.TEST_SEASON],
    }


def summary(y, p, d) -> dict:
    m = v11.metrics(y, p)
    m["tipin_o_e"] = v13.tipin_oe(d, p)
    return m


def show(label, m):
    print(f"  {label:40} log loss {m['log_loss']:.5f}  AUC {m['auc']:.4f}  "
          f"O/E {m['o_e']:.3f}  slope {m['cal_slope']:.3f}  tip-in O/E {m['tipin_o_e']:.3f}")


# ---------------------------------------------------------------- stage: select

def stage_select(parts) -> None:
    tr, va = parts["select_train"], parts["select_val"]
    y = va[TARGET].to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)

    print("\n--- v1.3 as it was chosen (no constraint, trees by AUC) on select val ---")
    m13 = fit(tr, V13_SETTINGS, monotone=False, n_trees=1000, val=va, stop_metric="auc")
    p13 = m13.predict_proba(va[ALL_FEATURES])[:, 1]
    ref = summary(y, p13, va) | {"n_trees": int(m13.best_iteration_)}
    show(f"v1.3 recipe ({ref['n_trees']} trees)", ref)

    print(f"\n--- Settings search: {BUDGET} tries, distance constraint on ---")
    done = pd.read_csv(SEARCH_CSV) if SEARCH_CSV.exists() else pd.DataFrame()
    for i, s in enumerate(tries(), start=1):
        if len(done) and i in set(done["try"]):
            continue
        t0 = time.time()
        m = fit(tr, s, monotone=True, val=va)
        p = m.predict_proba(va[ALL_FEATURES])[:, 1]
        met = summary(y, p, va)
        row = {"try": i, **s, "n_trees": int(m.best_iteration_),
               "log_loss": met["log_loss"], "auc": met["auc"], "brier": met["brier"],
               "o_e": met["o_e"], "cal_slope": met["cal_slope"],
               "tipin_o_e": met["tipin_o_e"], "seconds": round(time.time() - t0, 1)}
        pd.DataFrame([row]).to_csv(SEARCH_CSV, mode="a", header=not SEARCH_CSV.exists(),
                                   index=False)
        print(f"  try {i:2d}: log loss {met['log_loss']:.5f}  AUC {met['auc']:.4f}  "
              f"{row['n_trees']:4d} trees  {row['seconds']:5.1f} s")
    search = pd.read_csv(SEARCH_CSV).sort_values("log_loss")
    if len(search) != BUDGET:
        raise RuntimeError(f"search.csv has {len(search)} tries, expected {BUDGET}")
    best = search.iloc[0]
    keys = list(V13_SETTINGS)
    settings = {k: (best[k].item() if hasattr(best[k], "item") else best[k]) for k in keys}
    for k in ("num_leaves", "max_depth", "min_child_samples", "subsample_freq", "max_bin"):
        settings[k] = int(settings[k])
    print(f"\nBest: try {int(best['try'])}, log loss {best['log_loss']:.5f}, "
          f"{int(best['n_trees'])} trees")
    print(json.dumps(settings, indent=2))
    print("\nTop 10 tries:")
    print(search.head(10)[["try", "log_loss", "auc", "n_trees", "learning_rate",
                           "num_leaves", "min_child_samples", "max_depth",
                           "monotone_constraints_method"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print("\n--- What the distance constraint costs (same split, log loss) ---")
    winner = fit(tr, settings, monotone=True, val=va)
    p_win = winner.predict_proba(va[ALL_FEATURES])[:, 1]
    cost_rows = []
    for label, s in (("v1.3 settings", V13_SETTINGS), ("winner's settings", settings)):
        free = fit(tr, s, monotone=False, val=va)
        tied = winner if s is settings else fit(tr, s, monotone=True, val=va)
        pf = free.predict_proba(va[ALL_FEATURES])[:, 1]
        pt = tied.predict_proba(va[ALL_FEATURES])[:, 1]
        r = {"settings": label, "free_log_loss": log_loss(y, pf),
             "constrained_log_loss": log_loss(y, pt),
             "free_trees": int(free.best_iteration_), "constrained_trees": int(tied.best_iteration_)}
        r["cost"] = r["constrained_log_loss"] - r["free_log_loss"]
        cost_rows.append(r)
        free_check = monotone_check(lambda X: free.predict_proba(X)[:, 1],
                                    va.sample(2000, random_state=SEED))
        r["free_shots_with_a_rise"] = free_check["shots_with_a_rise"]
        r["free_largest_rise"] = free_check["largest_rise"]
    cost = pd.DataFrame(cost_rows)
    print(cost.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    cost.to_csv(OUT / "constraint_cost.csv", index=False)

    print("\n--- Distance check on the winner (2,000 select-val shots, 0-200 ft) ---")
    chk = monotone_check(lambda X: winner.predict_proba(X)[:, 1],
                         va.sample(2000, random_state=SEED))
    print(f"  {chk}")
    if chk["rises"]:
        raise RuntimeError("The constrained model's xG rises with distance somewhere.")

    print("\n--- Calibration, 5-fold cross-fit on select val (folds by game) ---")
    oof = crossfit_cal(p_win, y, va["game_id"].to_numpy())
    cal_rows = [{"method": "none", **{k: v for k, v in summary(y, p_win, va).items()
                                       if k in ("log_loss", "auc", "o_e", "cal_slope")}}]
    for method, q in oof.items():
        s = summary(y, q, va)
        cal_rows.append({"method": method, **{k: s[k] for k in
                                              ("log_loss", "auc", "o_e", "cal_slope")}})
    cal = pd.DataFrame(cal_rows)
    print(cal.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    cal.to_csv(OUT / "calibration_check.csv", index=False)
    best_cal = cal.sort_values("log_loss").iloc[0]["method"]
    keep = None if best_cal == "none" else best_cal
    print(f"\nCalibration kept: {keep or 'none (raw xG had the lowest log loss)'}")
    if keep:
        calibrator = fit_cal(keep, p_win, y)
        joblib.dump(calibrator, v1.ARTIFACT_DIR / f"{ARTIFACT_STEM}_calibrator.pkl")
        q = apply_cal(keep, calibrator, p_win)
        chk2 = monotone_check(
            lambda X: apply_cal(keep, calibrator, winner.predict_proba(X)[:, 1]),
            va.sample(2000, random_state=SEED))
        print(f"  distance check after calibration: {chk2}")
        if chk2["rises"]:
            raise RuntimeError("Calibrated xG rises with distance somewhere.")
        print(f"  (in-sample, all of select val: log loss {log_loss(y, q):.5f})")

    selection = {
        "version": VERSION, "settings": settings, "n_trees": int(best["n_trees"]),
        "best_try": int(best["try"]), "select_val_log_loss": float(best["log_loss"]),
        "v1_3_recipe_select_val": ref, "calibration": keep,
        "calibration_table": cal.to_dict("records"), "constraint_cost": cost_rows,
        "distance_check": chk,
    }
    with open(SELECTION_JSON, "w") as f:
        json.dump(selection, f, indent=2, default=float)
    print(f"\nFrozen in {SELECTION_JSON}. 2026-27 not touched.")


# ---------------------------------------------------------------- stage: final

def stage_final(df, parts, cutoff) -> None:
    with open(SELECTION_JSON) as f:
        sel = json.load(f)
    settings, n_trees, keep = sel["settings"], sel["n_trees"], sel["calibration"]
    test = parts["test"]
    y = test[TARGET].to_numpy()

    print(f"\n--- Refit on 2022-23 to 2025-26: {n_trees} trees, calibration {keep} ---")
    booster = fit(parts["refit_train"], settings, monotone=True, n_trees=n_trees)
    calibrator = (joblib.load(v1.ARTIFACT_DIR / f"{ARTIFACT_STEM}_calibrator.pkl")
                  if keep else None)
    model = Calibrated(booster, keep, calibrator)
    chk = monotone_check(lambda X: model.predict_proba(X)[:, 1],
                         parts["select_val"].sample(2000, random_state=SEED))
    print(f"  distance check on the refit model: {chk}")
    if chk["rises"]:
        raise RuntimeError("The refit model's xG rises with distance somewhere.")

    print("\n--- 2026-27 to date, the one test ---")
    p14 = model.predict_proba(test[ALL_FEATURES])[:, 1]
    old = joblib.load(v1.ARTIFACT_DIR / f"{v13.ARTIFACT_STEM}.pkl")
    p13 = old.predict_proba(test[v13.ALL_FEATURES])[:, 1]
    m14, m13 = summary(y, p14, test), summary(y, p13, test)
    show(VERSION, m14)
    show("v1.3-fenwick (saved, live)", m13)
    raw = summary(y, booster.predict_proba(test[ALL_FEATURES])[:, 1], test)
    if keep:
        show(f"{VERSION} before calibration", raw)

    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    rel = reliability_table(test[TARGET], p14)
    print("\n--- Reliability on 2026-27 ---")
    print(rel.to_string(index=False, float_format=fmt))

    frames = []
    for dim, fn in v12.SEGMENTS.items():
        seg = fn(test)
        a = v1f.segment_calibration(test[TARGET], p14, seg).set_index("segment")
        b = v1f.segment_calibration(test[TARGET], p13, seg).set_index("segment")
        tbl = pd.DataFrame({"n": a["n"], "oe_v1.4": a["o_e"], "oe_v1.3": b["o_e"]})
        tbl.index.name = "segment"
        frames.append(tbl.sort_values("n", ascending=False).reset_index().assign(dimension=dim))
    segs = pd.concat(frames, ignore_index=True)
    print("\n--- O/E by segment on 2026-27 ---")
    print(segs.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    p_train = model.predict_proba(parts["refit_train"][ALL_FEATURES])[:, 1]
    split_metrics = {
        "train": v11.metrics(parts["refit_train"][TARGET].values, p_train),
        "val": v11.metrics(
            parts["select_val"][TARGET].values,
            fit(parts["select_train"], settings, monotone=True, n_trees=n_trees)
            .predict_proba(parts["select_val"][ALL_FEATURES])[:, 1]),
        "test": v11.metrics(y, p14),
    }
    gain = (pd.DataFrame({"feature": ALL_FEATURES, "gain": booster.feature_importances_})
              .sort_values("gain", ascending=False))

    joblib.dump(model, v1.ARTIFACT_DIR / f"{ARTIFACT_STEM}.pkl")
    split_desc = {"select_train": f"2022-23 to 2024-25 + 2025-26 before {cutoff}",
                  "select_val": f"2025-26 from {cutoff}",
                  "refit_train": v13.TRAIN_SEASONS, "test": v13.TEST_SEASON}
    meta = {"version": VERSION, "table": v1f.TABLE, "half_life": HALF_LIFE,
            "features": {"numeric": NUMERIC, "boolean": BOOLEAN, "categorical": CATEGORICAL},
            "monotone": {"distance_ft": -1}, "settings": settings, "n_trees": n_trees,
            "calibration": keep, "split": split_desc, "splits": split_metrics,
            "calibration_max_gap": max_abs_gap(rel), "distance_check": chk}
    with open(v1.ARTIFACT_DIR / f"{ARTIFACT_STEM}_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    results_dir = write_results(
        version=VERSION,
        description=("v1.3-fenwick's features and recency weights (half-life 1 season), "
                     "settings from a 50-try search on late 2025-26 log loss, xG never rises "
                     f"with distance; calibration: {keep or 'none'}. Tested on 2026-27."),
        splits=split_metrics, reliability=rel, feature_importance=gain,
        hyperparameters={**settings, "n_trees": n_trees, "half_life": HALF_LIFE,
                         "monotone": "distance_ft -1", "calibration": keep},
        features=ALL_FEATURES,
        data_summary={"population": "fenwick", "table": v1f.TABLE, "n_shots": int(len(df)),
                      "seasons": [int(s) for s in sorted(df["season"].unique())],
                      "split": split_desc},
        notes=("Test is 2026-27 to date. 'val' is late 2025-26 with the selection fit, "
               "which also early-stopped and chose settings there, so it's optimistic."),
    )
    segs.to_csv(results_dir / "segments.csv", index=False)
    pd.DataFrame([{"model": VERSION, **m14}, {"model": "v1.3-fenwick", **m13}]).to_csv(
        results_dir / "test_vs_v1_3.csv", index=False)
    print(f"\nSaved model to {v1.ARTIFACT_DIR}/{ARTIFACT_STEM}* and results to {results_dir}/")
    print("Not scored into shot_xg; nothing exported.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=("select", "final"), required=True)
    args = ap.parse_args()
    print("=" * 60)
    print(f"{VERSION}, stage {args.stage}")
    print("=" * 60)
    df = v13.load_features()
    v1.assert_no_leakage(df)
    if "game_id" not in df.columns:
        raise RuntimeError("load_features() has no game_id; calibration folds need it.")
    cutoff, parts = splits(df)
    for name, part in parts.items():
        if name == "test" and args.stage == "select":
            print(f"{'test:':14} (not loaded into anything at this stage)")
            continue
        print(f"{name + ':':14} {part['game_date'].min()} to {part['game_date'].max()}, "
              f"{len(part):,} attempts, {part[TARGET].mean() * 100:.2f}% goals")
    if args.stage == "select":
        stage_select(parts)
    else:
        stage_final(df, parts, cutoff)


if __name__ == "__main__":
    main()
