"""v1.5-fenwick (shooter prior) and v1.6-fenwick (goalie prior), each measured on its own.

THE THREE CANDIDATES (all on the same rebuilt shot_features_fenwick)
--------------------------------------------------------------------
    baseline   v1.4's 17 features and frozen settings (results/v1_4-fenwick/
               selection.json), recency weights (half-life 1 season) and distance
               constraint, retrained here. So the gains below aren't mixed up with
               the stoppage fix (docs/FINDINGS.md) or with a refit: that was the
               v1.2 lesson, where a "new feature" gain turned out to be the refit.
    v1.5       baseline + shooter_prior
    v1.6       v1.5 + goalie_prior
Shooter gain = v1.5 - baseline; goalie gain = v1.6 - v1.5. No settings search.
Priors come from features/build_priors_fenwick.py (shot_priors_fenwick): only games
before the shot's date, shrunk toward the league average, 1.0 = average.

CHOOSING (2026-27 never looked at here)
---------------------------------------
Same split as v1.3 and v1.4: train on 2022-23 to 2024-25 plus 2025-26 before the
season's median game date, score 2025-26 from that date. Each candidate early-stops
there on log loss (up to 5,000 trees, as v1.4's search did), which fixes its tree
count. Lowest log loss wins. The differences come with a bootstrap over games
(1,000 resamples) so a gain smaller than its noise is visible as one.

STAGES
------
    --stage select   the three candidates; writes results/v1_5-fenwick/selection.json
    --stage final    refits the pick on 2022-23 to 2025-26 with its tree count, tests
                     it on 2026-27 once next to the saved, live v1.4. Saves the model
                     and results; scores nothing into shot_xg.

Usage (repo root, with the project venv's python):
    .venv/bin/python3 -m model.train_v1_5_fenwick --stage select
    .venv/bin/python3 -m model.train_v1_5_fenwick --stage final
"""
from __future__ import annotations

import argparse
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ingest.db import pg_conn
from model import train_v1 as v1
from model import train_v1_fenwick as v1f
from model import train_v1_1_fenwick as v11
from model import train_v1_2_fenwick as v12
from model import train_v1_3_fenwick as v13
from model import train_v1_4_fenwick as v14
from model.reporting import RESULTS_DIR, max_abs_gap, reliability_table, write_results

OUT = RESULTS_DIR / "v1_5-fenwick"
SELECTION_JSON = OUT / "selection.json"
TARGET = v14.TARGET
CATEGORICAL = list(v14.CATEGORICAL)
HALF_LIFE = v14.HALF_LIFE
with open(v14.SELECTION_JSON) as _f:
    SETTINGS = json.load(_f)["settings"]

CANDIDATES = {
    "baseline": v14.NUMERIC + v14.BOOLEAN + CATEGORICAL,
    "v1.5-fenwick": v14.NUMERIC + ["shooter_prior"] + v14.BOOLEAN + CATEGORICAL,
    "v1.6-fenwick": v14.NUMERIC + ["shooter_prior", "goalie_prior"] + v14.BOOLEAN + CATEGORICAL,
}
STEMS = {"v1.5-fenwick": "xg_v1_5_fenwick", "v1.6-fenwick": "xg_v1_6_fenwick"}
ALL_FEATURES = CANDIDATES["v1.5-fenwick"]   # what model.score_fenwick would use for v1.5
BOOT = 1000
SEED = 1515


def load_features() -> pd.DataFrame:
    df = v14.load_features()
    with pg_conn() as conn:
        pri = pd.read_sql("SELECT shot_id, shooter_prior, goalie_prior FROM shot_priors_fenwick",
                          conn)
    df = df.merge(pri, on="shot_id", how="left", validate="one_to_one")
    miss = df[["shooter_prior", "goalie_prior"]].isna().sum()
    if miss.any():
        raise RuntimeError(f"Missing priors: {miss.to_dict()}. "
                           "Run python3 -m features.build_priors_fenwick after the table build.")
    return df


def fit(train, features, n_trees=v14.MAX_TREES, val=None):
    params = dict(v14.FIXED, **SETTINGS, n_estimators=n_trees,
                  monotone_constraints=[-1 if f == "distance_ft" else 0 for f in features])
    model = lgb.LGBMClassifier(**params)
    kwargs = {"categorical_feature": CATEGORICAL,
              "sample_weight": v13.recency_weights(train["season"], HALF_LIFE)}
    if val is not None:
        kwargs.update(eval_set=[(val[features], val[TARGET])],
                      callbacks=[lgb.early_stopping(v14.STOP_ROUNDS, verbose=False)])
    model.fit(train[features], train[TARGET], **kwargs)
    return model


def game_bootstrap(y, pa, pb, games, n=BOOT, seed=SEED) -> dict:
    """Log loss(b) - log loss(a), with a 95% interval from resampling games."""
    eps = 1e-15
    ll = lambda p: -(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1)))  # noqa: E731
    diff = pd.DataFrame({"g": games, "d": ll(pb) - ll(pa)}).groupby("g")["d"].agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    s, c = diff["sum"].to_numpy(), diff["size"].to_numpy()
    boots = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return {"diff": float(s.sum() / c.sum()), "lo": float(np.quantile(boots, .025)),
            "hi": float(np.quantile(boots, .975)), "share_better": float((boots < 0).mean())}


def by_fifth(d, col, preds: dict) -> pd.DataFrame:
    """Goals / xG by fifths of a prior, for each model's xG."""
    q = pd.qcut(d[col].rank(method="first"), 5, labels=[f"{i}" for i in range(1, 6)])
    rows = []
    for k, grp in d.groupby(q, observed=True):
        m = grp.index
        row = {col + " fifth": k, "shots": len(grp), "prior range":
               f"{grp[col].min():.2f}-{grp[col].max():.2f}", "goals": int(grp[TARGET].sum())}
        for name, p in preds.items():
            row[f"O/E {name}"] = grp[TARGET].sum() / p.loc[m].sum()
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- stage: select

def stage_select(parts) -> None:
    tr, va = parts["select_train"], parts["select_val"]
    y = va[TARGET].to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\nv1.4's settings (try {json.load(open(v14.SELECTION_JSON))['best_try']}): {SETTINGS}")

    models, preds, rows = {}, {}, []
    for name, feats in CANDIDATES.items():
        m = fit(tr, feats, val=va)
        p = m.predict_proba(va[feats])[:, 1]
        models[name], preds[name] = m, pd.Series(p, index=va.index)
        met = v14.summary(y, p, va)
        chk = distance_check(m, feats, va)
        rows.append({"candidate": name, "n_trees": int(m.best_iteration_), **{
            k: met[k] for k in ("log_loss", "auc", "brier", "o_e", "cal_slope", "tipin_o_e")},
            "distance_rises": chk["rises"]})
        v14.show(f"{name} ({int(m.best_iteration_)} trees)", met)
        if chk["rises"]:
            raise RuntimeError(f"{name}: xG rises with distance somewhere: {chk}")
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "candidates.csv", index=False)

    games = va["game_id"].to_numpy()
    steps = {"shooter prior (v1.5 - baseline)": ("baseline", "v1.5-fenwick"),
             "goalie prior (v1.6 - v1.5)": ("v1.5-fenwick", "v1.6-fenwick"),
             "both (v1.6 - baseline)": ("baseline", "v1.6-fenwick")}
    gains = []
    print(f"\n--- Log loss change on late 2025-26, 95% interval from {BOOT} game resamples ---")
    for label, (a, b) in steps.items():
        g = game_bootstrap(y, preds[a].to_numpy(), preds[b].to_numpy(), games)
        gains.append({"step": label, **g})
        print(f"  {label:34} {g['diff']:+.5f}  ({g['lo']:+.5f} to {g['hi']:+.5f}), "
              f"better in {g['share_better'] * 100:.0f}% of resamples")
    pd.DataFrame(gains).to_csv(OUT / "gains.csv", index=False)

    fifths = []
    for col in ("shooter_prior", "goalie_prior"):
        t = by_fifth(va, col, preds)
        print(f"\n--- Goals / xG by {col} fifth, late 2025-26 ---")
        print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        fifths.append(t.assign(prior=col))
    pd.concat(fifths).to_csv(OUT / "oe_by_prior_fifth.csv", index=False)

    imp = []
    for name in ("v1.5-fenwick", "v1.6-fenwick"):
        feats = CANDIDATES[name]
        gain = pd.Series(models[name].booster_.feature_importance("gain"), index=feats)
        share = gain / gain.sum()
        for col in ("shooter_prior", "goalie_prior"):
            if col in feats:
                imp.append({"model": name, "feature": col, "gain_share": share[col],
                            "gain_rank": int((gain > gain[col]).sum() + 1)})
    imp = pd.DataFrame(imp)
    print("\n--- Gain importance of the priors ---")
    print(imp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    pick = table.sort_values("log_loss").iloc[0]
    print(f"\nPick: {pick['candidate']} (log loss {pick['log_loss']:.5f}, "
          f"{int(pick['n_trees'])} trees)")
    with open(SELECTION_JSON, "w") as f:
        json.dump({"settings_from": str(v14.SELECTION_JSON), "settings": SETTINGS,
                   "half_life": HALF_LIFE, "candidates": table.to_dict("records"),
                   "gains": gains, "importance": imp.to_dict("records"),
                   "pick": pick["candidate"], "n_trees": int(pick["n_trees"])},
                  f, indent=2, default=float)
    print(f"Frozen in {SELECTION_JSON}. 2026-27 not touched.")


def distance_check(model, feats, va) -> dict:
    """v1.4's distance sweep, for a model with its own feature list."""
    shots = va.sample(2000, random_state=v14.SEED)
    grid = np.arange(0, 201, 1.0)
    reps = shots.loc[shots.index.repeat(len(grid))].copy()
    reps["distance_ft"] = np.tile(grid, len(shots))
    p = model.predict_proba(reps[feats])[:, 1].reshape(len(shots), len(grid))
    step = np.diff(p, axis=1)
    return {"shots": len(shots), "steps": int(step.size), "rises": int((step > 1e-12).sum()),
            "largest_rise": float(step.max())}


# ---------------------------------------------------------------- stage: final

def stage_final(df, parts, cutoff) -> None:
    with open(SELECTION_JSON) as f:
        sel = json.load(f)
    version, n_trees = sel["pick"], sel["n_trees"]
    if version == "baseline":
        print("The baseline won: neither prior helped on late 2025-26. Nothing to test.")
        return
    feats = CANDIDATES[version]
    test = parts["test"]
    y = test[TARGET].to_numpy()

    print(f"\n--- {version}: refit on 2022-23 to 2025-26, {n_trees} trees ---")
    model = fit(parts["refit_train"], feats, n_trees=n_trees)
    chk = distance_check(model, feats, parts["select_val"])
    print(f"  distance check: {chk}")
    if chk["rises"]:
        raise RuntimeError("The refit model's xG rises with distance somewhere.")

    print("\n--- 2026-27 to date, the one test ---")
    p_new = model.predict_proba(test[feats])[:, 1]
    live = joblib.load(v1.ARTIFACT_DIR / f"{v14.ARTIFACT_STEM}.pkl")
    p_live = live.predict_proba(test[v14.ALL_FEATURES])[:, 1]
    m_new, m_live = v14.summary(y, p_new, test), v14.summary(y, p_live, test)
    v14.show(version, m_new)
    v14.show("v1.4-fenwick (saved, live)", m_live)
    g = game_bootstrap(y, p_live, p_new, test["game_id"].to_numpy())
    print(f"  log loss change vs v1.4: {g['diff']:+.5f} ({g['lo']:+.5f} to {g['hi']:+.5f})")

    rel = reliability_table(test[TARGET], p_new)
    print("\n--- Reliability on 2026-27 ---")
    print(rel.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    frames = []
    for dim, fn in v12.SEGMENTS.items():
        seg = fn(test)
        a = v1f.segment_calibration(test[TARGET], p_new, seg).set_index("segment")
        b = v1f.segment_calibration(test[TARGET], p_live, seg).set_index("segment")
        tbl = pd.DataFrame({"n": a["n"], f"oe_{version}": a["o_e"], "oe_v1.4": b["o_e"]})
        tbl.index.name = "segment"
        frames.append(tbl.sort_values("n", ascending=False).reset_index().assign(dimension=dim))
    segs = pd.concat(frames, ignore_index=True)
    print("\n--- O/E by segment on 2026-27 ---")
    print(segs.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    fifths = pd.concat([by_fifth(test, c, {version: pd.Series(p_new, index=test.index),
                                           "v1.4": pd.Series(p_live, index=test.index)})
                        .assign(prior=c) for c in ("shooter_prior", "goalie_prior")])
    print("\n--- Goals / xG by prior fifth on 2026-27 ---")
    print(fifths.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    stem = STEMS[version]
    joblib.dump(model, v1.ARTIFACT_DIR / f"{stem}.pkl")
    split_desc = {"select_train": f"2022-23 to 2024-25 + 2025-26 before {cutoff}",
                  "select_val": f"2025-26 from {cutoff}",
                  "refit_train": v13.TRAIN_SEASONS, "test": v13.TEST_SEASON}
    split_metrics = {
        "train": v11.metrics(parts["refit_train"][TARGET].values,
                             model.predict_proba(parts["refit_train"][feats])[:, 1]),
        "val": v11.metrics(parts["select_val"][TARGET].values,
                           fit(parts["select_train"], feats, n_trees=n_trees)
                           .predict_proba(parts["select_val"][feats])[:, 1]),
        "test": v11.metrics(y, p_new),
    }
    gain = (pd.DataFrame({"feature": feats, "gain": model.booster_.feature_importance("gain")})
              .sort_values("gain", ascending=False))
    meta = {"version": version, "table": v1f.TABLE, "priors_table": "shot_priors_fenwick",
            "half_life": HALF_LIFE, "features": feats, "monotone": {"distance_ft": -1},
            "settings": SETTINGS, "n_trees": n_trees, "split": split_desc,
            "splits": split_metrics, "calibration_max_gap": max_abs_gap(rel),
            "distance_check": chk}
    with open(v1.ARTIFACT_DIR / f"{stem}_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    results_dir = write_results(
        version=version,
        description=("v1.4's settings, recency weights and distance constraint on the "
                     "stoppage-fixed table, plus "
                     + ("a shooter prior" if version == "v1.5-fenwick"
                        else "shooter and goalie priors")
                     + " (games before the shot's date only, shrunk to the league average). "
                     "Tested on 2026-27."),
        splits=split_metrics, reliability=rel, feature_importance=gain,
        hyperparameters={**SETTINGS, "n_trees": n_trees, "half_life": HALF_LIFE,
                         "monotone": "distance_ft -1"},
        features=feats,
        data_summary={"population": "fenwick", "table": v1f.TABLE, "n_shots": int(len(df)),
                      "seasons": [int(s) for s in sorted(df["season"].unique())],
                      "split": split_desc},
        notes=("Test is 2026-27 to date. 'val' is late 2025-26, which also set the tree "
               "count and made the pick, so it's optimistic. Not live."),
    )
    segs.to_csv(results_dir / "segments.csv", index=False)
    fifths.to_csv(results_dir / "test_oe_by_prior_fifth.csv", index=False)
    pd.DataFrame([{"model": version, **m_new}, {"model": "v1.4-fenwick", **m_live},
                  {"model": "log loss change", "log_loss": g["diff"]}]).to_csv(
        results_dir / "test_vs_v1_4.csv", index=False)
    print(f"\nSaved {stem}* and results to {results_dir}/. Not scored into shot_xg.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=("select", "final"), required=True)
    args = ap.parse_args()
    print("=" * 60)
    print(f"v1.5 / v1.6-fenwick, stage {args.stage}")
    print("=" * 60)
    df = load_features()
    v1.assert_no_leakage(df)
    cutoff, parts = v14.splits(df)
    for name, part in parts.items():
        if name == "test" and args.stage == "select":
            print(f"{'test:':14} (not used at this stage)")
            continue
        print(f"{name + ':':14} {part['game_date'].min()} to {part['game_date'].max()}, "
              f"{len(part):,} attempts, {part[TARGET].mean() * 100:.2f}% goals")
    if args.stage == "select":
        stage_select(parts)
    else:
        stage_final(df, parts, cutoff)


if __name__ == "__main__":
    main()
