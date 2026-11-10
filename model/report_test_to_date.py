"""Report-only: every saved model's numbers on the season in progress, as it stands now.

Training-time results (results/<version>/) are frozen at the day each model was made.
This rescores the test season to date with each saved model, on today's table, so the
site's numbers can grow with the season. It trains, picks and saves no model and writes
nothing to the database.

Writes results/test_to_date/:
    summary.csv       one row per model: log loss, AUC, O/E, tip-in O/E, n, dates
    reliability.csv   the live model's ten equal-count groups
    segments.csv      O/E by segment for every model

Usage (repo root, with the project venv's python):
    .venv/bin/python3 -m model.report_test_to_date
"""
from __future__ import annotations

import importlib
import warnings

import joblib
import pandas as pd

from model import train_v1 as v1
from model import train_v1_fenwick as v1f
from model import train_v1_2_fenwick as v12
from model import train_v1_3_fenwick as v13
from model import train_v1_4_fenwick as v14
from model.reporting import RESULTS_DIR, reliability_table
from model.score_fenwick import DEFAULT_VERSION, MODELS

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
OUT = RESULTS_DIR / "test_to_date"
VERSIONS = ["v1.4.1-fenwick", "v1.4-fenwick", "v1.3-fenwick", "v1.2-fenwick"]
TARGET = v14.TARGET


def main() -> None:
    base = None
    preds = {}
    for version in VERSIONS:
        stem, module = MODELS[version]
        trainer = importlib.import_module(module)
        df = trainer.load_features()
        if version == DEFAULT_VERSION:
            totals = (len(df), df["game_id"].nunique(), df["game_date"].min().date(),
                      df["game_date"].max().date())
        df = df[df["season"] == v13.TEST_SEASON]
        model = joblib.load(v1.ARTIFACT_DIR / f"{stem}.pkl")
        p = pd.Series(model.predict_proba(df[trainer.ALL_FEATURES])[:, 1],
                      index=df["shot_id"].to_numpy())
        if base is None:
            base = df.reset_index(drop=True)
        preds[version] = p.reindex(base["shot_id"]).to_numpy()
        if pd.isna(preds[version]).any():
            raise RuntimeError(f"{version} is missing some test shots.")

    y = base[TARGET].to_numpy()
    first, last = base["game_date"].min().date(), base["game_date"].max().date()
    rows, frames = [], []
    for version, p in preds.items():
        m = v14.summary(y, p, base)
        rows.append({"model": version, "first_game": first, "last_game": last,
                     "games": base["game_id"].nunique(), "goals": int(y.sum()),
                     "xg": float(p.sum()), **m})
        for dim, fn in v12.SEGMENTS.items():
            seg = v1f.segment_calibration(base[TARGET], p, fn(base))
            frames.append(seg.assign(dimension=dim, model=version))
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    rel = reliability_table(base[TARGET], preds[DEFAULT_VERSION])
    rel.to_csv(OUT / "reliability.csv", index=False)
    segs = pd.concat(frames, ignore_index=True)
    segs.to_csv(OUT / "segments.csv", index=False)

    print(f"{v13.TEST_SEASON} to date: {first} to {last}, {summary.games[0]} games, "
          f"{len(base):,} attempts, {int(y.sum()):,} goals. Live model: {DEFAULT_VERSION}.")
    print(f"Whole table: {totals[0]:,} attempts in {totals[1]:,} games, {totals[2]} to {totals[3]}.")
    print(summary[["model", "xg", "log_loss", "auc", "o_e", "cal_slope", "tipin_o_e"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\nReliability, live model:")
    print(rel.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    worst = rel.loc[rel["pred_vs_actual_gap"].abs().idxmax()]
    print(f"Biggest gap: group {int(worst['bin'])}, {worst['mean_pred'] * 100:.2f}% predicted, "
          f"{worst['actual_rate'] * 100:.2f}% went in, "
          f"{abs(worst['pred_vs_actual_gap']) * 100:.2f} points.")
    wide = segs.pivot_table(index=["dimension", "segment"], columns="model", values="o_e")
    wide.insert(0, "n", segs[segs.model == DEFAULT_VERSION].set_index(["dimension", "segment"])["n"])
    print("\nO/E by segment:")
    print(wide[["n"] + VERSIONS].to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
