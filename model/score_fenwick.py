"""Score every row of shot_features_fenwick with a saved model into shot_xg.

The site's xG numbers come from this table, so each one names the model that
made it. A run replaces that version's rows in one transaction and leaves other
versions alone. Rerun it after every features.build_features_fenwick rebuild,
since new games only get xG here.

Usage (repo root, with the project venv's python):
    .venv/bin/python3 -m model.score_fenwick                         # v1.2-fenwick
    .venv/bin/python3 -m model.score_fenwick --version v1.1-fenwick
"""
from __future__ import annotations

import argparse
import importlib
import io
import time
from pathlib import Path

import joblib
import pandas as pd

from ingest.db import pg_conn
from model import train_v1 as v1

# version -> (artifact stem, trainer module that defines load_features / ALL_FEATURES)
MODELS = {
    "v1.1-fenwick": ("xg_v1_1_fenwick", "model.train_v1_1_fenwick"),
    "v1.2-fenwick": ("xg_v1_2_fenwick", "model.train_v1_2_fenwick"),
}
DEFAULT_VERSION = "v1.2-fenwick"
MIGRATION = Path(__file__).resolve().parent.parent / "db" / "migrations" / "003_shot_xg.sql"


def main() -> None:
    ap = argparse.ArgumentParser(description="Score shot_features_fenwick into shot_xg.")
    ap.add_argument("--version", default=DEFAULT_VERSION, choices=sorted(MODELS))
    args = ap.parse_args()

    stem, module = MODELS[args.version]
    trainer = importlib.import_module(module)
    df = trainer.load_features()
    model = joblib.load(v1.ARTIFACT_DIR / f"{stem}.pkl")

    t0 = time.time()
    xg = model.predict_proba(df[trainer.ALL_FEATURES])[:, 1]
    print(f"\nScored {len(df):,} attempts with {args.version} in {time.time() - t0:.1f} s")

    check = (pd.DataFrame({"season": df["season"], "goals": df[trainer.TARGET], "xg": xg})
               .groupby("season").sum())
    check["o_e"] = check["goals"] / check["xg"]
    print("Observed / expected by season (in-sample for the model's training seasons):")
    print(check.round({"xg": 1, "o_e": 3}).to_string())

    buf = io.StringIO()
    pd.DataFrame({"shot_id": df["shot_id"], "xg": xg}).to_csv(buf, index=False, header=False)
    buf.seek(0)
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(MIGRATION.read_text())
        cur.execute("DELETE FROM shot_xg WHERE model_version = %s", (args.version,))
        cur.execute("CREATE TEMP TABLE xg_load (shot_id BIGINT, xg DOUBLE PRECISION) "
                    "ON COMMIT DROP")
        cur.copy_expert("COPY xg_load FROM STDIN WITH (FORMAT csv)", buf)
        cur.execute("INSERT INTO shot_xg (model_version, shot_id, xg) "
                    "SELECT %s, shot_id, xg FROM xg_load", (args.version,))
        n = cur.rowcount
        conn.commit()
    print(f"\n{n:,} rows written to shot_xg for {args.version}.")


if __name__ == "__main__":
    main()
