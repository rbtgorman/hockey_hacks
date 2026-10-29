#!/usr/bin/env bash
# Refresh the period page's data for the season in progress.
# Run from the repo root; every step is safe to rerun. Afterwards, look at
# site/periods.html locally, then commit site/data/ and push to deploy.
set -euo pipefail
PY=.venv/bin/python
SEASON="${1:-20262027}"

$PY -m ingest.run --seasons "$SEASON"         # new official (OFF) games only
$PY -m ingest.run_stage_a --season "$SEASON"  # shift charts and rosters
$PY -m ingest.run_shifts_html                 # games whose shift chart came back empty
$PY -m features.build_features_fenwick        # full rebuild, about a minute
$PY -m model.score_fenwick                    # xG for every attempt into shot_xg
$PY -m export.period_tables                   # site/data/periods_*.json
git status --short site/data
