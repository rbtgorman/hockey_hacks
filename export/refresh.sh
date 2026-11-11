#!/usr/bin/env bash
# Refresh the period page's and the profiles' data for the season in progress.
# Run from the repo root; every step is safe to rerun. Afterwards, look at
# site/periods.html and site/profiles.html locally, then commit site/data/ and push to deploy.
#
# Two steps can stop the refresh before anything is scored or exported (both
# exit with an error, and `set -e` stops the script there):
#   ingest.check_ingest           games complete, goals add up, no dropped shots
#   features.build_features_fenwick  shot-table checks 3, 8, 11, 12 must be 0
# If one stops it, read what it printed; nothing on the site has changed.
set -euo pipefail
PY=.venv/bin/python
SEASON="${1:-20262027}"

$PY -m ingest.run --seasons "$SEASON"         # new official (OFF) games only
$PY -m ingest.run_stage_a --season "$SEASON"  # shift charts and rosters
$PY -m ingest.run_shifts_html                 # games whose shift chart came back empty
$PY -m ingest.check_ingest                    # stops on missing games, goals that don't add up, dropped shots
$PY -m features.build_features_fenwick        # full rebuild, about a minute; stops if a must-be-0 check isn't
$PY -m model.score_fenwick                    # xG for every attempt into shot_xg
$PY -m export.period_tables                   # site/data/periods_*.json
$PY -m export.profiles                        # site/data/profiles_index.json, teams/, players/
$PY -m ingest.check_ingest --report-warnings  # parse warnings, for reading
git status --short site/data
