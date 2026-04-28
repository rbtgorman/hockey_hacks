"""HockeyViz config — endpoints, seasons, paths."""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- API endpoints ---
NHL_WEB_BASE = "https://api-web.nhle.com/v1"
NHL_STATS_BASE = "https://api.nhle.com/stats/rest/en"

SCHEDULE_URL = NHL_WEB_BASE + "/schedule/{date}"           # date = YYYY-MM-DD
PBP_URL      = NHL_WEB_BASE + "/gamecenter/{game_id}/play-by-play"
LANDING_URL  = NHL_WEB_BASE + "/gamecenter/{game_id}/landing"   # scoring/penalties/stars only (2024-25+)
BOXSCORE_URL = NHL_WEB_BASE + "/gamecenter/{game_id}/boxscore"  # rosters + starters live here
SHIFTS_URL   = NHL_STATS_BASE + "/shiftcharts?cayenneExp=gameId={game_id}"
EDGE_SKATER  = NHL_WEB_BASE + "/edge/skater-detail/{player_id}/{season}/{game_type}"
EDGE_GOALIE  = NHL_WEB_BASE + "/edge/goalie-detail/{player_id}/{season}/{game_type}"

# --- Seasons: NHL encodes as YYYYYYYY (start+end year) ---
# Regular season runs roughly Oct 1 to mid-April
SEASONS = [
    # (season_id, start_date, end_date) — end_date covers playoffs too
    (20222023, "2022-10-07", "2023-06-20"),
    (20232024, "2023-10-10", "2024-06-25"),
    (20242025, "2024-10-04", "2025-06-30"),
]

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
RAW_DATA_DIR.mkdir(exist_ok=True)

# --- Postgres ---
# Expected env vars (put in .env file):
#   PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD
PG_DSN = (
    f"host={os.getenv('PG_HOST','localhost')} "
    f"port={os.getenv('PG_PORT','5432')} "
    f"dbname={os.getenv('PG_DB','hockeyviz')} "
    f"user={os.getenv('PG_USER','postgres')} "
    f"password={os.getenv('PG_PASSWORD','')}"
)

# --- HTTP behavior ---
# Be polite: NHL API is free. Don't hammer it.
HTTP_TIMEOUT = 20        # seconds
REQUEST_DELAY = 0.25     # between successful calls
USER_AGENT = "HockeyViz/0.1 (research; contact via github)"