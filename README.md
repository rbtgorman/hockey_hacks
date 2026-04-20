# HockeyViz Pro — Data Pipeline

Stage 1 of the project: pull 3 seasons of NHL play-by-play into Postgres, with raw JSONs cached to disk.

## Setup on your coding PC

```bash
# 1. Clone / sync this dir to the PC, then:
cd hockeyviz
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Create Postgres DB
createdb hockeyviz
# (if your PG needs auth, set env vars — see below)

# 3. Create .env in the project root
cat > .env <<EOF
PG_HOST=localhost
PG_PORT=5432
PG_DB=hockeyviz
PG_USER=postgres
PG_PASSWORD=your_password
EOF

# 4. Initialize schema
python -m ingest.run --init-schema
```

## Ingest

```bash
# Smoke test: pull 20 games from the most recent season
python -m ingest.run --seasons 20242025 --max-games 20

# Full run: all 3 seasons (~4,000 games, ~2GB JSON, ~2–4 hours depending on API latency)
python -m ingest.run --seasons all

# If it dies partway, just re-run — it resumes from ingest_log
python -m ingest.run --seasons all
```

## Verify

After the smoke test:

```sql
-- Connect with:  psql hockeyviz
SELECT COUNT(*) FROM games;
SELECT COUNT(*) FROM shots;
SELECT * FROM v_shot_rate_by_season;

-- Sanity: league-wide shooting % should be ~9-10%
SELECT
  COUNT(*) FILTER (WHERE is_goal) AS goals,
  COUNT(*) FILTER (WHERE is_sog) AS sog,
  ROUND(100.0 * COUNT(*) FILTER (WHERE is_goal) / NULLIF(COUNT(*) FILTER (WHERE is_sog), 0), 2) AS sh_pct
FROM shots;

-- Distance distribution should peak around 15-25ft (slot shots)
SELECT WIDTH_BUCKET(distance_ft, 0, 100, 10) AS bucket,
       COUNT(*) AS shots,
       ROUND(100.0 * COUNT(*) FILTER (WHERE is_goal) / COUNT(*), 2) AS goal_pct
FROM shots WHERE is_sog
GROUP BY 1 ORDER BY 1;
```

If league shooting % is ~9–10% and closer shots convert at higher rates than distant ones, the parser is working correctly. If those numbers look wrong, **stop before training the model** and ping me — the normalization convention in `parser.py::_normalize_coords` makes an assumption about which side the home team defends in period 1 that needs to be verified against a known game.

## Project layout

```
db/schema.sql            Postgres DDL
ingest/
  config.py              seasons + endpoints + PG DSN
  http_client.py         retries, caching to disk
  parser.py              NHL JSON -> normalized rows
  db.py                  batch upserts
  run.py                 orchestrator (this is what you run)
tests/
  test_parser.py         synthetic payload test (passes 26 checks)
raw_data/                cached JSONs (~2GB when full)
```

## Known limitations (for the next step)

- **Coordinate normalization assumes home team defends negative-x in period 1.** This is the NHL convention, but different arenas set it up differently. First QA check: pick one known game, verify that goals go in the expected direction.
- **Empty-net detection is per-shot**, not per-game-state. A shot is flagged `empty_net=True` only when the *defending* goalie is pulled at shot time.
- **`plays_context` stores shots too** — this is intentional, so feature engineering can look at "time since last shot attempt" without joining both tables.
- **No EDGE ingester yet.** Next step. EDGE is season-level per player, so we pull it once per season once we know which players appeared in the shots table.
