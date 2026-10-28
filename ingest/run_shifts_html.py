"""Recover shifts from NHL HTML TOI reports for games the shiftcharts API left empty.

Only touches games whose ingest_log row is kind='shifts', status='fail'
(57 games of 2024-25, April 8-15 2025, at the time of writing). Rows go into
the same shifts table through db.insert_shifts, with raw_json.source =
'htmlreport' so they can always be told apart from API rows.

Usage:
    python -m ingest.run_shifts_html --validate 2024021234 2024021200
        # parse-only: compare HTML against the API rows already stored, write nothing
    python -m ingest.run_shifts_html
        # ingest every failed game
"""
from __future__ import annotations

import argparse
import sys
import time

from tqdm import tqdm

from .config import RAW_DATA_DIR, REQUEST_DELAY
from .db import insert_shifts, log_ingest, pg_conn
from .http_client import FetchError, _http_get
from .parser_shifts_html import parse_shifts_html

HTML_REPORT_URL = "https://www.nhl.com/scores/htmlreports/{season}/{side}{game_no}.HTM"


def fetch_report(game_id: int, season: int, side: str) -> str:
    """side 'TH' (home) or 'TV' (visitor). Cached under raw_data/shifts_html/."""
    game_no = str(game_id)[4:]
    cache = RAW_DATA_DIR / f"shifts_html/{season}/{side}{game_no}.HTM"
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    url = HTML_REPORT_URL.format(season=season, side=side, game_no=game_no)
    resp = _http_get(url)
    if not resp.ok:
        raise FetchError(f"HTTP {resp.status_code}: {url}", status=resp.status_code)
    resp.encoding = "utf-8"  # the reports declare charset=UTF-8
    text = resp.text

    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(cache.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.rename(cache)
    time.sleep(REQUEST_DELAY)
    return text


def parse_game(cur, game_id: int):
    """Fetch and parse both sides of one game. Returns (shifts, warnings)."""
    cur.execute("SELECT season, home_team_id, away_team_id FROM games WHERE game_id = %s",
                (game_id,))
    season, home_id, away_id = cur.fetchone()

    shifts, warnings = [], []
    for side, team_id in (("TH", home_id), ("TV", away_id)):
        cur.execute("""
            SELECT sweater, player_id FROM game_rosters
            WHERE game_id = %s AND team_id = %s AND sweater IS NOT NULL
        """, (game_id, team_id))
        sweaters = {int(s): int(p) for s, p in cur.fetchall()}
        if not sweaters:
            warnings.append(f"{side}: no roster rows for team {team_id}")
            continue
        r = parse_shifts_html(fetch_report(game_id, season, side), game_id, team_id, sweaters)
        shifts.extend(r.shifts)
        warnings.extend(f"{side}: {w}" for w in r.warnings)
    return shifts, warnings


def validate(game_ids: list[int]) -> bool:
    """Compare HTML rows against stored API rows (typeCode 517 only). Writes nothing."""
    all_ok = True
    with pg_conn() as conn, conn.cursor() as cur:
        for gid in game_ids:
            cur.execute("""
                SELECT player_id, period, start_seconds, end_seconds FROM shifts
                WHERE game_id = %s AND raw_json->>'typeCode' = '517'
            """, (gid,))
            api = set(cur.fetchall())
            shifts, warnings = parse_game(cur, gid)
            html = {(s.player_id, s.period, s.start_seconds, s.end_seconds) for s in shifts}
            matched = len(api & html)
            pct = 100 * matched / len(api) if api else 0.0
            all_ok &= pct >= 98.0
            print(f"{gid}: api={len(api)} html={len(html)} matched={matched} "
                  f"({pct:.1f}% of api)  api-only={len(api - html)} html-only={len(html - api)} "
                  f"warnings={len(warnings)}")
            for w in warnings[:5]:
                print(f"    - {w}")
            for row in sorted(api ^ html)[:5]:
                print(f"    diff {'api ' if row in api else 'html'} {row}")
    return all_ok


def failed_games(conn) -> list[int]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT CAST(target_key AS BIGINT) FROM ingest_log
            WHERE kind = 'shifts' AND status = 'fail'
            ORDER BY 1
        """)
        return [r[0] for r in cur.fetchall()]


def run() -> None:
    with pg_conn() as conn:
        todo = failed_games(conn)
    print(f"Games with failed API shifts: {len(todo)}")

    ok = fail = total = 0
    for gid in tqdm(todo, desc="shifts_html", unit="game"):
        try:
            with pg_conn() as conn, conn.cursor() as cur:
                shifts, warnings = parse_game(cur, gid)
                if not shifts:
                    raise ValueError(f"no shifts parsed ({len(warnings)} warnings)")
                n = insert_shifts(cur, shifts)
                log_ingest(cur, "shifts", str(gid), "ok", shots_parsed=n,
                           error_msg="source=htmlreport")
                conn.commit()
            ok += 1
            total += len(shifts)
            if warnings:
                tqdm.write(f"  {gid}: {len(warnings)} warnings, first: {warnings[0]}")
        except Exception as e:  # noqa: BLE001
            fail += 1
            tqdm.write(f"  FAIL shifts_html {gid}: {type(e).__name__}: {e}")
    print(f"HTML shifts: {ok} ok, {fail} failed, {total} shift rows stored.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=int, nargs="+", metavar="GAME_ID",
                    help="Parse-only comparison against stored API shifts")
    args = ap.parse_args()

    if args.validate:
        sys.exit(0 if validate(args.validate) else 1)
    run()


if __name__ == "__main__":
    main()
