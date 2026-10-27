"""Audit which `details` keys appear on non-shot events, per season.

The point is drift detection. plays_context throws away every player id on hit,
penalty, giveaway, takeaway and faceoff events, and the fix is a parser change
driven by a key map. That map is only safe if the key names are stable across
every season being parsed. This project has already been bitten twice by the NHL
changing payload semantics between seasons (/landing losing roster data in the
2024-25 refresh, 'short' miss reasons appearing in 2023-24), so the map gets
built from observed payloads, never from convention.

Reads cached JSON under raw_data/pbp/ only. Makes no network calls.

Usage:
    python -m diagnostics.check_context_actor_keys
    python -m diagnostics.check_context_actor_keys --games 400
    python -m diagnostics.check_context_actor_keys --seasons 20242025 20252026

Output is a per-event-type matrix of key fill rates with one column per season,
plus a DRIFT section listing keys that are not present in every season.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PBP = PROJECT_ROOT / "raw_data" / "pbp"

# Event types whose actors we need and currently discard. 'goal' is here to
# confirm the audit claim that assist ids already sit in shots.raw_details_json.
WATCH = [
    "hit",
    "penalty",
    "giveaway",
    "takeaway",
    "faceoff",
    "blocked-shot",
    "missed-shot",
    "goal",
]

# A fill rate below this on an otherwise-present key is worth looking at by eye
# rather than wiring straight into the parser.
LOW_FILL = 0.50


def discover_seasons() -> list[int]:
    if not RAW_PBP.exists():
        raise SystemExit(f"No cached play-by-play under {RAW_PBP}")
    out = []
    for d in sorted(RAW_PBP.iterdir()):
        if d.is_dir():
            try:
                out.append(int(d.name))
            except ValueError:
                continue
    if not out:
        raise SystemExit(f"No season directories under {RAW_PBP}")
    return out


def scan_season(season: int, limit: int):
    """Return (event_counts, key_counts, n_files) for one season."""
    files = sorted((RAW_PBP / str(season)).glob("*.json"))[:limit]
    events: Counter[str] = Counter()
    keys: dict[str, Counter[str]] = defaultdict(Counter)

    for path in files:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"  [warn] unreadable {path.name}: {e}")
            continue
        for play in payload.get("plays", []) or []:
            etype = play.get("typeDescKey")
            if etype not in WATCH:
                continue
            events[etype] += 1
            details = play.get("details") or {}
            if not isinstance(details, dict):
                continue
            for key in details:
                keys[etype][key] += 1

    return events, keys, len(files)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=200,
                    help="Max games sampled per season (default 200)")
    ap.add_argument("--seasons", type=int, nargs="*", default=None,
                    help="Seasons to scan. Default: every season on disk")
    args = ap.parse_args()

    seasons = args.seasons if args.seasons else discover_seasons()

    per_season = {}
    for s in seasons:
        events, keys, n_files = scan_season(s, args.games)
        per_season[s] = {"events": events, "keys": keys, "n_files": n_files}
        print(f"Scanned {n_files:>4} games from {s}")

    thin = [s for s in seasons if per_season[s]["n_files"] < 20]
    if thin:
        print(f"\nNOTE: {', '.join(str(s) for s in thin)} sampled under 20 games. "
              "Treat those columns as key PRESENCE only. Fill rates are unreliable "
              "at that sample size, and presence is what the parser map needs.")

    width = max(10, max(len(str(s)) for s in seasons) + 2)

    for etype in WATCH:
        if not any(per_season[s]["events"][etype] for s in seasons):
            continue

        print(f"\n{'=' * 72}")
        print(f"=== {etype}")
        print(f"{'=' * 72}")

        header = f"{'key':<32}" + "".join(f"{s:>{width}}" for s in seasons)
        print(header)
        print(f"{'n events':<32}"
              + "".join(f"{per_season[s]['events'][etype]:>{width}}" for s in seasons))
        print("-" * len(header))

        all_keys = sorted({
            k
            for s in seasons
            for k in per_season[s]["keys"][etype]
        })

        for key in all_keys:
            is_player = key.lower().endswith("playerid")
            cells = ""
            for s in seasons:
                n = per_season[s]["events"][etype]
                c = per_season[s]["keys"][etype][key]
                cells += f"{(c / n):>{width}.1%}" if n else f"{'-':>{width}}"
            marker = " *" if is_player else "  "
            print(f"{key:<30}{marker}{cells}")

        # Drift: a key seen in some seasons but missing entirely from others.
        present = {
            key: [s for s in seasons
                  if per_season[s]["events"][etype]
                  and per_season[s]["keys"][etype][key] > 0]
            for key in all_keys
        }
        live_seasons = [s for s in seasons if per_season[s]["events"][etype]]
        drifted = {k: v for k, v in present.items() if len(v) != len(live_seasons)}

        low = []
        for key in all_keys:
            if not key.lower().endswith("playerid"):
                continue
            for s in live_seasons:
                n = per_season[s]["events"][etype]
                c = per_season[s]["keys"][etype][key]
                if 0 < c / n < LOW_FILL:
                    low.append((key, s, c / n))

        if drifted or low:
            print("\n  DRIFT")
            for key, seen in sorted(drifted.items()):
                missing = [s for s in live_seasons if s not in seen]
                print(f"    {key}: absent from {', '.join(str(m) for m in missing)}")
            for key, s, rate in low:
                print(f"    {key}: {rate:.1%} fill in {s}, below {LOW_FILL:.0%}")

    print(f"\n{'=' * 72}")
    print("* marks a player id key. These are the ones the parser map needs.")
    print("A key absent from any season means the map needs a per-season branch,")
    print("not a single constant. Low fill is not automatically wrong: penalty")
    print("drawn-by is legitimately empty on bench minors and too-many-men.")


if __name__ == "__main__":
    main()