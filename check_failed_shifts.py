"""Look at one of the failed-shift games to see what the API actually returned."""
import json
from pathlib import Path

CACHE_DIR = Path("raw_data/shifts/20242025")

# Were the JSONs cached at all? If failed games have no JSON on disk,
# the fetch failed. If they have JSON with empty data, the API returned empty.
sample_failed = ["2024021235", "2024021260", "2024021291"]
sample_ok = ["2024020500", "2024020001"]

print("=== Failed games — JSON status ===")
for gid in sample_failed:
    p = CACHE_DIR / f"{gid}.json"
    if p.exists():
        with p.open() as f:
            data = json.load(f)
        n = len(data.get("data", [])) if isinstance(data.get("data"), list) else "non-list"
        print(f"  {gid}: cached, top keys={list(data.keys())}, data len={n}")
    else:
        print(f"  {gid}: NO CACHE — fetch must have failed silently")

print("\n=== Successful games — JSON status (for comparison) ===")
for gid in sample_ok:
    p = CACHE_DIR / f"{gid}.json"
    if p.exists():
        with p.open() as f:
            data = json.load(f)
        n = len(data.get("data", [])) if isinstance(data.get("data"), list) else "non-list"
        print(f"  {gid}: cached, top keys={list(data.keys())}, data len={n}")
    else:
        print(f"  {gid}: NO CACHE")