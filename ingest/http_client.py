"""HTTP client for NHL API. Handles retries, rate limiting, and JSON disk caching."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import HTTP_TIMEOUT, REQUEST_DELAY, USER_AGENT

_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return _session


class FetchError(Exception):
    """Wraps HTTP failures with status code for caller inspection."""
    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status


@retry(
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _http_get(url: str) -> requests.Response:
    resp = _get_session().get(url, timeout=HTTP_TIMEOUT)
    # Retry on 5xx by raising a ConnectionError (so tenacity catches it);
    # 4xx we don't retry — those are permanent (e.g., game doesn't exist yet)
    if 500 <= resp.status_code < 600:
        raise requests.ConnectionError(f"{resp.status_code} from {url}")
    return resp


def fetch_json(url: str, cache_path: Path | None = None) -> dict:
    """Fetch JSON with optional disk cache.

    If cache_path is provided and the file exists, load from disk.
    Otherwise fetch, write to disk, return.
    Raises FetchError on 4xx or parsing failure.
    """
    if cache_path is not None and cache_path.exists():
        with cache_path.open("r") as f:
            return json.load(f)

    resp = _http_get(url)

    if resp.status_code == 404:
        raise FetchError(f"Not found: {url}", status=404)
    if not resp.ok:
        raise FetchError(f"HTTP {resp.status_code}: {url}", status=resp.status_code)

    try:
        data = resp.json()
    except ValueError as e:
        raise FetchError(f"Invalid JSON from {url}: {e}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to .tmp then rename (avoids half-written files on Ctrl-C)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(data, f)
        tmp.rename(cache_path)

    time.sleep(REQUEST_DELAY)  # be nice to the API
    return data
