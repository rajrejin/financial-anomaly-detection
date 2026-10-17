"""Cached, polite HTTP access. Raw responses are stored gzip-compressed and never refetched."""

from __future__ import annotations

import gzip
import logging
import time
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)
_last_request = 0.0


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=60), reraise=True)
def _get(url: str, params: dict[str, str]) -> httpx.Response:
    global _last_request
    wait = 0.5 - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    r = httpx.get(url, params=params, timeout=120, headers={"Accept": "*/*"})
    _last_request = time.time()
    if r.status_code in (413, 429, 500, 502, 503, 504):
        r.raise_for_status()
    return r


def fetch_cached(url: str, params: dict[str, str], cache_file: Path) -> bytes | None:
    """Return response body (bytes), reading from the gzip cache when present.

    Returns None for 404 (no results). Raises for other errors.
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists():
        return gzip.decompress(cache_file.read_bytes())
    log.info("fetching %s %s", url, params)
    r = _get(url, params)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    cache_file.write_bytes(gzip.compress(r.content))
    return r.content
