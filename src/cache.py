"""Valkey-backed cache for the MCP server.

Named caches:
- `page_cache`      — single source of truth per scraped/extracted page.
- `searxng_cache`   — raw SearXNG responses keyed on (query, time_range,
                      language, pageno). Filter-agnostic; filters apply
                      at response-shaping time.
- `seen_urls`       — previously-returned URLs for the `previously_seen`
                      flag on search results.
- `content_alias`   — content_hash → canonical URL, for exact-dupe
                      aliasing inside page_cache.

They replace the per-session in-memory `cachetools.TTLCache` trio that
used to hang off FastMCP Context state. A single connection pool backs
them all; keys are prefixed per-cache and carry a 3600 s TTL.

Valkey is an internal compose service alongside searxng and crawl4ai. If
the stack is up, Valkey is up — there is no in-request fallback path.
Tests inject a fake via :func:`set_client`.

Note on the connection URL: the `redis://` scheme is the RESP protocol
scheme used by the `redis-py` client, not a reference to the Redis
server binary. Valkey speaks the same wire protocol.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import redis.asyncio as _resp_client


_DEFAULT_URL = "redis://valkey:6379/0"
_DEFAULT_TTL_S = 3600
# Short TTL for failed / rejected page entries. Long enough to prevent
# immediate retry thrash on bad URLs, short enough that a transient
# upstream failure (CAPTCHA, 5xx, brief timeout) can recover in under a
# minute instead of being locked out for an hour.
FAILURE_TTL_S = 60

_client: _resp_client.Redis | None = None


def _get_client() -> _resp_client.Redis:
    global _client
    if _client is None:
        url = os.environ.get("VALKEY_URL", _DEFAULT_URL)
        _client = _resp_client.from_url(url, decode_responses=True)
    return _client


def set_client(client: _resp_client.Redis | None) -> None:
    """Swap the backing client — used by tests to inject a fake."""
    global _client
    _client = client


async def ping() -> bool:
    """Return True when the backing store responds to PING. Used by /ready."""
    try:
        return bool(await _get_client().ping())
    except Exception:
        return False


class KVCache:
    """Async dict-like facade with a key prefix and TTL.

    Values are JSON-serialized on write and parsed on read. Mirrors the
    read/write surface the old TTLCache exposed (contains / get / set)
    so call sites in core.py change minimally.

    Every `get` increments a per-cache hit/miss counter at
    `ws:stats:<prefix>:{hits,misses}`. These are plain Valkey INCR
    counters; no TTL (we want lifetime totals). Read via /metrics.
    """

    def __init__(self, prefix: str, *, ttl: int = _DEFAULT_TTL_S) -> None:
        self._prefix = prefix
        self._ttl = ttl
        self._hits_key = f"ws:stats:{prefix}:hits"
        self._misses_key = f"ws:stats:{prefix}:misses"

    def _key(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    async def contains(self, key: str) -> bool:
        return bool(await _get_client().exists(self._key(key)))

    async def get(self, key: str) -> Any | None:
        client = _get_client()
        raw = await client.get(self._key(key))
        if raw is None:
            await client.incr(self._misses_key)
            return None
        await client.incr(self._hits_key)
        return json.loads(raw)

    async def set(self, key: str, value: Any, *, ttl: int | None = None) -> None:
        """Write a value. `ttl` overrides the cache default when given
        (used for failure entries that should expire faster)."""
        await _get_client().set(
            self._key(key),
            json.dumps(value),
            ex=ttl if ttl is not None else self._ttl,
        )

    async def delete(self, key: str) -> None:
        await _get_client().delete(self._key(key))

    async def stats(self) -> dict[str, int]:
        """Return this cache's lifetime hit/miss counts."""
        client = _get_client()
        hits, misses = await asyncio.gather(
            client.get(self._hits_key), client.get(self._misses_key),
        )
        return {
            "hits": int(hits) if hits else 0,
            "misses": int(misses) if misses else 0,
        }


# Single source of truth per page. The scrape path and the extract path
# both read/write here with the same envelope shape so a scrape by
# `search` is an immediate hit for a later `extract` on the same URL.
page_cache = KVCache("ws:page")
# Raw SearXNG responses keyed on (query, time_range, language, pageno) —
# no filter params. Filters are applied at response-shaping time so a
# query cached with no filters can still serve a follow-up that adds an
# include_domains constraint without a cache miss.
searxng_cache = KVCache("ws:searxng")
# Stored as individual keys rather than a set so each URL carries its
# own TTL, matching the per-entry expiry TTLCache used to give us.
seen_urls = KVCache("ws:seen")
# content_hash → canonical normalized URL. When two URLs scrape byte-
# identical content we write only one full entry and alias the rest
# through this map. See core._page_set / core._page_get.
content_alias = KVCache("ws:content")
