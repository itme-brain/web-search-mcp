"""Unified page cache envelopes and scrape-cache admission."""

import logging
import re

from web_search_mcp.common import _normalize_url
from web_search_mcp.crawling.operations import _scrape
from web_search_mcp.extraction.html import _content_hash
from web_search_mcp.storage import cache as cache_module
from web_search_mcp.storage.cache import KVCache

log = logging.getLogger("web-search-mcp")
_MIN_CACHE_WORDS = 20

# Phrases that cluster on login walls / auth pages across most social and
# SaaS sites. A real article that happens to mention logging in may hit
# one or two; short pages dense with multiple hits are the content
# Crawl4AI got when the page redirected to or rendered auth UI instead of
# the post / thread / article the caller asked for.
_LOGIN_WALL_RE = re.compile(
    r"\b(log\s?in|sign\s?in|sign\s?up|"
    r"forgot\s+(?:your\s+)?(?:password|account)|"
    r"create\s+(?:new\s+)?account|"
    r"email\s+or\s+phone(?:\s+number)?)\b",
    re.IGNORECASE,
)
_LOGIN_WALL_MIN_HITS = 3
_LOGIN_WALL_MAX_WORDS = 400
_CAPTCHA_RE = re.compile(r"\b(captcha|robot check|verify you are human|unusual traffic|automated queries)\b", re.IGNORECASE)
_PAYWALL_RE = re.compile(r"\b(subscribe to continue|subscription required|paywall|members only|sign in to continue)\b", re.IGNORECASE)


def _content_diagnostic(content: str | None) -> str | None:
    if not content:
        return None
    if _CAPTCHA_RE.search(content):
        return "captcha_or_bot_check"
    if _PAYWALL_RE.search(content):
        return "paywall_or_subscription_wall"
    if _is_login_wall(content):
        return "login_wall"
    if len(content.split()) < _MIN_CACHE_WORDS:
        return "content_too_short"
    return None


def _is_login_wall(content: str | None) -> bool:
    """Heuristic: scraped content is auth-chrome rather than real content.

    Matches short pages that pack multiple login/signup phrases together
    (e.g. Facebook post URLs served to logged-out users, some paywalled
    sites, Twitter individual tweet pages). A real article with >400
    words that mentions "log in" once stays below the threshold.
    """
    if not content:
        return False
    if len(content.split()) >= _LOGIN_WALL_MAX_WORDS:
        return False
    return len(_LOGIN_WALL_RE.findall(content)) >= _LOGIN_WALL_MIN_HITS


async def _scrape_cached(url: str, cache: KVCache) -> dict:
    """Scrape with shared page cache. Returns the full envelope.

    The envelope is the same shape a fresh extract would cache, so the
    search pipeline and user-facing extract calls share a single entry
    per normalized URL. Callers that only need {content, title,
    metadata} read those fields; callers that need status / file_type /
    etc. read those too.

    Speculative fetches (search/crawl) apply a minimum-length gate:
    scraped pages with fewer than _MIN_CACHE_WORDS cache as failures.
    Blocks CAPTCHA walls / error shells from crowding the cache.
    """
    key = _normalize_url(url)
    existing = await _page_get(url, cache)
    if existing is not None:
        log.debug("page cache hit url=%s", url)
        return existing
    result = await _scrape(url)
    content = result.get("content")
    metadata = result.get("metadata") or {}
    # Length/auth/CAPTCHA/paywall gates — only applied at speculative
    # write time, not on user-directed extract (which wants whatever it
    # asked for). Search/crawl fall back to snippets so walls don't crowd
    # out real results.
    diagnostic = _content_diagnostic(content)
    if diagnostic:
        log.info("scrape rejected url=%s reason=%s", url, diagnostic)
        metadata = {**metadata, "diagnostic": diagnostic}
        content = None
    entry = _page_entry(
        url=url,
        content=content,
        title=result.get("title"),
        metadata=metadata,
    )
    await _page_set(url, entry, cache)
    return entry


async def _page_set(url: str, entry: dict, cache: KVCache) -> None:
    """Write a page entry, aliasing when content_hash already seen.

    If another URL has already cached a page with this exact content,
    write a lightweight alias entry instead of a duplicate full entry.
    Dangling detection happens on read via _page_get.

    Failed/rejected entries (status != "ok") get FAILURE_TTL_S so
    transient upstream issues recover quickly instead of being cached
    as broken for the full TTL.
    """
    key = _normalize_url(url)
    content_hash = entry.get("_content_hash")
    if not content_hash or entry.get("status") != "ok":
        # No hash (no content) or rejected entry → plain write, no dedup.
        ttl = cache_module.FAILURE_TTL_S if entry.get("status") != "ok" else None
        await cache.set(key, entry, ttl=ttl)
        return

    canonical_key = await cache_module.content_alias.get(content_hash)
    if canonical_key and canonical_key != key:
        canonical = await cache.get(canonical_key)
        canonical_hash = (canonical or {}).get("_content_hash")
        if canonical and canonical_hash == content_hash:
            # Existing canonical confirmed — alias through it.
            await cache.set(key, {
                "_schema_version": 1,
                "alias": canonical_key,
                "content_hash": content_hash,
            })
            return
        # Canonical missing or drifted — fall through and reclaim the hash.

    await cache.set(key, entry)
    await cache_module.content_alias.set(content_hash, key)


async def _page_get(url: str, cache: KVCache) -> dict | None:
    """Read a page entry, dereferencing aliases and detecting dangling.

    An alias entry (`{alias, content_hash}`) is resolved against the
    canonical URL. If the canonical is missing or its content_hash
    drifted (stale alias), return None so the caller re-scrapes.
    """
    key = _normalize_url(url)
    entry = await cache.get(key)
    if not entry:
        return None
    if "alias" not in entry:
        return entry
    canonical_key = entry["alias"]
    expected_hash = entry.get("content_hash")
    canonical = await cache.get(canonical_key)
    if not canonical:
        return None
    canonical_hash = canonical.get("_content_hash")
    if canonical_hash != expected_hash:
        return None
    # Preserve the caller's requested URL in the returned view.
    return {**canonical, "url": url}


def _page_entry(
    *,
    url: str,
    content: str | None,
    title: str | None,
    metadata: dict,
    content_type: str = "text/html",
    file_type: str = "html",
    status: str | None = None,
) -> dict:
    """Construct a unified page-cache envelope.

    Callers supply what they know; everything else is defaulted. `status`
    defaults to 'ok' when content is non-empty and 'error' otherwise.
    """
    if status is None:
        status = "ok" if content else "error"
    envelope: dict = {
        "_schema_version": 1,
        "status": status,
        "url": url,
        "content_type": content_type,
        "file_type": file_type,
        "title": title,
        "content": content,
        "total_chars": len(content) if content else 0,
        "metadata": metadata,
    }
    # Internal-only field: content fingerprint used by _page_set /
    # _page_get for exact-dupe aliasing. Prefixed with '_' so it never
    # gets confused with caller-facing metadata fields. Stripped from
    # the response in tool implementations when building structured_content
    # payload.
    if content and status == "ok":
        envelope["_content_hash"] = _content_hash(content)
    return envelope
