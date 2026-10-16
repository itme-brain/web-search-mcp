"""Crawl4AI result parsing, scrape, map, and deep-crawl orchestration."""

import asyncio
from collections import OrderedDict
from email.utils import parsedate_to_datetime
from fnmatch import fnmatchcase
import logging
import math
import re
import time
from urllib.parse import urljoin, urlparse
from weakref import WeakValueDictionary

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from web_search_mcp.common import _canonical_hostname, _domain_from_url, _normalize_url
from web_search_mcp.config.settings import CRAWL4AI_API_TOKEN, CRAWL4AI_URL, REQUEST_TIMEOUT, _HTTP_TIMEOUT
from web_search_mcp.crawling import client as crawl
from web_search_mcp.extraction.html import _build_document_metadata, _extract_markdown
from web_search_mcp.extraction.blocks import block_diagnostic

log = logging.getLogger("web-search-mcp")
_host_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
_rate_limits: OrderedDict[str, float] = OrderedDict()


def _retry_after_seconds(headers: dict) -> float:
    value = next((v for k, v in headers.items() if k.lower() == "retry-after"), None)
    if value is None:
        return 60
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return 60
    return max(1, seconds) if math.isfinite(seconds) else 60


def _crawl_failure_metadata(result: dict) -> dict | None:
    status = result.get("status_code")
    error = str(result.get("error_message") or result.get("error") or "")
    # Crawl4AI sometimes supplies the HTTP status only in its error message.
    if not isinstance(status, int) or status < 400:
        match = re.search(r"\bHTTP\s+([45]\d\d)\b", error, re.IGNORECASE)
        status = int(match[1]) if match else None
    if status is None and result.get("success") is not False:
        return None
    diagnostic = {429: "rate_limited", 403: "access_denied"}.get(status, "fetch_failed")
    metadata = {"diagnostic": diagnostic, "error": error[:500]}
    if status is not None:
        metadata["http_status"] = status
    if status == 429:
        metadata["retry_after_seconds"] = _retry_after_seconds(result.get("response_headers") or {})
    return metadata


def _extract_crawl_result(data: dict) -> dict:
    results = data.get("results")
    if results and isinstance(results, list):
        return results[0]
    return data.get("result", data)


def _extract_crawl_results(data: dict) -> list[dict]:
    results = data.get("results")
    if isinstance(results, list):
        return [result for result in results if isinstance(result, dict)]
    result = data.get("result", data)
    return [result] if isinstance(result, dict) else []


def _extract_crawl_title(result: dict) -> str | None:
    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        title = metadata.get("title")
        if title:
            return title
    title = result.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return None


# Visible link text used for accessibility shortcuts — useless as a map snippet
# since it describes the *mechanism* of the link, not its destination.
_A11Y_LINK_TEXTS = frozenset({
    "skip to main content",
    "skip to content",
    "skip to main",
    "skip navigation",
    "skip to navigation",
    "jump to content",
    "jump to main content",
    "jump to navigation",
    "main content",
})


def _clean_link_label(value: str | None) -> str | None:
    if not value:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.lower() in _A11Y_LINK_TEXTS:
        return None
    return stripped


def _extract_crawl_links(result: dict, base_url: str) -> list[dict]:
    link_groups = result.get("links")
    if not isinstance(link_groups, dict):
        return []

    links: list[dict] = []
    for link_type in ("internal", "external"):
        bucket = link_groups.get(link_type) or []
        if not isinstance(bucket, list):
            continue
        for link in bucket:
            if not isinstance(link, dict):
                continue
            href = link.get("href")
            if not isinstance(href, str) or not href.strip():
                continue
            absolute_url = urljoin(base_url, href.strip())
            parsed = urlparse(absolute_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                continue
            links.append({
                "url": absolute_url,
                "title": _clean_link_label(link.get("title")),
                "text": _clean_link_label(link.get("text")),
                "link_type": link_type,
            })
    return links


# ---------------------------------------------------------------------------
# Crawl4AI configs + HTTP layer (retry, poll, scrape, discover)
# ---------------------------------------------------------------------------
# `remove_overlay_elements` is deliberately OFF. Crawl4AI's overlay
# heuristic mis-classifies Wikipedia's main article body as an overlay
# and deletes it, leaving only <head> — trafilatura then returns 0
# chars and the search path silently falls back to SearXNG snippets.
# The heuristic is a net loss: trafilatura already drops boilerplate
# on its own, and Crawl4AI's own content filter picks up what's left.
_DEFAULT_CRAWL_CONFIG = {
    "type": "CrawlerRunConfig",
    "params": {
        "excluded_tags": ["nav", "footer", "header", "aside"],
        "markdown_generator": {
            "type": "DefaultMarkdownGenerator",
            "params": {
                "content_filter": {
                    "type": "PruningContentFilter",
                    "params": {
                        "threshold": 0.48,
                        "threshold_type": "fixed",
                        "min_word_threshold": 0,
                    },
                },
            },
        },
    },
}

# Separate config for link discovery (map/crawl). The default config
# strips nav/footer/header/aside to get clean content — but that's
# exactly where a docs site's link graph lives. Using the default here
# was making `map` return ~0 URLs on real sites. Keep the full page
# when we're only interested in hrefs.
_MAP_CRAWL_CONFIG = {
    "type": "CrawlerRunConfig",
    "params": {},
}


def _crawl_url_in_scope(
    url: str,
    *,
    root_url: str,
    same_domain_only: bool,
    include_patterns: list[str] | None,
) -> bool:
    if same_domain_only:
        root = urlparse(root_url)
        candidate = urlparse(url)
        if _canonical_hostname(candidate.hostname or "") != _canonical_hostname(root.hostname or ""):
            return False
        root_parts = [part for part in root.path.split("/") if part]
        candidate_parts = [part for part in candidate.path.split("/") if part]
        if root_parts:
            if _canonical_hostname(root.hostname or "") == "docs.rs" and len(root_parts) >= 3:
                # docs.rs canonicalizes `latest` to a concrete version. Keep
                # the crate and rustdoc subtree fixed while allowing that
                # version segment to change.
                if (
                    len(candidate_parts) < len(root_parts)
                    or candidate_parts[0] != root_parts[0]
                    or candidate_parts[2 : len(root_parts)] != root_parts[2:]
                ):
                    return False
            elif candidate_parts[: len(root_parts)] != root_parts:
                return False
    return not include_patterns or any(fnmatchcase(url, pattern) for pattern in include_patterns)


def _is_retryable_crawl_error(exc: BaseException) -> bool:
    """Retry only on Crawl4AI transient failures: network issues and 5xx.

    Crawl4AI 0.8.x has documented browser-pool flakiness (memory leaks,
    'target page context closed' after N requests). A single transient 5xx or
    reset connection shouldn't kill the scrape.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, (httpx.TransportError, httpx.TimeoutException))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(_is_retryable_crawl_error),
    reraise=True,
)
async def _crawl_post(
    client: httpx.AsyncClient,
    urls: list[str],
    priority: int,
    crawler_config: dict | None = None,
) -> dict:
    """POST to Crawl4AI's stream endpoint and collect NDJSON results.

    `urls` is the seed list. With a BFS deep_crawl_strategy in the
    config, Crawl4AI expands each seed independently.
    """
    return await crawl.crawl_post(
        client,
        crawl4ai_url=CRAWL4AI_URL,
        urls=urls,
        priority=priority,
        api_token=CRAWL4AI_API_TOKEN,
        crawler_config=crawler_config,
    )


async def _scrape_impl(url: str) -> dict:
    """Scrape a URL via Crawl4AI. Returns {content, title, metadata}."""
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        data = await _crawl_post(client, [url], priority=8)

        result = _extract_crawl_result(data)
        failure = _crawl_failure_metadata(result)
        if failure:
            return {"content": None, "title": _extract_crawl_title(result), "metadata": failure}
        content = _extract_markdown(result)
        diagnostic = block_diagnostic(content, title=_extract_crawl_title(result))
        if diagnostic:
            return {"content": None, "title": _extract_crawl_title(result), "metadata": {
                "diagnostic": diagnostic, "error": "Origin returned an access challenge instead of the requested document.",
            }}
        final_url = result.get("url") if isinstance(result.get("url"), str) else url
        return {
            "content": content,
            "title": _extract_crawl_title(result),
            "metadata": _build_document_metadata(
                result.get("html"), content, requested_url=url, final_url=final_url,
            ),
        }


async def _scrape(url: str) -> dict:
    """Scrape a URL via Crawl4AI, bounded by REQUEST_TIMEOUT seconds end-to-end.

    Returns {content, title, metadata}. On failure content is None.
    """
    host = urlparse(url).hostname or url
    lock = _host_locks.setdefault(host, asyncio.Lock())

    async def guarded_scrape() -> dict:
        # Serialize requests to one host across search/extract calls; other
        # hosts remain parallel. Do not hammer an origin after a 429.
        async with lock:
            remaining = _rate_limits.get(host, 0) - time.monotonic()
            if remaining > 0:
                return {"content": None, "title": None, "metadata": {
                    "diagnostic": "rate_limited", "http_status": 429,
                    "retry_after_seconds": math.ceil(remaining),
                }}
            _rate_limits.pop(host, None)
            result = await _scrape_impl(url)
            metadata = result.get("metadata") or {}
            if metadata.get("http_status") == 429:
                _rate_limits[host] = time.monotonic() + metadata.get("retry_after_seconds", 60)
                _rate_limits.move_to_end(host)
                if len(_rate_limits) > 1024:
                    _rate_limits.popitem(last=False)
            return result

    diagnostic = "fetch_failed"
    try:
        return await asyncio.wait_for(guarded_scrape(), timeout=REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        diagnostic = "fetch_timeout"
        log.warning("scrape timed out url=%s budget=%ss", url, REQUEST_TIMEOUT)
    except httpx.HTTPError as e:
        log.warning("scrape http error url=%s err=%s", url, e)
    except (ValueError, KeyError) as e:
        log.warning("scrape payload error url=%s err=%s", url, e)
    return {"content": None, "title": None, "metadata": {"diagnostic": diagnostic}}
async def _discover_page_links(url: str) -> dict:
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        data = await _crawl_post(client, [url], priority=6, crawler_config=_MAP_CRAWL_CONFIG)

        result = _extract_crawl_result(data)
        return {
            "status": "ok",
            "url": url,
            "title": _extract_crawl_title(result),
            "links": _extract_crawl_links(result, url),
        }


async def _deep_crawl(
    seeds: list[str],
    *,
    max_depth: int,
    max_pages: int,
    same_domain_only: bool,
    include_patterns: list[str] | None = None,
) -> list[dict]:
    """Perform bounded BFS in the MCP process using Crawl4AI page batches.

    Crawl4AI 0.9 rejects request-supplied ``deep_crawl_strategy`` objects at
    its HTTP trust boundary. Keeping traversal here preserves map/crawl
    behavior without weakening the crawler container's hardened API.
    """
    if not seeds:
        return []

    root_url = seeds[0]
    queued: set[str] = set()
    frontier: list[tuple[str, str | None]] = []
    for seed in seeds:
        normalized = _normalize_url(seed)
        if normalized not in queued:
            queued.add(normalized)
            frontier.append((seed, None))

    crawled_pages: list[dict] = []
    crawled_urls: set[str] = set()
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        for depth in range(max_depth + 1):
            if not frontier or len(crawled_pages) >= max_pages:
                break
            batch = frontier[: max_pages - len(crawled_pages)]
            requested_urls = [url for url, _parent in batch]
            parent_by_url = {_normalize_url(url): parent for url, parent in batch}
            data = await _crawl_post(
                client,
                requested_urls,
                priority=7,
                crawler_config=_MAP_CRAWL_CONFIG,
            )
            pages = _extract_crawl_results(data)
            next_frontier: list[tuple[str, str]] = []
            for index, page in enumerate(pages):
                page = dict(page)
                fallback_url = requested_urls[index] if index < len(requested_urls) else ""
                page_url = page.get("url") if isinstance(page.get("url"), str) else fallback_url
                if not page_url:
                    continue
                normalized_page_url = _normalize_url(page_url)
                if normalized_page_url in crawled_urls:
                    continue
                crawled_urls.add(normalized_page_url)
                queued.add(normalized_page_url)
                metadata = dict(page.get("metadata") or {})
                metadata["depth"] = depth
                parent = parent_by_url.get(_normalize_url(page_url))
                if parent:
                    metadata["parent_url"] = parent
                page["metadata"] = metadata
                crawled_pages.append(page)
                if depth >= max_depth or len(crawled_pages) + len(next_frontier) >= max_pages:
                    continue
                for link in _extract_crawl_links(page, page_url):
                    candidate = link["url"]
                    normalized = _normalize_url(candidate)
                    if normalized in queued or not _crawl_url_in_scope(
                        candidate,
                        root_url=root_url,
                        same_domain_only=same_domain_only,
                        include_patterns=include_patterns,
                    ):
                        continue
                    queued.add(normalized)
                    next_frontier.append((candidate, page_url))
                    if len(crawled_pages) + len(next_frontier) >= max_pages:
                        break
            frontier = next_frontier
    return crawled_pages
