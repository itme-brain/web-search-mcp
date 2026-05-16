"""Crawl4AI result parsing, scrape, map, and deep-crawl orchestration."""

import asyncio
import logging
from urllib.parse import urljoin, urlparse

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from web_search_mcp.common import _domain_from_url, _normalize_url, _registrable_domain
from web_search_mcp.config.settings import CRAWL4AI_URL, REQUEST_TIMEOUT, _HTTP_TIMEOUT
from web_search_mcp.crawling import client as crawl
from web_search_mcp.extraction.html import _build_document_metadata, _extract_markdown

log = logging.getLogger("web-search-mcp")
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


def _domain_filter_patterns(root_url: str, same_domain_only: bool) -> list[str]:
    if not same_domain_only:
        return []
    root_domain = _domain_from_url(root_url).lower()
    registrable = _registrable_domain(root_domain)
    schemes = ("http", "https")
    patterns: list[str] = []
    for scheme in schemes:
        patterns.append(f"{scheme}://{registrable}/*")
        patterns.append(f"{scheme}://*.{registrable}/*")
        patterns.append(f"{scheme}://{root_domain}/*")
    return patterns


def _crawl_filter_chain(
    *,
    root_url: str,
    same_domain_only: bool,
    include_patterns: list[str] | None,
) -> list[dict]:
    filters: list[dict] = []
    domain_patterns = _domain_filter_patterns(root_url, same_domain_only)
    if domain_patterns:
        filters.append({
            "type": "URLPatternFilter",
            "params": {"patterns": domain_patterns},
        })
    if include_patterns:
        filters.append({
            "type": "URLPatternFilter",
            "params": {"patterns": include_patterns},
        })
    filters.append({
        "type": "ContentTypeFilter",
        "params": {"allowed_types": ["text/html"]},
    })
    # NOTE: We intentionally do NOT add a ContentRelevanceFilter here.
    # Filtering by query relevance during BFS exploration prematurely
    # rejects pages that link to relevant content but don't themselves
    # contain the query terms.  Relevance filtering happens post-crawl
    # via the configured chunk reranker in crawl_impl instead.
    return filters


def _deep_crawl_config(
    *,
    root_url: str,
    max_depth: int,
    max_pages: int,
    same_domain_only: bool,
    include_patterns: list[str] | None = None,
) -> dict:
    # Discovery should keep the full page chrome where site topology
    # often lives. Starting from the content-pruned default config can
    # collapse map/crawl to the root page on real sites.
    return crawl.deep_crawl_config(
        root_url=root_url,
        max_depth=max_depth,
        max_pages=max_pages,
        same_domain_only=same_domain_only,
        include_patterns=include_patterns,
    )


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
        crawler_config=crawler_config,
    )


async def _scrape_impl(url: str) -> dict:
    """Scrape a URL via Crawl4AI. Returns {content, title, metadata}."""
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        data = await _crawl_post(client, [url], priority=8)

        result = _extract_crawl_result(data)
        content = _extract_markdown(result)
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
    empty = {"content": None, "title": None, "metadata": {}}
    try:
        return await asyncio.wait_for(_scrape_impl(url), timeout=REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        log.warning("scrape timed out url=%s budget=%ss", url, REQUEST_TIMEOUT)
    except httpx.HTTPError as e:
        log.warning("scrape http error url=%s err=%s", url, e)
    except (ValueError, KeyError) as e:
        log.warning("scrape payload error url=%s err=%s", url, e)
    return empty
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
    """BFS deep crawl starting from every seed URL.

    The BFS strategy's same_domain and include_pattern filters are keyed
    off the first seed (as the "root") for domain-scope purposes, since
    all seeds should already be in-scope at the call site.
    """
    if not seeds:
        return []
    crawler_config = _deep_crawl_config(
        root_url=seeds[0],
        max_depth=max_depth,
        max_pages=max_pages,
        same_domain_only=same_domain_only,
        include_patterns=include_patterns,
    )
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        data = await _crawl_post(client, seeds, priority=7, crawler_config=crawler_config)
        return _extract_crawl_results(data)
