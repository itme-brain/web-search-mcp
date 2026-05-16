"""Map tool implementation."""

import logging
import time
import uuid

from web_search_mcp.common import (
    _domain_from_url,
    _normalize_glob_patterns,
    _normalize_url,
    _validate_positive_int,
    _validate_urls,
    _warning,
)
from web_search_mcp.crawling.operations import _deep_crawl, _extract_crawl_title
from web_search_mcp.presentation import models
from web_search_mcp import observability
from web_search_mcp.config.settings import _MAX_MAP_URLS

log = logging.getLogger("web-search-mcp")


def _validated_response(model_cls, response: dict) -> dict:
    return models.dump_response(model_cls, response)

async def map_impl(
    url: str,
    max_urls: int = 25,
    include_patterns: list[str] | None = None,
    observe: bool = True,
) -> dict:
    """Discover an in-scope site tree rooted at one URL.

    Discovery is link-only: Crawl4AI walks the site graph without this
    tool returning page bodies. The result is a bounded tree the caller
    can use as a planning surface before spending crawl budget on
    selected nodes.
    """
    root_url = _validate_urls([url], maximum=1)[0]
    request_id = uuid.uuid4().hex
    max_urls = _validate_positive_int("max_urls", max_urls, maximum=_MAX_MAP_URLS)
    include_patterns = _normalize_glob_patterns(include_patterns, field_name="include_patterns")

    started = time.monotonic()
    warnings: list[dict] = []
    pages_visited = 0
    try:
        discovered_pages = await _deep_crawl(
            [root_url],
            max_depth=2,
            max_pages=max_urls,
            same_domain_only=True,
            include_patterns=include_patterns,
        )
        pages_visited = len({
            _normalize_url(page.get("url", ""))
            for page in discovered_pages
            if isinstance(page, dict) and page.get("url")
        }) or 1
    except Exception as exc:
        warnings.append(_warning("link_discovery_failed", "crawl4ai", str(exc)))
        discovered_pages = []

    results: list[dict] = []
    visited: set[str] = set()

    root_normalized = _normalize_url(root_url)
    visited.add(root_normalized)
    root_entry = {
        "url": root_url,
        "domain": _domain_from_url(root_url),
        "title": None,
        "link_text": None,
        "depth": 0,
        "discovered_from": None,
        "link_type": "seed",
    }
    results.append(root_entry)

    for page in discovered_pages:
        if len(results) >= max_urls:
            break
        if not isinstance(page, dict):
            continue
        page_url = page.get("url")
        if not isinstance(page_url, str) or not page_url:
            continue
        normalized_url = _normalize_url(page_url)
        metadata = page.get("metadata") if isinstance(page.get("metadata"), dict) else {}
        if normalized_url == root_normalized:
            root_entry["title"] = _extract_crawl_title(page)
            continue
        if normalized_url in visited:
            continue
        visited.add(normalized_url)
        depth = metadata.get("depth")
        if not isinstance(depth, int) or depth < 1:
            depth = 1
        parent_url = metadata.get("parent_url")
        if not isinstance(parent_url, str) or not parent_url:
            parent_url = root_url
        entry = {
            "url": page_url,
            "domain": _domain_from_url(page_url),
            "title": _extract_crawl_title(page),
            "link_text": None,
            "depth": depth,
            "discovered_from": parent_url,
            "link_type": "internal",
        }
        results.append(entry)

    for rank, entry in enumerate(results, start=1):
        entry["rank"] = rank

    response = {
        "url": root_url,
        "results": results,
        "meta": {
            "request_id": request_id,
            "max_urls_requested": max_urls,
            "urls_returned": len(results),
            "pages_visited": pages_visited,
            "warnings": warnings,
            "timings_ms": {
                "total": int((time.monotonic() - started) * 1000),
            },
        },
    }
    if observe:
        observability.observe_tool_response("map", response)
    log.info(
        "request_id=%s map url=%s returned=%d warnings=%d",
        request_id, root_url, len(results), len(warnings),
    )
    return _validated_response(models.MapResponseModel, response)
