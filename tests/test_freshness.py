import time
from unittest.mock import AsyncMock, patch

import pytest

from web_search_mcp.storage import cache as cache_module
from web_search_mcp.storage.pages import _scrape_cached
from web_search_mcp.tools.search import search_impl


def _rank_in_order(_query, documents):
    return [(idx, 1.0 - idx / 100) for idx in range(len(documents))]


@pytest.mark.asyncio
async def test_time_bounded_search_excludes_unverified_semantic_memory():
    search_response = {
        "results": [{"title": "Fresh", "url": "https://example.com/fresh", "content": "fresh snippet"}],
        "unresponsive_engines": [],
    }
    scrape = {
        "title": "Fresh",
        "content": "Fresh page content with enough distinct words to satisfy cache admission and provide useful evidence.",
        "metadata": {"date": "2026-08-08"},
    }
    semantic_search = AsyncMock(return_value=[{
        "url": "https://example.com/stale", "title": "Stale", "text": "stale evidence",
    }])
    with (
        patch("web_search_mcp.tools.search._search", AsyncMock(return_value=search_response)),
        patch("web_search_mcp.storage.pages._scrape", AsyncMock(return_value=scrape)),
        patch("web_search_mcp.tools.search._rerank_scored", AsyncMock(side_effect=_rank_in_order)),
        patch("web_search_mcp.tools.search.semantic.search", semantic_search),
    ):
        result = await search_impl("latest release", num_results=1, time_range="day")

    semantic_search.assert_not_awaited()
    assert result["meta"]["semantic_hits"] == 0
    assert all(item["retrieval_source"] != "semantic_memory" for item in result["results"])


@pytest.mark.asyncio
async def test_freshness_policy_rejects_untimestamped_page_cache_entry():
    url = "https://example.com/legacy"
    await cache_module.page_cache.set(url, {
        "_schema_version": 1, "status": "ok", "url": url,
        "content": "legacy cached content", "title": "Legacy", "metadata": {},
    })
    scrape = AsyncMock(return_value={
        "title": "Current",
        "content": "Current page content has enough distinct words to pass speculative cache admission without a diagnostic.",
        "metadata": {},
    })
    with patch("web_search_mcp.storage.pages._scrape", scrape):
        result = await _scrape_cached(url, cache_module.page_cache, max_age_seconds=300)

    scrape.assert_awaited_once()
    assert result["title"] == "Current"


@pytest.mark.asyncio
async def test_freshness_policy_reuses_recent_page_snapshot():
    url = "https://example.com/recent"
    await cache_module.page_cache.set(url, {
        "_schema_version": 1, "status": "ok", "url": url,
        "content": "recent cached content", "title": "Recent", "metadata": {},
        "retrieved_at": int(time.time()),
    })
    scrape = AsyncMock()
    with patch("web_search_mcp.storage.pages._scrape", scrape):
        result = await _scrape_cached(url, cache_module.page_cache, max_age_seconds=300)

    scrape.assert_not_awaited()
    assert result["title"] == "Recent"
