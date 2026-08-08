from unittest.mock import AsyncMock, patch

import pytest

from web_search_mcp.tools.search import _persist_search_memory


@pytest.mark.asyncio
async def test_search_memory_indexes_full_document_after_compact_response():
    full_content = "Complete page content that is longer than the returned passage."
    results = [{
        "url": "https://example.com/page",
        "title": "Page",
        "domain": "example.com",
        "source_type": "web",
        "scraped": True,
        "passages": [{"citation": "1.1", "text": "returned passage"}],
    }]
    entries = [{
        "url": "https://example.com/page",
        "title": "Page",
        "content": "truncated content",
        "full_content": full_content,
        "metadata": {"date": "2026-08-08"},
        "scraped": True,
    }]
    seen_set = AsyncMock()
    memory_set = AsyncMock()
    index_page = AsyncMock()

    with (
        patch("web_search_mcp.tools.search.cache_module.seen_urls.set", seen_set),
        patch("web_search_mcp.tools.search.cache_module.page_memory_cache.set", memory_set),
        patch("web_search_mcp.tools.search.semantic.index_page", index_page),
    ):
        await _persist_search_memory(
            results, ["https://example.com/page"], entries
        )

    assert memory_set.await_args.args[1]["content"] == full_content
    assert index_page.await_args.args[2] == full_content
