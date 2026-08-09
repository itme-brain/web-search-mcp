from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import make_search_results, server_module
from web_search_mcp.tools.search import _persist_search_memory


PATCH_SEARCH = "web_search_mcp.tools.search._search"
PATCH_SCRAPE = "web_search_mcp.storage.pages._scrape"
PATCH_RERANK = "web_search_mcp.tools.search._rerank_scored"


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


@pytest.mark.parametrize(
    "domain_filters",
    [
        {"include_domains": ["docs.python.org"]},
        {"exclude_domains": ["danielputtick.com"]},
    ],
)
@pytest.mark.asyncio
async def test_semantic_memory_respects_domain_filters(domain_filters):
    live_url = "https://docs.python.org/3/library/asyncio.html"
    semantic_hit = {
        "url": "https://www.danielputtick.com/writing/asyncio-basics.html",
        "title": "Asyncio basics",
        "text": "External cached discussion of Python asyncio tasks.",
        "metadata": {},
    }

    with (
        patch(PATCH_SEARCH, AsyncMock(return_value=make_search_results([live_url]))),
        patch(
            PATCH_SCRAPE,
            AsyncMock(return_value={
                "title": "asyncio documentation",
                "content": "Official Python asyncio and TaskGroup documentation.",
                "metadata": {},
            }),
        ),
        patch(
            PATCH_RERANK,
            AsyncMock(side_effect=lambda _query, documents: [
                (index, 0.9) for index in range(len(documents))
            ]),
        ),
        patch(
            "web_search_mcp.tools.search.semantic.search",
            AsyncMock(return_value=[semantic_hit]),
        ),
    ):
        response = await server_module.search_impl(
            query="python asyncio taskgroup",
            num_results=3,
            **domain_filters,
        )

    assert response["meta"]["semantic_hits"] == 0
    assert response["results"]
    assert {
        result["domain"] for result in response["results"]
    } == {"docs.python.org"}
