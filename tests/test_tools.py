from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Client

from tests.conftest import URLS_A, make_search_results, server_app, server_module

PATCH_SEARCH = "web_search_server._search"
PATCH_EXTRACT_URLS_IMPL = "web_search_server._extract_urls_impl"
PATCH_RERANK = "web_search_server._rerank_scored"


def _identity_rerank(_query: str, documents: list[str]) -> list[tuple[int, float]]:
    return [(i, 0.0) for i in range(len(documents))]


@pytest.mark.asyncio
async def test_extract_urls_returns_structured_result_from_tool():
    extract_mock = AsyncMock(return_value={
        "query": "example query",
        "results": [
            {
                "url": "https://example.com/a1",
                "normalized_url": "https://example.com/a1",
                "domain": "example.com",
                "status": "ok",
                "content_type": "text/html",
                "file_type": "html",
                "title": "Example",
                "content": "# Example\n\nPage content",
                "top_chunks": [],
                "cached": False,
                "error": None,
            },
        ],
        "meta": {
            "urls_requested": 1,
            "urls_succeeded": 1,
            "urls_failed": 0,
            "timings_ms": {"total": 5},
        },
    })

    with patch(PATCH_EXTRACT_URLS_IMPL, extract_mock):
        async with Client(server_app) as client:
            result = await client.call_tool(
                "extract",
                {"urls": ["https://example.com/a1"], "query": "example query"},
            )
            payload = result.data

    assert "example query" in payload
    assert "https://example.com/a1" in payload
    assert "Example" in payload


@pytest.mark.asyncio
async def test_search_validates_time_range():
    with pytest.raises(ValueError, match="invalid time_range"):
        await server_module._web_search_impl("test query", time_range="decade", ctx=None)


@pytest.mark.asyncio
async def test_search_validates_num_results():
    with pytest.raises(ValueError, match="num_results must be <= 10"):
        await server_module._web_search_impl("test query", num_results=50, ctx=None)


@pytest.mark.asyncio
async def test_search_validates_mode():
    with pytest.raises(ValueError, match="invalid mode"):
        await server_module._web_search_impl("test query", mode="fast", ctx=None)


@pytest.mark.asyncio
async def test_search_validates_domain_filters():
    with pytest.raises(ValueError, match="bare domains"):
        await server_module._web_search_impl("test query", include_domains=["example.com/path"], ctx=None)


@pytest.mark.asyncio
async def test_search_returns_structured_json():
    search_mock = AsyncMock(return_value=make_search_results(URLS_A[:2]))
    rerank_mock = MagicMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch("web_search_server._scrape", AsyncMock(return_value={"content": "# Page\n\ncontent", "title": None, "screenshot": None})),
        patch(PATCH_RERANK, rerank_mock),
    ):
        payload = await server_module._web_search_impl("test query", num_results=2, scrape_top=2, ctx=None)

    assert payload["query"] == "test query"
    assert payload["meta"]["reranker"]["name"] == "flashrank"
    assert payload["meta"]["num_results_returned"] == 2
    assert payload["results"][0]["url"] == "https://example.com/a1"
    assert payload["results"][0]["normalized_url"] == "https://example.com/a1"
    assert payload["results"][0]["scraped"] is True
    assert payload["mode"] == "balanced"
