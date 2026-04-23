from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import Client

from tests.conftest import URLS_A, make_search_results, server_app, server_module

PATCH_SEARCH = "core._search"
PATCH_EXTRACT_IMPL = "impls.extract_impl"
PATCH_RERANK = "core._rerank_scored"


def _identity_rerank(_query: str, documents: list[str]) -> list[tuple[int, float]]:
    return [(i, 0.5) for i in range(len(documents))]


@pytest.mark.asyncio
async def test_extract_urls_returns_structured_result_from_tool():
    extract_mock = AsyncMock(return_value={
        "query": "example query",
        "results": [
            {
                "url": "https://example.com/a1",
                "domain": "example.com",
                "status": "ok",
                "content_type": "text/html",
                "file_type": "html",
                "title": "Example",
                "content": "# Example\n\nPage content",
                "chars_shown": 22,
                "total_chars": 22,
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

    with patch(PATCH_EXTRACT_IMPL, extract_mock):
        async with Client(server_app) as client:
            result = await client.call_tool_mcp(
                "extract",
                {"urls": ["https://example.com/a1"], "query": "example query"},
            )
            payload = result.content[0].text
            structured = result.structuredContent

    assert "example query" in payload
    assert "https://example.com/a1" in payload
    assert "Example" in payload
    assert structured["results"][0]["url"] == "https://example.com/a1"


@pytest.mark.asyncio
async def test_search_validates_time_range():
    with pytest.raises(ValueError, match="invalid time_range"):
        await server_module.search_impl("test query", time_range="decade", ctx=None)


@pytest.mark.asyncio
async def test_search_validates_num_results():
    with pytest.raises(ValueError, match="num_results must be <= 10"):
        await server_module.search_impl("test query", num_results=50, ctx=None)


@pytest.mark.asyncio
async def test_search_validates_domain_filters():
    with pytest.raises(ValueError, match="bare domains"):
        await server_module.search_impl("test query", include_domains=["example.com/path"], ctx=None)


@pytest.mark.asyncio
async def test_extract_rejects_private_ip_urls():
    with pytest.raises(ValueError, match="private or reserved target"):
        await server_module.extract_impl(urls=["http://127.0.0.1/admin"], ctx=None)


@pytest.mark.asyncio
async def test_extract_rejects_hostnames_that_resolve_private():
    fake_addrinfo = [(2, 1, 6, "", ("10.0.0.8", 0))]
    with (
        patch("core.socket.getaddrinfo", return_value=fake_addrinfo),
        pytest.raises(ValueError, match="private or reserved target"),
    ):
        await server_module.extract_impl(urls=["http://internal.example.test/secret"], ctx=None)


@pytest.mark.asyncio
async def test_search_returns_structured_json():
    search_mock = AsyncMock(return_value=make_search_results(URLS_A[:2]))
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch("core._scrape", AsyncMock(return_value={"content": "# Page\n\nfull page body text with at least enough words to clear the speculative cache admission floor for tests.", "title": None, "screenshot": None})),
        patch(PATCH_RERANK, rerank_mock),
    ):
        payload = await server_module.search_impl("test query", num_results=2, ctx=None)

    assert payload["query"] == "test query"
    assert payload["meta"]["reranker"]["name"] == "flashrank"
    assert payload["meta"]["num_results_returned"] == 2
    assert payload["results"][0]["url"] == "https://example.com/a1"
    assert payload["results"][0]["scraped"] is True
