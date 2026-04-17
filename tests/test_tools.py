from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Client

from tests.conftest import SCRAPE_CONTENT, URLS_A, make_search_results, server_app, server_module

PATCH_SEARCH = "web_search_server._search"
PATCH_SCRAPE = "web_search_server._scrape"
PATCH_RERANK = "web_search_server._rerank_scored"


def _identity_rerank(_query: str, documents: list[str]) -> list[tuple[int, float]]:
    return [(i, 0.0) for i in range(len(documents))]


def _make_scrape_mock():
    async def _fake_scrape(url: str) -> str | None:
        return SCRAPE_CONTENT.get(url)
    return AsyncMock(side_effect=_fake_scrape)


@pytest.mark.asyncio
async def test_fetch_page_returns_content():
    scrape_mock = _make_scrape_mock()

    with patch(PATCH_SCRAPE, scrape_mock):
        async with Client(server_app) as client:
            result = await client.call_tool("fetch_page", {"url": "https://example.com/a1"})
            text = result.content[0].text

    assert "# Content from https://example.com/a1" in text
    assert "Page A1" in text
    scrape_mock.assert_called_once_with("https://example.com/a1")


@pytest.mark.asyncio
async def test_fetch_page_returns_error_on_failure():
    scrape_mock = AsyncMock(return_value=None)

    with patch(PATCH_SCRAPE, scrape_mock):
        async with Client(server_app) as client:
            result = await client.call_tool("fetch_page", {"url": "https://example.com/broken"})
            text = result.content[0].text

    assert "Failed to fetch" in text


@pytest.mark.asyncio
async def test_fetch_page_uses_scrape_cache():
    scrape_mock = _make_scrape_mock()

    with patch(PATCH_SCRAPE, scrape_mock):
        async with Client(server_app) as client:
            await client.call_tool("fetch_page", {"url": "https://example.com/a1"})
            scrape_mock.reset_mock()

            result = await client.call_tool("fetch_page", {"url": "https://example.com/a1"})
            text = result.content[0].text

    assert "Page A1" in text
    scrape_mock.assert_not_called()


@pytest.mark.asyncio
async def test_site_search_prepends_site_prefix():
    search_mock = AsyncMock(return_value=make_search_results(URLS_A[:2]))
    scrape_mock = _make_scrape_mock()
    rerank_mock = MagicMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        async with Client(server_app) as client:
            result = await client.call_tool(
                "site_search",
                {"query": "python tutorial", "site": "docs.python.org", "num_results": 2, "scrape_top": 2},
            )
            text = result.content[0].text

    search_mock.assert_called_once()
    call_args = search_mock.call_args
    assert "site:docs.python.org" in call_args[1].get("query", call_args[0][0] if call_args[0] else "")
    assert "# Search results for:" in text
