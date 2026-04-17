from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import Client

from tests.conftest import SCRAPE_CONTENT, URLS_A, URLS_B, make_search_results, server_app, server_module

PATCH_SEARCH = "web_search_server._search"
PATCH_SCRAPE = "web_search_server._scrape"
PATCH_RERANK = "web_search_server._rerank"

URLS_C = [
    "https://example.com/c1",
    "https://example.com/c2",
    "https://example.com/c3",
]

SCRAPE_CONTENT_EXTENDED = {
    **SCRAPE_CONTENT,
    "https://example.com/c1": "# Page C1\nContent of page C1",
    "https://example.com/c2": "# Page C2\nContent of page C2",
    "https://example.com/c3": "# Page C3\nContent of page C3",
    "https://example.com/new1": "# Page New1\nContent of new1",
}


def _identity_rerank(_query: str, documents: list[str]) -> list[int]:
    return list(range(len(documents)))


def _make_scrape_mock(content_map: dict[str, str | None] | None = None):
    mapping = content_map or SCRAPE_CONTENT_EXTENDED

    async def _fake_scrape(url: str) -> str | None:
        return mapping.get(url)

    mock = AsyncMock(side_effect=_fake_scrape)
    return mock


@pytest.fixture
def patched_backends():
    search_mock = AsyncMock(return_value=make_search_results(URLS_A))
    scrape_mock = _make_scrape_mock()
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        yield {
            "search": search_mock,
            "scrape": scrape_mock,
            "rerank": rerank_mock,
        }


@pytest.mark.asyncio
async def test_query_cache_hit_returns_same_output_without_calling_backends(patched_backends):
    async with Client(server_app) as client:
        result1 = await client.call_tool("web_search", {"query": "test query", "num_results": 3, "scrape_top": 3})
        text1 = result1.content[0].text

        patched_backends["search"].reset_mock()
        patched_backends["scrape"].reset_mock()
        patched_backends["rerank"].reset_mock()

        result2 = await client.call_tool("web_search", {"query": "test query", "num_results": 3, "scrape_top": 3})
        text2 = result2.content[0].text

    assert text1 == text2
    patched_backends["search"].assert_not_called()
    patched_backends["scrape"].assert_not_called()
    patched_backends["rerank"].assert_not_called()


@pytest.mark.asyncio
async def test_query_cache_key_normalizes_whitespace_and_case(patched_backends):
    async with Client(server_app) as client:
        result1 = await client.call_tool("web_search", {"query": "Test Query ", "num_results": 3, "scrape_top": 3})
        text1 = result1.content[0].text

        patched_backends["search"].reset_mock()
        patched_backends["scrape"].reset_mock()
        patched_backends["rerank"].reset_mock()

        result2 = await client.call_tool("web_search", {"query": "  test query", "num_results": 3, "scrape_top": 3})
        text2 = result2.content[0].text

    assert text1 == text2
    patched_backends["search"].assert_not_called()


@pytest.mark.asyncio
async def test_query_cache_miss_on_different_query_triggers_fresh_pipeline():
    search_call_count = 0

    async def _search_side_effect(query, **kwargs):
        nonlocal search_call_count
        search_call_count += 1
        if search_call_count == 1:
            return make_search_results(URLS_A)
        return make_search_results(URLS_C)

    search_mock = AsyncMock(side_effect=_search_side_effect)
    scrape_mock = _make_scrape_mock()
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        async with Client(server_app) as client:
            await client.call_tool("web_search", {"query": "test query", "num_results": 3, "scrape_top": 3})

            search_mock.reset_mock()
            scrape_mock.reset_mock()
            rerank_mock.reset_mock()

            await client.call_tool("web_search", {"query": "different query", "num_results": 3, "scrape_top": 3})

    search_mock.assert_called_once()
    assert scrape_mock.call_count == 3
    rerank_mock.assert_called_once()


@pytest.mark.asyncio
async def test_scrape_cache_reuse_skips_already_scraped_urls():
    search_call_count = 0

    async def _search_side_effect(query, **kwargs):
        nonlocal search_call_count
        search_call_count += 1
        if search_call_count == 1:
            return make_search_results(URLS_A)
        return make_search_results(URLS_B)

    search_mock = AsyncMock(side_effect=_search_side_effect)
    scrape_mock = _make_scrape_mock()
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        async with Client(server_app) as client:
            await client.call_tool("web_search", {"query": "query alpha", "num_results": 3, "scrape_top": 3})
            assert scrape_mock.call_count == 3

            scrape_mock.reset_mock()

            await client.call_tool("web_search", {"query": "query beta", "num_results": 3, "scrape_top": 3})

    scrape_urls_second = [call.args[0] for call in scrape_mock.call_args_list]
    assert "https://example.com/a2" not in scrape_urls_second
    assert "https://example.com/b1" in scrape_urls_second
    assert "https://example.com/b2" in scrape_urls_second


@pytest.mark.asyncio
async def test_previously_seen_urls_annotated_in_subsequent_results():
    search_call_count = 0

    async def _search_side_effect(query, **kwargs):
        nonlocal search_call_count
        search_call_count += 1
        if search_call_count == 1:
            return make_search_results(URLS_A)
        return make_search_results(URLS_B)

    search_mock = AsyncMock(side_effect=_search_side_effect)
    scrape_mock = _make_scrape_mock()
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        async with Client(server_app) as client:
            result1 = await client.call_tool(
                "web_search", {"query": "first search", "num_results": 3, "scrape_top": 3}
            )
            text1 = result1.content[0].text
            assert "*(previously seen)*" not in text1

            result2 = await client.call_tool(
                "web_search", {"query": "second search", "num_results": 3, "scrape_top": 3}
            )
            text2 = result2.content[0].text

    assert "*(previously seen)*" in text2
    for url in ["https://example.com/b1", "https://example.com/b2"]:
        lines_with_url = [line for line in text2.split("\n") if url in line]
        for line in lines_with_url:
            assert "*(previously seen)*" not in line


@pytest.mark.asyncio
async def test_none_scrape_result_cached_so_broken_url_not_retried():
    content_with_failure = dict(SCRAPE_CONTENT_EXTENDED)
    content_with_failure["https://example.com/a2"] = None

    search_mock = AsyncMock(return_value=make_search_results(URLS_A))
    scrape_mock = _make_scrape_mock(content_with_failure)
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        async with Client(server_app) as client:
            await client.call_tool("web_search", {"query": "fail test", "num_results": 3, "scrape_top": 3})
            assert scrape_mock.call_count == 3

            scrape_mock.reset_mock()

            search_mock.return_value = make_search_results(
                ["https://example.com/a2", "https://example.com/new1"], prefix="Retry"
            )

            await client.call_tool("web_search", {"query": "retry test", "num_results": 2, "scrape_top": 2})

    scrape_urls = [call.args[0] for call in scrape_mock.call_args_list]
    assert "https://example.com/a2" not in scrape_urls
    assert "https://example.com/new1" in scrape_urls


@pytest.mark.asyncio
async def test_tool_works_without_session_context():
    search_mock = AsyncMock(return_value=make_search_results(URLS_A[:2]))
    scrape_mock = _make_scrape_mock()
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        result = await server_module.web_search("direct call", num_results=2, scrape_top=2, ctx=None)

    assert "# Search results for: direct call" in result
    assert "example.com/a1" in result
    search_mock.assert_called_once()
