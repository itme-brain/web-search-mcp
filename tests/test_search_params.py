from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Client

from tests.conftest import URLS_A, make_search_results, server_app, server_module

PATCH_SEARCH = "web_search_server._search"
PATCH_SCRAPE = "web_search_server._scrape"
PATCH_RERANK = "web_search_server._rerank_scored"


def _identity_rerank(_query: str, documents: list[str]) -> list[tuple[int, float]]:
    return [(i, 0.0) for i in range(len(documents))]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class TestNormalizeCategories:
    def test_valid_single(self):
        assert server_module._normalize_categories(["news"]) == ["news"]

    def test_valid_multiple_sorted(self):
        assert server_module._normalize_categories(["videos", "news", "general"]) == [
            "general",
            "news",
            "videos",
        ]

    def test_deduplicates(self):
        assert server_module._normalize_categories(["news", "news"]) == ["news"]

    def test_strips_whitespace(self):
        assert server_module._normalize_categories(["  news  "]) == ["news"]

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="invalid category"):
            server_module._normalize_categories(["bogus"])

    def test_none_returns_none(self):
        assert server_module._normalize_categories(None) is None

    def test_empty_returns_none(self):
        assert server_module._normalize_categories([]) is None


class TestNormalizeSafesearch:
    def test_valid_values(self):
        assert server_module._normalize_safesearch(0) == 0
        assert server_module._normalize_safesearch(1) == 1
        assert server_module._normalize_safesearch(2) == 2

    def test_none(self):
        assert server_module._normalize_safesearch(None) is None

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="invalid safesearch"):
            server_module._normalize_safesearch(3)


# ---------------------------------------------------------------------------
# _search() parameter threading
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_passes_categories_to_searxng():
    mock_client_cls = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"results": []}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("web_search_server.httpx.AsyncClient", return_value=mock_client):
        await server_module._search("test", categories=["news", "general"], language="en", safesearch=1, pageno=2)

    call_kwargs = mock_client.get.call_args
    params = call_kwargs[1]["params"] if "params" in call_kwargs[1] else call_kwargs[0][1]
    assert "news,general" in params.get("categories", "")
    assert params.get("language") == "en"
    assert params.get("safesearch") == 1
    assert params.get("pageno") == 2


# ---------------------------------------------------------------------------
# Deep mode fetches page 2
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deep_mode_fetches_page2():
    call_count = 0
    page1 = make_search_results(URLS_A[:2], prefix="P1")
    page2 = make_search_results(["https://example.com/p2"], prefix="P2")

    async def fake_search(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if kwargs.get("pageno", 1) == 2:
            return page2
        return page1

    search_mock = AsyncMock(side_effect=fake_search)
    scrape_mock = AsyncMock(return_value={"content": "content", "title": None, "screenshot": None})
    rerank_mock = MagicMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        result = await server_module._web_search_impl(
            query="test", num_results=3, mode="deep",
        )

    assert search_mock.call_count == 2
    second_call = search_mock.call_args_list[1]
    assert second_call[1].get("pageno") == 2


# ---------------------------------------------------------------------------
# Cache key differs for new params
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_key_includes_new_params():
    search_mock = AsyncMock(return_value=make_search_results(URLS_A[:1]))
    scrape_mock = AsyncMock(return_value={"content": "# Page\n\ncontent", "title": None, "screenshot": None})
    rerank_mock = MagicMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        r1 = await server_module._web_search_impl(query="test", num_results=1, scrape_top=1)
        assert search_mock.call_count == 1

        # same query, but with categories — should not hit cache
        r2 = await server_module._web_search_impl(
            query="test", num_results=1, scrape_top=1, categories=["news"],
        )
        assert search_mock.call_count == 2


# ---------------------------------------------------------------------------
# web_search tool threads categories
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_web_search_tool_passes_categories():
    search_mock = AsyncMock(return_value=make_search_results(URLS_A[:2]))
    rerank_mock = MagicMock(side_effect=_identity_rerank)
    scrape_mock = AsyncMock(return_value={"content": "# Page\n\ncontent", "title": None, "screenshot": None})

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        async with Client(server_app) as client:
            result = await client.call_tool(
                "web_search",
                {"query": "test", "num_results": 2, "scrape_top": 1, "categories": ["news"]},
            )
            payload = result.data

    call_kwargs = search_mock.call_args_list[0][1]
    assert call_kwargs.get("categories") == ["news"]


# ---------------------------------------------------------------------------
# image_search tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_search_returns_image_fields():
    fake_results = [
        {
            "title": "Northern Lights",
            "url": "https://example.com/source",
            "img_src": "https://cdn.example.com/image.jpg",
            "thumbnail_src": "https://cdn.example.com/thumb.jpg",
            "resolution": "1920x1080",
            "img_format": "jpeg",
        },
    ]
    search_mock = AsyncMock(return_value=fake_results)

    with patch(PATCH_SEARCH, search_mock):
        async with Client(server_app) as client:
            result = await client.call_tool(
                "image_search",
                {"query": "aurora borealis", "num_results": 1},
            )
            payload = result.data

    assert "Northern Lights" in payload
    assert "https://cdn.example.com/image.jpg" in payload
    assert "https://example.com/source" in payload
    assert "1920x1080" in payload

    call_kwargs = search_mock.call_args[1]
    assert call_kwargs.get("categories") == ["images"]
