"""Pin the markdown/dict contract: metadata fields stay in the Python dict
for scripts/eval, but never leak into the LLM-facing markdown output."""

from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import URLS_A, make_search_results, server_module

PATCH_SEARCH = "core._search"
PATCH_SCRAPE = "core._scrape"
PATCH_RERANK = "core._rerank_scored"
PATCH_EXTRACT_URL_DOCUMENT = "core._extract_url_document"
PATCH_MAP_SITE_IMPL = "impls.map_impl"
PATCH_EXTRACT_URLS_IMPL = "impls.extract_impl"

# These fields may live in the Python dict for scripting access, but must
# never appear as literal keys in the LLM-facing markdown output.
_LEAKY_FIELDS = [
    "normalized_url",
    "search_rank",
    "previously_seen",
    "cached",
    "top_chunks",
    "score",
]


def _identity_rerank(_query: str, documents: list[str]) -> list[tuple[int, float]]:
    return [(i, 1.0 - i * 0.1) for i in range(len(documents))]


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_dict_carries_full_structured_fields():
    search_mock = AsyncMock(return_value=make_search_results(URLS_A[:2]))
    scrape_mock = AsyncMock(return_value={"content": "# Page\n\ncontent", "title": None})
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        payload = await server_module.search_impl("test", num_results=2, ctx=None)

    first = payload["results"][0]
    # search dict carries these for scripting access
    for field in ["normalized_url", "search_rank", "previously_seen", "top_chunks", "score"]:
        assert field in first, f"dict should carry {field} for scripting access"


@pytest.mark.asyncio
async def test_search_markdown_does_not_leak_metadata_fields():
    search_mock = AsyncMock(return_value=make_search_results(URLS_A[:2]))
    scrape_mock = AsyncMock(return_value={"content": "# Page\n\ncontent", "title": None})
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        markdown = await server_module.search.fn("test", num_results=2, ctx=None)

    for field in _LEAKY_FIELDS:
        assert f"{field}:" not in markdown, (
            f"{field!r} key leaked into LLM-facing markdown:\n{markdown}"
        )


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_dict_carries_full_structured_fields():
    extract_mock = AsyncMock(return_value={
        "status": "ok",
        "url": "https://example.com/a",
        "content_type": "text/html",
        "file_type": "html",
        "title": "A",
        "content": "# A\n\ncontent",
        "top_chunks": [{"text": "content", "score": 0.5}],
        "cached": False,
    })

    with patch(PATCH_EXTRACT_URL_DOCUMENT, extract_mock):
        payload = await server_module.extract_impl(
            urls=["https://example.com/a"], query="q",
        )

    result = payload["results"][0]
    # extract's dict carries a different subset of leaky fields than search.
    for field in ["normalized_url", "top_chunks", "cached"]:
        assert field in result, f"extract dict should carry {field}"


@pytest.mark.asyncio
async def test_extract_markdown_does_not_leak_metadata_fields():
    extract_mock = AsyncMock(return_value={
        "status": "ok",
        "url": "https://example.com/a",
        "content_type": "text/html",
        "file_type": "html",
        "title": "A",
        "content": "# A\n\ncontent",
        "top_chunks": [{"text": "content", "score": 0.5}],
        "cached": True,
    })

    with patch(PATCH_EXTRACT_URL_DOCUMENT, extract_mock):
        markdown = await server_module.extract.fn(
            ["https://example.com/a"], query="q",
        )

    for field in _LEAKY_FIELDS:
        assert f"{field}:" not in markdown, (
            f"{field!r} key leaked into LLM-facing markdown:\n{markdown}"
        )


# ---------------------------------------------------------------------------
# map
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_map_markdown_does_not_leak_metadata_fields():
    discover_mock = AsyncMock(return_value={
        "status": "ok",
        "url": "https://docs.example.com",
        "title": "Docs",
        "links": [
            {
                "url": "https://docs.example.com/guide",
                "title": "Guide",
                "text": "Guide",
                "link_type": "internal",
            },
        ],
    })

    with patch("core._discover_page_links", discover_mock):
        markdown = await server_module.map.fn(
            "https://docs.example.com", max_urls=5, max_depth=1,
        )

    for field in _LEAKY_FIELDS + ["link_type", "discovered_from", "domain"]:
        assert f"{field}:" not in markdown, (
            f"{field!r} key leaked into LLM-facing markdown:\n{markdown}"
        )


# ---------------------------------------------------------------------------
# crawl
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_crawl_markdown_does_not_leak_metadata_fields():
    map_mock = AsyncMock(return_value={
        "url": "https://docs.example.com",
        "results": [
            {
                "rank": 1,
                "url": "https://docs.example.com",
                "normalized_url": "https://docs.example.com",
                "domain": "docs.example.com",
                "title": "Home",
                "link_text": None,
                "depth": 0,
                "discovered_from": None,
                "link_type": "seed",
            },
        ],
        "meta": {"urls_returned": 1, "warnings": [], "timings_ms": {"total": 1}},
    })
    extract_mock = AsyncMock(return_value={
        "query": "q",
        "results": [
            {
                "url": "https://docs.example.com",
                "normalized_url": "https://docs.example.com",
                "domain": "docs.example.com",
                "status": "ok",
                "content_type": "text/html",
                "title": "Home",
                "content": "# Home",
                "top_chunks": [{"text": "home", "score": 0.9}],
                "cached": True,
                "error": None,
            },
        ],
        "meta": {
            "urls_requested": 1, "urls_succeeded": 1, "urls_failed": 0,
            "timings_ms": {"total": 1},
        },
    })

    with (
        patch(PATCH_MAP_SITE_IMPL, map_mock),
        patch(PATCH_EXTRACT_URLS_IMPL, extract_mock),
    ):
        markdown = await server_module.crawl.fn(
            "https://docs.example.com", query="q", max_urls=1,
        )

    for field in _LEAKY_FIELDS:
        assert f"{field}:" not in markdown, (
            f"{field!r} key leaked into LLM-facing markdown:\n{markdown}"
        )
