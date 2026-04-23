"""Pin the markdown/dict contract: metadata fields stay in the Python dict
for scripts/eval, but never leak into the LLM-facing markdown output."""

from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import URLS_A, make_search_results, server_module

PATCH_SEARCH = "core._search"
PATCH_SCRAPE = "core._scrape"
PATCH_RERANK = "core._rerank_scored"
PATCH_EXTRACT_URL_DOCUMENT = "core._extract_url_document"
PATCH_MAP_IMPL = "impls.map_impl"
PATCH_EXTRACT_IMPL = "impls.extract_impl"

# Fields that are part of the structured dict for scripting access but
# must never appear as literal keys in the LLM-facing markdown output.
# (normalized_url / search_rank / score were trimmed from the response
# entirely — they're internal bookkeeping, not agent-facing data.)
_LEAKY_FIELDS = [
    "previously_seen",
    "cached",
    "top_chunks",
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
    for field in ["previously_seen", "top_chunks", "scraped"]:
        assert field in first, f"dict should carry {field} for scripting access"
    # top_chunks is a flat list[str] now — the LLM reads ordering as relevance.
    assert all(isinstance(c, str) for c in first["top_chunks"])


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


def test_search_warning_lines_are_rendered_as_issues():
    markdown = server_module._format_search_results({
        "query": "web search tools tutorial",
        "results": [],
        "meta": {
            "num_results_returned": 4,
            "warnings": [
                {"type": "scrape_failed", "detail": "2 of 5 pages failed"},
                {"type": "low_relevance_filtered", "detail": "1 result(s) dropped below relevance threshold"},
            ],
        },
    })

    assert "issues: scrape failures: 2 of 5 pages failed; filtered low-relevance results: 1 result(s) dropped below relevance threshold" in markdown
    assert "status: degraded" not in markdown


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
        "top_chunks": ["content"],
        "cached": False,
    })

    with patch(PATCH_EXTRACT_URL_DOCUMENT, extract_mock):
        payload = await server_module.extract_impl(
            urls=["https://example.com/a"], query="q",
        )

    result = payload["results"][0]
    # extract's dict carries these for scripting access
    for field in ["top_chunks", "cached"]:
        assert field in result, f"extract dict should carry {field}"
    assert all(isinstance(c, str) for c in result["top_chunks"])


@pytest.mark.asyncio
async def test_extract_markdown_does_not_leak_metadata_fields():
    extract_mock = AsyncMock(return_value={
        "status": "ok",
        "url": "https://example.com/a",
        "content_type": "text/html",
        "file_type": "html",
        "title": "A",
        "content": "# A\n\ncontent",
        "top_chunks": ["content"],
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
            "https://docs.example.com", max_urls=5,
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
    deep_crawl_mock = AsyncMock(return_value=[
        {
            "url": "https://docs.example.com",
            "metadata": {"title": "Home", "depth": 0},
            "markdown": {"raw_markdown": "# Home\n\nWelcome to the docs."},
        },
    ])

    with patch("core._deep_crawl", deep_crawl_mock):
        markdown = await server_module.crawl.fn(
            "https://docs.example.com", max_urls=1,
        )

    for field in _LEAKY_FIELDS:
        assert f"{field}:" not in markdown, (
            f"{field!r} key leaked into LLM-facing markdown:\n{markdown}"
        )
