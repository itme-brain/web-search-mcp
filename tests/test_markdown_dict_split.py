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


def test_search_markdown_separates_results_with_horizontal_rules():
    """Each result must be visually separated so the LLM reads them as
    distinct units rather than one flowing document."""
    markdown = server_module._format_search_results({
        "query": "q",
        "results": [
            {"rank": 1, "title": "A", "url": "https://a", "domain": "a", "content": "Alpha body."},
            {"rank": 2, "title": "B", "url": "https://b", "domain": "b", "content": "Beta body."},
            {"rank": 3, "title": "C", "url": "https://c", "domain": "c", "content": "Gamma body."},
        ],
        "meta": {"num_results_returned": 3, "warnings": []},
    })
    # mdformat renders `---` as a run of underscores. Every adjacent pair
    # of result headings should have a thematic break between them.
    between_1_and_2 = markdown.split("## 1.")[1].split("## 2.")[0]
    between_2_and_3 = markdown.split("## 2.")[1].split("## 3.")[0]
    assert "___" in between_1_and_2
    assert "___" in between_2_and_3


def test_extract_markdown_separates_multiple_pages_with_rules():
    """Multi-page extract must put a horizontal rule between each page
    section so the LLM reads them as distinct documents."""
    markdown = server_module._format_extract_results({
        "query": None,
        "results": [
            {"url": "https://a", "domain": "a", "status": "ok", "title": "A",
             "content": "Body A.", "chars_shown": 7, "total_chars": 7,
             "top_chunks": [], "cached": False},
            {"url": "https://b", "domain": "b", "status": "ok", "title": "B",
             "content": "Body B.", "chars_shown": 7, "total_chars": 7,
             "top_chunks": [], "cached": False},
        ],
        "meta": {"urls_requested": 2, "urls_succeeded": 2, "urls_failed": 0,
                 "timings_ms": {"total": 1}},
    })
    between = markdown.split("## [A]")[1].split("## [B]")[0]
    assert "___" in between


def test_crawl_markdown_separates_multiple_pages_with_rules():
    """Same separator requirement for crawl's per-page sections."""
    markdown = server_module._format_crawl_results({
        "url": "https://root",
        "results": [
            {"rank": 1, "url": "https://root", "domain": "root", "title": "Root",
             "link_text": None, "depth": 0, "discovered_from": None,
             "link_type": "seed", "status": "ok", "content": "Root body.",
             "chars_shown": 10, "total_chars": 10, "cached": False},
            {"rank": 2, "url": "https://root/a", "domain": "root", "title": "A",
             "link_text": None, "depth": 1, "discovered_from": "https://root",
             "link_type": "internal", "status": "ok", "content": "Body A.",
             "chars_shown": 7, "total_chars": 7, "cached": False},
        ],
        "meta": {"max_urls_requested": 2, "urls_discovered": 2, "urls_returned": 2,
                 "urls_truncated_by_limit": 0, "urls_deduplicated": 0,
                 "urls_succeeded": 2, "urls_failed": 0, "warnings": [],
                 "timings_ms": {"total": 1}},
    })
    between = markdown.split("## [Root]")[1].split("## [A]")[0]
    assert "___" in between


def test_search_markdown_marks_snippet_fallback_results():
    """When a page fails to scrape, the formatter should mark the
    snippet-fallback result so the LLM can tell full-text from snippet."""
    markdown = server_module._format_search_results({
        "query": "q",
        "results": [
            {"rank": 1, "title": "Full", "url": "https://a", "domain": "a",
             "content": "Full scraped body.", "scraped": True},
            {"rank": 2, "title": "Snippet", "url": "https://b", "domain": "b",
             "content": "Short snippet.", "scraped": False},
        ],
        "meta": {"num_results_returned": 2, "warnings": []},
    })
    # Snippet-fallback result is marked; fully-scraped one is not.
    snippet_section = markdown.split("## 2.")[1]
    full_section = markdown.split("## 1.")[1].split("## 2.")[0]
    assert "snippet only" in snippet_section
    assert "snippet only" not in full_section


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

    # Warnings render as a labeled bullet list, not a semicolon-joined line.
    assert "issues:" in markdown
    assert "- partial scrape: 2 of 5 pages failed — snippet used instead" in markdown
    assert "- filtered low-relevance results: 1 result(s) dropped below relevance threshold" in markdown
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
