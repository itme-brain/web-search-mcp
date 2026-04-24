from unittest.mock import AsyncMock, patch

import pytest

import cache as cache_module
from tests.conftest import FakeContext, server_module

PATCH_EXTRACT_URL_DOCUMENT = "core._extract_url_document"


@pytest.mark.asyncio
async def test_extract_urls_validates_input():
    with pytest.raises(ValueError, match="urls must not be empty"):
        await server_module.extract.fn([], ctx=None)

    with pytest.raises(ValueError, match="invalid URL"):
        await server_module.extract.fn(["notaurl"], ctx=None)


@pytest.mark.asyncio
async def test_extract_impl_treats_null_string_query_as_none():
    """Buggy MCP clients sometimes serialize `query=None` as the string
    "null". It must not be forwarded as a literal query to the reranker."""
    extract_mock = AsyncMock(return_value={
        "status": "ok",
        "url": "https://example.com/page",
        "content_type": "text/html",
        "title": "Example",
        "content": "# Example\n\nContent.",
        "top_chunks": [],
        "cached": False,
    })

    with patch(PATCH_EXTRACT_URL_DOCUMENT, extract_mock):
        payload = await server_module.extract_impl(
            urls=["https://example.com/page"],
            query="null",
        )

    assert payload["query"] is None
    # `_extract_url_document` must receive None, not the string "null".
    assert extract_mock.call_args.args[1] is None


@pytest.mark.asyncio
async def test_extract_urls_returns_structured_results():
    fake_ctx = FakeContext()
    extract_mock = AsyncMock(side_effect=[
        {
            "status": "ok",
            "url": "https://example.com/page",
            "content_type": "text/html",
            "title": "Example Page",
            "content": "# Example\n\nUseful extracted content.",
            "top_chunks": [{"text": "Useful extracted content.", "score": 1.0}],
            "cached": False,
        },
        {
            "status": "handoff",
            "url": "https://example.com/file.pdf",
            "content_type": "application/pdf",
            "file_type": "pdf",
            "title": None,
            "content": "",
            "top_chunks": [],
            "cached": True,
            "handoff": {
                "handler": "files",
                "reason": "pdf extraction is delegated to the files MCP",
            },
        },
    ])

    with patch(PATCH_EXTRACT_URL_DOCUMENT, extract_mock):
        payload = await server_module.extract_impl(
            urls=["https://example.com/page", "https://example.com/file.pdf"],
            query="example query",
            ctx=fake_ctx,
        )

    assert payload["query"] == "example query"
    assert payload["meta"]["urls_requested"] == 2
    assert payload["meta"]["urls_succeeded"] == 2
    assert payload["meta"]["urls_failed"] == 0
    assert payload["results"][0]["status"] == "ok"
    assert payload["results"][0]["content_type"] == "text/html"
    assert payload["results"][1]["status"] == "handoff"
    assert payload["results"][1]["content_type"] == "application/pdf"
    assert payload["results"][1]["cached"] is True


@pytest.mark.asyncio
async def test_extract_urls_reports_partial_failures():
    extract_mock = AsyncMock(side_effect=[
        {
            "status": "ok",
            "url": "https://example.com/page",
            "content_type": "text/html",
            "title": None,
            "content": "# Example\n\nUseful extracted content.",
            "top_chunks": [],
            "cached": False,
        },
        {
            "status": "error",
            "url": "https://example.com/missing.pdf",
            "content_type": "application/pdf",
            "title": None,
            "content": "",
            "error": "404 not found",
            "top_chunks": [],
            "cached": False,
        },
    ])

    with patch(PATCH_EXTRACT_URL_DOCUMENT, extract_mock):
        payload = await server_module.extract_impl(
            urls=["https://example.com/page", "https://example.com/missing.pdf"],
        )

    assert payload["meta"]["urls_succeeded"] == 1
    assert payload["meta"]["urls_failed"] == 1
    assert payload["results"][1]["status"] == "error"
    assert payload["results"][1]["error"] == "404 not found"


@pytest.mark.asyncio
async def test_extract_urls_single_url_returns_markdown():
    fake_ctx = FakeContext()
    extract_mock = AsyncMock(return_value={
        "status": "handoff",
        "url": "https://example.com/file.docx",
        "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "file_type": "docx",
        "title": None,
        "content": "",
        "top_chunks": [],
        "cached": False,
        "handoff": {
            "handler": "files",
            "reason": "docx extraction is delegated to the files MCP",
        },
    })

    with patch(PATCH_EXTRACT_URL_DOCUMENT, extract_mock):
        payload = await server_module.extract.fn(
            ["https://example.com/file.docx"],
            query="example query",
            ctx=fake_ctx,
        )
    payload_text = payload.content[0].text

    assert "example query" in payload_text
    assert "https://example.com/file.docx" in payload_text
    assert "`files` MCP" in payload_text
    assert "docx extraction is delegated" in payload_text


@pytest.mark.asyncio
async def test_single_extract_markdown_omits_redundant_success_header():
    extract_mock = AsyncMock(return_value={
        "status": "ok",
        "url": "https://example.com/page",
        "content_type": "text/html",
        "file_type": "html",
        "title": "Example Page",
        "content": "# Example\n\nUseful extracted content.",
        "total_chars": 28,
        "total_chunks": 3,
        "shown_chunk_ids": [0, 1, 2],
        "chunk_mode": "document",
        "top_chunks": [],
        "cached": False,
    })

    with patch(PATCH_EXTRACT_URL_DOCUMENT, extract_mock):
        payload = await server_module.extract.fn(["https://example.com/page"], ctx=None)

    payload_text = payload.content[0].text
    assert not payload_text.startswith("succeeded:")
    assert "## [Example Page](https://example.com/page)" in payload_text


@pytest.mark.asyncio
async def test_extract_url_document_handoffs_binary_file_types():
    with patch("core._detect_file_type", AsyncMock(return_value=("pdf", "application/pdf"))):
        result = await server_module._extract_url_document(
            "https://example.com/manual.pdf",
            query=None,
            cache=cache_module.page_cache,
        )

    assert result["status"] == "handoff"
    assert result["file_type"] == "pdf"
    assert result["handoff"]["handler"] == "files"
    assert result["content"] == ""


@pytest.mark.asyncio
async def test_extract_cache_hit_preserves_handoff_metadata():
    await cache_module.page_cache.set("https://example.com/manual.pdf", {
        "status": "handoff",
        "url": "https://example.com/manual.pdf",
        "content_type": "application/pdf",
        "file_type": "pdf",
        "title": None,
        "content": "",
        "total_chars": 0,
        "handoff": {
            "handler": "files",
            "reason": "pdf extraction is delegated to the files MCP",
        },
    })

    result = await server_module._extract_url_document(
        "https://example.com/manual.pdf", query=None, cache=cache_module.page_cache,
    )

    assert result["cached"] is True
    assert result["status"] == "handoff"
    assert result["file_type"] == "pdf"
    assert result["handoff"]["handler"] == "files"


def test_guess_file_type_supports_handoff_and_text_formats():
    assert server_module._guess_file_type(
        "https://example.com/file.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ) == "docx"
    assert server_module._guess_file_type(
        "https://example.com/slides.pptx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ) == "pptx"
    assert server_module._guess_file_type(
        "https://example.com/sheet.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ) == "xlsx"
    assert server_module._guess_file_type("https://example.com/data.json", "application/json") == "json"
    assert server_module._guess_file_type("https://example.com/config.yaml", "application/yaml") == "yaml"
    assert server_module._guess_file_type("https://example.com/notes.txt", "text/plain; charset=utf-8") == "text"


@pytest.mark.asyncio
async def test_detect_file_type_prefers_head_content_type_when_specific():
    with (
        patch("core._head_content_type", AsyncMock(return_value="text/html")),
        patch("core._sniff_content_type", AsyncMock(return_value="application/pdf")),
    ):
        file_type, content_type = await server_module._detect_file_type("https://example.com/download")

    assert file_type == "html"
    assert content_type == "text/html"


@pytest.mark.asyncio
async def test_detect_file_type_falls_back_to_head_content_type():
    with (
        patch("core._sniff_content_type", AsyncMock(return_value=None)),
        patch("core._head_content_type", AsyncMock(return_value="text/plain; charset=utf-8")),
    ):
        file_type, content_type = await server_module._detect_file_type("https://example.com/notes")

    assert file_type == "text"
    assert content_type == "text/plain"


@pytest.mark.asyncio
async def test_detect_file_type_sniffs_when_head_is_generic():
    with (
        patch("core._head_content_type", AsyncMock(return_value="application/octet-stream")),
        patch("core._sniff_content_type", AsyncMock(return_value="application/pdf")),
    ):
        file_type, content_type = await server_module._detect_file_type("https://example.com/download")

    assert file_type == "pdf"
    assert content_type == "application/pdf"


@pytest.mark.asyncio
async def test_sniff_content_type_bails_when_range_is_ignored():
    class _FakeResponse:
        status_code = 200
        headers = {"content-length": "8192"}

        def raise_for_status(self):
            return None

        async def aiter_bytes(self):
            yield b"%PDF-1.7"

    class _FakeStream:
        async def __aenter__(self):
            return _FakeResponse()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, *args, **kwargs):
            return _FakeStream()

    with patch("core.httpx.AsyncClient", _FakeClient):
        assert await server_module._sniff_content_type("https://example.com/file.pdf") is None


@pytest.mark.asyncio
async def test_extract_url_document_handoffs_unknown_types_by_default():
    with patch("core._detect_file_type", AsyncMock(return_value=("unknown", "application/octet-stream"))):
        result = await server_module._extract_url_document(
            "https://example.com/blob.bin",
            query=None,
            cache=cache_module.page_cache,
        )

    assert result["status"] == "handoff"
    assert result["file_type"] == "unknown"
    assert result["handoff"]["handler"] == "files"


# ---------------------------------------------------------------------------
# Chunk-aware extract presentation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_markdown_surfaces_chunk_range_summary():
    full_length = 24000
    sliced = "x" * 8000
    extract_mock = AsyncMock(return_value={
        "status": "ok",
        "url": "https://example.com/long",
        "content_type": "text/html",
        "file_type": "html",
        "title": "Long Page",
        "content": sliced,
        "total_chars": full_length,
        "total_chunks": 9,
        "shown_chunk_ids": [0, 1, 2],
        "chunk_mode": "relevant",
    })

    with patch(PATCH_EXTRACT_URL_DOCUMENT, extract_mock):
        markdown = await server_module.extract.fn(["https://example.com/long"])
    markdown_text = markdown.content[0].text

    assert "document: chunks: 0..2 of 0..8 | mode: relevant | 8,000 of 24,000 chars" in markdown_text
    assert markdown_text.index("document: chunks: 0..2 of 0..8 | mode: relevant | 8,000 of 24,000 chars") < markdown_text.index("## [Long Page](https://example.com/long)")
    assert "_chunks: 0..2 of 0..8" not in markdown_text


@pytest.mark.asyncio
async def test_extract_no_chunk_summary_when_chunks_absent():
    short = "tiny content"
    extract_mock = AsyncMock(return_value={
        "status": "ok",
        "url": "https://example.com/short",
        "content_type": "text/html",
        "file_type": "html",
        "title": "Short",
        "content": short,
        "total_chars": len(short),
    })

    with patch(PATCH_EXTRACT_URL_DOCUMENT, extract_mock):
        markdown = await server_module.extract.fn(["https://example.com/short"])
    markdown_text = markdown.content[0].text

    assert "chunks:" not in markdown_text
    assert "mode:" not in markdown_text


@pytest.mark.asyncio
async def test_extract_rejects_negative_chunk_ids():
    with pytest.raises(ValueError, match="chunk_ids entries must be >= 0"):
        await server_module.extract_impl(
            urls=["https://example.com/a"], chunk_ids=[-1],
        )


@pytest.mark.asyncio
async def test_extract_response_includes_chunks_with_stable_ids():
    """The full chunk list with ids is returned so callers can cherry-pick."""
    # Three paragraphs → three chunks with ids 0, 1, 2.
    content = "Alpha paragraph one.\n\nBeta paragraph two.\n\nGamma paragraph three."
    await cache_module.page_cache.set("https://example.com/chunked", {
        "status": "ok",
        "url": "https://example.com/chunked",
        "content_type": "text/html",
        "file_type": "html",
        "title": "Chunked",
        "content": content,
        "total_chars": len(content),
    })

    result = await server_module._extract_url_document(
        "https://example.com/chunked",
        query=None,
        cache=cache_module.page_cache,
    )

    assert [c["id"] for c in result["chunks"]] == [0, 1, 2]
    assert result["chunks"][0]["text"] == "Alpha paragraph one."
    assert result["chunks"][2]["text"] == "Gamma paragraph three."
    assert result["shown_chunk_ids"] == [0, 1, 2]
    assert result["total_chunks"] == 3
    assert result["chunk_mode"] == "document"


@pytest.mark.asyncio
async def test_failure_entries_get_short_ttl():
    """Rejected/failed page entries expire in FAILURE_TTL_S, not the
    default 3600s, so transient upstream issues recover quickly."""
    url = "https://example.com/failed"
    fake_scrape = AsyncMock(return_value={"content": None, "title": None, "metadata": {}})
    with patch("core._scrape", fake_scrape):
        await server_module._scrape_cached(url, cache_module.page_cache)

    # Inspect the raw Valkey TTL on the key.
    normalized = server_module._normalize_url(url)
    client = cache_module._get_client()
    ttl_seconds = await client.ttl(f"ws:page:{normalized}")
    assert 0 < ttl_seconds <= cache_module.FAILURE_TTL_S


@pytest.mark.asyncio
async def test_success_entries_get_default_ttl():
    """Successful page entries keep the full default TTL."""
    url = "https://example.com/success"
    content = (
        "# OK\n\nA useful page with enough content to clear the "
        "minimum word count floor applied at cache admission time "
        "so this entry becomes a full successful cache write."
    )
    fake_scrape = AsyncMock(return_value={"content": content, "title": "OK", "metadata": {}})
    with patch("core._scrape", fake_scrape):
        await server_module._scrape_cached(url, cache_module.page_cache)

    normalized = server_module._normalize_url(url)
    client = cache_module._get_client()
    ttl_seconds = await client.ttl(f"ws:page:{normalized}")
    assert ttl_seconds > cache_module.FAILURE_TTL_S
    # And under the default to account for any small elapsed time.
    assert ttl_seconds <= 3600


@pytest.mark.asyncio
async def test_scrape_cache_rejects_under_length_floor():
    """Speculative scrape (search path) caches short content as a failure.

    CAPTCHA walls and 404 shells typically come back as <20-word
    responses. The cache treats these as misses so search doesn't
    surface them and extract doesn't thrash re-fetching them.
    """
    url = "https://captcha.example.com/blocked"
    fake_scrape = AsyncMock(return_value={
        "content": "Please verify you are human.",  # 5 words
        "title": "Access Blocked",
        "metadata": {},
    })
    with patch("core._scrape", fake_scrape):
        envelope = await server_module._scrape_cached(url, cache_module.page_cache)

    # Short content rejected at write time → cached as failure.
    assert envelope["status"] == "error"
    assert envelope["content"] is None


@pytest.mark.asyncio
async def test_content_hash_alias_collapses_duplicate_content_at_different_urls():
    """Two URLs with byte-identical content share one full entry.

    Writes the second URL as an alias; reads through the alias return
    the canonical's content.
    """
    content = (
        "# Canonical\n\nOne paragraph of genuinely useful content that "
        "clears the minimum word count floor applied at cache admission "
        "time so the entry becomes a full write and the dedup alias path "
        "through content_hash gets exercised by this test without hitting "
        "the speculative-admission rejection branch that we also test."
    )
    fake_scrape = AsyncMock(return_value={
        "content": content,
        "title": "Canonical",
        "metadata": {},
    })
    with patch("core._scrape", fake_scrape):
        # First URL writes the canonical entry.
        await server_module._scrape_cached(
            "https://example.com/canonical", cache_module.page_cache,
        )
        # Second URL scrapes identical content.
        aliased = await server_module._scrape_cached(
            "https://example.com/mirror", cache_module.page_cache,
        )

    # Raw Valkey peek: the alias entry is light, the canonical is full.
    raw_alias = await cache_module.page_cache.get(
        server_module._normalize_url("https://example.com/mirror")
    )
    raw_canonical = await cache_module.page_cache.get(
        server_module._normalize_url("https://example.com/canonical")
    )
    assert "alias" in raw_alias
    assert raw_alias["alias"] == server_module._normalize_url("https://example.com/canonical")
    assert raw_canonical.get("content") == content

    # Dereferenced read returns the canonical content but keeps the
    # requested URL in the returned envelope.
    resolved = await server_module._page_get(
        "https://example.com/mirror", cache_module.page_cache,
    )
    assert resolved["content"] == content
    assert resolved["url"] == "https://example.com/mirror"


@pytest.mark.asyncio
async def test_dangling_alias_treated_as_miss():
    """If the canonical disappears, the alias returns None (re-scrape)."""
    content = (
        "# Page\n\nGenuine content body with at least enough prose to "
        "comfortably pass the cache admission floor used in the scrape "
        "path so we are exercising the happy-path aliasing logic and "
        "not the short-content rejection branch in this test scenario."
    )
    fake_scrape = AsyncMock(return_value={
        "content": content, "title": "T", "metadata": {},
    })
    with patch("core._scrape", fake_scrape):
        await server_module._scrape_cached(
            "https://example.com/canonical", cache_module.page_cache,
        )
        await server_module._scrape_cached(
            "https://example.com/mirror", cache_module.page_cache,
        )

    # Simulate the canonical being evicted.
    await cache_module.page_cache.delete(
        server_module._normalize_url("https://example.com/canonical")
    )

    resolved = await server_module._page_get(
        "https://example.com/mirror", cache_module.page_cache,
    )
    assert resolved is None


@pytest.mark.asyncio
async def test_extract_sees_search_scrape_as_cache_hit():
    """Unified ws:page cache — search's scrape is an extract cache hit.

    search_impl calls _scrape_cached which writes the full page-envelope
    shape. extract_impl later reads that entry and must get cached=True,
    full content, and no re-scrape.
    """
    url = "https://docs.example.com/shared"
    # Populate via _scrape_cached the way search_impl does.
    fake_scrape = AsyncMock(return_value={
        "content": "# Shared\n\nthis is the shared page body used by both the search path and the extract path; it needs enough words to clear the cache admission floor for test purposes.",
        "title": "Shared",
        "metadata": {"word_count": 32},
    })
    with patch("core._scrape", fake_scrape):
        envelope = await server_module._scrape_cached(url, cache_module.page_cache)

    assert envelope["_schema_version"] == 1
    assert envelope["status"] == "ok"
    assert envelope["file_type"] == "html"

    # Now extract_url_document on the same URL should hit the cache
    # without calling _scrape or _detect_file_type again.
    extract_scrape = AsyncMock(side_effect=AssertionError("should not re-scrape"))
    extract_detect = AsyncMock(side_effect=AssertionError("should not re-detect"))
    with (
        patch("core._scrape", extract_scrape),
        patch("core._detect_file_type", extract_detect),
    ):
        result = await server_module._extract_url_document(
            url, query=None, cache=cache_module.page_cache,
        )

    assert result["cached"] is True
    assert result["status"] == "ok"
    assert "shared page body" in result["content"]


@pytest.mark.asyncio
async def test_extract_cache_collapses_url_variants():
    """www./trailing-slash variants must share one cache entry.

    Caching on raw URL strings means equivalent requests miss each other.
    Normalized-URL cache keys collapse them so search's scrape cache and
    a user-facing extract call can share hits.
    """
    content = "Once cached, any variant of this URL should hit."
    # Pre-populate with the canonical form.
    canonical = "https://example.com/docs/page"
    await cache_module.page_cache.set(canonical, {
        "status": "ok",
        "url": canonical,
        "content_type": "text/html",
        "file_type": "html",
        "title": "Shared",
        "content": content,
        "total_chars": len(content),
    })

    variants = [
        "https://www.example.com/docs/page",
        "https://example.com/docs/page/",
        "https://example.com/docs/page?utm_source=nope",
    ]
    for variant in variants:
        result = await server_module._extract_url_document(
            variant,
            query=None,
            cache=cache_module.page_cache,
        )
        assert result["cached"] is True, f"variant missed cache: {variant!r}"
        assert result["title"] == "Shared"


@pytest.mark.asyncio
async def test_extract_chunk_ids_returns_only_selected_chunks():
    """chunk_ids=[0,2] joins chunks 0 and 2 into `content`, skips rerank."""
    content = "Alpha paragraph one.\n\nBeta paragraph two.\n\nGamma paragraph three."
    await cache_module.page_cache.set("https://example.com/chunked", {
        "status": "ok",
        "url": "https://example.com/chunked",
        "content_type": "text/html",
        "file_type": "html",
        "title": "Chunked",
        "content": content,
        "total_chars": len(content),
    })

    result = await server_module._extract_url_document(
        "https://example.com/chunked",
        query="query that would otherwise rerank",
        cache=cache_module.page_cache,
        chunk_ids=[0, 2],
    )

    # chunk_ids short-circuits rerank.
    assert result["top_chunks"] == []
    assert "Alpha paragraph one." in result["content"]
    assert "Gamma paragraph three." in result["content"]
    assert "Beta paragraph two." not in result["content"]
    # The full chunk list is still surfaced so the caller can iterate.
    assert [c["id"] for c in result["chunks"]] == [0, 1, 2]
    assert result["shown_chunk_ids"] == [0, 2]
    assert result["chunk_mode"] == "selected"


@pytest.mark.asyncio
async def test_extract_markdown_uses_compact_chunk_ranges():
    long_content = "B" * 12000
    extract_mock = AsyncMock(return_value={
        "status": "ok",
        "url": "https://example.com/page",
        "content_type": "text/html",
        "file_type": "html",
        "title": "Page",
        "content": long_content[:8000],
        "total_chars": 12000,
        "total_chunks": 9,
        "shown_chunk_ids": [3, 4, 5],
        "chunk_mode": "selected",
    })

    with patch(PATCH_EXTRACT_URL_DOCUMENT, extract_mock):
        markdown = await server_module.extract.fn(["https://example.com/page"])
    markdown_text = markdown.content[0].text

    assert "document: chunks: 3..5 of 0..8 | mode: selected | 8,000 of 12,000 chars" in markdown_text


@pytest.mark.asyncio
async def test_extract_markdown_renders_non_consecutive_chunks_as_list():
    """Cherry-picked chunks like [5, 30, 60] must render as a comma list,
    not `5..60` — the range form implies everything in between was pulled."""
    extract_mock = AsyncMock(return_value={
        "status": "ok",
        "url": "https://example.com/page",
        "content_type": "text/html",
        "file_type": "html",
        "title": "Page",
        "content": "X" * 2000,
        "total_chars": 30000,
        "total_chunks": 100,
        "shown_chunk_ids": [5, 30, 60],
        "chunk_mode": "selected",
    })

    with patch(PATCH_EXTRACT_URL_DOCUMENT, extract_mock):
        markdown = await server_module.extract.fn(["https://example.com/page"])
    markdown_text = markdown.content[0].text

    assert "chunks: 5, 30, 60 of 0..99" in markdown_text
    assert "5..60" not in markdown_text


def test_chunk_range_collapses_runs_but_preserves_gaps():
    r = server_module._chunk_range
    assert r([]) is None
    assert r([7]) == "7"
    assert r([20, 21, 22, 23, 24]) == "20..24"
    assert r([5, 30, 60]) == "5, 30, 60"
    assert r([5, 6, 7, 30, 60, 61]) == "5..7, 30, 60..61"
    # Unsorted and duplicated input normalizes the same way.
    assert r([60, 5, 30, 5]) == "5, 30, 60"
