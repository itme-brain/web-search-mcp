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
async def test_extract_url_document_handoffs_binary_file_types():
    with patch("core._detect_file_type", AsyncMock(return_value=("pdf", "application/pdf"))):
        result = await server_module._extract_url_document(
            "https://example.com/manual.pdf",
            query=None,
            cache=cache_module.extract_cache,
        )

    assert result["status"] == "handoff"
    assert result["file_type"] == "pdf"
    assert result["handoff"]["handler"] == "files"
    assert result["content"] == ""


@pytest.mark.asyncio
async def test_extract_cache_hit_preserves_handoff_metadata():
    await cache_module.extract_cache.set("https://example.com/manual.pdf", {
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
        "https://example.com/manual.pdf", query=None, cache=cache_module.extract_cache,
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
async def test_detect_file_type_prefers_magic_sniffing():
    with (
        patch("core._sniff_content_type", AsyncMock(return_value="application/pdf")),
        patch("core._head_content_type", AsyncMock(return_value="text/html")),
    ):
        file_type, content_type = await server_module._detect_file_type("https://example.com/download")

    assert file_type == "pdf"
    assert content_type == "application/pdf"


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
async def test_extract_url_document_handoffs_unknown_types_by_default():
    with patch("core._detect_file_type", AsyncMock(return_value=("unknown", "application/octet-stream"))):
        result = await server_module._extract_url_document(
            "https://example.com/blob.bin",
            query=None,
            cache=cache_module.extract_cache,
        )

    assert result["status"] == "handoff"
    assert result["file_type"] == "unknown"
    assert result["handoff"]["handler"] == "files"


# ---------------------------------------------------------------------------
# Truncation signal + offset continuation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_markdown_signals_truncation_with_next_offset():
    """When content is truncated, markdown footer must tell the LLM
    how to continue (offset=N).

    _extract_url_document returns content already sliced to <= MAX
    plus the full total_chars alongside — simulate that shape.
    """
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
    })

    with patch(PATCH_EXTRACT_URL_DOCUMENT, extract_mock):
        markdown = await server_module.extract.fn(["https://example.com/long"])
    markdown_text = markdown.content[0].text

    assert "chars shown" in markdown_text
    assert "offset=8000" in markdown_text, f"truncation footer missing offset hint:\n{markdown_text}"


@pytest.mark.asyncio
async def test_extract_no_signal_when_content_fits():
    """Short content that wasn't truncated must not emit a truncation footer."""
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

    assert "chars shown" not in markdown_text
    assert "end of document" not in markdown_text


@pytest.mark.asyncio
async def test_extract_offset_bypasses_rerank_and_slices_raw():
    """offset>0 returns the raw content slice, skipping query rerank."""
    long_content = "A" * 20000
    await cache_module.extract_cache.set("https://example.com/long", {
        "status": "ok",
        "url": "https://example.com/long",
        "content_type": "text/html",
        "file_type": "html",
        "title": "Long",
        "content": long_content,
        "total_chars": len(long_content),
    })

    result = await server_module._extract_url_document(
        "https://example.com/long",
        query="something",
        cache=cache_module.extract_cache,
        offset=8000,
    )
    assert result["content"] == long_content[8000:16000]
    # Offset skips rerank — no top_chunks.
    assert result["top_chunks"] == []


@pytest.mark.asyncio
async def test_extract_rejects_negative_offset():
    with pytest.raises(ValueError, match="offset must be >= 0"):
        await server_module.extract_impl(
            urls=["https://example.com/a"], offset=-1,
        )


@pytest.mark.asyncio
async def test_extract_rejects_chunk_ids_with_offset():
    with pytest.raises(ValueError, match="mutually exclusive"):
        await server_module.extract_impl(
            urls=["https://example.com/a"], offset=100, chunk_ids=[0],
        )


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
    await cache_module.extract_cache.set("https://example.com/chunked", {
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
        cache=cache_module.extract_cache,
    )

    assert [c["id"] for c in result["chunks"]] == [0, 1, 2]
    assert result["chunks"][0]["text"] == "Alpha paragraph one."
    assert result["chunks"][2]["text"] == "Gamma paragraph three."


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
    await cache_module.extract_cache.set(canonical, {
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
            cache=cache_module.extract_cache,
        )
        assert result["cached"] is True, f"variant missed cache: {variant!r}"
        assert result["title"] == "Shared"


@pytest.mark.asyncio
async def test_extract_chunk_ids_returns_only_selected_chunks():
    """chunk_ids=[0,2] joins chunks 0 and 2 into `content`, skips rerank."""
    content = "Alpha paragraph one.\n\nBeta paragraph two.\n\nGamma paragraph three."
    await cache_module.extract_cache.set("https://example.com/chunked", {
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
        cache=cache_module.extract_cache,
        chunk_ids=[0, 2],
    )

    # chunk_ids short-circuits rerank.
    assert result["top_chunks"] == []
    assert "Alpha paragraph one." in result["content"]
    assert "Gamma paragraph three." in result["content"]
    assert "Beta paragraph two." not in result["content"]
    # The full chunk list is still surfaced so the caller can iterate.
    assert [c["id"] for c in result["chunks"]] == [0, 1, 2]


@pytest.mark.asyncio
async def test_extract_offset_reaching_end_shows_end_marker():
    long_content = "B" * 12000
    extract_mock = AsyncMock(return_value={
        "status": "ok",
        "url": "https://example.com/page",
        "content_type": "text/html",
        "file_type": "html",
        "title": "Page",
        "content": long_content[8000:],  # simulating the slice returned
        "total_chars": 12000,
    })

    with patch(PATCH_EXTRACT_URL_DOCUMENT, extract_mock):
        markdown = await server_module.extract.fn(
            ["https://example.com/page"], offset=8000,
        )
    markdown_text = markdown.content[0].text

    assert "end of document" in markdown_text
    assert "12,000 chars total" in markdown_text
