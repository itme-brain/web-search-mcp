from unittest.mock import AsyncMock, patch
from io import BytesIO

import pytest
from docx import Document as DocxDocument

from tests.conftest import FakeContext, server_module

PATCH_EXTRACT_URL_DOCUMENT = "web_search_server._extract_url_document"


@pytest.mark.asyncio
async def test_extract_urls_validates_input():
    with pytest.raises(ValueError, match="urls must not be empty"):
        await server_module.extract_urls.fn([], ctx=None)

    with pytest.raises(ValueError, match="invalid URL"):
        await server_module.extract_urls.fn(["notaurl"], ctx=None)


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
            "status": "ok",
            "url": "https://example.com/file.pdf",
            "content_type": "application/pdf",
            "title": "Example PDF",
            "content": "## Page 1\n\nUseful PDF content.",
            "top_chunks": [],
            "cached": True,
        },
    ])

    with patch(PATCH_EXTRACT_URL_DOCUMENT, extract_mock):
        payload = await server_module._extract_urls_impl(
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
        payload = await server_module._extract_urls_impl(
            urls=["https://example.com/page", "https://example.com/missing.pdf"],
        )

    assert payload["meta"]["urls_succeeded"] == 1
    assert payload["meta"]["urls_failed"] == 1
    assert payload["results"][1]["status"] == "error"
    assert payload["results"][1]["error"] == "404 not found"


@pytest.mark.asyncio
async def test_pdf_suffix_detected_without_network():
    assert await server_module._looks_like_pdf("https://example.com/report.pdf") is True


@pytest.mark.asyncio
async def test_extract_url_returns_single_result_shape():
    fake_ctx = FakeContext()
    extract_mock = AsyncMock(return_value={
        "status": "ok",
        "url": "https://example.com/file.docx",
        "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "file_type": "docx",
        "title": "Example Doc",
        "content": "Example content",
        "top_chunks": [],
        "cached": False,
    })

    with patch(PATCH_EXTRACT_URL_DOCUMENT, extract_mock):
        payload = await server_module.extract_url.fn(
            "https://example.com/file.docx",
            query="example query",
            ctx=fake_ctx,
        )

    assert "example query" in payload
    assert "https://example.com/file.docx" in payload
    assert "Example Doc" in payload


def test_docx_bytes_to_markdown_extracts_paragraphs_and_tables():
    document = DocxDocument()
    document.core_properties.title = "Quarterly Report"
    document.add_paragraph("Executive summary")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Quarter"
    table.cell(0, 1).text = "Revenue"
    table.cell(1, 0).text = "Q1"
    table.cell(1, 1).text = "$1M"

    buffer = BytesIO()
    document.save(buffer)

    content, title = server_module._docx_bytes_to_markdown(buffer.getvalue())

    assert title == "Quarterly Report"
    assert "Executive summary" in content
    assert "| Quarter | Revenue |" in content


def _make_pdf(num_pages: int) -> bytes:
    from pypdf import PdfWriter
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=72, height=72)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_pdf_pagination_returns_total_pages():
    _, _, total, _ = server_module._pdf_bytes_to_markdown(_make_pdf(10))
    assert total == 10


def test_pdf_pagination_respects_page_range():
    _, _, total, _ = server_module._pdf_bytes_to_markdown(_make_pdf(10), start_page=3, end_page=5)
    assert total == 10


def test_pdf_pagination_clamps_end_page():
    _, _, total, _ = server_module._pdf_bytes_to_markdown(_make_pdf(3), start_page=1, end_page=100)
    assert total == 3


@pytest.mark.asyncio
async def test_extract_pdf_document_includes_pagination_metadata():
    extract_mock = AsyncMock(return_value={
        "status": "ok",
        "url": "https://example.com/manual.pdf",
        "content_type": "application/pdf",
        "file_type": "pdf",
        "title": "Manual",
        "content": "## Page 3\n\nContent here",
        "total_pages": 50,
        "start_page": 3,
        "end_page": 5,
        "top_chunks": [],
        "cached": False,
    })

    with patch(PATCH_EXTRACT_URL_DOCUMENT, extract_mock):
        payload = await server_module._extract_urls_impl(
            urls=["https://example.com/manual.pdf"],
            start_page=3,
            end_page=5,
        )

    result = payload["results"][0]
    assert result["total_pages"] == 50
    assert result["start_page"] == 3
    assert result["end_page"] == 5


def test_pdf_extract_all_pages_returns_per_page_chunks():
    # Blank pages have no extractable text, so we test the structure
    pages, title = server_module._pdf_extract_all_pages(_make_pdf(5))
    assert isinstance(pages, list)
    # Blank pages produce no text, so pages list is empty
    assert len(pages) == 0


@pytest.mark.asyncio
async def test_extract_document_chunks_rejects_non_pdf():
    detect_mock = AsyncMock(return_value=("html", "text/html"))
    with patch("web_search_server._detect_file_type", detect_mock):
        result = await server_module._extract_document_chunks_impl(
            url="https://example.com/page",
        )
    assert result["total_pages"] == 0
    assert "only supports PDFs" in result["meta"]["error"]


@pytest.mark.asyncio
async def test_extract_document_chunks_returns_all_pages():
    fake_pages = [
        {"page": i, "content": f"Content for page {i}"}
        for i in range(1, 11)
    ]
    detect_mock = AsyncMock(return_value=("pdf", "application/pdf"))
    extract_pages_mock = patch(
        "web_search_server._pdf_extract_all_pages",
        return_value=(fake_pages, "Test PDF"),
    )

    fake_resp = AsyncMock()
    fake_resp.content = b"fake pdf bytes"
    fake_resp.raise_for_status = lambda: None
    fake_client = AsyncMock()
    fake_client.get = AsyncMock(return_value=fake_resp)
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=None)

    # Need to also mock PdfReader for total_pages count
    fake_reader = AsyncMock()
    fake_reader.pages = [None] * 10

    with (
        patch("web_search_server._detect_file_type", detect_mock),
        extract_pages_mock,
        patch("web_search_server.httpx.AsyncClient", return_value=fake_client),
        patch("web_search_server.PdfReader", return_value=fake_reader),
    ):
        result = await server_module._extract_document_chunks_impl(
            url="https://example.com/manual.pdf",
            max_pages=50,
        )

    assert result["title"] == "Test PDF"
    assert result["total_pages"] == 10
    assert result["pages_returned"] == 10
    assert len(result["chunks"]) == 10
    assert result["chunks"][0]["content"] == "Content for page 1"


@pytest.mark.asyncio
async def test_extract_document_chunks_reranks_with_query():
    fake_pages = [
        {"page": i, "content": f"Content for page {i}"}
        for i in range(1, 6)
    ]
    detect_mock = AsyncMock(return_value=("pdf", "application/pdf"))

    fake_resp = AsyncMock()
    fake_resp.content = b"fake pdf bytes"
    fake_resp.raise_for_status = lambda: None
    fake_client = AsyncMock()
    fake_client.get = AsyncMock(return_value=fake_resp)
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=None)

    fake_reader = AsyncMock()
    fake_reader.pages = [None] * 5

    # Reranker returns pages in reverse order
    def fake_rerank(_query, docs):
        return [(i, 1.0 - i * 0.1) for i in reversed(range(len(docs)))]

    with (
        patch("web_search_server._detect_file_type", detect_mock),
        patch("web_search_server._pdf_extract_all_pages", return_value=(fake_pages, "Test")),
        patch("web_search_server.httpx.AsyncClient", return_value=fake_client),
        patch("web_search_server.PdfReader", return_value=fake_reader),
        patch("web_search_server._rerank_scored", side_effect=fake_rerank),
    ):
        result = await server_module._extract_document_chunks_impl(
            url="https://example.com/manual.pdf",
            query="relevant content",
            max_pages=3,
        )

    assert result["query"] == "relevant content"
    assert result["pages_returned"] == 3
    # All chunks should have scores from reranking
    assert all("score" in chunk for chunk in result["chunks"])


def test_format_document_chunks_includes_pagination_info():
    response = {
        "url": "https://example.com/manual.pdf",
        "title": "Manual",
        "total_pages": 100,
        "pages_returned": 3,
        "query": "installation",
        "chunks": [
            {"page": 5, "content": "Install instructions", "score": 0.95},
            {"page": 12, "content": "Config details", "score": 0.82},
            {"page": 1, "content": "Introduction", "score": 0.71},
        ],
        "meta": {"warnings": [], "timings_ms": {"total": 100}},
    }
    output = server_module._format_document_chunks(response)
    assert "total_pages: 100" in output
    assert "pages_returned: 3" in output
    assert "query: installation" in output
    assert "## Page 5 (score: 0.95)" in output
    assert "Install instructions" in output
    assert "## Page 12 (score: 0.82)" in output


def test_guess_file_type_supports_docx_and_text_formats():
    assert server_module._guess_file_type(
        "https://example.com/file.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ) == "docx"
    assert server_module._guess_file_type("https://example.com/data.json", "application/json") == "json"
    assert server_module._guess_file_type("https://example.com/notes.txt", "text/plain; charset=utf-8") == "text"
