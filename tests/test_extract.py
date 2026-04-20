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
async def test_extract_urls_single_url_returns_markdown():
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
        payload = await server_module.extract_urls.fn(
            ["https://example.com/file.docx"],
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


def test_pdf_extract_all_pages_returns_per_page_chunks():
    import pymupdf
    doc = pymupdf.Document()
    for _ in range(5):
        doc.new_page(width=72, height=72)
    pdf_bytes = doc.tobytes()
    # Blank pages have no extractable text
    pages, title, total_pages = server_module._pdf_extract_all_pages(pdf_bytes)
    assert isinstance(pages, list)
    assert len(pages) == 0
    assert total_pages == 5


@pytest.mark.asyncio
async def test_extract_pdf_returns_per_page_chunks_with_metadata():
    fake_pages = [
        {"page": i, "content": f"Content for page {i}"}
        for i in range(1, 11)
    ]

    fake_resp = AsyncMock()
    fake_resp.content = b"fake pdf bytes"
    fake_resp.raise_for_status = lambda: None
    fake_client = AsyncMock()
    fake_client.get = AsyncMock(return_value=fake_resp)
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("web_search_server._pdf_extract_all_pages", return_value=(fake_pages, "Test PDF", 10)),
        patch("web_search_server.httpx.AsyncClient", return_value=fake_client),
    ):
        result = await server_module._extract_pdf_document("https://example.com/manual.pdf")

    assert result["title"] == "Test PDF"
    assert result["total_pages"] == 10
    assert result["pages_returned"] > 0
    assert "## Page 1" in result["content"]


@pytest.mark.asyncio
async def test_extract_pdf_reranks_pages_with_query():
    fake_pages = [
        {"page": i, "content": f"Content for page {i}"}
        for i in range(1, 6)
    ]

    fake_resp = AsyncMock()
    fake_resp.content = b"fake pdf bytes"
    fake_resp.raise_for_status = lambda: None
    fake_client = AsyncMock()
    fake_client.get = AsyncMock(return_value=fake_resp)
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=None)

    # Reranker returns pages in reverse relevance order
    def fake_rerank(_query, docs):
        return [(i, 1.0 - i * 0.1) for i in reversed(range(len(docs)))]

    with (
        patch("web_search_server._pdf_extract_all_pages", return_value=(fake_pages, "Test", 5)),
        patch("web_search_server.httpx.AsyncClient", return_value=fake_client),
        patch("web_search_server._rerank_scored", side_effect=fake_rerank),
    ):
        result = await server_module._extract_pdf_document(
            "https://example.com/manual.pdf",
            query="relevant content",
        )

    # Reranked content should have score annotations
    assert "(score:" in result["content"]
    assert result["pages_returned"] > 0


@pytest.mark.asyncio
async def test_extract_urls_surfaces_pdf_pagination_metadata():
    extract_mock = AsyncMock(return_value={
        "status": "ok",
        "url": "https://example.com/manual.pdf",
        "content_type": "application/pdf",
        "file_type": "pdf",
        "title": "Manual",
        "content": "## Page 1\n\nContent",
        "total_pages": 50,
        "pages_returned": 3,
        "top_chunks": [],
        "cached": False,
    })

    with patch(PATCH_EXTRACT_URL_DOCUMENT, extract_mock):
        payload = await server_module._extract_urls_impl(
            urls=["https://example.com/manual.pdf"],
        )

    result = payload["results"][0]
    assert result["total_pages"] == 50
    assert result["pages_returned"] == 3


def test_guess_file_type_supports_docx_and_text_formats():
    assert server_module._guess_file_type(
        "https://example.com/file.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ) == "docx"
    assert server_module._guess_file_type("https://example.com/data.json", "application/json") == "json"
    assert server_module._guess_file_type("https://example.com/notes.txt", "text/plain; charset=utf-8") == "text"
