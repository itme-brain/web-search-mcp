"""PDF download and text extraction."""

from io import BytesIO

import httpx

from web_search_mcp.common import _normalize_url
from web_search_mcp.config.settings import MAX_PDF_BYTES, REQUEST_TIMEOUT, _DOWNLOAD_CHUNK_BYTES, _WHITESPACE
from web_search_mcp.extraction.file_types import _content_type_without_charset
from web_search_mcp.extraction.html import _detect_language
from web_search_mcp.http import policy as http_policy

PdfReader = None
def _clean_pdf_metadata_value(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _build_pdf_metadata(reader, content: str) -> dict:
    metadata: dict = {
        "page_count": len(reader.pages),
        "word_count": len(content.split()) if content else 0,
    }
    document_info = reader.metadata
    if document_info:
        for source_key, target_key in (
            ("title", "title"),
            ("author", "author"),
            ("subject", "description"),
            ("creator", "creator"),
            ("producer", "producer"),
        ):
            value = _clean_pdf_metadata_value(getattr(document_info, source_key, None))
            if value:
                metadata[target_key] = value
    language = _detect_language(content)
    if language:
        metadata["language"] = language
    return {k: v for k, v in metadata.items() if v is not None}

async def _download_document_bytes(url: str, *, max_bytes: int) -> tuple[bytes, str | None, str]:
    """Fetch a bounded binary document into memory for local extraction."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
        async with client.stream("GET", url, headers=http_policy.identity_headers()) as resp:
            resp.raise_for_status()
            content_length = resp.headers.get("content-length")
            if content_length is not None and int(content_length) > max_bytes:
                raise ValueError(f"document is too large for local extraction ({content_length} bytes)")

            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.aiter_bytes(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"document is too large for local extraction (>{max_bytes} bytes)")
                chunks.append(chunk)
            return (
                b"".join(chunks),
                _content_type_without_charset(resp.headers.get("content-type")),
                str(resp.url),
            )


def _get_pdf_reader_cls():
    """Import pypdf only when PDF extraction is actually used."""
    global PdfReader
    if PdfReader is None:
        from pypdf import PdfReader as _PdfReader
        PdfReader = _PdfReader
    return PdfReader


def _extract_pdf_markdown(reader) -> str:
    page_sections: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text(
                extraction_mode="layout",
                layout_mode_space_vertically=False,
            )
        except TypeError:
            page_text = page.extract_text()
        if not page_text:
            continue
        page_text = _WHITESPACE.sub(" ", page_text).strip()
        if page_text:
            page_sections.append(f"## Page {index}\n\n{page_text}")
    return "\n\n".join(page_sections)


async def _extract_pdf_document(url: str, content_type: str | None) -> dict:
    pdf_bytes, response_content_type, final_url = await _download_document_bytes(
        url, max_bytes=MAX_PDF_BYTES,
    )
    reader = _get_pdf_reader_cls()(BytesIO(pdf_bytes), strict=False)
    if reader.is_encrypted:
        try:
            decrypted = reader.decrypt("")
        except Exception:
            decrypted = 0
        if not decrypted:
            return {
                "status": "error",
                "url": url,
                "content_type": response_content_type or content_type or "application/pdf",
                "file_type": "pdf",
                "title": None,
                "content": "",
                "total_chars": 0,
                "metadata": {},
                "error": "encrypted pdf requires a password",
            }

    content = _extract_pdf_markdown(reader)
    if not content:
        return {
            "status": "error",
            "url": url,
            "content_type": response_content_type or content_type or "application/pdf",
            "file_type": "pdf",
            "title": None,
            "content": "",
            "total_chars": 0,
            "metadata": {"page_count": len(reader.pages)},
            "error": "no extractable text found in pdf",
        }

    metadata = _build_pdf_metadata(reader, content)
    if final_url and _normalize_url(final_url) != _normalize_url(url):
        metadata["final_url"] = final_url
    title = metadata.get("title")
    return {
        "status": "ok",
        "url": url,
        "content_type": response_content_type or content_type or "application/pdf",
        "file_type": "pdf",
        "title": title,
        "content": content,
        "total_chars": len(content),
        "metadata": metadata,
    }
