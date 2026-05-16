"""Local and provider-backed document extraction."""

import mimetypes

import httpx
import magic

from web_search_mcp.common import _chunk_text, _normalize_url
from web_search_mcp.config.settings import _CHUNK_GAP, _MAX_EXTRACT_CONTENT_CHARS, _SNIFF_MAX_BYTES, _HTTP_TIMEOUT, REQUEST_TIMEOUT
from web_search_mcp.crawling.operations import _scrape
from web_search_mcp.extraction.file_types import _content_type_without_charset, _guess_file_type
from web_search_mcp.extraction.pdf import _extract_pdf_document
from web_search_mcp.extraction.providers import wikimedia
from web_search_mcp.http import policy as http_policy
from web_search_mcp.storage.cache import KVCache
from web_search_mcp.storage.pages import _page_entry, _page_get, _page_set
async def _head_content_type(url: str) -> str | None:
    try:
        async with httpx.AsyncClient(
            timeout=_HTTP_TIMEOUT,
            follow_redirects=True,
            headers=http_policy.browser_compatible_headers(),
        ) as client:
            resp = await client.head(url)
            resp.raise_for_status()
            return resp.headers.get("content-type")
    except httpx.HTTPError:
        return None


async def _sniff_content_type(url: str) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
            async with client.stream(
                "GET",
                url,
                headers={
                    **http_policy.identity_headers(),
                    "Range": f"bytes=0-{_SNIFF_MAX_BYTES - 1}",
                },
            ) as resp:
                resp.raise_for_status()
                if resp.status_code != httpx.codes.PARTIAL_CONTENT:
                    return None
                length = resp.headers.get("content-length")
                if length is not None and int(length) > _SNIFF_MAX_BYTES:
                    return None
                chunks: list[bytes] = []
                total = 0
                async for chunk in resp.aiter_bytes():
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > _SNIFF_MAX_BYTES:
                        return None
                    chunks.append(chunk)
            if not chunks:
                return None
            detected = magic.from_buffer(b"".join(chunks), mime=True)
            return _content_type_without_charset(detected)
    except (httpx.HTTPError, OSError, ValueError):
        return None


async def _detect_file_type(url: str) -> tuple[str, str | None]:
    header_content_type = await _head_content_type(url)
    content_type = _content_type_without_charset(header_content_type)
    guessed_content_type = _content_type_without_charset(mimetypes.guess_type(url)[0])
    if content_type in {None, "application/octet-stream"}:
        content_type = guessed_content_type or content_type
    if content_type in {None, "application/octet-stream"}:
        content_type = await _sniff_content_type(url) or content_type
    return _guess_file_type(url, content_type), content_type


# ---------------------------------------------------------------------------
# Per-file-type extractors
# ---------------------------------------------------------------------------
_LOCAL_EXTRACT_TYPES = {"text", "markdown", "json", "yaml", "xml", "csv"}


def _unsupported_file_document(url: str, file_type: str, content_type: str | None) -> dict:
    return {
        "status": "unsupported",
        "url": url,
        "content_type": content_type,
        "file_type": file_type,
        "title": None,
        "content": "",
        "total_chars": 0,
        "metadata": {},
        "error": f"local {file_type} extraction is not supported yet",
    }


async def _extract_web_document(url: str) -> dict:
    result = await _scrape(url)
    content = result["content"]
    if not content:
        return {
            "status": "error",
            "url": url,
            "content_type": "text/html",
            "file_type": "html",
            "title": None,
            "content": "",
            "total_chars": 0,
            "metadata": {},
            "error": "extraction failed",
        }
    return {
        "status": "ok",
        "url": url,
        "content_type": "text/html",
        "file_type": "html",
        "title": result.get("title"),
        "content": content,
        "total_chars": len(content),
        "metadata": result.get("metadata") or {},
    }


async def _extract_text_document(url: str, file_type: str) -> dict:
    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT,
        follow_redirects=True,
        headers=http_policy.browser_compatible_headers(accept="text/plain,text/*;q=0.9,*/*;q=0.5"),
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return {
            "status": "ok",
            "url": url,
            "content_type": _content_type_without_charset(resp.headers.get("content-type"))
            or mimetypes.guess_type(url)[0]
            or "text/plain",
            "file_type": file_type,
            "title": None,
            "content": resp.text,
            "total_chars": len(resp.text),
            "metadata": {"word_count": len(resp.text.split())} if resp.text else {},
        }

async def _prepare_document_content(
    content: str,
    chunk_ids: list[int] | None = None,
) -> tuple[str, list[dict], list[dict], list[int], str, bool]:
    """Prepare full-document extract display content."""
    chunks = [{"id": i, "text": text} for i, text in enumerate(_chunk_text(content))]

    if chunk_ids is not None:
        wanted = set(chunk_ids)
        selected = [c for c in chunks if c["id"] in wanted]
        display = _CHUNK_GAP.join(c["text"] for c in selected)
        return display, [], chunks, [c["id"] for c in selected], "selected", False

    display = content[:_MAX_EXTRACT_CONTENT_CHARS]
    truncated = len(content) > len(display)
    shown_ids = [c["id"] for c in chunks]
    return display, [], chunks, shown_ids, "document", truncated


async def _extract_url_document(
    url: str,
    cache: KVCache,
    chunk_ids: list[int] | None = None,
) -> dict:
    # Normalized URL is the cache key so www./trailing-slash variants
    # collapse. The stored entry keeps the caller's original URL for
    # display (see cached_entry["url"] below).
    cached = await _page_get(url, cache)
    if cached is not None:
        raw = cached.get("content") or ""
        content, top_chunks, chunks, shown_chunk_ids, chunk_mode, truncated = await _prepare_document_content(
            raw, chunk_ids=chunk_ids,
        )
        return {
            **cached,
            "content": content,
            "top_chunks": top_chunks,
            "chunks": chunks,
            "shown_chunk_ids": shown_chunk_ids,
            "total_chunks": len(chunks),
            "chunk_mode": chunk_mode,
            "truncated": truncated,
            "cached": True,
        }
    key = _normalize_url(url)

    file_type = "unknown"
    content_type = None
    extracted = await wikimedia.extract_document(url)
    if extracted is None:
        try:
            file_type, content_type = await _detect_file_type(url)
            if file_type == "html":
                extracted = await _extract_web_document(url)
            elif file_type == "pdf":
                extracted = await _extract_pdf_document(url, content_type)
            elif file_type in _LOCAL_EXTRACT_TYPES:
                extracted = await _extract_text_document(url, file_type)
            else:
                extracted = _unsupported_file_document(url, file_type, content_type)
        except Exception as exc:
            extracted = {
                "status": "error",
                "url": url,
                "content_type": content_type,
                "file_type": file_type,
                "title": None,
                "content": "",
                "total_chars": 0,
                "metadata": {},
                "error": str(exc),
            }

    if extracted["status"] == "ok":
        # Cache successful local extracts so repeated calls do not
        # re-sniff/reclassify the same resource.
        raw = extracted.get("content", "")
        total_chars = extracted.get("total_chars", len(raw))
        cached_entry = _page_entry(
            url=url,
            content=raw,
            title=extracted.get("title"),
            metadata=extracted.get("metadata") or {},
            content_type=extracted.get("content_type") or "text/html",
            file_type=extracted.get("file_type") or "html",
            status=extracted["status"],
        )
        # Preserve the upstream's total_chars (e.g. local text documents
        # that know their own length) rather than deriving from content.
        cached_entry["total_chars"] = total_chars
        await _page_set(url, cached_entry, cache)
        content, top_chunks, chunks, shown_chunk_ids, chunk_mode, truncated = await _prepare_document_content(
            raw, chunk_ids=chunk_ids,
        )
        extracted["content"] = content
        extracted["total_chars"] = total_chars
        extracted["top_chunks"] = top_chunks
        extracted["chunks"] = chunks
        extracted["shown_chunk_ids"] = shown_chunk_ids
        extracted["total_chunks"] = len(chunks)
        extracted["chunk_mode"] = chunk_mode
        extracted["truncated"] = truncated
        extracted["cached"] = False
        return extracted

    extracted.setdefault("total_chars", 0)
    extracted["top_chunks"] = []
    extracted["chunks"] = []
    extracted["shown_chunk_ids"] = []
    extracted["total_chunks"] = 0
    extracted["chunk_mode"] = None
    extracted["truncated"] = False
    extracted["cached"] = False
    return extracted
