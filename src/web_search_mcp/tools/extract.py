"""Extract tool implementation."""

import asyncio
import logging
import time
import uuid

from web_search_mcp.storage import cache as cache_module
from web_search_mcp.common import _domain_from_url, _validate_urls
from web_search_mcp.extraction import documents as document_extractors
from web_search_mcp.presentation import models
from web_search_mcp import observability
from web_search_mcp.config.settings import _MAX_EXTRACT_URLS

log = logging.getLogger("web-search-mcp")


def _validated_response(model_cls, response: dict) -> dict:
    return models.dump_response(model_cls, response)

async def extract_impl(
    urls: list[str],
    chunk_ids: list[int] | None = None,
    observe: bool = True,
) -> dict:
    """Extract full cleaned documents with per-URL status reporting.

    Uses Crawl4AI for web pages and local fetch for text-like resources.
    Binary document formats are classified here and handed off via
    structured metadata rather than parsed locally.

    `chunk_ids` is an internal escape hatch for tests/debugging; public
    MCP extract always reads the document body.
    """
    urls = _validate_urls(urls, maximum=_MAX_EXTRACT_URLS)
    request_id = uuid.uuid4().hex
    if chunk_ids is not None and any(i < 0 for i in chunk_ids):
        raise ValueError("chunk_ids entries must be >= 0")
    started = time.monotonic()

    page_cache = cache_module.page_cache

    documents = await asyncio.gather(*[
        document_extractors._extract_url_document(
            url, page_cache,
            chunk_ids=chunk_ids,
        )
        for url in urls
    ])

    results: list[dict] = []
    urls_succeeded = 0
    urls_failed = 0
    for document in documents:
        if document["status"] == "ok":
            urls_succeeded += 1
        else:
            urls_failed += 1
        url = document["url"]
        content = document.get("content", "")
        total_chars = document.get("total_chars", len(content))
        entry = {
            "url": url,
            "domain": _domain_from_url(url),
            "status": document["status"],
            "content_type": document.get("content_type"),
            "file_type": document.get("file_type"),
            "title": document.get("title"),
            "content": content,
            "chars_shown": len(content),
            "total_chars": total_chars,
            "truncated": document.get("truncated", len(content) < total_chars),
            "total_chunks": document.get("total_chunks"),
            "shown_chunk_ids": document.get("shown_chunk_ids", []),
            "chunk_mode": document.get("chunk_mode"),
            "top_chunks": [
                c["text"] if isinstance(c, dict) else c
                for c in document.get("top_chunks", [])
            ],
            "chunks": document.get("chunks", []),
            "cached": document.get("cached", False),
            "error": document.get("error"),
        }
        metadata = document.get("metadata") or {}
        if metadata:
            entry["metadata"] = metadata
        results.append(entry)

    response = {
        "query": None,
        "results": results,
        "meta": {
            "request_id": request_id,
            "urls_requested": len(urls),
            "urls_succeeded": urls_succeeded,
            "urls_failed": urls_failed,
            "timings_ms": {
                "total": int((time.monotonic() - started) * 1000),
            },
        },
    }
    if observe:
        observability.observe_tool_response("extract", response)
    log.info(
        "request_id=%s extract requested=%d succeeded=%d failed=%d",
        request_id, len(urls), urls_succeeded, urls_failed,
    )
    return _validated_response(models.ExtractResponseModel, response)
