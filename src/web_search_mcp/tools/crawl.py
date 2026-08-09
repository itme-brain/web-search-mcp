"""Crawl tool implementation."""

from collections import defaultdict
import logging
import time
import uuid
from urllib.parse import urlparse

from web_search_mcp.common import _coerce_optional_str, _dedup_pages, _validate_positive_int, _warning
from web_search_mcp.ranking.service import _rerank_scored
from web_search_mcp.presentation import models
from web_search_mcp import observability
from web_search_mcp.config.settings import _MAX_EXTRACT_URLS, _MAX_MAP_URLS
from web_search_mcp.tools.extract import extract_impl
from web_search_mcp.tools.map import map_impl

log = logging.getLogger("web-search-mcp")
_QUERY_DISCOVERY_MULTIPLIER = 4


def _validated_response(model_cls, response: dict) -> dict:
    return models.dump_response(model_cls, response)


async def _select_nodes_for_query(
    nodes: list[dict],
    *,
    query: str,
    maximum: int,
) -> tuple[list[dict], list[dict]]:
    """Rank cheap map metadata before spending the extraction budget."""
    if len(nodes) <= maximum:
        return nodes, []
    documents = [
        "\n".join(filter(None, (
            node.get("title"),
            node.get("link_text"),
            urlparse(node["url"]).path.replace("/", " "),
        )))
        for node in nodes
    ]
    try:
        scored = await _rerank_scored(query, documents)
    except Exception as exc:
        log.warning("crawl URL preselection failed query=%r err=%s", query, exc)
        return nodes[:maximum], [_warning("crawl_preselection_failed", "reranker", str(exc))]
    selected_indexes = [index for index, _score in scored[:maximum]]
    return [nodes[index] for index in selected_indexes], []

async def crawl_impl(
    url: str,
    max_urls: int = 10,
    include_patterns: list[str] | None = None,
    query: str | None = None,
) -> dict:
    """Discover a site tree, then extract content for each discovered node.

    When `query` is set, results are reordered by per-page best-chunk
    relevance score (configured cross-encoder) instead of BFS discovery
    order, and each result's `content` carries the joined top chunks
    rather than the document head.
    """
    effective_max_urls = _validate_positive_int(
        "max_urls",
        max_urls,
        maximum=min(_MAX_MAP_URLS, _MAX_EXTRACT_URLS),
    )
    request_id = uuid.uuid4().hex
    normalized_query = _coerce_optional_str(query)
    started = time.monotonic()
    discovery_limit = (
        min(effective_max_urls * _QUERY_DISCOVERY_MULTIPLIER, _MAX_MAP_URLS)
        if normalized_query else effective_max_urls
    )
    tree = await map_impl(
        url=url,
        max_urls=discovery_limit,
        include_patterns=include_patterns,
        observe=False,
    )
    root_url = tree["url"]
    selected_nodes = tree["results"]
    selection_warnings: list[dict] = []
    if normalized_query:
        selected_nodes, selection_warnings = await _select_nodes_for_query(
            selected_nodes,
            query=normalized_query,
            maximum=effective_max_urls,
        )
    urls = [entry["url"] for entry in selected_nodes]

    extracted = await extract_impl(
        urls=urls,
        chunk_ids=None,
        observe=False,
    )
    urls_succeeded = extracted["meta"]["urls_succeeded"]
    urls_failed = extracted["meta"]["urls_failed"]
    doc_by_url = {entry["url"]: entry for entry in extracted["results"]}

    score_by_url: dict[str, float | None] = {}
    if normalized_query:
        chunk_docs: list[str] = []
        chunk_to_url: list[str] = []
        for u, doc in doc_by_url.items():
            if doc.get("status") != "ok":
                continue
            chunks = doc.get("chunks") or []
            for chunk in chunks:
                text = chunk.get("text") if isinstance(chunk, dict) else None
                if text:
                    chunk_docs.append(text)
                    chunk_to_url.append(u)
        if chunk_docs:
            try:
                scored = await _rerank_scored(normalized_query, chunk_docs)
            except Exception as exc:
                log.warning("crawl rerank failed query=%r err=%s", normalized_query, exc)
                scored = []
            top_by_url: dict[str, list[dict]] = defaultdict(list)
            for chunk_idx, score in scored:
                u = chunk_to_url[chunk_idx]
                if score_by_url.get(u) is None or score > (score_by_url[u] or 0):
                    score_by_url[u] = score
                if len(top_by_url[u]) < 3:
                    top_by_url[u].append({"text": chunk_docs[chunk_idx], "score": score})
            for u, top in top_by_url.items():
                doc_by_url[u]["top_chunks"] = top

    results: list[dict] = []
    for node in selected_nodes:
        document = doc_by_url.get(node["url"], {})
        top_chunks_raw = document.get("top_chunks", []) or []
        top_chunks = [
            c["text"] if isinstance(c, dict) else c
            for c in top_chunks_raw
        ]
        full_content = document.get("content", "")
        content = (
            "\n\n[\u2026]\n\n".join(top_chunks)
            if normalized_query and top_chunks
            else full_content
        )
        merged = {
            "url": node["url"],
            "domain": node["domain"],
            "title": document.get("title") or node.get("title"),
            "link_text": node.get("link_text"),
            "depth": node["depth"],
            "discovered_from": node.get("discovered_from"),
            "link_type": node["link_type"],
            "status": document.get("status", "error"),
            "content_type": document.get("content_type"),
            "content": content,
            "chars_shown": len(content),
            "total_chars": document.get("total_chars", 0),
            "top_chunks": top_chunks,
            "cached": document.get("cached", False),
            "error": document.get("error"),
        }
        metadata = document.get("metadata") or {}
        if metadata:
            merged["metadata"] = metadata
        results.append(merged)

    results, urls_deduplicated = _dedup_pages(results)

    if normalized_query:
        # Pages with a real score sort by score desc; pages without one
        # (extract failures, no chunks) trail in stable order.
        results.sort(key=lambda r: (
            score_by_url.get(r["url"]) is None,
            -(score_by_url.get(r["url"]) or 0.0),
        ))

    for rank, entry in enumerate(results, start=1):
        entry["rank"] = rank

    warnings = [*tree["meta"].get("warnings", []), *selection_warnings]
    sparse = len(results) < min(effective_max_urls, 3) or urls_succeeded == 0
    sparsity_reason = None
    if urls_succeeded == 0:
        sparsity_reason = "No pages could be extracted."
    elif len(results) < min(effective_max_urls, 3):
        sparsity_reason = f"Only {len(results)} in-scope page(s) were discovered under this root."
    if sparsity_reason:
        warnings.append(_warning("crawl_sparse", "crawl", sparsity_reason))
    response = {
        "url": root_url,
        "query": normalized_query,
        "results": results,
        "meta": {
            "request_id": request_id,
            "max_urls_requested": effective_max_urls,
            "urls_discovered": len(tree["results"]),
            "urls_returned": len(results),
            "urls_truncated_by_limit": max(0, len(tree["results"]) - len(selected_nodes)),
            "urls_deduplicated": urls_deduplicated,
            "sparse": sparse,
            "sparsity_reason": sparsity_reason,
            "urls_succeeded": urls_succeeded,
            "urls_failed": urls_failed,
            "warnings": warnings,
            "timings_ms": {
                "total": int((time.monotonic() - started) * 1000),
            },
        },
    }
    observability.observe_tool_response("crawl", response)
    log.info(
        "request_id=%s crawl url=%s query=%r discovered=%d returned=%d dedup=%d succeeded=%d failed=%d",
        request_id,
        root_url,
        normalized_query,
        response["meta"]["urls_discovered"],
        len(results),
        response["meta"]["urls_deduplicated"],
        urls_succeeded,
        urls_failed,
    )
    return _validated_response(models.CrawlResponseModel, response)
