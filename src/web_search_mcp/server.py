"""MCP entry point, health routes, resources, and six tool wrappers.

Run with `python -m web_search_mcp.server` inside the container.
"""

import asyncio
from typing import Annotated, Literal

from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from fastmcp_tasks import TasksExtension
from mcp.types import ResourceLink, TextContent
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

from web_search_mcp.http import policy as http_policy
from web_search_mcp.storage import cache
from web_search_mcp.presentation import models
from web_search_mcp.preprocessing import lfm
from web_search_mcp import observability
from web_search_mcp.storage import semantic
from web_search_mcp.config.settings import (
    CRAWL4AI_URL,
    MAX_RESULTS,
    RERANK_MODEL,
    SEARXNG_URL,
)
from web_search_mcp.ranking.service import RERANK_NAME
from web_search_mcp.search_client import _probe_dependency
from web_search_mcp.presentation.formatters import (
    _format_crawl_results,
    _format_extract_results,
    _format_map_results,
    _format_search_results,
)
from web_search_mcp.tools.search import search_impl, research_impl
from web_search_mcp.tools.extract import extract_impl
from web_search_mcp.tools.map import map_impl
from web_search_mcp.tools.crawl import crawl_impl  # noqa: F401
from web_search_mcp.tools.evidence import read_evidence_impl
from web_search_mcp.storage import evidence as evidence_store


mcp = FastMCP("Web Search", version="0.8.2", strict_input_validation=True)
mcp.add_extension(TasksExtension())

Query = Annotated[str, Field(min_length=1, max_length=1000)]
Url = Annotated[str, Field(min_length=8, max_length=4096, pattern=r"^https?://")]
EvidenceReference = Annotated[str, Field(min_length=1, max_length=256)]
EvidenceChunkStart = Annotated[int, Field(ge=0, le=100_000)]
EvidenceChunkCount = Annotated[int, Field(ge=1, le=10)]
ResultCount = Annotated[int, Field(ge=1, le=MAX_RESULTS)]
MapCount = Annotated[int, Field(ge=1, le=50)]
CrawlCount = Annotated[int, Field(ge=1, le=20)]
TimeRange = Literal["day", "week", "month", "year"] | None
DomainList = Annotated[list[Annotated[str, Field(min_length=1, max_length=253)]], Field(max_length=20)] | None
SourceTypeList = Annotated[
    list[Literal["docs", "official_docs", "repo", "issue", "mailing_list", "qa", "blog", "pdf", "paper", "web"]],
    Field(max_length=10),
] | None
READ_ONLY_OPEN_WORLD = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,
}
READ_ONLY_CLOSED_WORLD = {**READ_ONLY_OPEN_WORLD, "openWorldHint": False}

__all__ = ["mcp", "search_impl", "research_impl", "extract_impl", "map_impl", "crawl_impl", "read_evidence_impl"]


def _semantic_status() -> dict:
    """Return semantic-index config without importing heavy deps at startup."""
    try:
        from web_search_mcp.storage import semantic
    except Exception as exc:
        return {"enabled": False, "status": "error", "detail": str(exc)}
    return {
        "enabled": semantic.ENABLED,
        "backend": semantic.BACKEND,
        "model": semantic.MODEL_NAME,
        "device": semantic.DEVICE,
        "top_k": semantic.TOP_K,
        "min_score": semantic.MIN_SCORE,
        "max_chunks_per_page": semantic.MAX_CHUNKS_PER_PAGE,
        "index_name": semantic._INDEX_NAME,
    }


def _tool_result(response: dict, formatter) -> ToolResult:
    """Return curated markdown plus the structured dict payload."""
    content = [TextContent(type="text", text=formatter(response))]
    linked_uris: set[str] = set()
    passage_links = 0
    for result in response.get("results", []):
        resource_uri = result.get("resource_uri")
        if resource_uri and resource_uri not in linked_uris:
            content.append(ResourceLink(
                type="resource_link",
                uri=resource_uri,
                name=result.get("title") or result.get("document_id") or "retrieved document",
                description="Complete cleaned document already retrieved by web-search-mcp",
                mimeType="text/markdown",
            ))
            linked_uris.add(resource_uri)
        for passage in result.get("passages", []):
            passage_uri = passage.get("resource_uri")
            if (
                not passage_uri
                or passage_uri in linked_uris
                or passage_links >= 12
            ):
                continue
            passage_name = passage.get("citation") or passage_links + 1
            content.append(ResourceLink(
                type="resource_link",
                uri=passage_uri,
                name=(
                    f"{result.get('title') or 'retrieved document'} "
                    f"passage {passage_name}"
                ),
                description="Focused evidence passage already retrieved by web-search-mcp",
                mimeType="text/markdown",
            ))
            linked_uris.add(passage_uri)
            passage_links += 1
    return ToolResult(
        content=content,
        structured_content=response,
        meta={
            "dev.web-search/request-id": response.get("meta", {}).get("request_id"),
            "dev.web-search/timings-ms": response.get("meta", {}).get("timings_ms", {}),
            "dev.web-search/cache": {
                "semantic_hits": response.get("meta", {}).get("semantic_hits", 0),
            },
        },
    )


@mcp.resource("web-search://documents/{document_id}", mime_type="text/markdown")
async def retrieved_document(document_id: str) -> str:
    """Read a complete document that was already retrieved and persisted."""
    record = await evidence_store.get_document(document_id)
    if record is None:
        raise ValueError("retrieved document not found or expired")
    return record["content"]


@mcp.resource("web-search://chunks/{chunk_id}", mime_type="text/markdown")
async def retrieved_chunk(chunk_id: str) -> str:
    """Read one targeted chunk that was already retrieved and persisted."""
    record = await evidence_store.get_chunk(chunk_id)
    if record is None:
        raise ValueError("retrieved chunk not found or expired")
    return record["text"]


# ---------------------------------------------------------------------------
# Container healthchecks
# ---------------------------------------------------------------------------
@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "reranker": {"name": RERANK_NAME, "model": RERANK_MODEL},
        "preprocessing": lfm.status(),
    })


@mcp.custom_route("/metrics", methods=["GET"])
async def metrics(_: Request) -> JSONResponse:
    """Per-cache lifetime hit/miss counts.

    Plain INCR counters; no TTL. Reset by flushing Valkey.
    """
    page, searxng, seen, page_memory, semantic_index = await asyncio.gather(
        cache.page_cache.stats(),
        cache.searxng_cache.stats(),
        cache.seen_urls.stats(),
        cache.page_memory_cache.stats(),
        semantic.stats(),
    )
    return JSONResponse({
        "caches": {
            "page": page,
            "searxng": searxng,
            "seen_urls": seen,
            "page_memory": page_memory,
        },
        "semantic_index": semantic_index,
    })


@mcp.custom_route("/metrics/prometheus", methods=["GET"])
async def prometheus_metrics(_: Request) -> PlainTextResponse:
    """Prometheus text metrics for tool/stage observability."""
    page, searxng, seen, page_memory, semantic_index = await asyncio.gather(
        cache.page_cache.stats(),
        cache.searxng_cache.stats(),
        cache.seen_urls.stats(),
        cache.page_memory_cache.stats(),
        semantic.stats(),
    )
    gauges = {
        "web_search_mcp_cache_hits": (page["hits"], {"cache": "page"}),
        "web_search_mcp_cache_misses": (page["misses"], {"cache": "page"}),
        "web_search_mcp_searxng_cache_hits": (searxng["hits"], {"cache": "searxng"}),
        "web_search_mcp_searxng_cache_misses": (searxng["misses"], {"cache": "searxng"}),
        "web_search_mcp_seen_url_cache_hits": (seen["hits"], {"cache": "seen_urls"}),
        "web_search_mcp_seen_url_cache_misses": (seen["misses"], {"cache": "seen_urls"}),
        "web_search_mcp_page_memory_cache_hits": (page_memory["hits"], {"cache": "page_memory"}),
        "web_search_mcp_page_memory_cache_misses": (page_memory["misses"], {"cache": "page_memory"}),
        "web_search_mcp_semantic_index_ready": (1 if semantic_index.get("index_ready") else 0, {}),
        "web_search_mcp_semantic_indexed_chunks": (semantic_index.get("indexed_chunks", 0), {}),
    }
    return PlainTextResponse(
        observability.prometheus_text(gauges),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@mcp.custom_route("/ready", methods=["GET"])
async def ready(_: Request) -> JSONResponse:
    searxng = await _probe_dependency(f"{SEARXNG_URL}/healthz")
    crawl4ai = await _probe_dependency(f"{CRAWL4AI_URL}/health")
    valkey_ok = await cache.ping()
    valkey = {"status": "ok"} if valkey_ok else {"status": "error"}
    ready_ok = (
        searxng["status"] == "ok"
        and crawl4ai["status"] == "ok"
        and valkey_ok
    )
    payload = {
        "status": "ok" if ready_ok else "degraded",
        "dependencies": {
            "searxng": searxng,
            "crawl4ai": crawl4ai,
            "valkey": valkey,
            "reranker": {"status": "ok", "name": RERANK_NAME, "model": RERANK_MODEL},
            "semantic_index": _semantic_status(),
            "preprocessing": lfm.status(),
        },
    }
    return JSONResponse(payload, status_code=200 if ready_ok else 503)


# ---------------------------------------------------------------------------
# MCP tools (thin wrappers: call impl → format → return ToolResult)
# ---------------------------------------------------------------------------
@mcp.tool(output_schema=models.SearchResponseModel.model_json_schema(), annotations=READ_ONLY_OPEN_WORLD)
async def search(
    query: Query,
    num_results: ResultCount = 5,
    time_range: TimeRange = None,
    include_domains: DomainList = None,
    exclude_domains: DomainList = None,
) -> ToolResult:
    """Find web sources with compact evidence. Use first.

    Args:
        query: Plain search text; use `site:domain.com` to limit a site.
        num_results: Sources to return. Use 3-5 normally.
        time_range: Optional: `day`, `week`, `month`, or `year`.
        include_domains: Keep only these bare domains.
        exclude_domains: Drop these bare domains.
    """
    response = await search_impl(
        query=query,
        num_results=num_results,
        profile="search",
        time_range=time_range,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
    )
    return _tool_result(response, _format_search_results)


@mcp.tool(output_schema=models.ExtractResponseModel.model_json_schema(), annotations=READ_ONLY_OPEN_WORLD)
async def extract(url: Url) -> ToolResult:
    """Read one URL.

    Args:
        url: URL to read.
    """
    response = await extract_impl(urls=[url], chunk_ids=None)
    return _tool_result(response, _format_extract_results)


@mcp.tool(output_schema=models.MapResponseModel.model_json_schema(), annotations=READ_ONLY_OPEN_WORLD)
async def map(
    url: Url,
    max_urls: MapCount = 25,
) -> ToolResult:
    """List URLs on one site. Does not read page content.

    Args:
        url: Site/root URL.
        max_urls: URLs to return, 1-50.
    """
    response = await map_impl(
        url=url,
        max_urls=max_urls,
        include_patterns=None,
    )
    return _tool_result(response, _format_map_results)


@mcp.tool(
    output_schema=models.SearchResponseModel.model_json_schema(),
    annotations=READ_ONLY_OPEN_WORLD,
    task=True,
)
async def research(
    query: Query,
    num_results: ResultCount = 8,
    time_range: TimeRange = None,
    source_types: SourceTypeList = None,
) -> ToolResult:
    """Broader/slower search for hard questions.

    Args:
        query: Research question.
        num_results: Sources to return. Default 8.
        time_range: Optional: `day`, `week`, `month`, or `year`.
        source_types: Optional kinds to keep: docs, official_docs, repo, issue,
            mailing_list, qa, blog, pdf, paper, or web.
    """
    response = await research_impl(
        query=query,
        num_results=num_results,
        time_range=time_range,
        source_types=source_types,
    )
    return _tool_result(response, _format_search_results)


@mcp.tool(
    output_schema=models.CrawlResponseModel.model_json_schema(),
    annotations=READ_ONLY_OPEN_WORLD,
    task=True,
)
async def crawl(
    url: Url,
    query: Query | None = None,
    max_urls: CrawlCount = 10,
) -> ToolResult:
    """Read several pages from one site/docs tree.

    Args:
        url: Site/root URL.
        query: Optional focus question for ranking pages/chunks.
        max_urls: Pages to read, 1-20.
    """
    response = await crawl_impl(
        url=url,
        max_urls=max_urls,
        include_patterns=None,
        query=query,
    )
    return _tool_result(response, _format_crawl_results)


@mcp.tool(
    output_schema=models.EvidenceReadResponseModel.model_json_schema(),
    annotations=READ_ONLY_CLOSED_WORLD,
)
async def read_evidence(
    reference: EvidenceReference,
    chunk_start: EvidenceChunkStart | None = None,
    max_chunks: EvidenceChunkCount | None = None,
) -> ToolResult:
    """Expand cached evidence, optionally reading a bounded document chunk range.

    Args:
        reference: Document or chunk reference returned by search/research.
        chunk_start: Optional zero-based document chunk offset. Supplying it
            enables a bounded read.
        max_chunks: Optional document chunks to return, 1-10. Supplying it
            starts at chunk zero by default.
    """
    response = await read_evidence_impl(
        reference,
        chunk_start=chunk_start,
        max_chunks=max_chunks,
    )
    content = [TextContent(type="text", text=response["content"])]
    linked_uris = response["chunk_resource_uris"] or [response["resource_uri"]]
    for index, resource_uri in enumerate(linked_uris):
        content.append(ResourceLink(
            type="resource_link",
            uri=resource_uri,
            name=(
                f"{response.get('title') or response['document_id']} chunk "
                f"{(response.get('chunk_start') or 0) + index}"
                if response["chunk_resource_uris"]
                else (
                    response.get("title")
                    or response.get("chunk_id")
                    or response["document_id"]
                )
            ),
            mimeType="text/markdown",
        ))
    return ToolResult(
        content=content,
        structured_content=response,
    )


for _tool in (search, extract, map, research, crawl, read_evidence):
    if not hasattr(_tool, "fn"):
        _tool.fn = _tool


if __name__ == "__main__":
    # Initialize dynamic Firefox UA at startup so all tools use the latest version string
    asyncio.run(http_policy.ensure_user_agent_initialized())
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8000,
        stateless_http=True,
    )
