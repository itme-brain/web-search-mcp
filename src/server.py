"""MCP entry point: FastMCP instance, /health + /ready routes, and the five @mcp.tool wrappers.

Run with `python server.py` inside the container (WORKDIR /app, where
the sibling modules live).
"""

import asyncio

from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

# Module-qualified imports so `unittest.mock.patch("impls.X")` catches
# the calls made here — `from impls import X` would bind X locally and
# require a second patch target.
import cache
import impls
import models
import observability
import semantic
from core import (
    CRAWL4AI_URL,
    RERANK_NAME,
    RERANK_MODEL,
    SEARXNG_URL,
    _probe_dependency,
)
from formatters import (
    _format_crawl_results,
    _format_extract_results,
    _format_map_results,
    _format_search_results,
)
# Re-export impls as server-level attributes so `from server
# import search_impl` still works for Python scripters.
from impls import crawl_impl, extract_impl, map_impl, research_impl, search_impl  # noqa: F401


mcp = FastMCP("Web Search", version="0.6.3")

__all__ = ["mcp", "search_impl", "research_impl", "extract_impl", "map_impl", "crawl_impl"]


def _semantic_status() -> dict:
    """Return semantic-index config without importing heavy deps at startup."""
    try:
        import semantic
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
    return ToolResult(
        content=formatter(response),
        structured_content=response,
    )


# ---------------------------------------------------------------------------
# Container healthchecks
# ---------------------------------------------------------------------------
@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "reranker": {"name": RERANK_NAME, "model": RERANK_MODEL}})


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
        },
    }
    return JSONResponse(payload, status_code=200 if ready_ok else 503)


# ---------------------------------------------------------------------------
# MCP tools (thin wrappers: call impl → format → return ToolResult)
# ---------------------------------------------------------------------------
@mcp.tool(output_schema=models.SearchResponseModel.model_json_schema())
async def search(
    query: str,
    num_results: int = 5,
    time_range: str | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> ToolResult:
    """Find web sources with compact evidence. Use first.

    Args:
        query: Plain search text; use `site:domain.com` to limit a site.
        num_results: Sources to return. Use 3-5 normally.
        time_range: Optional: `day`, `week`, `month`, or `year`.
        include_domains: Keep only these bare domains.
        exclude_domains: Drop these bare domains.
    """
    response = await impls.search_impl(
        query=query,
        num_results=num_results,
        profile="search",
        time_range=time_range,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
    )
    return _tool_result(response, _format_search_results)


@mcp.tool(output_schema=models.ExtractResponseModel.model_json_schema())
async def extract(url: str) -> ToolResult:
    """Read one URL.

    Args:
        url: URL to read.
    """
    response = await impls.extract_impl(urls=[url], chunk_ids=None)
    return _tool_result(response, _format_extract_results)


@mcp.tool(output_schema=models.MapResponseModel.model_json_schema())
async def map(
    url: str,
    max_urls: int = 25,
) -> ToolResult:
    """List URLs on one site. Does not read page content.

    Args:
        url: Site/root URL.
        max_urls: URLs to return, 1-50.
    """
    response = await impls.map_impl(
        url=url,
        max_urls=max_urls,
        include_patterns=None,
    )
    return _tool_result(response, _format_map_results)


@mcp.tool(output_schema=models.SearchResponseModel.model_json_schema())
async def research(
    query: str,
    num_results: int = 8,
    time_range: str | None = None,
    source_types: list[str] | None = None,
) -> ToolResult:
    """Broader/slower search for hard questions.

    Args:
        query: Research question.
        num_results: Sources to return. Default 8.
        time_range: Optional: `day`, `week`, `month`, or `year`.
        source_types: Optional kinds to keep: docs, repo, issue, mailing_list, qa, blog, web.
    """
    response = await impls.research_impl(
        query=query,
        num_results=num_results,
        time_range=time_range,
        source_types=source_types,
    )
    return _tool_result(response, _format_search_results)


@mcp.tool(output_schema=models.CrawlResponseModel.model_json_schema())
async def crawl(
    url: str,
    query: str | None = None,
    max_urls: int = 10,
) -> ToolResult:
    """Read several pages from one site/docs tree.

    Args:
        url: Site/root URL.
        query: Optional focus question for ranking pages/chunks.
        max_urls: Pages to read, 1-20.
    """
    response = await impls.crawl_impl(
        url=url,
        max_urls=max_urls,
        include_patterns=None,
        query=query,
    )
    return _tool_result(response, _format_crawl_results)


for _tool in (search, extract, map, research, crawl):
    if not hasattr(_tool, "fn"):
        _tool.fn = _tool


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8000,
    )
