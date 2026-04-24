"""MCP entry point: FastMCP instance, /health + /ready routes, and the
four @mcp.tool wrappers.

Run with `python server.py` inside the container (WORKDIR /app, where
the sibling modules live).
"""

import asyncio

from fastmcp import Context, FastMCP
from fastmcp.tools.tool import ToolResult
from starlette.requests import Request
from starlette.responses import JSONResponse

# Module-qualified imports so `unittest.mock.patch("impls.X")` catches
# the calls made here — `from impls import X` would bind X locally and
# require a second patch target.
import cache
import impls
import models
from core import (
    CRAWL4AI_URL,
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
# Re-export the four impls as server-level attributes so `from server
# import search_impl` still works for Python scripters.
from impls import crawl_impl, extract_impl, map_impl, search_impl  # noqa: F401


mcp = FastMCP("Web Search", version="0.3.1")

__all__ = ["mcp", "search_impl", "extract_impl", "map_impl", "crawl_impl"]


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
    return JSONResponse({"status": "ok", "reranker": {"name": "flashrank", "model": RERANK_MODEL}})


@mcp.custom_route("/metrics", methods=["GET"])
async def metrics(_: Request) -> JSONResponse:
    """Per-cache lifetime hit/miss counts.

    Plain INCR counters; no TTL. Reset by flushing Valkey.
    """
    page, searxng, seen = await asyncio.gather(
        cache.page_cache.stats(),
        cache.searxng_cache.stats(),
        cache.seen_urls.stats(),
    )
    return JSONResponse({
        "caches": {
            "page": page,
            "searxng": searxng,
            "seen_urls": seen,
        },
    })


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
            "reranker": {"status": "ok", "name": "flashrank", "model": RERANK_MODEL},
        },
    }
    return JSONResponse(payload, status_code=200 if ready_ok else 503)


# ---------------------------------------------------------------------------
# MCP tools (thin wrappers: call impl → format → return ToolResult)
# ---------------------------------------------------------------------------
@mcp.tool(output_schema=models.SearchResponseModel.model_json_schema())
async def search(
    query: str,
    time_range: str | None = None,
    language: str | None = "en",
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    ctx: Context | None = None,
 ) -> ToolResult:
    """Search the web.

    Args:
        query: Search query. Use normal search text. To scope to one site, prefix with `site:<domain>`.
        time_range: Optional recency filter. One of `day`, `week`, `month`, or `year`. Prefer this over putting dates in the query.
        language: Optional language code such as `en`, `de`, or `fr`. Pass `None` or `\"\"` for no language filter.
        include_domains: Optional bare domains to keep, such as `["docs.python.org"]`.
        exclude_domains: Optional bare domains to exclude.
    """
    response = await impls.search_impl(
        query=query,
        time_range=time_range,
        language=language,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        ctx=ctx,
    )
    return _tool_result(response, _format_search_results)


@mcp.tool(output_schema=models.ExtractResponseModel.model_json_schema())
async def extract(
    urls: list[str],
    query: str | None = None,
    chunk_ids: list[int] | None = None,
    ctx: Context | None = None,
) -> ToolResult:
    """Read content from one or more URLs.

    Args:
        urls: URLs to read. Always pass a list, even for one URL.
        query: Optional query used to return the most relevant chunks from each document.
        chunk_ids: Optional chunk ids from a prior response. When provided, returns those exact chunks instead of reranking.
    """
    response = await impls.extract_impl(
        urls=urls, query=query, chunk_ids=chunk_ids, ctx=ctx,
    )
    return _tool_result(response, _format_extract_results)


@mcp.tool(output_schema=models.MapResponseModel.model_json_schema())
async def map(
    url: str,
    max_urls: int = 25,
    include_patterns: list[str] | None = None,
) -> ToolResult:
    """Discover a site tree.

    Args:
        url: URL to use as the root of the returned tree.
        max_urls: Maximum URLs to include, from `1` to `50`.
        include_patterns: Optional shell-glob patterns matched against the full URL. Only matching URLs are kept.
    """
    response = await impls.map_impl(
        url=url,
        max_urls=max_urls,
        include_patterns=include_patterns,
    )
    return _tool_result(response, _format_map_results)


@mcp.tool(output_schema=models.CrawlResponseModel.model_json_schema())
async def crawl(
    url: str,
    max_urls: int = 10,
    include_patterns: list[str] | None = None,
) -> ToolResult:
    """Map a site tree and read its pages.

    Args:
        url: URL to use as the root of the crawl.
        max_urls: Maximum pages to include, from `1` to `20`.
        include_patterns: Optional shell-glob patterns matched against the full URL. Only matching URLs are kept.
    """
    response = await impls.crawl_impl(
        url=url,
        max_urls=max_urls,
        include_patterns=include_patterns,
    )
    return _tool_result(response, _format_crawl_results)


for _tool in (search, extract, map, crawl):
    if not hasattr(_tool, "fn"):
        _tool.fn = _tool


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8000,
    )
