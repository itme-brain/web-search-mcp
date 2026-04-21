"""MCP entry point: FastMCP instance, /health + /ready routes, and the
four @mcp.tool wrappers.

Run with `python server.py` inside the container (WORKDIR /app, where
the sibling modules live).
"""

from fastmcp import Context, FastMCP
from fastmcp.tools.tool import ToolResult
from starlette.requests import Request
from starlette.responses import JSONResponse

# Module-qualified imports so `unittest.mock.patch("impls.X")` catches
# the calls made here — `from impls import X` would bind X locally and
# require a second patch target.
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


mcp = FastMCP("Web Search", version="0.1.0")

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


@mcp.custom_route("/ready", methods=["GET"])
async def ready(_: Request) -> JSONResponse:
    searxng = await _probe_dependency(f"{SEARXNG_URL}/healthz")
    crawl4ai = await _probe_dependency(f"{CRAWL4AI_URL}/health")
    ready_ok = searxng["status"] == "ok" and crawl4ai["status"] == "ok"
    payload = {
        "status": "ok" if ready_ok else "degraded",
        "dependencies": {
            "searxng": searxng,
            "crawl4ai": crawl4ai,
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
    num_results: int = 10,
    time_range: str | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    ctx: Context | None = None,
 ) -> ToolResult:
    """Search the web, scrape and rerank the top results.

    Args:
        query: Search query. Prefix with `site:<domain>` to scope to one site.
        num_results: How many results to return (1-10).
        time_range: Recency filter — `day` / `week` / `month` / `year`. Omit for no filter. Do NOT put dates or years in the query text; use this instead.
        include_domains: Keep only results from these bare domains (e.g. `["docs.python.org"]`).
        exclude_domains: Drop results from these bare domains.
    """
    response = await impls.search_impl(
        query=query,
        num_results=num_results,
        time_range=time_range,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        ctx=ctx,
    )
    return _tool_result(response, _format_search_results)


@mcp.tool(output_schema=models.ExtractResponseModel.model_json_schema())
async def extract(
    urls: list[str],
    query: str | None = None,
    offset: int = 0,
    ctx: Context | None = None,
) -> ToolResult:
    """Fetch full content for one or more URLs.

    Args:
        urls: URLs to fetch. Always a list — pass `["https://..."]` for one URL.
        query: Optional. Reranks the document's chunks and returns the top matches. Omit to get raw content from the top of the document.
        offset: Byte offset into the document. When the output footer says `N of M chars shown — pass offset=N to continue`, call again with that `offset` to read the next slice. `offset > 0` bypasses `query` reranking in favor of raw continuation.
    """
    response = await impls.extract_impl(urls=urls, query=query, offset=offset, ctx=ctx)
    return _tool_result(response, _format_extract_results)


@mcp.tool(output_schema=models.MapResponseModel.model_json_schema())
async def map(
    url: str,
    max_urls: int = 25,
    max_depth: int = 1,
    include_patterns: list[str] | None = None,
    same_domain_only: bool = True,
) -> ToolResult:
    """Discover URLs on a site — link graph, no body content.

    Args:
        url: Root URL to start discovery from.
        max_urls: Maximum URLs to discover (1-50).
        max_depth: How many link hops to follow from the root (1-2).
        include_patterns: Shell-glob patterns against the full URL; keep only matches (e.g. `["https://docs.example.com/api/*"]`).
        same_domain_only: If True, restrict to the root's registrable domain (e.g. `docs.pydantic.dev` and `pydantic.dev` count as same). If False, follow every in-scope link.
    """
    response = await impls.map_impl(
        url=url,
        max_urls=max_urls,
        max_depth=max_depth,
        include_patterns=include_patterns,
        same_domain_only=same_domain_only,
    )
    return _tool_result(response, _format_map_results)


@mcp.tool(output_schema=models.CrawlResponseModel.model_json_schema())
async def crawl(
    url: str,
    query: str | None = None,
    max_urls: int = 10,
    max_depth: int = 1,
    include_patterns: list[str] | None = None,
    same_domain_only: bool = True,
    ctx: Context | None = None,
) -> ToolResult:
    """Discover URLs on a site and fetch their content.

    Args:
        url: Root URL to start crawl from.
        query: Optional. Reranks discovered pages by best-chunk relevance so the most useful pages appear first.
        max_urls: Maximum pages to crawl (1-20).
        max_depth: How many link hops to follow from the root (1-2).
        include_patterns: Shell-glob patterns against the full URL; keep only matches.
        same_domain_only: If True, restrict to the root's registrable domain. If False, follow every in-scope link.
    """
    response = await impls.crawl_impl(
        url=url,
        query=query,
        max_urls=max_urls,
        max_depth=max_depth,
        include_patterns=include_patterns,
        same_domain_only=same_domain_only,
        ctx=ctx,
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
