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
# Re-export the four impls as server-level attributes so `from server
# import search_impl` still works for Python scripters.
from impls import crawl_impl, extract_impl, map_impl, research_impl, search_impl  # noqa: F401


mcp = FastMCP("Web Search", version="0.3.2")

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
    page, searxng, seen, semantic_cache = await asyncio.gather(
        cache.page_cache.stats(),
        cache.searxng_cache.stats(),
        cache.seen_urls.stats(),
        __import__("semantic").stats(),
    )
    return JSONResponse({
        "caches": {
            "page": page,
            "searxng": searxng,
            "seen_urls": seen,
        },
        "semantic_cache": semantic_cache,
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
            "reranker": {"status": "ok", "name": RERANK_NAME, "model": RERANK_MODEL},
            "semantic_cache": _semantic_status(),
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
    ctx: Context | None = None,
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
async def extract(
    urls: list[str],
    query: str | None = None,
    ctx: Context | None = None,
) -> ToolResult:
    """Read known URLs. Use after search for more context.

    Args:
        urls: URL list, even for one URL.
        query: Optional focus question for relevant chunks.
    """
    response = await impls.extract_impl(
        urls=urls, query=query, chunk_ids=None,
    )
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
    ctx: Context | None = None,
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
