import asyncio
import hashlib
import json
import logging
import os

import httpx
from fastmcp import Context, FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("web-search-mcp")

mcp = FastMCP("Web Search")

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://searxng:8080")
CRAWL4AI_URL = os.environ.get("CRAWL4AI_URL", "http://crawl4ai:11235")
RERANKER_URL = os.environ.get("RERANKER_URL", "http://jina-reranker:8000")
MAX_SCRAPE = int(os.environ.get("MAX_SCRAPE", "5"))
SCRAPE_TIMEOUT = int(os.environ.get("SCRAPE_TIMEOUT", "30"))
SCRAPE_HTTP_TIMEOUT = int(os.environ.get("SCRAPE_HTTP_TIMEOUT", "15"))
SEARCH_TIMEOUT = int(os.environ.get("SEARCH_TIMEOUT", "10"))
RERANK_TIMEOUT = int(os.environ.get("RERANK_TIMEOUT", "10"))
MAX_CONTENT_CHARS = int(os.environ.get("MAX_CONTENT_CHARS", "4000"))

STATE_SCRAPE_CACHE = "scrape_cache"
STATE_QUERY_CACHE = "query_cache"
STATE_SEEN_URLS = "seen_urls"


def _query_cache_key(query: str, num_results: int, scrape_top: int, time_range: str | None) -> str:
    raw = json.dumps([query.lower().strip(), num_results, scrape_top, time_range], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


async def _search(query: str, num_results: int = 10, time_range: str | None = None) -> list[dict]:
    """Query SearXNG and return raw results."""
    params = {
        "q": query,
        "format": "json",
        "number_of_results": num_results,
    }
    if time_range:
        params["time_range"] = time_range.strip('"')

    async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as client:
        resp = await client.get(f"{SEARXNG_URL}/search", params=params)
        resp.raise_for_status()
        return resp.json().get("results", [])


async def _scrape_impl(url: str) -> str | None:
    async with httpx.AsyncClient(timeout=SCRAPE_HTTP_TIMEOUT) as client:
        resp = await client.post(
            f"{CRAWL4AI_URL}/crawl",
            json={"urls": url, "priority": 8},
        )
        resp.raise_for_status()
        data = resp.json()

        task_id = data.get("task_id")
        if task_id:
            while True:
                await asyncio.sleep(1)
                status_resp = await client.get(f"{CRAWL4AI_URL}/task/{task_id}")
                status_resp.raise_for_status()
                status_data = status_resp.json()
                status = status_data.get("status")
                if status == "completed":
                    result = status_data.get("result", {})
                    return result.get("markdown") or result.get("cleaned_html")
                if status == "failed":
                    return None

        result = data.get("result", data)
        return result.get("markdown")


async def _scrape(url: str) -> str | None:
    """Scrape a URL via Crawl4AI, bounded by SCRAPE_TIMEOUT seconds end-to-end."""
    try:
        return await asyncio.wait_for(_scrape_impl(url), timeout=SCRAPE_TIMEOUT)
    except asyncio.TimeoutError:
        log.warning("scrape timed out url=%s budget=%ss", url, SCRAPE_TIMEOUT)
    except httpx.HTTPError as e:
        log.warning("scrape http error url=%s err=%s", url, e)
    except (ValueError, KeyError) as e:
        log.warning("scrape payload error url=%s err=%s", url, e)
    return None


async def _scrape_cached(url: str, cache: dict[str, str | None]) -> str | None:
    """Scrape with per-session cache. Returns cached content on hit, scrapes on miss."""
    if url in cache:
        log.debug("scrape cache hit url=%s", url)
        return cache[url]
    content = await _scrape(url)
    cache[url] = content
    return content


async def _rerank(query: str, documents: list[str]) -> list[int]:
    """Rerank documents using Jina Reranker. Returns indices sorted by relevance; falls back to original order on failure."""
    try:
        async with httpx.AsyncClient(timeout=RERANK_TIMEOUT) as client:
            resp = await client.post(
                f"{RERANKER_URL}/v1/rerank",
                json={
                    "query": query,
                    "documents": documents,
                    "top_n": len(documents),
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            return [r["index"] for r in sorted(results, key=lambda x: x["relevance_score"], reverse=True)]
    except httpx.HTTPError as e:
        log.warning("rerank http error err=%s; falling back to original order", e)
    except (ValueError, KeyError) as e:
        log.warning("rerank payload error err=%s; falling back to original order", e)
    return list(range(len(documents)))


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


@mcp.prompt()
def web_search_agent(topic: str) -> str:
    """Use web search to research a topic and provide an informed answer."""
    return f"""You have access to a web_search tool that searches the internet, scrapes pages, and returns relevant content ranked by relevance.

Use web_search to find current, accurate information about: {topic}

Guidelines:
- Use specific, targeted search queries
- Use time_range='day' or 'week' for recent events
- Synthesize information from multiple results
- Always cite sources with URLs
- If results are insufficient, refine your query and search again"""


@mcp.tool
async def web_search(
    query: str,
    num_results: int = 10,
    scrape_top: int = MAX_SCRAPE,
    time_range: str | None = None,
    ctx: Context | None = None,
) -> str:
    """Search the web, scrape top results, and return content ranked by relevance.

    Pipeline: SearXNG search -> Crawl4AI scrape -> Jina Reranker -> formatted output.
    Results are cached within the session — repeated or overlapping queries reuse
    previously scraped content instead of re-fetching.

    Args:
        query: The search query.
        num_results: Number of search results to fetch (default 10).
        scrape_top: Number of top results to scrape for full content (default 5).
        time_range: Optional time filter: 'day', 'week', 'month', or 'year'.
    """
    # --- session state ---
    scrape_cache: dict[str, str | None] = {}
    query_cache: dict[str, str] = {}
    seen_urls: set[str] = set()
    if ctx:
        scrape_cache = await ctx.get_state(STATE_SCRAPE_CACHE) or {}
        query_cache = await ctx.get_state(STATE_QUERY_CACHE) or {}
        seen_urls = set(await ctx.get_state(STATE_SEEN_URLS) or [])

    # --- exact query cache ---
    qkey = _query_cache_key(query, num_results, scrape_top, time_range)
    if qkey in query_cache:
        log.info("query cache hit query=%r", query)
        return query_cache[qkey]

    # --- search ---
    results = await _search(query, num_results=num_results, time_range=time_range)
    if not results:
        return f"No results found for: {query}"

    # --- scrape (cache-aware) ---
    to_scrape = min(scrape_top, len(results))
    scrape_tasks = [_scrape_cached(r["url"], scrape_cache) for r in results[:to_scrape]]
    scraped = await asyncio.gather(*scrape_tasks)

    entries = []
    for i, result in enumerate(results[:to_scrape]):
        content = scraped[i][:MAX_CONTENT_CHARS] if scraped[i] else result.get("content", "")
        entries.append({
            "title": result.get("title", "Untitled"),
            "url": result.get("url", ""),
            "content": content,
            "scraped": scraped[i] is not None,
        })

    for result in results[to_scrape:]:
        entries.append({
            "title": result.get("title", "Untitled"),
            "url": result.get("url", ""),
            "content": result.get("content", ""),
            "scraped": False,
        })

    # --- rerank ---
    documents = [e["content"] for e in entries]
    ranked_indices = await _rerank(query, documents)

    # --- format with novelty annotations ---
    sections = []
    new_urls: list[str] = []
    for rank, idx in enumerate(ranked_indices, 1):
        entry = entries[idx]
        url = entry["url"]
        seen_tag = " *(previously seen)*" if url in seen_urls else ""
        section = f"## {rank}. [{entry['title']}]({url}){seen_tag}\n"
        section += f"\n{entry['content']}\n"
        sections.append(section)
        new_urls.append(url)

    header = f"# Search results for: {query}\n\n"
    output = header + "\n---\n\n".join(sections)

    # --- persist session state ---
    seen_urls.update(new_urls)
    query_cache[qkey] = output
    if ctx:
        await ctx.set_state(STATE_SCRAPE_CACHE, scrape_cache)
        await ctx.set_state(STATE_QUERY_CACHE, query_cache)
        await ctx.set_state(STATE_SEEN_URLS, list(seen_urls))

    cache_stats = f"{len(scrape_cache)} URLs cached, {len(query_cache)} queries cached, {len(seen_urls)} URLs seen"
    log.info("query=%r %s", query, cache_stats)

    return output


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8000,
    )
