import asyncio
import hashlib
import json
import logging
import os
import re
from collections import defaultdict

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
RERANK_TIMEOUT = int(os.environ.get("RERANK_TIMEOUT", "30"))
MAX_CONTENT_CHARS = int(os.environ.get("MAX_CONTENT_CHARS", "8000"))
CHUNK_MIN_CHARS = int(os.environ.get("CHUNK_MIN_CHARS", "50"))
CHUNK_MAX_CHARS = int(os.environ.get("CHUNK_MAX_CHARS", "500"))
MAX_CHUNKS_PER_PAGE = int(os.environ.get("MAX_CHUNKS_PER_PAGE", "10"))
TOP_CHUNKS_PER_PAGE = int(os.environ.get("TOP_CHUNKS_PER_PAGE", "3"))

STATE_SCRAPE_CACHE = "scrape_cache"
STATE_QUERY_CACHE = "query_cache"
STATE_SEEN_URLS = "seen_urls"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _query_cache_key(query: str, num_results: int, scrape_top: int, time_range: str | None) -> str:
    raw = json.dumps([query.lower().strip(), num_results, scrape_top, time_range], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _chunk_text(text: str) -> list[str]:
    """Split text into paragraph-sized chunks suitable for reranking."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= CHUNK_MAX_CHARS:
            if len(para) >= CHUNK_MIN_CHARS:
                chunks.append(para)
            continue
        sentences = _SENTENCE_SPLIT.split(para)
        current = ""
        for sent in sentences:
            if current and len(current) + len(sent) + 1 > CHUNK_MAX_CHARS:
                if len(current) >= CHUNK_MIN_CHARS:
                    chunks.append(current)
                current = sent
            else:
                current = f"{current} {sent}".strip() if current else sent
        if current and len(current) >= CHUNK_MIN_CHARS:
            chunks.append(current)
    return chunks[:MAX_CHUNKS_PER_PAGE]


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


async def _rerank_scored(query: str, documents: list[str]) -> list[tuple[int, float]]:
    """Rerank documents. Returns (index, score) pairs sorted by descending relevance."""
    if not documents:
        return []
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
            return [
                (r["index"], float(r["relevance_score"]))
                for r in sorted(results, key=lambda x: x["relevance_score"], reverse=True)
            ]
    except httpx.HTTPError as e:
        log.warning("rerank http error err=%r; falling back to original order", str(e))
    except (ValueError, KeyError) as e:
        log.warning("rerank payload error err=%s; falling back to original order", e)
    return [(i, 0.0) for i in range(len(documents))]


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

    Pipeline: SearXNG search -> Crawl4AI scrape -> chunk -> Jina Reranker -> formatted output.
    Scraped pages are split into paragraphs and reranked at the chunk level, so only
    the most query-relevant excerpts from each page are returned.
    Results are cached within the session.

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

    # --- build entries ---
    entries: list[dict] = []
    for i, result in enumerate(results[:to_scrape]):
        raw = scraped[i][:MAX_CONTENT_CHARS] if scraped[i] else None
        entries.append({
            "title": result.get("title", "Untitled"),
            "url": result.get("url", ""),
            "content": raw or result.get("content", ""),
            "scraped": raw is not None,
        })

    for result in results[to_scrape:]:
        entries.append({
            "title": result.get("title", "Untitled"),
            "url": result.get("url", ""),
            "content": result.get("content", ""),
            "scraped": False,
        })

    # --- chunk scraped pages, keep snippets as single chunks ---
    all_chunks: list[str] = []
    chunk_to_entry: list[int] = []
    for i, entry in enumerate(entries):
        if entry["scraped"] and entry["content"]:
            chunks = _chunk_text(entry["content"])
        else:
            chunks = [entry["content"]] if entry["content"] else []
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_to_entry.append(i)

    # --- rerank at the chunk level ---
    scored = await _rerank_scored(query, all_chunks)

    # --- group scores by entry, select top-K chunks per page ---
    entry_chunks: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for chunk_idx, score in scored:
        eidx = chunk_to_entry[chunk_idx]
        entry_chunks[eidx].append((all_chunks[chunk_idx], score))

    for eidx in entry_chunks:
        entry_chunks[eidx].sort(key=lambda x: x[1], reverse=True)
        entry_chunks[eidx] = entry_chunks[eidx][:TOP_CHUNKS_PER_PAGE]

    # --- rank pages by best chunk score ---
    entry_best: dict[int, float] = {
        eidx: chunks[0][1] for eidx, chunks in entry_chunks.items() if chunks
    }
    ranked_entry_idxs = sorted(entry_best, key=entry_best.get, reverse=True)
    for i in range(len(entries)):
        if i not in entry_best:
            ranked_entry_idxs.append(i)

    # --- format with novelty annotations ---
    sections: list[str] = []
    new_urls: list[str] = []
    for rank, eidx in enumerate(ranked_entry_idxs, 1):
        entry = entries[eidx]
        url = entry["url"]
        seen_tag = " *(previously seen)*" if url in seen_urls else ""

        top = entry_chunks.get(eidx, [])
        if top:
            body = "\n\n".join(chunk for chunk, _ in top)
        else:
            body = entry["content"]

        section = f"## {rank}. [{entry['title']}]({url}){seen_tag}\n\n{body}\n"
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

    total_chunks = len(all_chunks)
    log.info(
        "query=%r chunks=%d pages=%d scrape_cache=%d query_cache=%d seen=%d",
        query, total_chunks, len(entries), len(scrape_cache), len(query_cache), len(seen_urls),
    )

    return output


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8000,
    )
