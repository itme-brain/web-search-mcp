import os
import asyncio
import httpx
from fastmcp import FastMCP

mcp = FastMCP("Web Search")

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://searxng:8080")
CRAWL4AI_URL = os.environ.get("CRAWL4AI_URL", "http://crawl4ai:11235")
RERANKER_URL = os.environ.get("RERANKER_URL", "http://jina-reranker:8000")
MAX_SCRAPE = int(os.environ.get("MAX_SCRAPE", "5"))
SCRAPE_TIMEOUT = int(os.environ.get("SCRAPE_TIMEOUT", "30"))
MAX_CONTENT_CHARS = int(os.environ.get("MAX_CONTENT_CHARS", "4000"))


async def _search(query: str, num_results: int = 10, time_range: str | None = None) -> list[dict]:
    """Query SearXNG and return raw results."""
    params = {
        "q": query,
        "format": "json",
        "number_of_results": num_results,
    }
    if time_range:
        params["time_range"] = time_range.strip('"')

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{SEARXNG_URL}/search", params=params)
        resp.raise_for_status()
        return resp.json().get("results", [])


async def _scrape(url: str) -> str | None:
    """Scrape a URL via Crawl4AI and return markdown content."""
    try:
        async with httpx.AsyncClient(timeout=SCRAPE_TIMEOUT) as client:
            resp = await client.post(
                f"{CRAWL4AI_URL}/crawl",
                json={
                    "urls": url,
                    "priority": 8,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            # Handle async task response
            task_id = data.get("task_id")
            if task_id:
                for _ in range(SCRAPE_TIMEOUT):
                    await asyncio.sleep(1)
                    status_resp = await client.get(f"{CRAWL4AI_URL}/task/{task_id}")
                    status_data = status_resp.json()
                    if status_data.get("status") == "completed":
                        result = status_data.get("result", {})
                        return result.get("markdown", result.get("cleaned_html"))
                    elif status_data.get("status") == "failed":
                        return None
                return None

            # Handle sync response
            result = data.get("result", data)
            return result.get("markdown")
    except Exception:
        return None


async def _rerank(query: str, documents: list[str]) -> list[int]:
    """Rerank documents using Jina Reranker. Returns indices sorted by relevance."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
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
    except Exception:
        # Fallback: return original order
        return list(range(len(documents)))



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
) -> str:
    """Search the web, scrape top results, and return content ranked by relevance.

    Pipeline: SearXNG search -> Crawl4AI scrape -> Jina Reranker -> formatted output.

    Args:
        query: The search query.
        num_results: Number of search results to fetch (default 10).
        scrape_top: Number of top results to scrape for full content (default 5).
        time_range: Optional time filter: 'day', 'week', 'month', or 'year'.
    """
    # Step 1: Search
    results = await _search(query, num_results=num_results, time_range=time_range)
    if not results:
        return f"No results found for: {query}"

    # Step 2: Scrape top N concurrently
    to_scrape = min(scrape_top, len(results))
    scrape_tasks = [_scrape(r["url"]) for r in results[:to_scrape]]
    scraped = await asyncio.gather(*scrape_tasks)

    # Build content list for reranking
    entries = []
    for i, result in enumerate(results[:to_scrape]):
        content = scraped[i][:MAX_CONTENT_CHARS] if scraped[i] else result.get("content", "")
        entries.append({
            "title": result.get("title", "Untitled"),
            "url": result.get("url", ""),
            "content": content,
            "scraped": scraped[i] is not None,
        })

    # Add remaining results (not scraped) with snippets only
    for result in results[to_scrape:]:
        entries.append({
            "title": result.get("title", "Untitled"),
            "url": result.get("url", ""),
            "content": result.get("content", ""),
            "scraped": False,
        })

    # Step 3: Rerank by relevance
    documents = [e["content"] for e in entries]
    ranked_indices = await _rerank(query, documents)

    # Step 4: Format output in ranked order
    sections = []
    for rank, idx in enumerate(ranked_indices, 1):
        entry = entries[idx]
        section = f"## {rank}. [{entry['title']}]({entry['url']})\n"
        section += f"\n{entry['content']}\n"
        sections.append(section)

    header = f"# Search results for: {query}\n\n"
    return header + "\n---\n\n".join(sections)


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8000,
    )
