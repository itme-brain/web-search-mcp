# web-search-mcp

A self-contained MCP server that gives any MCP-compatible model (Claude, local llama.cpp, etc.) the same kind of search pipeline that paid "answer engines" use — fully FOSS, fully self-hosted.

Pipeline:

```
query -> SearXNG -> Crawl4AI (scrape top N) -> FlashRank -> ranked JSON
```

All services run in one `docker compose` project. Only the MCP server port is exposed on the host; SearXNG and Crawl4AI stay internal, and reranking happens inside the MCP server process via FlashRank.

## Requirements

- Docker (daemon must be running on the host — Nix cannot provide this on non-NixOS)
- Either Nix (recommended — gives you the full toolchain with one command) or host-installed `docker compose`, `just`, `openssl`
- First boot downloads the FlashRank model into the MCP container image cache. Allow extra startup time the first time the MCP server initializes the reranker.

## Quick start with Nix (works on NixOS and non-NixOS Linux/macOS)

```sh
git clone <this-repo> web-search-mcp
cd web-search-mcp

# One-shot: renders settings.yml with a random secret, builds images,
# starts the stack, and waits for the MCP /ready probe to succeed.
nix run .#deploy

# Or drop into a devshell and use just recipes.
nix develop
just up
just test
just eval
just logs
```

To stop:

```sh
nix run .#teardown          # same as `just down`
nix run .#teardown -- -v    # also wipe volumes (reranker cache, searxng cache)
```

## Quick start without Nix

```sh
git clone <this-repo> web-search-mcp
cd web-search-mcp
just setup                  # renders .env + searxng/config/settings.yml with a random secret
docker compose up -d --build
nix develop -c pytest -q    # or `just test`
```

## Hooking your MCP client into it

The server speaks MCP over streamable HTTP at `http://<host>:${MCP_HOST_PORT:-8002}/mcp`.

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "web-search": {
      "type": "streamable-http",
      "url": "http://localhost:8002/mcp"
    }
  }
}
```

### Claude Code (CLI)

```sh
claude mcp add --transport http web-search http://localhost:8002/mcp
```

### Cline / Continue / Cursor / OpenWebUI

All of these accept a streamable-HTTP MCP entry. Use the same URL as above; the exact JSON key differs per client (`"transport": "http"`, `"type": "streamable-http"`, etc.) — consult your client's docs.

### Generic MCP client (Python, for scripting)

```python
import asyncio
from fastmcp import Client

async def main():
    async with Client("http://localhost:8002/mcp") as c:
        result = await c.call_tool("web_search", {"query": "current LLM context window records", "num_results": 5})
        print(result.data)

asyncio.run(main())
```

## Configuration

`env.sample` is copied to `.env` by `just setup` / `nix run .#deploy`. Knobs:

| Var | Default | Purpose |
|---|---|---|
| `MCP_HOST_PORT` | `8002` | Host port the MCP server binds to |
| `MAX_NUM_RESULTS` | `10` | Hard upper bound accepted by the MCP tool |
| `MAX_SCRAPE` | `5` | How many top results to scrape for full content |
| `MAX_SCRAPE_TOP` | `5` | Hard upper bound accepted by the MCP tool |
| `MAX_EXTRACT_URLS` | `20` | Hard upper bound accepted by `extract_urls` |
| `MAX_MAP_URLS` | `50` | Hard upper bound accepted by `map_site` |
| `MAX_MAP_DEPTH` | `2` | Hard upper bound accepted by `map_site` |
| `SCRAPE_TIMEOUT` | `30` | Total per-URL scrape budget (seconds, end-to-end) |
| `SCRAPE_HTTP_TIMEOUT` | `15` | Per-request HTTP timeout inside the scrape loop |
| `SEARCH_TIMEOUT` | `10` | SearXNG request timeout |
| `MAX_CONTENT_CHARS` | `4000` | Max chars per scraped page passed to the reranker |
| `LOG_LEVEL` | `INFO` | MCP log level (`DEBUG` for scrape/rerank failure detail) |
| `RERANK_MODEL` | `ms-marco-MiniLM-L-12-v2` | FlashRank model id loaded by the MCP server |
| `SEARXNG_IMAGE` / `CRAWL4AI_IMAGE` | pinned | Override to bump upstream image versions |

SearXNG engine allowlist, safesearch, etc. live in `searxng/config/settings.yml.template`. Edit there to change defaults for new deployments; `settings.yml` itself is generated on first boot and gitignored.

## Tools the MCP exposes

- Tool: `web_search(query, num_results=10, scrape_top=5, time_range=None, mode="balanced", include_domains=None, exclude_domains=None)` — returns structured JSON with ranked results, snippets, extracted content, per-page top chunks, and response metadata. `time_range` takes `day`, `week`, `month`, `year`. `mode` is `balanced` or `deep`.
- Tool: `site_search(query, site, num_results=10, scrape_top=5, mode="balanced", include_domains=None, exclude_domains=None)` — same JSON schema, but scoped with `site:<domain>`.
- Tool: `extract_url(url, query=None)` — structured single-URL extraction for pages and supported files.
- Tool: `extract_urls(urls, query=None)` — batch extraction with per-URL statuses. Uses Crawl4AI for web pages and `pypdf` for PDFs, returning markdown content only in v1.
- Tool: `map_site(url, max_urls=25, max_depth=1, include_patterns=None, exclude_patterns=None, same_domain_only=True)` — discovers candidate URLs from a site using Crawl4AI link extraction and returns a structured site map.
- Tool: `crawl_site(url, query=None, max_urls=10, max_depth=1, include_patterns=None, exclude_patterns=None, same_domain_only=True)` — maps a site, then extracts each discovered page into a single structured response.

Mode behavior:

- `balanced` keeps the current pipeline shape: fetch `num_results`, scrape up to `scrape_top`, rerank, and return the top results.
- `deep` increases recall by fetching a larger candidate set first, then filtering, deduplicating, scraping, and reranking back down to `num_results`.

Domain filters:

- `include_domains=["docs.python.org"]` keeps only matching domains and subdomains.
- `exclude_domains=["reddit.com"]` removes matching domains and subdomains.
- Domain filter entries must be bare domains, not full URLs.

Ranking behavior:

- Results are deduplicated by normalized URL and also by same-domain same-title matches.
- Final ranking is diversity-aware, so one domain is less likely to dominate the top results when multiple relevant sources are available.

Extraction behavior:

- `extract_url` and `extract_urls` share the same extraction pipeline and response fields.
- `extract_urls` supports partial success: one failed URL does not fail the whole call.
- PDF handling is explicit and library-backed via `pypdf`.
- `docx` handling is explicit and library-backed via `python-docx`.
- Plain-text and structured-text files like `txt`, `md`, `json`, `xml`, and `csv` are fetched and decoded through `httpx`.
- Results include per-URL `status`, `content_type`, `file_type`, `title`, `content`, and `error`.

Map behavior:

- `map_site` is a bounded discovery tool, not a full-content crawl.
- It follows links with Crawl4AI and returns normalized candidate URLs, titles, discovery depth, and the source URL each page was found from.
- `same_domain_only=True` keeps discovery within the root host and its subdomains.
- `include_patterns` and `exclude_patterns` accept shell-style globs against full URLs.

Crawl behavior:

- `crawl_site` composes `map_site` and `extract_urls`; it is not a separate crawling engine.
- It keeps the same bounded controls as `map_site`, then returns per-page extraction status, content, and top chunks in one response.
- If `query` is provided, crawled pages are re-ordered by their best chunk score after extraction.

`web_search` response shape:

```json
{
  "query": "current llm context window records",
  "time_range": "month",
  "mode": "balanced",
  "include_domains": [],
  "exclude_domains": [],
  "results": [
    {
      "rank": 1,
      "search_rank": 3,
      "title": "Example Title",
      "url": "https://example.com/post",
      "normalized_url": "https://example.com/post",
      "domain": "example.com",
      "snippet": "Short search-engine snippet.",
      "content": "Most relevant extracted page content.",
      "top_chunks": [
        {
          "text": "Most relevant excerpt from the page.",
          "score": 0.92
        }
      ],
      "score": 0.92,
      "scraped": true,
      "previously_seen": false
    }
  ],
  "meta": {
    "num_results_requested": 5,
    "num_results_returned": 5,
    "scrape_top": 5,
    "candidate_count": 5,
    "search_backend": "searxng",
    "reranker": {
      "name": "flashrank",
      "model": "ms-marco-MiniLM-L-12-v2"
    },
    "degraded": false,
    "warnings": [],
    "timings_ms": {
      "search": 120,
      "scrape": 640,
      "rerank": 18,
      "total": 790
    }
  }
}
```

When an upstream dependency partially fails, the MCP returns the same JSON shape with `meta.degraded=true` and a populated `meta.warnings` array instead of hard-failing the whole tool.

## Layout

```
docker-compose.yml              # the bundled stack
flake.nix                       # devshell + deploy/teardown apps
justfile                        # user-facing recipes (up, down, setup, logs, ...)
env.sample                      # copy to .env
eval/                           # benchmark queries, runner, scorer
mcp/                            # the MCP server (FastMCP + httpx)
searxng/config/
  settings.yml.template         # tracked; settings.yml is generated from this
```

## License

MIT
