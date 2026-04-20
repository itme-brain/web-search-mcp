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

### Generic MCP client (Python, over HTTP)

```python
import asyncio
from fastmcp import Client

async def main():
    async with Client("http://localhost:8002/mcp") as c:
        result = await c.call_tool("search", {"query": "current LLM context window records", "num_results": 5})
        print(result.data)

asyncio.run(main())
```

### Scripting against the Python API (in-process)

When you want the raw Python dict (not markdown), import the impls directly. The four public impls are `search_impl`, `extract_impl`, `map_impl`, and `crawl_impl`:

```python
import asyncio, sys
sys.path.insert(0, "mcp")  # or set PYTHONPATH=mcp
from server import search_impl

async def main():
    response = await search_impl("current LLM context windows", num_results=5)
    for r in response["results"]:
        print(r["rank"], r["title"], r["url"])

asyncio.run(main())
```

MCP clients get markdown; Python callers get structured dicts. The `@mcp.tool` wrappers are thin formatters around these impls.

## Configuration

`env.sample` is copied to `.env` by `just setup` / `nix run .#deploy`. Knobs:

| Var | Default | Purpose |
|---|---|---|
| `MCP_HOST_PORT` | `8002` | Host port the MCP server binds to |
| `SEARXNG_URL` | `http://searxng:8080` | SearXNG endpoint (set by compose) |
| `CRAWL4AI_URL` | `http://crawl4ai:11235` | Crawl4AI endpoint (set by compose) |
| `REQUEST_TIMEOUT` | `30` | Timeout budget for search and scrape requests (seconds) |
| `MAX_RESULTS` | `10` | Max search results to return |
| `MAX_SCRAPE` | `5` | Max results to fetch full content for |
| `RERANK_MODEL` | `ms-marco-MiniLM-L-12-v2` | FlashRank reranker model |
| `LOG_LEVEL` | `INFO` | MCP log level (`DEBUG` for failure detail) |
| `SEARXNG_IMAGE` / `CRAWL4AI_IMAGE` | pinned | Override to bump upstream image versions |

SearXNG engine allowlist, safesearch, etc. live in `searxng/config/settings.yml.template`. Edit there to change defaults for new deployments; `settings.yml` itself is generated on first boot and gitignored.

## Tools the MCP exposes

The surface is intentionally small: four tools that compose. Pick the one that matches what you know.

| You want | Tool |
|---|---|
| To find sources on the open web (you don't have URLs yet) | `search` |
| To fetch full content for URLs you already have | `extract` |
| To discover URLs on a known site without fetching content | `map` |
| Both discovery **and** content from a known site in one call | `crawl` |

### `search(query, num_results=10, time_range=None, include_domains=None, exclude_domains=None)`

Web search → scrape top results → rerank → return ranked markdown.

- Scope to a single site with `site:<domain>` in the query (e.g. `site:docs.python.org asyncio`).
- For recency, set `time_range` (`day` / `week` / `month` / `year`) — do NOT put dates in the query text.
- `include_domains` / `exclude_domains` hard-filter by bare domain.
- Fetches a second page from SearXNG automatically if page 1 is short of `num_results` after dedup/filter. Always scrapes `min(num_results, MAX_SCRAPE)` top results.

Examples:

```json
{"query": "python asyncio taskgroup"}
{"query": "site:docs.python.org asyncio taskgroup"}
{"query": "ai safety paper", "time_range": "month", "num_results": 5}
{"query": "rust async runtime", "exclude_domains": ["reddit.com", "quora.com"]}
```

### `extract(urls, query=None)`

Fetch full content for the URLs you pass in. Always takes a list — `["https://..."]` for a single URL.

- HTML, PDF, DOCX, and plain-text files are all handled natively.
- PDFs are fully extracted with per-page chunking. Pass `query` to rerank pages by relevance; the response reports `total_pages` and `pages_returned`.
- Partial success: one failed URL does not fail the whole call. Results include per-URL `status`, `content_type`, `file_type`, `title`, `content`, and `error`.

Examples:

```json
{"urls": ["https://docs.python.org/3/library/asyncio-task.html"]}
{"urls": ["https://arxiv.org/pdf/2301.00001.pdf"], "query": "attention mechanism"}
{"urls": ["https://a.example/doc.html", "https://b.example/paper.pdf"]}
```

### `map(url, max_urls=25, max_depth=1, include_patterns=None, exclude_patterns=None, same_domain_only=True)`

Cheap discovery of URLs on a site — no body content, just the link graph.

- Returns normalized candidate URLs with titles, discovery depth, and the source URL each page was found from.
- `same_domain_only=True` keeps discovery within the root host and its subdomains.
- `include_patterns` / `exclude_patterns` accept shell-style globs against the full URL.
- This is a bounded survey tool, not a full-content crawler.

Examples:

```json
{"url": "https://docs.example.com"}
{"url": "https://docs.example.com", "include_patterns": ["https://docs.example.com/api/*"]}
{"url": "https://example.com", "max_depth": 2, "max_urls": 50}
```

### `crawl(url, query=None, max_urls=10, max_depth=1, include_patterns=None, exclude_patterns=None, same_domain_only=True)`

`map` + `extract` composed — discover the URL set, then fetch each page's content in one response.

- Same bounded controls as `map`.
- If `query` is set, pages are re-ordered by their best chunk score after extraction so the most relevant ones appear first.

Examples:

```json
{"url": "https://docs.example.com", "query": "authentication flow"}
{"url": "https://blog.example.com", "include_patterns": ["*/2026/*"], "max_urls": 8}
```

### Ranking + dedup (applies to `search` and `crawl` with `query`)

- Results are deduplicated by normalized URL and by same-domain same-title matches.
- Final ranking is diversity-aware, so one domain is less likely to dominate the top results when multiple relevant sources are available.

### Response shape

All tools return markdown optimized for LLM consumption. Example for `search`:

```
query: current llm context window records
time_range: month
results: 5

---

## [Example Title](https://example.com/post)

Most relevant extracted page content.

---

## [Another Result](https://example.com/other)

More content here.
```

When an upstream dependency partially fails, the response header includes `status: degraded` and a `warnings:` line instead of hard-failing the whole tool.

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
