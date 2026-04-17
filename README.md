# web-search-mcp

A self-contained MCP server that gives any MCP-compatible model (Claude, local llama.cpp, etc.) the same kind of search pipeline that paid "answer engines" use — fully FOSS, fully self-hosted.

Pipeline:

```
query -> SearXNG -> Crawl4AI (scrape top N) -> Jina reranker -> ranked markdown
```

All four services run in one `docker compose` project. Only the MCP server port is exposed on the host; SearXNG, Crawl4AI, and the reranker stay internal.

## Requirements

- Docker (daemon must be running on the host — Nix cannot provide this on non-NixOS)
- Either Nix (recommended — gives you the full toolchain with one command) or host-installed `docker compose`, `just`, `openssl`
- First boot downloads the reranker model (~600 MB multilingual v2) into a cached volume. Allow a few minutes before the stack reports healthy.

## Quick start with Nix (works on NixOS and non-NixOS Linux/macOS)

```sh
git clone <this-repo> web-search-mcp
cd web-search-mcp

# One-shot: renders settings.yml with a random secret, builds images,
# starts the stack, and waits for the MCP /health probe to succeed.
nix run .#deploy

# Or drop into a devshell and use just recipes.
nix develop
just up
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
        print(result.content[0].text)

asyncio.run(main())
```

## Configuration

`env.sample` is copied to `.env` by `just setup` / `nix run .#deploy`. Knobs:

| Var | Default | Purpose |
|---|---|---|
| `MCP_HOST_PORT` | `8002` | Host port the MCP server binds to |
| `MAX_SCRAPE` | `5` | How many top results to scrape for full content |
| `SCRAPE_TIMEOUT` | `30` | Total per-URL scrape budget (seconds, end-to-end) |
| `SCRAPE_HTTP_TIMEOUT` | `15` | Per-request HTTP timeout inside the scrape loop |
| `SEARCH_TIMEOUT` | `10` | SearXNG request timeout |
| `RERANK_TIMEOUT` | `10` | Reranker request timeout |
| `MAX_CONTENT_CHARS` | `4000` | Max chars per scraped page passed to the reranker |
| `LOG_LEVEL` | `INFO` | MCP log level (`DEBUG` for scrape/rerank failure detail) |
| `JINA_MODEL` | `jinaai/jina-reranker-v2-base-multilingual` | HF model id for the reranker |
| `SEARXNG_IMAGE` / `CRAWL4AI_IMAGE` | pinned | Override to bump upstream image versions |

SearXNG engine allowlist, safesearch, etc. live in `searxng/config/settings.yml.template`. Edit there to change defaults for new deployments; `settings.yml` itself is generated on first boot and gitignored.

## Tools + prompts the MCP exposes

- Tool: `web_search(query, num_results=10, scrape_top=5, time_range=None)` — returns ranked markdown with titles, URLs, and scraped content. `time_range` takes `day`, `week`, `month`, `year`.
- Prompt: `web_search_agent(topic)` — a ready-made system-prompt snippet that teaches the model how to use `web_search`.

## Layout

```
docker-compose.yml              # the bundled stack
flake.nix                       # devshell + deploy/teardown apps
justfile                        # user-facing recipes (up, down, setup, logs, ...)
env.sample                      # copy to .env
mcp/                            # the MCP server (FastMCP + httpx)
jina-reranker/                  # local cross-encoder reranker (FastAPI, CPU)
searxng/config/
  settings.yml.template         # tracked; settings.yml is generated from this
```

## License

MIT
