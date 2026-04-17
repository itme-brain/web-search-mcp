# web-search-mcp

A self-contained MCP server that gives any MCP-compatible model (Claude, local llama.cpp, etc.) the same kind of search pipeline that paid "answer engines" use — fully FOSS, fully self-hosted.

Pipeline:

```
query -> SearXNG -> Crawl4AI (scrape top N) -> Jina reranker -> ranked markdown
```

All four services run in one `docker compose` project. Only the MCP server port is exposed on the host; SearXNG, Crawl4AI, and the reranker stay internal.

## Requirements

- Docker (daemon must be running on the host — Nix cannot provide this on non-NixOS)
- Either Nix (recommended — gives you the full toolchain with one command) or host-installed `docker compose` + `just`

## Quick start with Nix (works on NixOS and non-NixOS Linux/macOS)

```sh
git clone <this-repo> web-search-mcp
cd web-search-mcp

# One-shot: builds images, starts stack, waits for readiness.
nix run .#deploy

# Or drop into a devshell and use `just` recipes.
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
cp env.sample .env
docker compose up -d --build
```

## Configuration

Copy `env.sample` to `.env` and edit:

| Var | Default | Purpose |
|---|---|---|
| `MCP_HOST_PORT` | `8002` | Host port the MCP server binds to |
| `MAX_SCRAPE` | `5` | How many top results to fetch full content for |
| `SCRAPE_TIMEOUT` | `30` | Per-URL scrape timeout (seconds) |
| `MAX_CONTENT_CHARS` | `4000` | Max chars per scraped page before rerank |
| `JINA_MODEL` | `jinaai/jina-reranker-v2-base-multilingual` | HF model id for the reranker |

SearXNG is configured in `searxng/config/settings.yml`. Edit engine list, safesearch, etc. there.

## Using it

Point your MCP client at `http://<host>:${MCP_HOST_PORT}/mcp`. The server exposes one tool:

- `web_search(query, num_results=10, scrape_top=5, time_range=None)` — returns ranked markdown with titles, URLs, and scraped content.

And one prompt:

- `web_search_agent(topic)` — a ready-made system-prompt snippet that teaches the model to use `web_search`.

## Layout

```
docker-compose.yml     # the bundled stack
flake.nix              # devshell + deploy/teardown apps
justfile               # user-facing recipes (up, down, logs, smoke, ...)
env.sample             # copy to .env
mcp/                   # the MCP server (FastMCP + httpx)
jina-reranker/         # local cross-encoder reranker (FastAPI, CPU)
searxng/config/        # SearXNG settings.yml
```

## License

MIT
