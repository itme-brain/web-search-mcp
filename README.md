# web-search-mcp

Self-hosted MCP web search for LLMs — no API keys, no per-query costs.

SearXNG searches configurable engines in parallel (9 by default), Crawl4AI scrapes the results, FlashRank reranks locally. Everything runs in `docker compose`.

## Install

```sh
git clone https://github.com/itme-brain/web-search-mcp && cd web-search-mcp
```

**With Nix:**

```sh
nix run .#deploy
```

**Without Nix** (requires `docker compose`, `just`, `uv`):

```sh
uv venv .venv && uv pip sync --python .venv/bin/python requirements.txt
just setup
docker compose up -d --build
```

## Connect your MCP client

The server speaks streamable HTTP at `http://localhost:8002/mcp`.

**Claude Desktop** — add to `claude_desktop_config.json`:

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

**Claude Code:**

```sh
claude mcp add --transport http web-search http://localhost:8002/mcp
```

## Tools

| Tool | Purpose |
|---|---|
| `search` | Find sources on the open web |
| `extract` | Fetch full content for URLs you already have |
| `map` | Discover URLs on a site (links only, no content) |
| `crawl` | `map` + `extract` in one call |

## Configuration

`just setup` generates `.env` from `env.sample`. See `env.sample` for available knobs. SearXNG engine config lives in `searxng/config/settings.yml.template`.

## Layout

```
Dockerfile                          MCP server image
docker-compose.yml                  full stack (MCP + SearXNG + Crawl4AI)
env.sample                          default environment variables
flake.nix                           Nix devshell + deploy/teardown
justfile                            task runner recipes
requirements.in / .txt              Python deps (uv-compiled, hash-locked)
src/
  server.py                         FastMCP entry point and tool wrappers
  impls.py                          search, extract, map, crawl implementations
  core.py                           HTTP clients, reranker, caching, text processing
  models.py                         Pydantic response models
  formatters.py                     dict → markdown rendering
searxng/config/
  settings.yml.template             engine allowlist, weights, safesearch
tests/                              pytest suite
eval/                               benchmark queries, scorer, live smoke tests
```
