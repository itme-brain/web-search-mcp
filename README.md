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
| `search` | Start here for unknown/current facts. Returns ranked sources with evidence passages. |
| `extract` | Read known URLs in more detail after `search`. |
| `map` | List URLs on one site; does not read page content. |
| `research` | Hard/broad questions. Multi-query search, compact brief, cited highlights. |
| `crawl` | Read several pages from one site/docs tree. |

Small-model agent rule of thumb: use `search` first with `num_results=3..5`; use `extract` only for sources that need more context. Use `mode=fast` for cheap link checks, `mode=deep` by default, and `research` for hard/broad questions. Raise `max_passages`, `max_chars_per_result`, or `include_raw_content` when compact evidence may omit needed context. Use `source_types=["docs","repo"]` to reduce noise.

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
