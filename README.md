# web-search-mcp

Self-hosted MCP web search for local and hosted LLM agents.

What it does:
- web search with compact evidence and citations
- full-page document extraction for known URLs
- site mapping and small-tree crawling
- local reranking, caching, and optional semantic retrieval
- no paid search API required

The server is designed for small tool-using models: few tools, sane defaults, transparent output, and minimal knobs exposed to the model.

*This server is intended to run behind a reverse proxy that handles TLS or on a trusted local network.*
*It is not recommended to expose the raw MCP port directly to the internet.*

## Install

Tags:
- https://github.com/itme-brain/web-search-mcp/tags

Download a tagged version from GitHub, extract it, and `cd` into the extracted directory.

**Without Nix**

Requires `docker compose`, `just`, and `uv`.

```sh
uv venv .venv && uv pip sync --python .venv/bin/python requirements.txt
just setup
docker compose up -d --build
```

**With Nix**

```sh
nix run .#deploy
```

The MCP endpoint will be available at:

```text
http://localhost:8002/mcp
```

## Connect

### Pi

Add this to your Pi MCP server config:

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

## Tool guide

| Tool | Use it for |
|---|---|
| `search` | Default first step. Find sources and compact evidence for a question. |
| `extract` | Read one known URL as a full cleaned document. |
| `map` | List URLs under one site/root without reading page content. |
| `research` | Broader, slower, multi-source search for harder questions. |
| `crawl` | Read a small site/docs subtree, optionally ranked for a query. |

Rule of thumb:
- use `search` first
- use `extract` to read the best page in full
- use `research` for hard, comparative, or current questions
- use `map` then `crawl` for docs/site exploration

## Config example

Copy `env.sample` to `.env` and start with the defaults. The top section is all most users need.

```sh
cp env.sample .env
```

Core defaults:

```dotenv
MCP_HOST_PORT=8002
REQUEST_TIMEOUT=30
MAX_RESULTS=20
MAX_SCRAPE=10

RERANK_BACKEND=flashrank
RERANK_MODEL=ms-marco-MiniLM-L-12-v2
RERANK_DEVICE=cpu

ENABLE_SEMANTIC_CACHE=1
```

Notes:
- Advanced config options are documented in `env.sample`

## Layout

```
Dockerfile                          MCP server image
docker-compose.yml                  full stack (MCP + SearXNG + Crawl4AI)
env.sample                          default environment variables
flake.nix                           Nix devshell + deploy/teardown
justfile                            task runner recipes
requirements.in / .txt              Python deps (uv-compiled, hash-locked)
src/
  server.py                         FastMCP entry point, tool wrappers, health/metrics
  impls.py                          orchestration for search, extract, map, crawl
  core.py                           HTTP clients, scraping/extraction, shared helpers
  crawl.py                          site mapping/crawling helpers
  rerank.py                         rerank utilities and scoring helpers
  rerankers.py                      local reranker backend adapters
  text_utils.py                     text chunking/cleaning utilities
  urls.py                           URL normalization and filtering helpers
  validators.py                     parameter validation helpers
  models.py                         Pydantic response models
  formatters.py                     dict → markdown rendering
searxng/config/
  settings.yml.template             engine allowlist, weights, safesearch
tests/                              pytest suite
eval/                               benchmark queries, scorer, live smoke tests
```
