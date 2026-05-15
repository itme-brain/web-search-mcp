# web-search-mcp

Self-hosted MCP web search for LLMs — no API keys, no per-query costs.

SearXNG searches configurable engines in parallel (9 by default), Crawl4AI scrapes the results, and a local reranker orders evidence. Everything runs in `docker compose`.

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
| `extract` | Read the full cleaned body of one known URL after `search`. |
| `map` | List URLs on one site; does not read page content. |
| `research` | Hard/broad questions. Multi-query search, compact brief, cited evidence. |
| `crawl` | Read several pages from one site/docs tree. |

Small-model agent rule of thumb: use `search` first with `num_results=3..5`; use `extract` to read one selected source as a full document, calling it again for additional URLs. Use `research` for hard/broad/current questions. Use `map` to plan a docs/site read, then `crawl` a small tree. Use `site:domain.com terms` in `search` for focused docs/site lookup. The MCP tools expose few knobs on purpose; retrieval depth, passage limits, and raw-content fallbacks are sane internal defaults.

`search`/`research` are the compression tools; `extract` is the document reader. `extract` handles HTML, common text formats, and born-digital PDFs locally.
PDF downloads are capped by `MAX_PDF_BYTES` before parsing so large files do
not exhaust memory; scanned/image-only PDFs require a future OCR backend.

## Configuration

`just setup` generates `.env` from `env.sample`. See `env.sample` for available knobs. SearXNG engine config lives in `searxng/config/settings.yml.template`.

Semantic retrieval is enabled in the compose stack by default and can augment live search. The cache layers are intentionally separate: `page_cache` stores canonical fetched/extracted documents, `page_memory` (`ws:page_memory`) stores retrieval-ready page text, and `semantic.py` owns the TTL-bounded Valkey Search HNSW chunk/vector index over recently scraped chunks. Normal searches write only to the configured cache TTL and Valkey maxmemory/LRU policy still bounds growth; there is no separate permanent vector database.

Observability endpoints:

- `/metrics`: JSON page, SearXNG, seen-URL, page-memory, and semantic-index counters for quick inspection.
- `/metrics/prometheus`: Prometheus text metrics for tool requests, warnings,
  stage latency histograms, cache counters, and semantic-index gauges.
Each tool response also includes `meta.request_id`, which is mirrored in
server logs for correlation.

Reranking is local and pluggable. The compose default is the English
Sentence Transformers CrossEncoder `cross-encoder/ms-marco-MiniLM-L4-v2`,
which matched the larger MiniLM rerankers on the bundled eval set while
using less CPU time:

```sh
RERANK_BACKEND=sentence-transformers
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L4-v2
RERANK_DEVICE=cpu
```

Only the configured `RERANK_BACKEND` and `RERANK_MODEL` are loaded. Other
models listed here are examples; they are not downloaded unless selected.

FlashRank remains available as the smallest ONNX-based backend:

```sh
RERANK_BACKEND=flashrank
RERANK_MODEL=ms-marco-MiniLM-L-12-v2
```

Tested English reranker options:

| Backend | Model | Use when |
|---|---|---|
| `sentence-transformers` | `cross-encoder/ms-marco-MiniLM-L4-v2` | Default balance of quality and CPU latency. |
| `sentence-transformers` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Slightly larger CPU model; matched L4 quality in the bundled eval but ran slower. |
| `sentence-transformers` | `cross-encoder/ms-marco-MiniLM-L12-v2` | Larger MiniLM model; no bundled-eval gain over L4/L6 in local tests. |
| `sentence-transformers` | `cross-encoder/ms-marco-MiniLM-L2-v2` | Fastest tested CrossEncoder, but lower usefulness on the bundled eval. |
| `flashrank` | `ms-marco-MiniLM-L-12-v2` | Small ONNX-based backend; available for compatibility and comparison. |

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
