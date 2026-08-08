# web-search-mcp

Self-hosted MCP web search for local and hosted LLM agents.

What it does:

- web search with compact evidence and citations
- stable document/chunk handles for targeted follow-up reads
- full-page document extraction for known URLs
- site mapping and small-tree crawling
- intent-aware candidate reranking, freshness-safe caching, and optional semantic retrieval
- optional LFM2.5 preprocessing for research planning and cited evidence compression
- no paid search API required

The server is designed for small tool-using models: few tools, sane defaults, transparent output, and minimal knobs exposed to the model.

*This server is intended to run behind a reverse proxy that handles TLS or on a trusted local network.*
*It is not recommended to expose the raw MCP port directly to the internet.*

## Install

Tags:

- https://github.com/itme-brain/web-search-mcp/tags

Download a tagged version from GitHub, extract it, and `cd` into the extracted directory.

Nix provides the pinned development and deployment tools. The host must provide
a running Docker daemon.

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
| `read_evidence` | Expand a stable document or chunk reference already returned by search. |

Rule of thumb:

- use `search` first
- use `read_evidence` to expand an already-retrieved passage/document without another web fetch
- use `extract` when you already know the URL
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
- `research` and `crawl` support optional MCP background-task execution
- FastMCP 4 is deliberately pinned to the `4.0.0b1` beta; the server negotiates
  both modern sessionless and legacy session-based MCP transports

### Optional LFM2.5 preprocessing

The preprocessing layer is fail-open and applies only to `research`. It can
refine query intent/search variants and compress retrieved passages into
validated cited statements. Deterministic extraction, retrieval, reranking, and
the original evidence remain authoritative.

The pinned artifact is
[`LiquidAI/LFM2.5-2.6B-GGUF`](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF),
file `LFM2.5-2.6B-Q8_0.gguf`. Serve it from an OpenAI-compatible llama.cpp
endpoint, preferably with that filename as its model alias, then configure:

```dotenv
ENABLE_LFM_PREPROCESSING=1
LFM_BASE_URL=http://your-llama-host:8000/v1
LFM_MODEL=LFM2.5-2.6B-Q8_0.gguf
LFM_HF_REPO=LiquidAI/LFM2.5-2.6B-GGUF
LFM_HF_FILE=LFM2.5-2.6B-Q8_0.gguf
```

Set `LFM_API_KEY` when the endpoint requires authentication. The API key is
used only as a bearer token and is never included in health/result metadata.

The stack also includes a profile-gated, CPU-only llama.cpp sidecar. It uses an
upstream image pinned to build `b10326`, persists the 2.87 GB Q8 model in the
`lfm-models` volume, exposes no host port, and explicitly sets both
`--device none` and `--n-gpu-layers 0`. Start it with:

```sh
nix develop --command just up-lfm
nix develop --command just lfm-health
```

The defaults cap the sidecar at 8 CPUs and 6 GB RAM with one 8192-token slot.
Tune `LFM_CPUS`, `LFM_MEMORY_LIMIT`, `LFM_THREADS`, and `LFM_CTX_SIZE` in `.env`
for the production host. The first start downloads the model and can take
several minutes. To keep using an existing external llama-server instead, run
the normal stack and set `ENABLE_LFM_PREPROCESSING=1` plus its `LFM_BASE_URL`.

## Developer workflows

```sh
nix develop --command just setup-python
nix develop --command just test
nix develop --command just up-lfm
nix develop --command just smoke --url http://localhost:8002/mcp
```

### Production canary

The `canary` Compose profile runs the candidate MCP on port `18002` beside the
production service. It shares the existing SearXNG and Crawl4AI services, but
uses Valkey databases 2 and 3 so retrieval caches, evidence, and background
tasks remain isolated from production.

```sh
nix develop --command just canary-up
nix develop --command just canary-health
```

To verify the complete agent loop, configure the production llama.cpp endpoint
without exposing its key in command history:

```dotenv
AGENT_LLM_BASE_URL=http://localhost:8000/v1
AGENT_LLM_MODEL=your-production-model-alias
AGENT_LLM_API_KEY=
```

Then run `nix develop --command just canary-agent-smoke`. The bounded smoke
offers only `search`, `research`, and `read_evidence`, executes requested MCP
calls, and requires the model to return a final answer. It reports tool names
and answer size, not model reasoning or retrieved content. Stop only the canary
with `nix develop --command just canary-stop`.

## Layout

```
Dockerfile                          MCP server image
docker-compose.yml                  full stack (MCP + SearXNG + Crawl4AI)
env.sample                          default environment variables
flake.nix                           Nix devshell + deploy/teardown
justfile                            task runner recipes
requirements.in / .txt              Python deps (uv-compiled, hash-locked)
src/
  web_search_mcp/
    server.py                       FastMCP entry point, tool wrappers, health/metrics
    common.py                       shared warnings, validation, URL, and dedup helpers
    search_client.py                SearXNG and dependency-probe HTTP clients
    config/                         settings, search budgets, freshness policy
    http/                           request policy, URL utilities, target validators
    storage/                        Valkey cache, durable evidence, semantic index
    extraction/                     HTML/text/PDF extraction and provider adapters
    crawling/                       Crawl4AI client/config/result parsing
    ranking/                        intent, reranker lifecycle, evidence, source quality
    preprocessing/                  optional LFM query planning/evidence digest
    presentation/                   Pydantic models and markdown formatters
    tools/                          search, research, extract, map, crawl, evidence tools
searxng/config/
  settings.yml.template             engine allowlist, weights, safesearch
tests/                              pytest suite
eval/                               benchmark queries, scorer, live smoke tests
```
