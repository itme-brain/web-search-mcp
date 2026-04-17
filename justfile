set dotenv-load := true
set shell := ["bash", "-euo", "pipefail", "-c"]

compose := "docker compose"

# Show available recipes.
default:
    @just --list

# Build and start the full stack detached.
up:
    {{ compose }} up -d --build

# Stop and remove containers, keep volumes.
down:
    {{ compose }} down

# Stop and remove containers AND volumes (wipes the reranker cache and searxng cache).
nuke:
    {{ compose }} down -v

# Tail logs for all services (Ctrl-C to exit).
logs:
    {{ compose }} logs -f

# Tail logs for a single service, e.g. `just logs-one web-search-mcp`.
logs-one service:
    {{ compose }} logs -f {{ service }}

# Show container status.
ps:
    {{ compose }} ps

# Rebuild images from scratch and recreate containers.
rebuild:
    {{ compose }} build --no-cache
    {{ compose }} up -d --force-recreate

# Restart all services without rebuilding.
restart:
    {{ compose }} restart

# Hit the MCP healthcheck via the configured host port.
health:
    curl -sf "http://localhost:${MCP_HOST_PORT:-8002}/mcp" >/dev/null && echo "MCP reachable on port ${MCP_HOST_PORT:-8002}" || (echo "MCP not reachable" >&2; exit 1)

# One-shot smoke test — requires the stack to be running.
smoke query="hello world":
    curl -sf -X POST "http://localhost:${MCP_HOST_PORT:-8002}/mcp/tools/web_search" \
        -H 'Content-Type: application/json' \
        -d '{"query": "{{ query }}", "num_results": 3, "scrape_top": 2}' | head -c 2000
