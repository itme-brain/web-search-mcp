set dotenv-load := true
set shell := ["bash", "-euo", "pipefail", "-c"]

compose := "docker compose"

# Show available recipes.
default:
    @just --list

# Generate .env from env.sample and searxng settings.yml from the template (idempotent).
setup:
    @if [ ! -f .env ]; then \
        echo ">> copying env.sample to .env"; \
        cp env.sample .env; \
    fi
    @if [ ! -f searxng/config/settings.yml ]; then \
        echo ">> rendering searxng/config/settings.yml with a random secret_key"; \
        sed "s|ultrasecretkey|$(openssl rand -hex 32)|" searxng/config/settings.yml.template > searxng/config/settings.yml; \
    fi

# Build and start the full stack detached.
up: setup
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

# Show container status + health for each service.
ps:
    {{ compose }} ps

# Rebuild images from scratch and recreate containers.
rebuild: setup
    {{ compose }} build --no-cache
    {{ compose }} up -d --force-recreate

# Restart all services without rebuilding.
restart:
    {{ compose }} restart

# Check that the MCP's /health endpoint is reachable on the configured host port.
health:
    @curl -fsS "http://localhost:${MCP_HOST_PORT:-8002}/health" && echo
