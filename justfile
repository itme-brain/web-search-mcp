set dotenv-load := true
set shell := ["bash", "-euo", "pipefail", "-c"]

compose := "docker compose"

# Show available recipes.
default:
    @just --list

# Create/sync the local Python virtualenv from requirements.txt via uv.
setup-python:
    @echo ">> syncing .venv from requirements.txt"
    @if [ ! -x .venv/bin/python ]; then \
        uv venv --python "$(command -v python)" .venv; \
    fi
    @uv pip sync --python .venv/bin/python requirements.txt

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

# Check that the MCP's /ready endpoint is reachable on the configured host port.
health:
    @curl -fsS "http://localhost:${MCP_HOST_PORT:-8002}/ready" && echo

# Run the Python test suite through the uv-managed virtualenv.
test: setup-python
    .venv/bin/pytest -q

# Run the benchmark query set and write a JSONL run under eval/runs/.
eval: setup-python
    .venv/bin/python eval/run_eval.py

# Score a saved eval run, e.g. `just eval-score eval/runs/20260420T000000Z.jsonl`.
eval-score run_file: setup-python
    .venv/bin/python eval/score.py {{ run_file }}

# Live end-to-end smoke: one call per tool against the running stack.
# Pass --full to include PDF extraction and a degraded-mode engine spike.
smoke *args: setup-python
    .venv/bin/python eval/live_smoke.py {{ args }}

# Regenerate requirements.txt (hash-locked) from requirements.in via uv.
# Run this after editing requirements.in. `nix run .#deploy` will also
# auto-regen when .in is newer than .txt.
lock:
    uv pip compile --generate-hashes requirements.in -o requirements.txt
