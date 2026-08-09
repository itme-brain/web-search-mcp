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

# Generate host-only service credentials and configuration (idempotent).
setup:
    @if [ ! -f .env ]; then \
        echo ">> copying env.sample to .env"; \
        cp env.sample .env; \
    fi
    @if [ ! -f searxng/config/settings.yml ]; then \
        echo ">> rendering searxng/config/settings.yml with a random secret_key"; \
        sed "s|ultrasecretkey|$(openssl rand -hex 32)|" searxng/config/settings.yml.template > searxng/config/settings.yml; \
    fi
    @if ! grep -Eq '^CRAWL4AI_API_TOKEN=.+$' .env; then \
        echo ">> generating an internal Crawl4AI API token"; \
        token="$(openssl rand -hex 32)"; \
        if grep -q '^CRAWL4AI_API_TOKEN=' .env; then \
            sed -i "s|^CRAWL4AI_API_TOKEN=.*|CRAWL4AI_API_TOKEN=${token}|" .env; \
        else \
            printf '\nCRAWL4AI_API_TOKEN=%s\n' "${token}" >> .env; \
        fi; \
    fi

# Build and start the full stack detached.
up: setup
    {{ compose }} up -d --build

# Start the stack with the CPU-only LFM2.5 Q8 preprocessing sidecar.
up-lfm: setup
    COMPOSE_PROFILES=lfm ENABLE_LFM_PREPROCESSING=1 {{ compose }} up -d --build

# Start a candidate MCP beside production on port 18002 with isolated Valkey DBs.
canary-up: setup
    COMPOSE_PROFILES=canary {{ compose }} up -d --build web-search-mcp-canary

# Stop the canary without changing production services or shared volumes.
canary-stop:
    {{ compose }} --profile canary stop web-search-mcp-canary

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

# Check the internal LFM sidecar from the MCP container.
lfm-health:
    {{ compose }} exec -T web-search-mcp python -c "import urllib.request; print(urllib.request.urlopen('http://lfm:8080/health', timeout=5).read().decode())"

# Check the candidate MCP's readiness endpoint.
canary-health:
    @curl -fsS "http://localhost:${MCP_CANARY_HOST_PORT:-18002}/ready" && echo

# Run the Python test suite through the uv-managed virtualenv.
test: setup-python
    .venv/bin/pytest -q

# Run one test module or node id, e.g. `just test-target tests/test_tools.py`.
test-target target: setup-python
    .venv/bin/pytest -q "{{ target }}"

# Probe the optional OpenAI-compatible preprocessing endpoint. LFM_API_KEY is
# inherited from `.env` when required by the backend.
lfm-smoke endpoint model="LFM2.5-2.6B-Q8_0.gguf": setup-python
    PYTHONPATH=src .venv/bin/python scripts/lfm_smoke.py --base-url "{{ endpoint }}" --model "{{ model }}"

# Let an OpenAI-compatible production model discover and call the canary MCP.
# Configure AGENT_LLM_BASE_URL, AGENT_LLM_MODEL, and optionally AGENT_LLM_API_KEY.
canary-agent-smoke *args: setup-python
    PYTHONPATH=src .venv/bin/python scripts/llama_mcp_smoke.py --mcp-url "http://localhost:${MCP_CANARY_HOST_PORT:-18002}/mcp" {{ args }}

# Run the benchmark query set and write a JSONL run under eval/runs/.
eval: setup-python
    .venv/bin/python eval/run_eval.py

# Score a saved eval run, e.g. `just eval-score eval/runs/20260420T000000Z.jsonl`.
eval-score run_file: setup-python
    .venv/bin/python eval/score.py {{ run_file }}

# Compare reranker backends on eval/queries.json.
# Override specs with e.g.
# `just eval-rerankers --spec flashrank=flashrank:ms-marco-MiniLM-L-12-v2 --spec minilm=sentence-transformers:cross-encoder/ms-marco-MiniLM-L6-v2`.
eval-rerankers *args: setup-python
    .venv/bin/python eval/benchmark_rerankers.py {{ args }}

# Lightweight retrieval smoke focused on small-model output shape.
small-eval: setup-python
    .venv/bin/python eval/small_model_eval.py

# Live end-to-end smoke: one call per tool against the running stack.
# Pass --full to include PDF extraction and a degraded-mode engine spike.
smoke *args: setup-python
    .venv/bin/python eval/live_smoke.py {{ args }}

# Regenerate requirements.txt (hash-locked) from requirements.in via uv.
# Run this after editing requirements.in. `nix run .#deploy` will also
# auto-regen when .in is newer than .txt.
lock:
    uv pip compile --prerelease=allow --generate-hashes requirements.in -o requirements.txt
