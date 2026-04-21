FROM python:3.12.8-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Pinned uv binary — keeps the install stack fully reproducible. Version
# matches the uv shipped in the flake devshell so dev + build use the
# same resolver.
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /usr/local/bin/uv

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# --require-hashes forces pip/uv to verify every wheel's sha256 against
# requirements.txt before install; any tampered or swapped wheel fails
# the build instead of landing silently in the image.
COPY requirements.txt .
RUN uv pip install --system --no-cache --require-hashes -r requirements.txt

COPY src/*.py ./

CMD ["python", "server.py"]
