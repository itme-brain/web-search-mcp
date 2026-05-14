"""Optional CPU-only semantic cache backed by Valkey.

This is deliberately simple: no vector DB service, no GPU use. Page
chunks are embedded with a small sentence-transformers model and stored
in Valkey. Search embeds the query, scans cached vectors, and returns the
nearest chunks to merge with live web candidates.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any

import cache
import core

log = logging.getLogger("web-search-mcp")

ENABLED = os.environ.get("ENABLE_SEMANTIC_INDEX", "0").lower() in {"1", "true", "yes", "on"}
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu")
TOP_K = int(os.environ.get("SEMANTIC_TOP_K", "20"))
_INDEX_KEY = "ws:semantic:index"
_MODEL: Any | None = None
_MODEL_LOCK: asyncio.Lock | None = None


def _lock() -> asyncio.Lock:
    global _MODEL_LOCK
    if _MODEL_LOCK is None:
        _MODEL_LOCK = asyncio.Lock()
    return _MODEL_LOCK


async def _model() -> Any:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    async with _lock():
        if _MODEL is not None:
            return _MODEL
        from sentence_transformers import SentenceTransformer

        log.info("loading embedding model=%s device=%s", MODEL_NAME, DEVICE)
        _MODEL = await asyncio.to_thread(SentenceTransformer, MODEL_NAME, device=DEVICE)
        return _MODEL


async def _embed(texts: list[str], *, is_query: bool = False):
    import numpy as np

    model = await _model()
    prefix = "query: " if is_query else "passage: "
    prepared = [prefix + text.replace("\n", " ") for text in texts]
    vectors = await asyncio.to_thread(
        model.encode,
        prepared,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def _chunk_id(url: str, text: str) -> str:
    return hashlib.sha256(f"{core._normalize_url(url)}\n{text}".encode()).hexdigest()


async def index_page(url: str, title: str, content: str, metadata: dict | None = None) -> None:
    """Embed and store chunks for one page. No-op unless enabled."""
    if not ENABLED or not content:
        return
    chunks = core._chunk_text(content)[:40]
    if not chunks:
        return
    if cache._DEFAULT_TTL_S == 0:
        return
    vectors = await _embed(chunks, is_query=False)
    client = cache._get_client()  # internal service module; intentional shared Valkey connection
    normalized = core._normalize_url(url)
    domain = core._domain_from_url(url)
    pipe = client.pipeline()
    now = int(time.time())
    ids: list[str] = []
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        cid = _chunk_id(url, chunk)
        ids.append(cid)
        payload = {
            "id": cid,
            "url": url,
            "normalized_url": normalized,
            "domain": domain,
            "title": title,
            "chunk_index": idx,
            "text": chunk,
            "metadata": metadata or {},
            "updated_at": now,
            "model": MODEL_NAME,
            "vector": vector.tolist(),
        }
        pipe.set(f"ws:semantic:chunk:{cid}", json.dumps(payload), ex=cache._DEFAULT_TTL_S or None)
    if ids:
        pipe.sadd(_INDEX_KEY, *ids)
    await pipe.execute()


async def search(query: str, *, top_k: int | None = None) -> list[dict]:
    """Return cached semantic chunks nearest to query. No-op unless enabled."""
    if not ENABLED:
        return []
    client = cache._get_client()
    ids = list(await client.smembers(_INDEX_KEY))
    if not ids:
        return []
    keys = [f"ws:semantic:chunk:{cid}" for cid in ids]
    raws = await client.mget(keys)
    records = []
    for raw in raws:
        if not raw:
            continue
        record = json.loads(raw)
        if record.get("model") == MODEL_NAME:
            records.append(record)
    if not records:
        return []
    import numpy as np

    qvec = (await _embed([query], is_query=True))[0]
    vectors = np.asarray([r["vector"] for r in records], dtype=np.float32)
    scores = vectors @ qvec
    limit = top_k or TOP_K
    order = np.argsort(-scores)[:limit]
    out: list[dict] = []
    for i in order:
        record = records[int(i)]
        record = {k: v for k, v in record.items() if k != "vector"}
        record["score"] = float(scores[int(i)])
        out.append(record)
    return out
