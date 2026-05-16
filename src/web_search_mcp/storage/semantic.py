"""Optional CPU-only semantic index backed by Valkey Search.

Page chunks are embedded with a small sentence-transformers model and
stored as Valkey HASH records. Valkey Search maintains an HNSW vector
index over those hashes, giving us bounded TTL semantics without a
separate vector database service.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any

from redis.exceptions import ResponseError

from web_search_mcp.storage import cache
from web_search_mcp.common import _chunk_text, _domain_from_url, _normalize_url

log = logging.getLogger("web-search-mcp")

ENABLED = os.environ.get("ENABLE_SEMANTIC_CACHE", "0").lower() in {"1", "true", "yes", "on"}
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu")
TOP_K = int(os.environ.get("SEMANTIC_TOP_K", "20"))
MIN_SCORE = float(os.environ.get("SEMANTIC_MIN_SCORE", "0.45"))
MAX_CHUNKS_PER_PAGE = int(os.environ.get("SEMANTIC_MAX_CHUNKS_PER_PAGE", "40"))
BACKEND = os.environ.get("SEMANTIC_BACKEND", "valkey-search").strip().lower()
_KEY_PREFIX = "ws:semantic:chunk:"
_MODEL_KEY = hashlib.sha256(MODEL_NAME.encode()).hexdigest()[:12]
_INDEX_NAME = f"ws:semantic:idx:{_MODEL_KEY}"
_MODEL: Any | None = None
_MODEL_LOCK: asyncio.Lock | None = None
_INDEX_READY = False
_INDEX_DIM: int | None = None
_LAST_RESULT_COUNT = 0
_LAST_STALE_PRUNED = 0
_LAST_ERROR: str | None = None


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
    return hashlib.sha256(f"{_normalize_url(url)}\n{text}".encode()).hexdigest()


def _vector_blob(vector: Any) -> bytes:
    import numpy as np

    return np.asarray(vector, dtype=np.float32).tobytes()


def _score_from_distance(distance: float) -> float:
    # Valkey Search COSINE returns distance as 1 - cosine_similarity.
    return 1.0 - distance


def _chunk_key(chunk_id: str) -> str:
    return f"{_KEY_PREFIX}{chunk_id}"


def _response_error_text(exc: ResponseError) -> str:
    return str(exc).lower()


async def _ensure_index(dim: int) -> bool:
    """Create the Valkey Search HNSW index if needed.

    Returns False when the connected Valkey does not have Search loaded.
    """
    global _INDEX_READY, _INDEX_DIM, _LAST_ERROR
    if _INDEX_READY and _INDEX_DIM == dim:
        return True
    if BACKEND != "valkey-search":
        _LAST_ERROR = f"unsupported semantic backend: {BACKEND}"
        return False

    client = cache._get_client()
    try:
        await client.execute_command("FT.INFO", _INDEX_NAME)
        _INDEX_READY = True
        _INDEX_DIM = dim
        _LAST_ERROR = None
        return True
    except ResponseError as exc:
        message = _response_error_text(exc)
        if "unknown index" not in message and "no such index" not in message:
            _LAST_ERROR = str(exc)
            log.warning("valkey search index check failed: %s", exc)
            return False

    try:
        await client.execute_command(
            "FT.CREATE",
            _INDEX_NAME,
            "ON", "HASH",
            "PREFIX", "1", _KEY_PREFIX,
            "SCHEMA",
            "vector", "VECTOR", "HNSW", "10",
            "TYPE", "FLOAT32",
            "DIM", str(dim),
            "DISTANCE_METRIC", "COSINE",
            "M", "16",
            "EF_CONSTRUCTION", "200",
            "model", "TAG",
            "domain", "TAG",
            "updated_at", "NUMERIC",
            "url", "TEXT", "NOSTEM",
            "title", "TEXT",
            "text", "TEXT",
        )
    except ResponseError as exc:
        message = _response_error_text(exc)
        if "index already exists" in message:
            _INDEX_READY = True
            _INDEX_DIM = dim
            _LAST_ERROR = None
            return True
        if "unknown command" in message or "wrong number of arguments" in message:
            _LAST_ERROR = "valkey-search module is not loaded"
        else:
            _LAST_ERROR = str(exc)
        log.warning("valkey search index creation failed: %s", exc)
        return False

    _INDEX_READY = True
    _INDEX_DIM = dim
    _LAST_ERROR = None
    return True


async def index_page(url: str, title: str, content: str, metadata: dict | None = None) -> None:
    """Embed and store chunks for one page. No-op unless enabled."""
    if not ENABLED or not content:
        return
    chunks = _chunk_text(content)[:MAX_CHUNKS_PER_PAGE]
    if not chunks:
        return
    if cache.SEMANTIC_INDEX_TTL_S == 0:
        return
    vectors = await _embed(chunks, is_query=False)
    if vectors.size == 0:
        return
    if not await _ensure_index(int(vectors.shape[1])):
        return
    client = cache._get_client()  # internal service module; intentional shared Valkey connection
    normalized = _normalize_url(url)
    domain = _domain_from_url(url)
    pipe = client.pipeline()
    now = int(time.time())
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        cid = _chunk_id(url, chunk)
        key = _chunk_key(cid)
        pipe.hset(key, mapping={
            "id": cid,
            "url": url,
            "normalized_url": normalized,
            "domain": domain,
            "title": title or "",
            "chunk_index": idx,
            "text": chunk,
            "metadata": json.dumps(metadata or {}),
            "updated_at": now,
            "model": MODEL_NAME,
            "vector": _vector_blob(vector),
        })
        pipe.expire(key, cache.SEMANTIC_INDEX_TTL_S)
    await pipe.execute()


async def stats() -> dict[str, int | str | bool | None]:
    client = cache._get_client()
    indexed_chunks = 0
    if ENABLED and _INDEX_READY:
        try:
            info = await client.execute_command("FT.INFO", _INDEX_NAME)
            if isinstance(info, list):
                pairs = dict(zip(info[::2], info[1::2]))
                indexed_chunks = int(pairs.get("num_docs", 0))
        except Exception as exc:
            log.debug("valkey search stats failed: %s", exc)
    return {
        "enabled": ENABLED,
        "backend": BACKEND,
        "model": MODEL_NAME,
        "device": DEVICE,
        "top_k": TOP_K,
        "min_score": MIN_SCORE,
        "max_chunks_per_page": MAX_CHUNKS_PER_PAGE,
        "index_name": _INDEX_NAME,
        "index_ready": _INDEX_READY,
        "indexed_chunks": indexed_chunks,
        "last_result_count": _LAST_RESULT_COUNT,
        "last_stale_pruned": _LAST_STALE_PRUNED,
        "last_error": _LAST_ERROR,
    }


async def search(query: str, *, top_k: int | None = None) -> list[dict]:
    """Return cached semantic chunks nearest to query. No-op unless enabled."""
    global _LAST_RESULT_COUNT, _LAST_STALE_PRUNED, _LAST_ERROR
    if not ENABLED:
        return []
    qvec = await _embed([query], is_query=True)
    if qvec.size == 0:
        return []
    if not await _ensure_index(int(qvec.shape[1])):
        return []
    client = cache._get_client()
    limit = top_k or TOP_K
    try:
        response = await client.execute_command(
            "FT.SEARCH",
            _INDEX_NAME,
            f"*=>[KNN {limit} @vector $query_vec AS distance]",
            "PARAMS", "2", "query_vec", _vector_blob(qvec[0]),
            "SORTBY", "distance",
            "RETURN", "9",
            "id", "url", "domain", "title", "chunk_index", "text", "metadata", "updated_at", "distance",
            "DIALECT", "2",
        )
    except ResponseError as exc:
        _LAST_ERROR = str(exc)
        log.warning("valkey search query failed: %s", exc)
        return []

    out: list[dict] = []
    total = response[0] if isinstance(response, list) and response else 0
    _LAST_RESULT_COUNT = int(total or 0)
    _LAST_STALE_PRUNED = 0
    for _key, fields in zip(response[1::2], response[2::2]):
        if not isinstance(fields, list):
            continue
        record = dict(zip(fields[::2], fields[1::2]))
        distance = float(record.get("distance", 1.0))
        score = _score_from_distance(distance)
        if score < MIN_SCORE:
            continue
        record["score"] = score
        record["metadata"] = json.loads(record.get("metadata") or "{}")
        record["chunk_index"] = int(record.get("chunk_index", 0))
        record["updated_at"] = int(record.get("updated_at", 0))
        out.append(record)
        if len(out) >= limit:
            break
    return out
