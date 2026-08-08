"""Durable content-addressed evidence handles backed by Valkey."""

from __future__ import annotations

import hashlib
import re
import time
from urllib.parse import urlparse

from web_search_mcp.common import _chunk_text, _normalize_url
from web_search_mcp.storage import cache

SCHEMA_VERSION = 1
_ID_RE = re.compile(r"^[0-9a-f]{64}$")


def document_uri(document_id: str) -> str:
    return f"web-search://documents/{document_id}"


def chunk_uri(chunk_id: str) -> str:
    return f"web-search://chunks/{chunk_id}"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _validate_id(value: str, *, kind: str) -> str:
    if not _ID_RE.fullmatch(value):
        raise ValueError(f"invalid {kind} id")
    return value


async def persist_document(
    *, url: str, title: str | None, content: str, metadata: dict | None = None,
) -> dict:
    """Persist a complete cleaned document and its stable chunks."""
    normalized_url = _normalize_url(url)
    content_hash = _digest(content)
    document_id = _digest(f"{SCHEMA_VERSION}\n{normalized_url}\n{content_hash}")
    retrieved_at = int(time.time())
    chunk_specs: list[dict] = []
    chunk_records = []
    for index, text in enumerate(_chunk_text(content)):
        chunk_id = _digest(f"{document_id}\n{index}\n{_digest(text)}")
        spec = {"id": chunk_id, "index": index, "uri": chunk_uri(chunk_id)}
        chunk_specs.append(spec)
        chunk_records.append((chunk_id, {
            "schema_version": SCHEMA_VERSION,
            "kind": "chunk",
            "id": chunk_id,
            "document_id": document_id,
            "url": url,
            "title": title,
            "text": text,
            "index": index,
            "retrieved_at": retrieved_at,
        }))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "document",
        "id": document_id,
        "uri": document_uri(document_id),
        "url": url,
        "normalized_url": normalized_url,
        "title": title,
        "content": content,
        "content_hash": content_hash,
        "metadata": metadata or {},
        "retrieved_at": retrieved_at,
        "chunks": chunk_specs,
    }
    await cache.chunk_cache.set_many(chunk_records)
    await cache.document_cache.set(document_id, manifest)
    return manifest


async def get_document(document_id: str) -> dict | None:
    return await cache.document_cache.get(_validate_id(document_id, kind="document"))


async def get_chunk(chunk_id: str) -> dict | None:
    return await cache.chunk_cache.get(_validate_id(chunk_id, kind="chunk"))


async def resolve(reference: str) -> dict:
    """Resolve a document/chunk URI or a raw stable ID."""
    value = reference.strip()
    parsed = urlparse(value)
    if parsed.scheme == "web-search":
        parts = [part for part in parsed.path.split("/") if part]
        namespace = parsed.netloc
        if namespace == "documents" and len(parts) == 1:
            record = await get_document(parts[0])
        elif namespace == "chunks" and len(parts) == 1:
            record = await get_chunk(parts[0])
        else:
            raise ValueError("invalid web-search evidence URI")
    else:
        _validate_id(value, kind="evidence")
        record = await get_document(value) or await get_chunk(value)
    if record is None:
        raise ValueError("evidence reference not found or expired")
    return record
