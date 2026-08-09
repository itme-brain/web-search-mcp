"""Targeted expansion for durable document and chunk evidence handles."""

import asyncio

from web_search_mcp.presentation import models
from web_search_mcp.storage import evidence as evidence_store


async def read_evidence_impl(
    reference: str,
    *,
    chunk_start: int | None = None,
    max_chunks: int | None = None,
) -> dict:
    if chunk_start is not None and chunk_start < 0:
        raise ValueError("chunk_start must be non-negative")
    if max_chunks is not None and not 1 <= max_chunks <= 10:
        raise ValueError("max_chunks must be between 1 and 10")

    record = await evidence_store.resolve(reference)
    is_document = record["kind"] == "document"
    bounded = chunk_start is not None or max_chunks is not None
    if not is_document and bounded:
        raise ValueError("chunk_start and max_chunks apply only to document references")

    document_id = record["id"] if is_document else record["document_id"]
    resource_uri = (
        evidence_store.document_uri(record["id"])
        if is_document else evidence_store.chunk_uri(record["id"])
    )
    content = record["content"] if is_document else record["text"]
    chunk_resource_uris: list[str] = []
    resolved_start: int | None = None
    chunks_returned: int | None = None
    total_chunks: int | None = None
    next_chunk_start: int | None = None
    truncated = False

    if is_document:
        chunks = record.get("chunks", [])
        total_chunks = len(chunks)
        if bounded:
            resolved_start = chunk_start or 0
            limit = max_chunks or 3
            if total_chunks and resolved_start >= total_chunks:
                raise ValueError(f"chunk_start must be less than total_chunks ({total_chunks})")
            selected = chunks[resolved_start:resolved_start + limit]
            chunk_records = await asyncio.gather(
                *(evidence_store.get_chunk(spec["id"]) for spec in selected),
            )
            if any(item is None for item in chunk_records):
                raise ValueError("one or more evidence chunks are missing or expired")
            content = "\n\n".join(item["text"] for item in chunk_records if item)
            chunk_resource_uris = [spec["uri"] for spec in selected]
            chunks_returned = len(selected)
            end = resolved_start + chunks_returned
            next_chunk_start = end if end < total_chunks else None
            truncated = resolved_start > 0 or next_chunk_start is not None
        else:
            chunks_returned = total_chunks

    payload = {
        "reference": reference,
        "kind": record["kind"],
        "document_id": document_id,
        "chunk_id": None if is_document else record["id"],
        "url": record["url"],
        "title": record.get("title"),
        "content": content,
        "resource_uri": resource_uri,
        "chunk_start": resolved_start,
        "chunks_returned": chunks_returned,
        "total_chunks": total_chunks,
        "next_chunk_start": next_chunk_start,
        "truncated": truncated,
        "chunk_resource_uris": chunk_resource_uris,
    }
    return models.dump_response(models.EvidenceReadResponseModel, payload)
