"""Targeted expansion for durable document and chunk evidence handles."""

from web_search_mcp.presentation import models
from web_search_mcp.storage import evidence as evidence_store


async def read_evidence_impl(reference: str) -> dict:
    record = await evidence_store.resolve(reference)
    is_document = record["kind"] == "document"
    document_id = record["id"] if is_document else record["document_id"]
    resource_uri = (
        evidence_store.document_uri(record["id"])
        if is_document else evidence_store.chunk_uri(record["id"])
    )
    payload = {
        "reference": reference,
        "kind": record["kind"],
        "document_id": document_id,
        "chunk_id": None if is_document else record["id"],
        "url": record["url"],
        "title": record.get("title"),
        "content": record["content"] if is_document else record["text"],
        "resource_uri": resource_uri,
    }
    return models.dump_response(models.EvidenceReadResponseModel, payload)
