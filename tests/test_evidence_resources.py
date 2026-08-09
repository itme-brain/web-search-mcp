from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import Client

from tests.conftest import server_app
from web_search_mcp.storage import evidence
from web_search_mcp.tools.evidence import read_evidence_impl


@pytest.mark.asyncio
async def test_document_and_chunk_ids_are_stable_for_identical_content():
    kwargs = {
        "url": "https://example.com/article",
        "title": "Article",
        "content": "First evidence paragraph.\n\nSecond evidence paragraph.",
        "metadata": {"date": "2026-08-08"},
    }
    first = await evidence.persist_document(**kwargs)
    second = await evidence.persist_document(**kwargs)

    assert first["id"] == second["id"]
    assert [item["id"] for item in first["chunks"]] == [item["id"] for item in second["chunks"]]


@pytest.mark.asyncio
async def test_persist_document_batches_chunk_writes(monkeypatch):
    batched = AsyncMock()
    monkeypatch.setattr(evidence.cache.chunk_cache, "set_many", batched)

    manifest = await evidence.persist_document(
        url="https://example.com/batched",
        title="Batched",
        content="one\n\n" + "two " * 500,
    )

    batched.assert_awaited_once()
    records = batched.await_args.args[0]
    assert [record[0] for record in records] == [chunk["id"] for chunk in manifest["chunks"]]


@pytest.mark.asyncio
async def test_tool_only_expansion_resolves_chunk_reference():
    manifest = await evidence.persist_document(
        url="https://example.com/article",
        title="Article",
        content="Targeted evidence content.",
    )
    chunk = manifest["chunks"][0]

    response = await read_evidence_impl(chunk["uri"])

    assert response["kind"] == "chunk"
    assert response["chunk_id"] == chunk["id"]
    assert response["content"] == "Targeted evidence content."


@pytest.mark.asyncio
async def test_document_expansion_can_return_bounded_chunks_with_continuation():
    manifest = await evidence.persist_document(
        url="https://example.com/long-article",
        title="Long article",
        content="First evidence block.\n\nSecond evidence block.\n\nThird evidence block.",
    )

    response = await read_evidence_impl(manifest["uri"], chunk_start=1, max_chunks=1)

    assert response["content"] == "Second evidence block."
    assert response["chunk_start"] == 1
    assert response["chunks_returned"] == 1
    assert response["total_chunks"] == 3
    assert response["next_chunk_start"] == 2
    assert response["truncated"] is True
    assert response["chunk_resource_uris"] == [manifest["chunks"][1]["uri"]]


@pytest.mark.asyncio
async def test_document_expansion_remains_full_by_default():
    content = "First evidence block.\n\nSecond evidence block."
    manifest = await evidence.persist_document(
        url="https://example.com/full-article",
        title="Full article",
        content=content,
    )

    response = await read_evidence_impl(manifest["uri"])

    assert response["content"] == content
    assert response["chunk_start"] is None
    assert response["chunks_returned"] == 2
    assert response["next_chunk_start"] is None
    assert response["truncated"] is False


@pytest.mark.asyncio
async def test_mcp_bounded_expansion_returns_chunk_resource_links():
    manifest = await evidence.persist_document(
        url="https://example.com/bounded-resource",
        title="Bounded resource",
        content="First evidence block.\n\nSecond evidence block.\n\nThird evidence block.",
    )

    async with Client(server_app) as client:
        result = await client.call_tool_mcp(
            "read_evidence",
            {"reference": manifest["uri"], "max_chunks": 2},
        )

    links = [
        str(item.uri)
        for item in result.content
        if getattr(item, "type", None) == "resource_link"
    ]
    assert links == [item["uri"] for item in manifest["chunks"][:2]]
    assert result.structured_content["next_chunk_start"] == 2


@pytest.mark.asyncio
async def test_mcp_resource_template_reads_persisted_document():
    manifest = await evidence.persist_document(
        url="https://example.com/resource",
        title="Resource",
        content="# Persisted\n\nComplete resource body.",
    )

    async with Client(server_app) as client:
        templates = await client.list_resource_templates()
        contents = await client.read_resource(manifest["uri"])

    assert any("web-search://documents/" in str(template.uri_template) for template in templates)
    assert contents[0].text == "# Persisted\n\nComplete resource body."


@pytest.mark.asyncio
async def test_search_returns_document_and_chunk_resource_links():
    search_response = {
        "results": [{"title": "Page", "url": "https://example.com/page", "content": "snippet"}],
        "unresponsive_engines": [],
    }
    scrape_response = {
        "title": "Page",
        "content": (
            "Full page evidence contains enough distinct words for the speculative cache admission policy "
            "and ranking pipeline while preserving a useful complete document for targeted expansion later."
        ),
        "metadata": {},
    }

    def rank_in_order(_query, documents):
        return [(idx, 0.9) for idx in range(len(documents))]

    with (
        patch("web_search_mcp.tools.search._search", AsyncMock(return_value=search_response)),
        patch("web_search_mcp.storage.pages._scrape", AsyncMock(return_value=scrape_response)),
        patch("web_search_mcp.tools.search._rerank_scored", AsyncMock(side_effect=rank_in_order)),
    ):
        async with Client(server_app) as client:
            result = await client.call_tool_mcp("search", {"query": "page evidence", "num_results": 1})

    structured = result.structured_content["results"][0]
    assert structured["document_id"]
    assert structured["resource_uri"].startswith("web-search://documents/")
    assert structured["passages"][0]["chunk_id"]
    links = {
        str(item.uri)
        for item in result.content
        if getattr(item, "type", None) == "resource_link"
    }
    assert structured["resource_uri"] in links
    assert structured["passages"][0]["resource_uri"] in links
