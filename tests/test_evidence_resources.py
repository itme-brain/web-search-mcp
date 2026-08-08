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
    assert any(getattr(item, "type", None) == "resource_link" for item in result.content)
