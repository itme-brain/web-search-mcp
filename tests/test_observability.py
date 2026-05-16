from unittest.mock import AsyncMock, patch

import pytest

from web_search_mcp import observability
from tests.conftest import URLS_A, make_search_results, server_module


def _identity_rerank(_query: str, documents: list[str]) -> list[tuple[int, float]]:
    return [(i, 0.5) for i in range(len(documents))]


@pytest.fixture(autouse=True)
def reset_metrics():
    observability.reset()
    yield
    observability.reset()


@pytest.mark.asyncio
async def test_search_response_has_request_id_and_records_metrics():
    search_mock = AsyncMock(return_value=make_search_results(URLS_A[:1]))
    scrape_mock = AsyncMock(return_value={
        "content": (
            "# Page\n\nfull page body text with enough words for reranking and caching "
            "plus additional useful terms that clearly exceed the speculative cache word floor"
        ),
        "title": "Page",
        "metadata": {},
    })
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch("web_search_mcp.tools.search._search", search_mock),
        patch("web_search_mcp.storage.pages._scrape", scrape_mock),
        patch("web_search_mcp.tools.search._rerank_scored", rerank_mock),
    ):
        payload = await server_module.search_impl("observability query", num_results=1)

    assert payload["meta"]["request_id"]
    assert "semantic" in payload["meta"]["timings_ms"]

    text = observability.prometheus_text()
    assert 'web_search_mcp_tool_requests_total{profile="search",status="ok",tool="search"} 1' in text
    assert 'web_search_mcp_stage_duration_ms_count{stage="total",tool="search"} 1' in text


def test_prometheus_text_renders_warning_and_semantic_counters():
    observability.observe_tool_response("search", {
        "meta": {
            "profile": "search",
            "degraded": True,
            "semantic_hits": 2,
            "warnings": [{"type": "rerank_failed", "source": "test", "detail": "boom"}],
            "timings_ms": {"total": 123, "rerank": 45},
        }
    })

    text = observability.prometheus_text({
        "web_search_mcp_semantic_index_ready": (1, {}),
    })

    assert 'web_search_mcp_tool_requests_total{profile="search",status="degraded",tool="search"} 1' in text
    assert 'web_search_mcp_warnings_total{source="test",tool="search",type="rerank_failed"} 1' in text
    assert 'web_search_mcp_semantic_hits_total{tool="search"} 2' in text
    assert "web_search_mcp_semantic_index_ready 1" in text


@pytest.mark.asyncio
async def test_prometheus_route_includes_cache_and_semantic_gauges():
    with (
        patch.object(server_module.cache.page_cache, "stats", AsyncMock(return_value={"hits": 3, "misses": 4})),
        patch.object(server_module.cache.searxng_cache, "stats", AsyncMock(return_value={"hits": 5, "misses": 6})),
        patch.object(server_module.cache.seen_urls, "stats", AsyncMock(return_value={"hits": 7, "misses": 8})),
        patch.object(server_module.semantic, "stats", AsyncMock(return_value={"index_ready": True, "indexed_chunks": 9})),
    ):
        response = await server_module.prometheus_metrics(None)

    text = response.body.decode()
    assert response.media_type == "text/plain; version=0.0.4; charset=utf-8"
    assert 'web_search_mcp_cache_hits{cache="page"} 3' in text
    assert "web_search_mcp_semantic_index_ready 1" in text
    assert "web_search_mcp_semantic_indexed_chunks 9" in text
