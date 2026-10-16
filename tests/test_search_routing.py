"""Coverage-driven routing and bounded fetch replacement regressions."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from web_search_mcp.config import routing
from web_search_mcp.config.freshness import policy_for
from web_search_mcp import search_client
from web_search_mcp.tools import search
from tests.conftest import make_search_results


def candidates(count=8, *, host=None):
    return make_search_results([
        f"https://{host or f'source{i}.example'}/article{i}" for i in range(count)
    ])


async def collect(**overrides):
    steps = []
    args = dict(
        search_queries=["test documentation"], profile="search", candidate_budget=12,
        searx_pages=2, time_range=None, language="en", include_domains=None,
        exclude_domains=None, source_types=None, freshness_policy=policy_for(None),
        intent_name="technical_documentation", requested_results=2, retrieval_steps=steps,
    )
    args.update(overrides)
    result = await search._collect_search_candidates(**args)
    return result, steps


@pytest.mark.asyncio
async def test_healthy_relevant_diverse_primary_skips_fallback(monkeypatch):
    upstream = AsyncMock(return_value=candidates())
    monkeypatch.setattr(search, "_search", upstream)
    result, steps = await collect()
    assert len(result[0]) == 8
    assert upstream.await_count == 1
    assert steps[0]["engines"] == ["bing", "yandex"]
    assert not result[3]


@pytest.mark.asyncio
@pytest.mark.parametrize("cause", ["few", "one_host", "irrelevant", "filtered"])
async def test_inadequate_primary_broadens(monkeypatch, cause):
    first = candidates(1 if cause == "few" else 8, host="excluded.example" if cause in {"one_host", "filtered"} else None)
    upstream = AsyncMock(side_effect=[first, candidates()])
    monkeypatch.setattr(search, "_search", upstream)
    if cause == "irrelevant":
        monkeypatch.setattr(search, "_rerank_scored", AsyncMock(return_value=[(i, 0.001) for i in range(8)]))
    kwargs = {"exclude_domains": ["excluded.example"]} if cause == "filtered" else {}
    _, steps = await collect(**kwargs)
    assert upstream.await_count == 2
    assert steps[1]["engines"] == ["github", "stackoverflow", "fynd"]


@pytest.mark.asyncio
async def test_single_domain_request_does_not_require_other_hosts(monkeypatch):
    upstream = AsyncMock(return_value=candidates(host="docs.example"))
    monkeypatch.setattr(search, "_search", upstream)
    await collect(include_domains=["docs.example"])
    assert upstream.await_count == 1


@pytest.mark.asyncio
async def test_primary_failure_recovers_through_fallback(monkeypatch):
    upstream = AsyncMock(side_effect=[RuntimeError("unavailable"), candidates()])
    monkeypatch.setattr(search, "_search", upstream)
    result, steps = await collect()
    assert not result[3]
    assert steps[0]["failed"]
    assert not steps[1]["failed"]
    assert result[2][0]["type"] == "search_failed"


@pytest.mark.asyncio
async def test_research_retains_broad_engines_pages_and_variants(monkeypatch):
    upstream = AsyncMock(return_value=candidates())
    monkeypatch.setattr(search, "_search", upstream)
    _, steps = await collect(profile="research", searx_pages=3, search_queries=["one", "two"])
    assert upstream.await_count == 6
    assert {s["page"] for s in steps} == {1, 2, 3}
    assert all(s["engines"] == ["bing", "yandex", "github", "stackoverflow", "fynd"] for s in steps)


@pytest.mark.asyncio
async def test_engine_selection_has_separate_cache_entries(monkeypatch):
    upstream = AsyncMock(return_value=candidates())
    monkeypatch.setattr(search, "_search", upstream)
    args = dict(num_results=12, time_range=None, language="en", pageno=1)
    for engines in [("bing",), ("arxiv",), ("bing",)]:
        await search._searxng_cached("same query", engines=engines, **args)
    assert upstream.await_count == 2


@pytest.mark.asyncio
async def test_engine_selection_reaches_searxng(monkeypatch):
    response = MagicMock()
    response.json.return_value = {"results": []}
    client = AsyncMock()
    client.get.return_value = response
    client.__aenter__.return_value = client
    monkeypatch.setattr(search_client.httpx, "AsyncClient", lambda **kw: client)
    await search_client._search("test", engines=("bing news", "google news"))
    assert client.get.call_args.kwargs["params"]["engines"] == "bing news,google news"


@pytest.mark.asyncio
async def test_failed_fetches_are_replaced_and_snippets_retained(monkeypatch):
    scraped = AsyncMock(side_effect=[
        {"content": None, "metadata": {"diagnostic": "access_denied", "http_status": 403}},
        {"content": "Accepted page " * 30},
        {"content": "Replacement page " * 30},
    ])
    monkeypatch.setattr(search, "_scrape_cached", scraped)
    entries, attempts, warnings = await search._scrape_search_entries(candidates(8)["results"], 2, policy_for(None))
    assert attempts == 3
    assert scraped.await_count == 3
    assert not entries[0]["scraped"]
    assert entries[0]["content"].startswith("snippet")
    assert entries[0]["metadata"]["http_status"] == 403
    assert sum(e["scraped"] for e in entries) == 2
    assert any(w["type"] == "scrape_failed" for w in warnings)


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [3, 5, 9])
async def test_fetch_failure_storm_is_bounded(monkeypatch, cap):
    monkeypatch.setattr(search, "MAX_SCRAPE", cap)
    scraped = AsyncMock(side_effect=RuntimeError("crawler unavailable"))
    monkeypatch.setattr(search, "_scrape_cached", scraped)
    _, attempts, _ = await search._scrape_search_entries(candidates(20)["results"], 3, policy_for(None))
    assert attempts == min(cap, 6)
    assert scraped.await_count == min(cap, 6)


@pytest.mark.asyncio
@pytest.mark.parametrize("target", [1, 2, 3])
async def test_successful_fetches_stop_at_target(monkeypatch, target):
    scraped = AsyncMock(return_value={"content": "Useful content " * 30})
    monkeypatch.setattr(search, "_scrape_cached", scraped)
    _, attempts, _ = await search._scrape_search_entries(candidates(20)["results"], target, policy_for(None))
    assert attempts == scraped.await_count == target


def test_specialist_routes_and_research_have_no_duplicate_engines():
    assert routing.engine_stages("current_events", "search")[0] == ("bing news", "google news")
    assert routing.engine_stages("academic_research", "search")[0] == ("bing", "arxiv")
    for intent in ("technical_documentation", "current_events", "academic_research", "factual_lookup", "general_web_research"):
        engines = routing.engine_stages(intent, "research")[0]
        assert len(engines) == len(set(engines))


@pytest.mark.asyncio
async def test_fallback_candidates_survive_primary_pool_limit(monkeypatch):
    primary = candidates(12, host="primary.example")
    fallback = candidates(12, host="fallback.example")
    monkeypatch.setattr(search, "_search", AsyncMock(side_effect=[primary, fallback]))
    result, _ = await collect()
    retained = result[0][:12]
    assert {r["url"].split('/')[2] for r in retained} == {"primary.example", "fallback.example"}


@pytest.mark.asyncio
async def test_all_engine_requests_failing_reports_degraded(monkeypatch):
    monkeypatch.setattr(search, "_search", AsyncMock(side_effect=RuntimeError("offline")))
    result, steps = await collect()
    assert result[0] == []
    assert result[3]
    assert len(steps) == 2
    assert all(step["failed"] for step in steps)


@pytest.mark.asyncio
async def test_documentation_date_filter_keeps_technical_engines(monkeypatch):
    monkeypatch.setattr(search, "_search", AsyncMock(return_value=candidates()))
    monkeypatch.setattr(search, "_scrape_cached", AsyncMock(return_value={"content": "Documentation text " * 30}))
    result = await search.search_impl("Python asyncio API documentation", num_results=2, time_range="month")
    assert result["meta"]["intent"] == "technical_documentation"
    assert result["meta"]["retrieval_steps"][0]["engines"] == ["bing", "yandex"]
