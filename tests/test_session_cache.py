from unittest.mock import AsyncMock, patch

import pytest
from tests.conftest import SCRAPE_CONTENT, URLS_A, URLS_B, make_search_results, server_module

PATCH_SEARCH = "core._search"
PATCH_SCRAPE = "core._scrape"
PATCH_RERANK = "core._rerank_scored"

URLS_C = [
    "https://example.com/c1",
    "https://example.com/c2",
    "https://example.com/c3",
]

SCRAPE_CONTENT_EXTENDED = {
    **SCRAPE_CONTENT,
    "https://example.com/c1": "# Page C1\n\nThis is the full content of page C1 with enough text to pass the minimum chunk size threshold for reranking.",
    "https://example.com/c2": "# Page C2\n\nThis is the full content of page C2 with enough text to pass the minimum chunk size threshold for reranking.",
    "https://example.com/c3": "# Page C3\n\nThis is the full content of page C3 with enough text to pass the minimum chunk size threshold for reranking.",
    "https://example.com/new1": "# Page New1\n\nThis is the full content of page New1 with enough text to pass the minimum chunk size threshold for reranking.",
}


def _identity_rerank(_query: str, documents: list[str]) -> list[tuple[int, float]]:
    return [(i, 0.5) for i in range(len(documents))]


def _make_scrape_mock(content_map: dict[str, str | None] | None = None):
    mapping = content_map or SCRAPE_CONTENT_EXTENDED

    async def _fake_scrape(url: str, crawl_config=None) -> dict:
        content = mapping.get(url)
        return {"content": content, "title": None, "screenshot": None}

    mock = AsyncMock(side_effect=_fake_scrape)
    return mock


def _make_domain_results() -> dict:
    return {
        "results": [
            {"title": "Example A", "url": "https://example.com/a", "content": "snippet a"},
            {"title": "Docs A", "url": "https://docs.python.org/3/tutorial/", "content": "snippet docs"},
            {"title": "Example B", "url": "https://blog.example.com/b", "content": "snippet b"},
        ],
        "unresponsive_engines": [],
    }


def _make_diversity_results() -> dict:
    return {
        "results": [
            {"title": "A1", "url": "https://alpha.com/1", "content": "alpha one"},
            {"title": "A2", "url": "https://alpha.com/2", "content": "alpha two"},
            {"title": "B1", "url": "https://beta.com/1", "content": "beta one"},
            {"title": "A3", "url": "https://alpha.com/3", "content": "alpha three"},
        ],
        "unresponsive_engines": [],
    }


@pytest.fixture
def patched_backends():
    search_mock = AsyncMock(return_value=make_search_results(URLS_A))
    scrape_mock = _make_scrape_mock()
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        yield {
            "search": search_mock,
            "scrape": scrape_mock,
            "rerank": rerank_mock,
        }


@pytest.mark.asyncio
async def test_query_cache_hit_returns_same_output_without_calling_backends(patched_backends, fake_ctx):
    payload1 = await server_module.search_impl("test query", num_results=3, ctx=fake_ctx)

    patched_backends["search"].reset_mock()
    patched_backends["scrape"].reset_mock()
    patched_backends["rerank"].reset_mock()

    payload2 = await server_module.search_impl("test query", num_results=3, ctx=fake_ctx)

    assert payload1 == payload2
    patched_backends["search"].assert_not_called()
    patched_backends["scrape"].assert_not_called()
    patched_backends["rerank"].assert_not_called()


@pytest.mark.asyncio
async def test_query_cache_key_normalizes_whitespace_and_case(patched_backends, fake_ctx):
    payload1 = await server_module.search_impl("Test Query ", num_results=3, ctx=fake_ctx)

    patched_backends["search"].reset_mock()
    patched_backends["scrape"].reset_mock()
    patched_backends["rerank"].reset_mock()

    payload2 = await server_module.search_impl("  test query", num_results=3, ctx=fake_ctx)

    assert payload1 == payload2
    patched_backends["search"].assert_not_called()


@pytest.mark.asyncio
async def test_query_cache_miss_on_different_query_triggers_fresh_pipeline(fake_ctx):
    search_call_count = 0

    async def _search_side_effect(query, **kwargs):
        nonlocal search_call_count
        search_call_count += 1
        if search_call_count == 1:
            return make_search_results(URLS_A)
        return make_search_results(URLS_C)

    search_mock = AsyncMock(side_effect=_search_side_effect)
    scrape_mock = _make_scrape_mock()
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        await server_module.search_impl("test query", num_results=3, ctx=fake_ctx)

        search_mock.reset_mock()
        scrape_mock.reset_mock()
        rerank_mock.reset_mock()

        await server_module.search_impl("different query", num_results=3, ctx=fake_ctx)

    search_mock.assert_called_once()
    assert scrape_mock.call_count == 3
    rerank_mock.assert_called_once()


@pytest.mark.asyncio
async def test_scrape_cache_reuse_skips_already_scraped_urls(fake_ctx):
    search_call_count = 0

    async def _search_side_effect(query, **kwargs):
        nonlocal search_call_count
        search_call_count += 1
        if search_call_count == 1:
            return make_search_results(URLS_A)
        return make_search_results(URLS_B)

    search_mock = AsyncMock(side_effect=_search_side_effect)
    scrape_mock = _make_scrape_mock()
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        await server_module.search_impl("query alpha", num_results=3, ctx=fake_ctx)
        assert scrape_mock.call_count == 3

        scrape_mock.reset_mock()

        await server_module.search_impl("query beta", num_results=3, ctx=fake_ctx)

    scrape_urls_second = [call.args[0] for call in scrape_mock.call_args_list]
    assert "https://example.com/a2" not in scrape_urls_second
    assert "https://example.com/b1" in scrape_urls_second
    assert "https://example.com/b2" in scrape_urls_second


@pytest.mark.asyncio
async def test_previously_seen_urls_annotated_in_subsequent_results(fake_ctx):
    search_call_count = 0

    async def _search_side_effect(query, **kwargs):
        nonlocal search_call_count
        search_call_count += 1
        if search_call_count == 1:
            return make_search_results(URLS_A)
        return make_search_results(URLS_B)

    search_mock = AsyncMock(side_effect=_search_side_effect)
    scrape_mock = _make_scrape_mock()
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        payload1 = await server_module.search_impl("first search", num_results=3, ctx=fake_ctx)
        assert all(not item["previously_seen"] for item in payload1["results"])

        payload2 = await server_module.search_impl("second search", num_results=3, ctx=fake_ctx)

    seen_flags = {item["url"]: item["previously_seen"] for item in payload2["results"]}
    assert seen_flags["https://example.com/a2"] is True
    assert seen_flags["https://example.com/b1"] is False
    assert seen_flags["https://example.com/b2"] is False


@pytest.mark.asyncio
async def test_none_scrape_result_cached_so_broken_url_not_retried(fake_ctx):
    content_with_failure = dict(SCRAPE_CONTENT_EXTENDED)
    content_with_failure["https://example.com/a2"] = None

    search_mock = AsyncMock(return_value=make_search_results(URLS_A))
    scrape_mock = _make_scrape_mock(content_with_failure)
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        await server_module.search_impl("fail test", num_results=3, ctx=fake_ctx)
        assert scrape_mock.call_count == 3

        scrape_mock.reset_mock()

        search_mock.return_value = make_search_results(
            ["https://example.com/a2", "https://example.com/new1"], prefix="Retry"
        )

        await server_module.search_impl("retry test", num_results=2, ctx=fake_ctx)

    scrape_urls = [call.args[0] for call in scrape_mock.call_args_list]
    assert "https://example.com/a2" not in scrape_urls
    assert "https://example.com/new1" in scrape_urls


@pytest.mark.asyncio
async def test_tool_works_without_session_context():
    search_mock = AsyncMock(return_value=make_search_results(URLS_A[:2]))
    scrape_mock = _make_scrape_mock()
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        result = await server_module.search_impl("direct call", num_results=2, ctx=None)

    assert result["query"] == "direct call"
    assert result["results"][0]["url"] == "https://example.com/a1"
    search_mock.assert_called_once()


@pytest.mark.asyncio
async def test_rerank_failure_falls_back_to_search_order():
    search_mock = AsyncMock(return_value=make_search_results(URLS_A[:2]))
    scrape_mock = _make_scrape_mock()
    rerank_mock = AsyncMock(side_effect=RuntimeError("rerank offline"))

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        payload = await server_module.search_impl("fallback query", num_results=2, ctx=None)

    assert payload["meta"]["degraded"] is True
    assert any(w["type"] == "rerank_failed" for w in payload["meta"]["warnings"])
    assert [item["url"] for item in payload["results"]] == URLS_A[:2]


@pytest.mark.asyncio
async def test_search_failure_returns_empty_degraded_payload():
    search_mock = AsyncMock(side_effect=RuntimeError("search offline"))
    scrape_mock = _make_scrape_mock()
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        payload = await server_module.search_impl("failed query", num_results=2, ctx=None)

    assert payload["results"] == []
    assert payload["meta"]["degraded"] is True
    assert any(w["type"] == "search_failed" for w in payload["meta"]["warnings"])


@pytest.mark.asyncio
async def test_include_domains_filters_results(fake_ctx):
    search_mock = AsyncMock(return_value=_make_domain_results())
    scrape_mock = _make_scrape_mock({
        "https://docs.python.org/3/tutorial/": "# Docs\n\nUseful python docs content for reranking.",
    })
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        payload = await server_module.search_impl(
            "python tutorial",
            num_results=2,
            include_domains=["docs.python.org"],
            ctx=fake_ctx,
        )

    assert [item["domain"] for item in payload["results"]] == ["docs.python.org"]
    assert payload["include_domains"] == ["docs.python.org"]


@pytest.mark.asyncio
async def test_exclude_domains_filters_results(fake_ctx):
    search_mock = AsyncMock(return_value=_make_domain_results())
    scrape_mock = _make_scrape_mock({
        "https://docs.python.org/3/tutorial/": "# Docs\n\nUseful python docs content for reranking.",
    })
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        payload = await server_module.search_impl(
            "python tutorial",
            num_results=3,
            exclude_domains=["example.com"],
            ctx=fake_ctx,
        )

    assert [item["domain"] for item in payload["results"]] == ["docs.python.org"]
    assert payload["exclude_domains"] == ["example.com"]


@pytest.mark.asyncio
async def test_scrape_top_is_auto_bounded_by_num_results(fake_ctx):
    search_mock = AsyncMock(return_value=make_search_results(URLS_A))
    scrape_mock = _make_scrape_mock()
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        payload = await server_module.search_impl(
            "bounded", num_results=2, ctx=fake_ctx,
        )

    assert payload["meta"]["scrape_top"] == 2


@pytest.mark.asyncio
async def test_final_results_are_domain_diversified(fake_ctx):
    search_mock = AsyncMock(return_value=_make_diversity_results())
    scrape_mock = _make_scrape_mock({
        "https://alpha.com/1": "# A1\n\nAlpha one covers rockets, launch pads, fuel checks, mission control, and countdown sequencing with plenty of unique detail.",
        "https://alpha.com/2": "# A2\n\nAlpha two covers climate models, rainfall patterns, ocean temperatures, adaptation plans, and policy tradeoffs in detail.",
        "https://beta.com/1": "# B1\n\nBeta one covers Python packaging, virtual environments, lockfiles, dependency resolution, and testing workflows in detail.",
        "https://alpha.com/3": "# A3\n\nAlpha three covers database indexes, write amplification, query planning, disk usage, and cache invalidation in detail.",
    })

    def _ordered_rerank(_query: str, documents: list[str]) -> list[tuple[int, float]]:
        return [(idx, float(len(documents) - idx)) for idx in range(len(documents))]

    rerank_mock = AsyncMock(side_effect=_ordered_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        payload = await server_module.search_impl(
            "diverse query",
            num_results=4,
            ctx=fake_ctx,
        )

    assert [item["domain"] for item in payload["results"][:4]] == [
        "alpha.com",
        "beta.com",
        "alpha.com",
        "alpha.com",
    ]


def test_session_caches_are_bounded():
    """Session caches must have a maxsize + TTL cap to prevent unbounded growth."""
    from cachetools import TTLCache

    cache = server_module._new_cache()
    assert isinstance(cache, TTLCache)
    assert cache.maxsize == server_module._CACHE_MAXSIZE
    assert cache.ttl == server_module._CACHE_TTL_S


@pytest.mark.asyncio
async def test_persisted_state_is_json_serializable(fake_ctx):
    """Every value written to ctx state must survive json.dumps round-trip.

    Real FastMCP Context rejects non-serializable state at runtime — so a
    full search_impl run that completes under FakeContext must leave every
    state key in a JSON-dumpable shape. Pre-fix this regressed: we stored
    raw TTLCache objects and live MCP calls failed with 'not serializable'.
    """
    import json

    search_mock = AsyncMock(return_value=make_search_results(URLS_A[:2]))
    scrape_mock = _make_scrape_mock()
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        await server_module.search_impl("test", num_results=2, ctx=fake_ctx)

    for key in (
        server_module.STATE_SCRAPE_CACHE,
        server_module.STATE_QUERY_CACHE,
        server_module.STATE_SEEN_URLS,
    ):
        value = fake_ctx.get_state(key)
        assert value is not None, f"state key {key!r} was never written"
        # Should not raise — hardened FakeContext also enforces this,
        # but explicit assertion here documents the contract.
        json.dumps(value)
