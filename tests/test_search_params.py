from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import URLS_A, make_search_results, server_module

PATCH_SEARCH = "web_search_server._search"
PATCH_SCRAPE = "web_search_server._scrape"
PATCH_RERANK = "web_search_server._rerank_scored"


def _identity_rerank(_query: str, documents: list[str]) -> list[tuple[int, float]]:
    return [(i, 0.0) for i in range(len(documents))]


# ---------------------------------------------------------------------------
# _search() threads minimal params to SearXNG
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_passes_time_range_and_pageno_to_searxng():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"results": []}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("web_search_server.httpx.AsyncClient", return_value=mock_client):
        await server_module._search("test", num_results=5, time_range="week", pageno=2)

    call_kwargs = mock_client.get.call_args
    params = call_kwargs[1]["params"] if "params" in call_kwargs[1] else call_kwargs[0][1]
    assert params.get("q") == "test"
    assert params.get("time_range") == "week"
    assert params.get("pageno") == 2
    assert params.get("number_of_results") == 5


# ---------------------------------------------------------------------------
# Page 2 is fetched only when page 1 (after dedup+filter) is short
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_page2_fetched_when_page1_underfills():
    # Page 1 returns 1 result after dedup; num_results=3 → page 2 needed.
    page1 = make_search_results(URLS_A[:1], prefix="P1")
    page2 = make_search_results(
        ["https://example.com/p2a", "https://example.com/p2b"], prefix="P2"
    )

    async def fake_search(*args, **kwargs):
        return page2 if kwargs.get("pageno", 1) == 2 else page1

    search_mock = AsyncMock(side_effect=fake_search)
    scrape_mock = AsyncMock(return_value={"content": "content", "title": None, "screenshot": None})
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        result = await server_module.search_impl(query="test", num_results=3)

    assert search_mock.call_count == 2
    second_call = search_mock.call_args_list[1]
    assert second_call[1].get("pageno") == 2
    assert result["meta"]["num_results_returned"] == 3


@pytest.mark.asyncio
async def test_page2_skipped_when_page1_already_full():
    # Page 1 already yields 3 unique, non-filtered results → no page 2.
    page1 = make_search_results(URLS_A[:3], prefix="P1")

    search_mock = AsyncMock(return_value=page1)
    scrape_mock = AsyncMock(return_value={"content": "content", "title": None, "screenshot": None})
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        await server_module.search_impl(query="test", num_results=3)

    assert search_mock.call_count == 1


# ---------------------------------------------------------------------------
# Cache key differentiates on retained params
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_key_includes_time_range():
    search_mock = AsyncMock(return_value=make_search_results(URLS_A[:1]))
    scrape_mock = AsyncMock(return_value={"content": "# Page\n\ncontent", "title": None, "screenshot": None})
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        await server_module.search_impl(query="test", num_results=1)
        assert search_mock.call_count == 1

        # same query, different time_range — should NOT hit cache
        await server_module.search_impl(query="test", num_results=1, time_range="week")
        assert search_mock.call_count == 2


# ---------------------------------------------------------------------------
# SearXNG unresponsive_engines is surfaced as per-engine warnings
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unresponsive_engines_emitted_as_warnings():
    """Engines that CAPTCHA'd or errored at SearXNG should show up named."""
    search_mock = AsyncMock(return_value=make_search_results(
        URLS_A[:2],
        unresponsive_engines=[["google", "CAPTCHA"], ["brave", "HTTP 503"]],
    ))
    scrape_mock = AsyncMock(return_value={"content": "# Page\n\ncontent", "title": None, "screenshot": None})
    rerank_mock = AsyncMock(side_effect=_identity_rerank)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        payload = await server_module.search_impl(query="test", num_results=2)

    # Per-engine failures should NOT flip degraded — multi-engine hedge
    # means one upstream CAPTCHAing is business as usual.
    assert payload["meta"]["degraded"] is False
    warnings = payload["meta"]["warnings"]
    engine_warnings = [w for w in warnings if w["type"] == "engine_unresponsive"]
    details = [w["detail"] for w in engine_warnings]
    assert "google: CAPTCHA" in details
    assert "brave: HTTP 503" in details


@pytest.mark.asyncio
async def test_search_separates_reranked_chunks_with_ellipsis_gap():
    """Top-K chunks from different parts of a page must be marked as
    discontinuous so the LLM doesn't read them as continuous prose."""
    page_with_two_distinct_paragraphs = (
        "Paragraph one covers rockets and launch pads and mission control.\n\n"
        "Completely unrelated paragraph about database indexes and write "
        "amplification and cache invalidation strategies."
    )
    search_mock = AsyncMock(return_value=make_search_results(URLS_A[:1]))
    scrape_mock = AsyncMock(return_value={
        "content": page_with_two_distinct_paragraphs,
        "title": "Mixed",
        "screenshot": None,
    })
    # Rerank returns both chunks with equal scores so both are kept.
    def rerank_both(_q, docs):
        return [(i, 1.0) for i in range(len(docs))]
    rerank_mock = AsyncMock(side_effect=rerank_both)

    with (
        patch(PATCH_SEARCH, search_mock),
        patch(PATCH_SCRAPE, scrape_mock),
        patch(PATCH_RERANK, rerank_mock),
    ):
        payload = await server_module.search_impl(query="anything", num_results=1)

    content = payload["results"][0]["content"]
    # When 2+ chunks are kept, they should be separated by the gap marker.
    assert "[…]" in content, f"expected ellipsis gap in content:\n{content}"


def test_dedup_unresponsive_engines_handles_list_and_dict_shapes():
    """SearXNG sometimes ships list pairs, sometimes dicts, sometimes dupes."""
    entries = [
        ["google", "CAPTCHA"],
        ("brave", "HTTP 503"),
        {"name": "duckduckgo", "error": "timeout"},
        {"engine": "bing", "reason": "timeout"},
        ["google", "CAPTCHA"],  # dup of first
        [""],                    # empty engine, skip
        "not-a-shape",           # not a list/tuple/dict, skip
    ]
    result = server_module._dedup_unresponsive_engines(entries)
    assert result == [
        ("google", "CAPTCHA"),
        ("brave", "HTTP 503"),
        ("duckduckgo", "timeout"),
        ("bing", "timeout"),
    ]
