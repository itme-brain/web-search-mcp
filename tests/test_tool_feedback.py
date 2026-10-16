"""Regressions for real model feedback: strict dates and honest failure signals."""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from tests.conftest import make_search_results
from tests.test_crawl import _map_result, _extract_result
from web_search_mcp.config.freshness import filter_dated_entries
from web_search_mcp.crawling import operations
from web_search_mcp.extraction.blocks import block_diagnostic
from web_search_mcp.presentation.formatters import _format_extract_results, _format_search_results
from web_search_mcp.storage import cache, pages
from web_search_mcp.tools import search, crawl, extract

BODY = "Useful documentation about Python asyncio TaskGroup exceptions and handling child tasks. " * 20
BLOCK = "# Sorry, you have been blocked\nYou are unable to access example.com. Cloudflare Ray ID: abc123. Performance & security by Cloudflare."


def test_strict_date_window_excludes_old_unknown_and_future_sources():
    now = datetime(2026, 10, 16, 12, tzinfo=timezone.utc)
    entries = [{"url": name, "metadata": {"date": value}} for name, value in [
        ("recent", "2026-10-16T10:00:00+02:00"), ("old", "2024-01-01"),
        ("future", "2026-11-01"), ("unknown", None), ("invalid", "yesterday maybe"),
        ("rfc822", "Thu, 15 Oct 2026 12:00:00 GMT"),
    ]]
    eligible, report = filter_dated_entries(entries, "week", now=now)
    assert [e["url"] for e in eligible] == ["recent", "rfc822"]
    assert report["excluded_undated"] == report["excluded_outside_window"] == 2


@pytest.mark.asyncio
async def test_dated_search_recovers_from_news_captcha_without_relaxing_dates(monkeypatch):
    now = datetime.now(timezone.utc)
    payload = make_search_results([f"https://source{i}.example/doc" for i in range(4)])
    for item, date in zip(payload["results"], [now - timedelta(hours=1), now - timedelta(days=100), None, now + timedelta(days=3)]):
        item["publishedDate"] = date.isoformat() if date else None
    upstream = AsyncMock(side_effect=[{"results": [], "unresponsive_engines": [["bing news", "CAPTCHA"]]}, payload])
    monkeypatch.setattr(search, "_search", upstream)
    monkeypatch.setattr(search, "_scrape_cached", AsyncMock(return_value={"content": BODY}))
    result = await search.search_impl("latest Python announcement", num_results=1, time_range="week")
    assert [r["url"] for r in result["results"]] == ["https://source0.example/doc"]
    assert upstream.call_args_list[0].kwargs["time_range"] == "week"
    assert upstream.call_args_list[1].kwargs["time_range"] is None
    assert result["meta"]["date_filter"]["policy"] == "strict"
    assert not result["meta"]["degraded"]


@pytest.mark.asyncio
async def test_recovered_fetch_failure_is_warning_without_retry_signal(monkeypatch):
    monkeypatch.setattr(search, "_search", AsyncMock(return_value=make_search_results([
        "https://blocked.example/a", "https://docs.example/b", "https://other.example/c",
    ])))
    async def scrape(url, *args, **kwargs):
        return {"content": None, "metadata": {"http_status": 403}} if "blocked" in url else {"content": BODY}
    monkeypatch.setattr(search, "_scrape_cached", scrape)
    result = await search.search_impl("Python asyncio documentation", num_results=2)
    assert len(result["results"]) == 2
    assert not result["meta"]["degraded"]
    assert any(w["type"] == "scrape_failed" for w in result["meta"]["warnings"])
    assert not any("retry" in action.lower() for action in result["meta"]["next_actions"])


@pytest.mark.asyncio
async def test_unrecovered_fetch_failure_recommends_retry(monkeypatch):
    monkeypatch.setattr(search, "_search", AsyncMock(return_value=make_search_results(["https://blocked.example/a"])))
    monkeypatch.setattr(search, "_scrape_cached", AsyncMock(return_value={"content": None}))
    result = await search.search_impl("Python asyncio documentation", num_results=3)
    assert result["meta"]["degraded"]
    assert any("retry" in action.lower() for action in result["meta"]["next_actions"])


@pytest.mark.asyncio
async def test_cloudflare_http200_shell_is_not_a_document(monkeypatch):
    monkeypatch.setattr(operations, "_crawl_post", AsyncMock(return_value={"results": [{
        "success": True, "status_code": 200, "markdown": BLOCK,
    }]}))
    result = await operations._scrape_impl("https://blocked.example/a")
    assert result["content"] is None
    assert result["metadata"]["diagnostic"] == "cloudflare_challenge"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["ok", "error"])
async def test_cached_cloudflare_body_never_appears_as_content_or_error_none(status):
    url = "https://blocked.example/cached"
    await cache.page_cache.set(url, {"status": status, "url": url, "content": BLOCK,
                                    "title": "Attention Required! | Cloudflare", "metadata": {}, "error": None})
    result = await extract.extract_impl([url])
    document = result["results"][0]
    assert document["status"] == "error"
    assert document["content"] == ""
    assert document["error"]
    text = _format_extract_results(result)
    assert "Error:** None" not in text
    assert "Cloudflare Ray ID" not in text


def test_articles_about_cloudflare_are_not_rejected():
    assert block_diagnostic("Cloudflare troubleshooting guide. Example errors contain Cloudflare Ray ID and say 'Sorry, you have been blocked'. " * 20,
                            title="Cloudflare troubleshooting guide") is None


def test_search_passages_from_snippets_are_labelled_as_snippets():
    text = _format_search_results({"query": "test", "meta": {}, "results": [{
        "title": "Article", "url": "https://example.com", "rank": 1,
        "scraped": False, "passages": [{"text": "Search snippet evidence", "citation": "1.1"}],
    }]})
    assert "snippet" in text
    assert "scraped" not in text


@pytest.mark.asyncio
async def test_sparse_leaf_crawl_explains_scope_and_suggests_parent(monkeypatch):
    url = "https://docs.example/library/asyncio.html"
    monkeypatch.setattr(crawl, "map_impl", AsyncMock(return_value={"url": url,
        "results": [_map_result(url, depth=0, link_type="seed")], "meta": {"warnings": []}}))
    monkeypatch.setattr(crawl, "extract_impl", AsyncMock(return_value={
        "results": [_extract_result(url, content=BODY)],
        "meta": {"urls_succeeded": 1, "urls_failed": 0}}))
    result = await crawl.crawl_impl(url, max_urls=5)
    reason = result["meta"]["sparsity_reason"]
    assert "leaf page" in reason
    assert "does not measure the size of the site" in reason
    assert "https://docs.example/library/" in reason
    assert "extract" in reason
