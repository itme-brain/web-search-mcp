import asyncio
from datetime import datetime, timezone
from email.utils import format_datetime
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client

from web_search_mcp.crawling import operations
from web_search_mcp.extraction import documents
from web_search_mcp.storage import cache, pages
from web_search_mcp.tools import search as search_tool
from tests.conftest import server_app


@pytest.fixture(autouse=True)
def clear_cooldowns():
    operations._rate_limits.clear()
    yield
    operations._rate_limits.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("status, diagnostic", [(429, "rate_limited"), (403, "access_denied")])
async def test_origin_error_is_not_returned_as_page_content(monkeypatch, status, diagnostic):
    monkeypatch.setattr(operations, "_crawl_post", AsyncMock(return_value={"results": [{
        "success": False, "status_code": status,
        "markdown": "This is an error page, not the requested article. " * 10,
        "error_message": f"Blocked by anti-bot protection: HTTP {status}",
        "response_headers": {"Retry-After": "120"},
    }]}))
    result = await operations._scrape_impl("https://blocked.example/article")
    assert result["content"] is None
    assert result["metadata"]["diagnostic"] == diagnostic
    assert result["metadata"]["http_status"] == status
    if status == 429:
        assert result["metadata"]["retry_after_seconds"] == 120


def test_status_can_be_recovered_from_crawler_error():
    metadata = operations._crawl_failure_metadata({
        "success": False, "error_message": "Blocked: HTTP 429 Too Many Requests",
    })
    assert metadata["http_status"] == 429
    assert metadata["retry_after_seconds"] == 60


def test_retry_after_http_date(monkeypatch):
    monkeypatch.setattr(operations.time, "time", lambda: 1000)
    date = format_datetime(datetime.fromtimestamp(1120, timezone.utc), usegmt=True)
    assert operations._retry_after_seconds({"retry-after": date}) == 120


@pytest.mark.asyncio
async def test_rate_limit_covers_other_paths_but_not_other_hosts(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(operations.time, "monotonic", lambda: clock[0])
    limited = {"content": None, "metadata": {
        "diagnostic": "rate_limited", "http_status": 429, "retry_after_seconds": 120,
    }}
    scrape = AsyncMock(return_value=limited)
    monkeypatch.setattr(operations, "_scrape_impl", scrape)
    await operations._scrape("https://blocked.example/a")
    second = await operations._scrape("https://blocked.example/b")
    assert scrape.await_count == 1
    assert second["metadata"]["diagnostic"] == "rate_limited"
    await operations._scrape("https://other.example/a")
    assert scrape.await_count == 2
    clock[0] += 121
    await operations._scrape("https://blocked.example/b")
    assert scrape.await_count == 3


@pytest.mark.asyncio
async def test_same_host_fetches_are_serialized(monkeypatch):
    active = 0
    peak = 0

    async def scrape(_url):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return {"content": "ok", "metadata": {}}

    monkeypatch.setattr(operations, "_scrape_impl", scrape)
    await asyncio.gather(*(operations._scrape(f"https://same.example/{i}") for i in range(4)))
    assert peak == 1


@pytest.mark.asyncio
async def test_cache_preserves_origin_failure_diagnostic(monkeypatch):
    monkeypatch.setattr(pages, "_scrape", AsyncMock(return_value={
        "content": "Too Many Requests", "metadata": {
            "diagnostic": "rate_limited", "http_status": 429,
        },
    }))
    result = await pages._scrape_cached("https://limited.example/page", cache.page_cache)
    assert result["status"] == "error"
    assert result["metadata"]["diagnostic"] == "rate_limited"
    assert result["metadata"]["http_status"] == 429


@pytest.mark.asyncio
async def test_extract_preserves_rate_limit_details(monkeypatch):
    metadata = {"diagnostic": "rate_limited", "http_status": 429, "retry_after_seconds": 120}
    monkeypatch.setattr(documents, "_scrape", AsyncMock(return_value={
        "content": None, "metadata": metadata,
    }))
    result = await documents._extract_web_document("https://limited.example/page")
    assert result["status"] == "error"
    assert result["metadata"] == metadata
    assert result["error"] == "rate_limited"


@pytest.mark.asyncio
async def test_mcp_search_accepts_rate_limit_metadata_and_falls_back_to_snippet(monkeypatch):
    monkeypatch.setattr(search_tool, "_search", AsyncMock(return_value={
        "results": [{"url": "https://limited.example/article", "title": "Article",
                     "content": "Useful article snippet about Python asyncio semaphores."}],
        "unresponsive_engines": [],
    }))
    monkeypatch.setattr(pages, "_scrape", AsyncMock(return_value={
        "content": None, "metadata": {"diagnostic": "rate_limited", "http_status": 429,
                                      "retry_after_seconds": 120, "error": "HTTP 429"},
    }))
    async with Client(server_app) as client:
        result = await client.call_tool_mcp("search", {"query": "Python asyncio semaphores", "num_results": 1})
    assert not result.is_error
    data = result.structured_content
    assert data["results"][0]["metadata"]["http_status"] == 429
    assert data["results"][0]["metadata"]["retry_after_seconds"] == 120
    assert data["results"][0]["scraped"] is False
    assert any("rate_limited" in item["detail"] for item in data["meta"]["warnings"])
