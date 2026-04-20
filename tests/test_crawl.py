from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import FakeContext, server_module

PATCH_MAP_SITE_IMPL = "web_search_server._map_site_impl"
PATCH_EXTRACT_URLS_IMPL = "web_search_server._extract_urls_impl"


@pytest.mark.asyncio
async def test_crawl_site_validates_max_urls():
    with pytest.raises(ValueError, match="max_urls must be <= "):
        await server_module.crawl.fn("https://docs.example.com", max_urls=1000)


@pytest.mark.asyncio
async def test_crawl_site_combines_map_and_extract_results():
    fake_ctx = FakeContext()
    map_mock = AsyncMock(return_value={
        "url": "https://docs.example.com",
        "results": [
            {
                "rank": 1,
                "url": "https://docs.example.com",
                "normalized_url": "https://docs.example.com",
                "domain": "docs.example.com",
                "title": "Docs Home",
                "link_text": None,
                "depth": 0,
                "discovered_from": None,
                "link_type": "seed",
            },
            {
                "rank": 2,
                "url": "https://docs.example.com/guide",
                "normalized_url": "https://docs.example.com/guide",
                "domain": "docs.example.com",
                "title": "Guide",
                "link_text": "Guide",
                "depth": 1,
                "discovered_from": "https://docs.example.com",
                "link_type": "internal",
            },
        ],
        "meta": {
            "urls_returned": 2,
            "warnings": [],
            "timings_ms": {"total": 5},
        },
    })
    extract_mock = AsyncMock(return_value={
        "query": "installation guide",
        "results": [
            {
                "url": "https://docs.example.com",
                "normalized_url": "https://docs.example.com",
                "domain": "docs.example.com",
                "status": "ok",
                "content_type": "text/html",
                "title": "Docs Home",
                "content": "# Docs",
                "top_chunks": [{"text": "home chunk", "score": 0.3}],
                "cached": False,
                "error": None,
            },
            {
                "url": "https://docs.example.com/guide",
                "normalized_url": "https://docs.example.com/guide",
                "domain": "docs.example.com",
                "status": "ok",
                "content_type": "text/html",
                "title": "Install Guide",
                "content": "# Guide",
                "top_chunks": [{"text": "guide chunk", "score": 0.9}],
                "cached": True,
                "error": None,
            },
        ],
        "meta": {
            "urls_requested": 2,
            "urls_succeeded": 2,
            "urls_failed": 0,
            "timings_ms": {"total": 7},
        },
    })

    with (
        patch(PATCH_MAP_SITE_IMPL, map_mock),
        patch(PATCH_EXTRACT_URLS_IMPL, extract_mock),
    ):
        payload = await server_module.crawl.fn(
            "https://docs.example.com",
            query="installation guide",
            max_urls=2,
            ctx=fake_ctx,
        )

    assert "https://docs.example.com" in payload
    assert "installation guide" in payload
    assert "Install Guide" in payload
    assert "https://docs.example.com/guide" in payload


@pytest.mark.asyncio
async def test_crawl_site_preserves_map_order_without_query():
    map_mock = AsyncMock(return_value={
        "url": "https://docs.example.com",
        "results": [
            {
                "rank": 1,
                "url": "https://docs.example.com",
                "normalized_url": "https://docs.example.com",
                "domain": "docs.example.com",
                "title": "Docs Home",
                "link_text": None,
                "depth": 0,
                "discovered_from": None,
                "link_type": "seed",
            },
            {
                "rank": 2,
                "url": "https://docs.example.com/guide",
                "normalized_url": "https://docs.example.com/guide",
                "domain": "docs.example.com",
                "title": "Guide",
                "link_text": "Guide",
                "depth": 1,
                "discovered_from": "https://docs.example.com",
                "link_type": "internal",
            },
        ],
        "meta": {
            "urls_returned": 2,
            "warnings": [{"type": "link_discovery_failed", "source": "crawl4ai", "detail": "test"}],
            "timings_ms": {"total": 5},
        },
    })
    extract_mock = AsyncMock(return_value={
        "query": None,
        "results": [
            {
                "url": "https://docs.example.com",
                "normalized_url": "https://docs.example.com",
                "domain": "docs.example.com",
                "status": "ok",
                "content_type": "text/html",
                "title": None,
                "content": "# Docs",
                "top_chunks": [],
                "cached": False,
                "error": None,
            },
            {
                "url": "https://docs.example.com/guide",
                "normalized_url": "https://docs.example.com/guide",
                "domain": "docs.example.com",
                "status": "error",
                "content_type": "text/html",
                "title": None,
                "content": "",
                "top_chunks": [],
                "cached": False,
                "error": "extraction failed",
            },
        ],
        "meta": {
            "urls_requested": 2,
            "urls_succeeded": 1,
            "urls_failed": 1,
            "timings_ms": {"total": 7},
        },
    })

    with (
        patch(PATCH_MAP_SITE_IMPL, map_mock),
        patch(PATCH_EXTRACT_URLS_IMPL, extract_mock),
    ):
        payload = await server_module.crawl.fn(
            "https://docs.example.com",
            max_urls=2,
        )

    assert "https://docs.example.com" in payload
    assert "https://docs.example.com/guide" in payload
    assert "warnings:" in payload
