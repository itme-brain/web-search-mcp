from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import FakeContext, server_module

PATCH_DEEP_CRAWL = "core._deep_crawl"


@pytest.mark.asyncio
async def test_crawl_validates_max_urls():
    with pytest.raises(ValueError, match="max_urls must be <= "):
        await server_module.crawl.fn("https://docs.example.com", max_urls=1000)


@pytest.mark.asyncio
async def test_crawl_combines_map_and_extract_results():
    fake_ctx = FakeContext()
    deep_crawl_mock = AsyncMock(return_value=[
        {
            "url": "https://docs.example.com",
            "metadata": {"title": "Docs Home", "depth": 0},
            "markdown": {"raw_markdown": "# Docs\n\nhome chunk"},
        },
        {
            "url": "https://docs.example.com/guide",
            "metadata": {"title": "Install Guide", "depth": 1, "parent_url": "https://docs.example.com"},
            "markdown": {"raw_markdown": "# Guide\n\ninstallation guide chunk"},
        },
    ])

    with patch(PATCH_DEEP_CRAWL, deep_crawl_mock):
        payload = await server_module.crawl.fn(
            "https://docs.example.com",
            query="installation guide",
            max_urls=2,
            ctx=fake_ctx,
        )
    payload_text = payload.content[0].text

    assert "https://docs.example.com" in payload_text
    assert "installation guide" in payload_text
    assert "Install Guide" in payload_text
    assert "https://docs.example.com/guide" in payload_text


@pytest.mark.asyncio
async def test_crawl_preserves_map_order_without_query():
    deep_crawl_mock = AsyncMock(return_value=[
        {
            "url": "https://docs.example.com",
            "metadata": {"title": "Docs Home", "depth": 0},
            "markdown": {"raw_markdown": "# Docs"},
        },
        {
            "url": "https://docs.example.com/guide",
            "metadata": {"title": "Guide", "depth": 1, "parent_url": "https://docs.example.com"},
            "markdown": {"raw_markdown": "# Guide"},
        },
    ])

    with patch(PATCH_DEEP_CRAWL, deep_crawl_mock):
        payload = await server_module.crawl.fn(
            "https://docs.example.com",
            max_urls=2,
    )
    payload_text = payload.content[0].text

    assert "https://docs.example.com" in payload_text
    assert "https://docs.example.com/guide" in payload_text
    assert "warnings:" not in payload_text


@pytest.mark.asyncio
async def test_crawl_query_ranking_prefers_specific_docs_page_over_site_root():
    deep_crawl_mock = AsyncMock(return_value=[
        {
            "url": "https://docs.djangoproject.com",
            "metadata": {"title": "Django documentation", "depth": 0, "score": 0.35},
            "markdown": {"raw_markdown": "# Django documentation\n\nDatabase docs and navigation links."},
        },
        {
            "url": "https://docs.djangoproject.com/en/5.2/topics/db/models/",
            "metadata": {
                "title": "Django ORM models",
                "depth": 1,
                "parent_url": "https://docs.djangoproject.com",
                "score": 0.93,
            },
            "markdown": {"raw_markdown": "# Models\n\nThe Django ORM maps models to database tables."},
        },
    ])

    with patch(PATCH_DEEP_CRAWL, deep_crawl_mock):
        payload = await server_module.crawl_impl(
            "https://docs.djangoproject.com",
            query="ORM",
        )

    assert payload["results"][0]["url"] == "https://docs.djangoproject.com/en/5.2/topics/db/models/"
    assert payload["results"][1]["url"] == "https://docs.djangoproject.com"


@pytest.mark.asyncio
async def test_crawl_passes_include_patterns_to_deep_crawl():
    deep_crawl_mock = AsyncMock(return_value=[
        {
            "url": "https://docs.example.com",
            "metadata": {"title": "Docs Home", "depth": 0},
            "markdown": {"raw_markdown": "# Docs"},
        },
        {
            "url": "https://blog.example.com/post",
            "metadata": {"title": "Blog", "depth": 1},
            "markdown": {"raw_markdown": "# Blog"},
        },
        {
            "url": "https://docs.example.com/api/auth",
            "metadata": {"title": "Auth", "depth": 1},
            "markdown": {"raw_markdown": "# Auth"},
        },
    ])

    with patch(PATCH_DEEP_CRAWL, deep_crawl_mock):
        await server_module.crawl_impl(
            "https://docs.example.com",
            include_patterns=["https://docs.example.com/api/*"],
            same_domain_only=True,
            max_urls=5,
        )

    assert deep_crawl_mock.call_args.kwargs["include_patterns"] == ["https://docs.example.com/api/*"]
    assert deep_crawl_mock.call_args.kwargs["same_domain_only"] is True
