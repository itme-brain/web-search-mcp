from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import server_module

PATCH_DEEP_CRAWL = "core._deep_crawl"


@pytest.mark.asyncio
async def test_crawl_validates_max_urls():
    with pytest.raises(ValueError, match="max_urls must be <= "):
        await server_module.crawl.fn("https://docs.example.com", max_urls=1000)


@pytest.mark.asyncio
async def test_crawl_combines_discovery_and_extraction():
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
            max_urls=2,
        )
    payload_text = payload.content[0].text

    assert "https://docs.example.com" in payload_text
    assert "Install Guide" in payload_text
    assert "https://docs.example.com/guide" in payload_text


@pytest.mark.asyncio
async def test_crawl_preserves_discovery_order():
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
async def test_crawl_collapses_same_body_at_different_urls():
    body = (
        "# Intro\n\nThe Model Context Protocol standardizes how applications "
        "expose tools and data to large language models. It gives developers a "
        "common integration surface so that any MCP-aware client can connect to "
        "any MCP-aware server without bespoke glue code each time."
    )
    deep_crawl_mock = AsyncMock(return_value=[
        {
            "url": "https://example.com/docs/getting-started/intro",
            "metadata": {"title": "Intro", "depth": 0},
            "markdown": {"raw_markdown": body},
        },
        {
            "url": "https://example.com/docs/getting-started",
            "metadata": {"title": "Intro", "depth": 1, "parent_url": "https://example.com"},
            "markdown": {"raw_markdown": body + "\n\n_mirror_"},
        },
        {
            "url": "https://example.com/",
            "metadata": {"title": "Intro", "depth": 1, "parent_url": "https://example.com"},
            "markdown": {"raw_markdown": body + "\n\n_home_"},
        },
    ])

    with patch(PATCH_DEEP_CRAWL, deep_crawl_mock):
        payload = await server_module.crawl_impl(
            "https://example.com",
            max_urls=5,
        )

    assert payload["meta"]["urls_deduplicated"] == 2
    assert payload["meta"]["urls_returned"] == 1
    assert payload["results"][0]["url"] == "https://example.com/docs/getting-started/intro"


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
            max_urls=5,
        )

    assert deep_crawl_mock.call_args.kwargs["include_patterns"] == ["https://docs.example.com/api/*"]
    assert deep_crawl_mock.call_args.kwargs["same_domain_only"] is True
