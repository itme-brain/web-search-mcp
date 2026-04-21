from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import server_module

PATCH_DEEP_CRAWL = "core._deep_crawl"


@pytest.mark.asyncio
async def test_map_validates_url():
    with pytest.raises(ValueError, match="invalid URL"):
        await server_module.map_impl("notaurl")


@pytest.mark.asyncio
async def test_map_discovers_same_org_links():
    deep_crawl_mock = AsyncMock(return_value=[
        {
            "url": "https://docs.example.com",
            "metadata": {"title": "Docs Home", "depth": 0},
        },
        {
            "url": "https://docs.example.com/guide",
            "metadata": {"title": "Guide", "depth": 1, "parent_url": "https://docs.example.com"},
        },
        {
            "url": "https://blog.example.com/post",
            "metadata": {"title": "Blog", "depth": 1, "parent_url": "https://docs.example.com"},
        },
        {
            "url": "https://example.com/about",
            "metadata": {"title": "About", "depth": 1, "parent_url": "https://docs.example.com"},
        },
    ])

    with patch(PATCH_DEEP_CRAWL, deep_crawl_mock):
        payload = await server_module.map_impl(
            "https://docs.example.com",
            max_urls=10,
            max_depth=1,
        )

    returned_urls = set(result["url"] for result in payload["results"])
    # all share registrable domain example.com
    assert "https://docs.example.com" in returned_urls
    assert "https://docs.example.com/guide" in returned_urls
    assert "https://blog.example.com/post" in returned_urls
    assert "https://example.com/about" in returned_urls
    assert payload["meta"]["same_domain_only"] is True
    assert deep_crawl_mock.call_args.kwargs["same_domain_only"] is True


def test_registrable_domain_basic():
    """Registrable domains should follow the public suffix list."""
    rd = server_module._registrable_domain
    assert rd("docs.pydantic.dev") == "pydantic.dev"
    assert rd("pydantic.dev") == "pydantic.dev"
    assert rd("logfire.pydantic.dev") == "pydantic.dev"
    assert rd("a.b.c.example.com") == "example.com"
    assert rd("example.com") == "example.com"
    assert rd("www.example.com") == "example.com"
    assert rd("docs.service.example.co.uk") == "example.co.uk"


@pytest.mark.asyncio
async def test_map_applies_depth_and_patterns():
    deep_crawl_mock = AsyncMock(return_value=[
        {
            "url": "https://docs.example.com",
            "metadata": {"title": "Docs Home", "depth": 0},
        },
        {
            "url": "https://docs.example.com/docs/start",
            "metadata": {"title": "Start", "depth": 1, "parent_url": "https://docs.example.com"},
        },
        {
            "url": "https://docs.example.com/blog/post",
            "metadata": {"title": "Blog", "depth": 1, "parent_url": "https://docs.example.com"},
        },
        {
            "url": "https://docs.example.com/docs/advanced",
            "metadata": {"title": "Advanced", "depth": 2, "parent_url": "https://docs.example.com/docs/start"},
        },
    ])

    with patch(PATCH_DEEP_CRAWL, deep_crawl_mock):
        payload = await server_module.map_impl(
            "https://docs.example.com",
            max_urls=10,
            max_depth=2,
            include_patterns=["https://docs.example.com/docs/*"],
        )

    returned_urls = [result["url"] for result in payload["results"]]
    assert returned_urls == [
        "https://docs.example.com",
        "https://docs.example.com/docs/start",
        "https://docs.example.com/blog/post",
        "https://docs.example.com/docs/advanced",
    ]
    assert deep_crawl_mock.call_args.kwargs["include_patterns"] == ["https://docs.example.com/docs/*"]


@pytest.mark.asyncio
async def test_map_can_include_external_links():
    deep_crawl_mock = AsyncMock(return_value=[
        {
            "url": "https://docs.example.com",
            "metadata": {"title": "Docs Home", "depth": 0},
        },
        {
            "url": "https://other.example.net/page",
            "metadata": {"title": "Other", "depth": 1, "parent_url": "https://docs.example.com"},
        },
    ])

    with patch(PATCH_DEEP_CRAWL, deep_crawl_mock):
        payload = await server_module.map_impl(
            "https://docs.example.com",
            same_domain_only=False,
            max_urls=10,
            max_depth=1,
        )

    returned_urls = [result["url"] for result in payload["results"]]
    assert returned_urls == [
        "https://docs.example.com",
        "https://other.example.net/page",
    ]


@pytest.mark.asyncio
async def test_map_uses_prefetch_deep_crawl():
    captured: dict = {}

    async def fake_deep_crawl(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return []

    with patch(PATCH_DEEP_CRAWL, side_effect=fake_deep_crawl):
        await server_module.map_impl(
            "https://example.com",
            max_urls=5,
            max_depth=2,
            include_patterns=["https://example.com/docs/*"],
        )

    assert captured["url"] == "https://example.com"
    assert captured["prefetch"] is True
    assert captured["same_domain_only"] is True
    assert captured["include_patterns"] == ["https://example.com/docs/*"]
    assert captured["max_pages"] == 5
    assert captured["max_depth"] == 2


@pytest.mark.asyncio
async def test_deep_crawl_config_uses_prefetch_and_preserves_link_graph():
    cfg = server_module._deep_crawl_config(
        root_url="https://example.com",
        max_depth=2,
        max_pages=5,
        same_domain_only=True,
        include_patterns=["https://example.com/docs/*"],
        prefetch=True,
    )

    params = cfg["params"]
    assert params["prefetch"] is True
    excluded = params.get("excluded_tags", [])
    for tag in ("nav", "footer", "header", "aside"):
        assert tag not in excluded, (
            f"{tag!r} in excluded_tags would hide the link graph: {excluded}"
        )
