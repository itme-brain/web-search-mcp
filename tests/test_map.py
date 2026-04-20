from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import server_module

PATCH_DISCOVER_PAGE_LINKS = "core._discover_page_links"


@pytest.mark.asyncio
async def test_map_validates_url():
    with pytest.raises(ValueError, match="invalid URL"):
        await server_module.map_impl("notaurl")


@pytest.mark.asyncio
async def test_map_discovers_same_org_links():
    """same_domain_only=True uses registrable-domain match ("same org"):
    docs.example.com, blog.example.com, api.example.com all pass; an
    entirely different TLD/site does not."""
    discover_mock = AsyncMock(side_effect=[
        {
            "status": "ok",
            "url": "https://docs.example.com",
            "title": "Docs Home",
            "links": [
                {
                    "url": "https://docs.example.com/guide",
                    "title": "Guide",
                    "text": "Guide",
                    "link_type": "internal",
                },
                {
                    "url": "https://blog.example.com/post",
                    "title": "Blog",
                    "text": "Blog",
                    "link_type": "external",
                },
                {
                    "url": "https://example.com/about",
                    "title": "About",
                    "text": "About (parent domain)",
                    "link_type": "external",
                },
                {
                    "url": "https://other.example.net/page",
                    "title": "Other",
                    "text": "Other",
                    "link_type": "external",
                },
            ],
        },
    ])

    with patch(PATCH_DISCOVER_PAGE_LINKS, discover_mock):
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
    # different registrable domain (example.net), excluded
    assert "https://other.example.net/page" not in returned_urls
    assert payload["meta"]["same_domain_only"] is True


def test_registrable_domain_basic():
    """Last two labels win for common TLDs."""
    rd = server_module._registrable_domain
    assert rd("docs.pydantic.dev") == "pydantic.dev"
    assert rd("pydantic.dev") == "pydantic.dev"
    assert rd("logfire.pydantic.dev") == "pydantic.dev"
    assert rd("a.b.c.example.com") == "example.com"
    assert rd("example.com") == "example.com"
    assert rd("www.example.com") == "example.com"


@pytest.mark.asyncio
async def test_map_applies_depth_and_patterns():
    discover_mock = AsyncMock(side_effect=[
        {
            "status": "ok",
            "url": "https://docs.example.com",
            "title": "Docs Home",
            "links": [
                {
                    "url": "https://docs.example.com/docs/start",
                    "title": "Start",
                    "text": "Start",
                    "link_type": "internal",
                },
                {
                    "url": "https://docs.example.com/blog/post",
                    "title": "Blog",
                    "text": "Blog",
                    "link_type": "internal",
                },
            ],
        },
        {
            "status": "ok",
            "url": "https://docs.example.com/docs/start",
            "title": "Start",
            "links": [
                {
                    "url": "https://docs.example.com/docs/advanced",
                    "title": "Advanced",
                    "text": "Advanced",
                    "link_type": "internal",
                },
            ],
        },
    ])

    with patch(PATCH_DISCOVER_PAGE_LINKS, discover_mock):
        payload = await server_module.map_impl(
            "https://docs.example.com",
            max_urls=10,
            max_depth=2,
            include_patterns=["https://docs.example.com/docs/*"],
            exclude_patterns=["*advanced*"],
        )

    returned_urls = [result["url"] for result in payload["results"]]
    assert returned_urls == [
        "https://docs.example.com",
        "https://docs.example.com/docs/start",
    ]


@pytest.mark.asyncio
async def test_map_can_include_external_links():
    discover_mock = AsyncMock(side_effect=[
        {
            "status": "ok",
            "url": "https://docs.example.com",
            "title": "Docs Home",
            "links": [
                {
                    "url": "https://other.example.net/page",
                    "title": "Other",
                    "text": "Other",
                    "link_type": "external",
                },
            ],
        },
        {
            "status": "ok",
            "url": "https://other.example.net/page",
            "title": "Other",
            "links": [],
        },
    ])

    with patch(PATCH_DISCOVER_PAGE_LINKS, discover_mock):
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
async def test_discover_page_links_preserves_nav_footer_header():
    """Link discovery must NOT strip nav/footer/header/aside — those are
    exactly where a docs site's link graph lives. Stripping them with the
    default content-extraction config made map return ~0 URLs on real
    sites."""
    captured: dict = {}

    async def fake_post(client, url, priority, crawler_config=None):
        captured["crawler_config"] = crawler_config
        return {"results": [{"links": {"internal": [], "external": []}}]}

    with patch("core._crawl_post", side_effect=fake_post):
        await server_module._discover_page_links("https://example.com")

    cfg = captured["crawler_config"]
    assert cfg is not None, "map config was not passed through"
    excluded = cfg["params"].get("excluded_tags", [])
    for tag in ("nav", "footer", "header", "aside"):
        assert tag not in excluded, (
            f"{tag!r} in excluded_tags would hide the link graph: {excluded}"
        )
