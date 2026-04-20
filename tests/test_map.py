from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import server_module

PATCH_DISCOVER_PAGE_LINKS = "web_search_server._discover_page_links"


@pytest.mark.asyncio
async def test_map_validates_url():
    with pytest.raises(ValueError, match="invalid URL"):
        await server_module.map_impl("notaurl")


@pytest.mark.asyncio
async def test_map_discovers_same_domain_links_only():
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

    returned_urls = [result["url"] for result in payload["results"]]
    assert returned_urls == [
        "https://docs.example.com",
        "https://docs.example.com/guide",
    ]
    assert all("blog.example.com" not in url for url in returned_urls)
    assert all("other.example.net" not in url for url in returned_urls)
    assert payload["meta"]["same_domain_only"] is True


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
