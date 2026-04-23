from unittest.mock import AsyncMock, patch

import core
import pytest

from tests.conftest import server_module

PATCH_DEEP_CRAWL = "core._deep_crawl"


def _page(
    url: str,
    *,
    title: str | None = None,
    depth: int = 1,
    parent: str | None = None,
) -> dict:
    metadata = {"depth": depth}
    if parent is not None:
        metadata["parent_url"] = parent
    if title is not None:
        metadata["title"] = title
    return {
        "url": url,
        "metadata": metadata,
        "title": title,
    }


@pytest.mark.asyncio
async def test_map_validates_url():
    with pytest.raises(ValueError, match="invalid URL"):
        await server_module.map_impl("notaurl")


@pytest.mark.asyncio
async def test_map_returns_root_plus_tree_nodes():
    mock = AsyncMock(return_value=[
        _page("https://docs.example.com", title="Docs Home", depth=0),
        _page("https://docs.example.com/guide", title="Guide", depth=1, parent="https://docs.example.com"),
        _page("https://docs.example.com/guide/install", title="Install", depth=2, parent="https://docs.example.com/guide"),
        _page("https://docs.example.com/api", title="API", depth=1, parent="https://docs.example.com"),
    ])

    with patch(PATCH_DEEP_CRAWL, mock):
        payload = await server_module.map_impl("https://docs.example.com", max_urls=10)

    urls = [r["url"] for r in payload["results"]]
    assert urls[0] == "https://docs.example.com"
    assert payload["results"][0]["depth"] == 0
    assert payload["results"][0]["link_type"] == "seed"
    assert "https://docs.example.com/guide" in urls
    assert "https://docs.example.com/guide/install" in urls
    assert "https://docs.example.com/api" in urls
    install = next(r for r in payload["results"] if r["url"].endswith("/install"))
    assert install["depth"] == 2
    assert install["discovered_from"] == "https://docs.example.com/guide"
    assert payload["meta"]["pages_visited"] == 4


@pytest.mark.asyncio
async def test_map_honors_max_urls():
    mock = AsyncMock(return_value=[
        _page("https://docs.example.com", title="Root", depth=0),
        *[
            _page(f"https://docs.example.com/page-{i}", title=f"P{i}", depth=1, parent="https://docs.example.com")
            for i in range(20)
        ],
    ])

    with patch(PATCH_DEEP_CRAWL, mock):
        payload = await server_module.map_impl("https://docs.example.com", max_urls=5)

    assert len(payload["results"]) == 5
    assert payload["meta"]["urls_returned"] == 5


@pytest.mark.asyncio
async def test_map_applies_include_patterns_through_deep_crawl():
    mock = AsyncMock(return_value=[])

    with patch(PATCH_DEEP_CRAWL, mock):
        await server_module.map_impl(
            "https://docs.example.com",
            max_urls=10,
            include_patterns=["https://docs.example.com/api/*"],
        )

    assert mock.call_args.kwargs["include_patterns"] == ["https://docs.example.com/api/*"]


@pytest.mark.asyncio
async def test_map_deduplicates_on_normalized_url():
    mock = AsyncMock(return_value=[
        _page("https://docs.example.com", title="Root", depth=0),
        _page("https://docs.example.com/guide/", title="A", depth=1, parent="https://docs.example.com"),
        _page("https://docs.example.com/guide", title="B", depth=1, parent="https://docs.example.com"),
        _page("https://docs.example.com/guide?utm_source=x", title="C", depth=1, parent="https://docs.example.com"),
    ])

    with patch(PATCH_DEEP_CRAWL, mock):
        payload = await server_module.map_impl("https://docs.example.com", max_urls=10)

    guide_count = sum(1 for r in payload["results"] if "/guide" in r["url"])
    assert guide_count == 1


@pytest.mark.asyncio
async def test_map_sparse_discovery_returns_root_only():
    mock = AsyncMock(return_value=[])

    with patch(PATCH_DEEP_CRAWL, mock):
        payload = await server_module.map_impl("https://docs.example.com", max_urls=10)

    assert len(payload["results"]) == 1
    assert payload["results"][0]["url"] == "https://docs.example.com"
    assert payload["meta"]["warnings"] == []
    assert payload["meta"]["pages_visited"] == 1


@pytest.mark.asyncio
async def test_map_discovery_failure_emits_warning_and_returns_root():
    mock = AsyncMock(side_effect=RuntimeError("crawl4ai offline"))

    with patch(PATCH_DEEP_CRAWL, mock):
        payload = await server_module.map_impl("https://docs.example.com", max_urls=5)

    assert len(payload["results"]) == 1
    assert payload["results"][0]["url"] == "https://docs.example.com"
    assert any(w["type"] == "link_discovery_failed" for w in payload["meta"]["warnings"])
    assert payload["meta"]["pages_visited"] == 0


@pytest.mark.asyncio
async def test_map_max_urls_exceeds_available_returns_all():
    mock = AsyncMock(return_value=[
        _page("https://docs.example.com", title="Root", depth=0),
        _page("https://docs.example.com/a", title="A", depth=1, parent="https://docs.example.com"),
        _page("https://docs.example.com/b", title="B", depth=1, parent="https://docs.example.com"),
    ])

    with patch(PATCH_DEEP_CRAWL, mock):
        payload = await server_module.map_impl("https://docs.example.com", max_urls=50)

    assert len(payload["results"]) == 3
    assert payload["meta"]["urls_returned"] == 3


def test_registrable_domain_basic():
    rd = server_module._registrable_domain
    assert rd("docs.pydantic.dev") == "pydantic.dev"
    assert rd("pydantic.dev") == "pydantic.dev"
    assert rd("logfire.pydantic.dev") == "pydantic.dev"
    assert rd("a.b.c.example.com") == "example.com"
    assert rd("example.com") == "example.com"
    assert rd("www.example.com") == "example.com"
    assert rd("docs.service.example.co.uk") == "example.co.uk"


def test_deep_crawl_config_uses_discovery_base_config():
    config = core._deep_crawl_config(
        root_url="https://docs.example.com",
        max_depth=3,
        max_pages=25,
        same_domain_only=True,
        include_patterns=None,
    )

    params = config["params"]
    assert params["remove_overlay_elements"] is True
    assert "excluded_tags" not in params
    assert "markdown_generator" not in params
    assert params["deep_crawl_strategy"]["type"] == "BFSDeepCrawlStrategy"
