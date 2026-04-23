from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import server_module

PATCH_DISCOVER = "core._discover_page_links"


def _discovery(root_title: str, links: list[dict]) -> dict:
    """Shape what _discover_page_links returns."""
    return {
        "status": "ok",
        "url": "https://ignored-by-impl",
        "title": root_title,
        "links": links,
    }


@pytest.mark.asyncio
async def test_map_validates_url():
    with pytest.raises(ValueError, match="invalid URL"):
        await server_module.map_impl("notaurl")


@pytest.mark.asyncio
async def test_map_returns_root_plus_in_scope_links():
    """Happy path: root at depth 0, in-scope links at depth 1, ranked."""
    mock = AsyncMock(return_value=_discovery("Docs Home", [
        {"url": "https://docs.example.com/guide", "title": "Guide", "text": "Guide", "link_type": "internal"},
        {"url": "https://blog.example.com/post", "title": "Blog", "text": "Blog", "link_type": "external"},
        {"url": "https://example.com/about", "title": "About", "text": "About", "link_type": "external"},
    ]))

    with patch(PATCH_DISCOVER, mock):
        payload = await server_module.map_impl("https://docs.example.com", max_urls=10)

    urls = [r["url"] for r in payload["results"]]
    assert urls[0] == "https://docs.example.com"
    assert payload["results"][0]["depth"] == 0
    assert payload["results"][0]["link_type"] == "seed"
    # All three links share the registrable domain example.com and are in-scope.
    assert "https://docs.example.com/guide" in urls
    assert "https://blog.example.com/post" in urls
    assert "https://example.com/about" in urls
    assert all(r["rank"] == i + 1 for i, r in enumerate(payload["results"]))


@pytest.mark.asyncio
async def test_map_honors_max_urls():
    """Return at most max_urls entries including the root."""
    mock = AsyncMock(return_value=_discovery("Many", [
        {"url": f"https://docs.example.com/page-{i}", "title": f"P{i}", "text": None, "link_type": "internal"}
        for i in range(20)
    ]))

    with patch(PATCH_DISCOVER, mock):
        payload = await server_module.map_impl("https://docs.example.com", max_urls=5)

    assert len(payload["results"]) == 5
    assert payload["meta"]["urls_returned"] == 5


@pytest.mark.asyncio
async def test_map_drops_cross_domain_links():
    """Links outside the registrable domain are filtered out."""
    mock = AsyncMock(return_value=_discovery("Home", [
        {"url": "https://docs.example.com/in-scope", "title": "In", "text": None, "link_type": "internal"},
        {"url": "https://unrelated.org/external", "title": "Out", "text": None, "link_type": "external"},
    ]))

    with patch(PATCH_DISCOVER, mock):
        payload = await server_module.map_impl("https://docs.example.com", max_urls=10)

    urls = [r["url"] for r in payload["results"]]
    assert "https://docs.example.com/in-scope" in urls
    assert "https://unrelated.org/external" not in urls


@pytest.mark.asyncio
async def test_map_applies_include_patterns():
    """include_patterns acts as a glob filter on the full URL."""
    mock = AsyncMock(return_value=_discovery("Home", [
        {"url": "https://docs.example.com/api/auth", "title": "Auth", "text": None, "link_type": "internal"},
        {"url": "https://docs.example.com/blog/post", "title": "Blog", "text": None, "link_type": "internal"},
        {"url": "https://docs.example.com/api/users", "title": "Users", "text": None, "link_type": "internal"},
    ]))

    with patch(PATCH_DISCOVER, mock):
        payload = await server_module.map_impl(
            "https://docs.example.com",
            max_urls=10,
            include_patterns=["https://docs.example.com/api/*"],
        )

    urls = [r["url"] for r in payload["results"]]
    # Root passes through regardless; only api/* links join it.
    assert "https://docs.example.com/api/auth" in urls
    assert "https://docs.example.com/api/users" in urls
    assert "https://docs.example.com/blog/post" not in urls


@pytest.mark.asyncio
async def test_map_deduplicates_on_normalized_url():
    """URL variants that normalize to the same thing collapse."""
    mock = AsyncMock(return_value=_discovery("Home", [
        # Trailing slash is stripped by _normalize_url.
        {"url": "https://docs.example.com/guide/", "title": "A", "text": None, "link_type": "internal"},
        {"url": "https://docs.example.com/guide", "title": "B", "text": None, "link_type": "internal"},
        # utm_* is dropped by _normalize_url.
        {"url": "https://docs.example.com/guide?utm_source=x", "title": "C", "text": None, "link_type": "internal"},
    ]))

    with patch(PATCH_DISCOVER, mock):
        payload = await server_module.map_impl("https://docs.example.com", max_urls=10)

    guide_count = sum(1 for r in payload["results"] if "/guide" in r["url"])
    assert guide_count == 1, "trailing-slash + utm variants should collapse to one entry"


@pytest.mark.asyncio
async def test_map_sparse_discovery_returns_root_only_with_no_fallback():
    """Regression guard: if discovery returns zero links we return the
    root as the single result and do NOT invoke any secondary path."""
    mock = AsyncMock(return_value=_discovery("Lonely", []))

    with patch(PATCH_DISCOVER, mock):
        payload = await server_module.map_impl("https://docs.example.com", max_urls=10)

    assert len(payload["results"]) == 1
    assert payload["results"][0]["url"] == "https://docs.example.com"
    assert payload["meta"]["warnings"] == []
    mock.assert_called_once()


@pytest.mark.asyncio
async def test_map_discovery_failure_emits_warning_and_returns_root():
    """If Crawl4AI itself fails, surface the warning but still return the
    root as a single entry — the caller knows what they asked for."""
    mock = AsyncMock(side_effect=RuntimeError("crawl4ai offline"))

    with patch(PATCH_DISCOVER, mock):
        payload = await server_module.map_impl("https://docs.example.com", max_urls=5)

    assert len(payload["results"]) == 1
    assert payload["results"][0]["url"] == "https://docs.example.com"
    assert any(w["type"] == "link_discovery_failed" for w in payload["meta"]["warnings"])


@pytest.mark.asyncio
async def test_map_max_urls_exceeds_available_returns_all():
    """Graceful: when fewer links exist than requested, return what's
    available rather than erroring."""
    mock = AsyncMock(return_value=_discovery("Small", [
        {"url": "https://docs.example.com/a", "title": "A", "text": None, "link_type": "internal"},
        {"url": "https://docs.example.com/b", "title": "B", "text": None, "link_type": "internal"},
    ]))

    with patch(PATCH_DISCOVER, mock):
        payload = await server_module.map_impl("https://docs.example.com", max_urls=50)

    assert len(payload["results"]) == 3  # root + 2 discovered
    assert payload["meta"]["urls_returned"] == 3


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
