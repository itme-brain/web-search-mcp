from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import server_module

PATCH_DISCOVER = "core._discover_page_links"
PATCH_SCRAPE_CACHED = "core._scrape_cached"
PATCH_DEEP_CRAWL = "core._deep_crawl"


def _discovery(title: str, links: list[dict]) -> dict:
    return {
        "status": "ok",
        "url": "https://ignored",
        "title": title,
        "links": links,
    }


def _scrape(content: str, title: str | None = None) -> dict:
    return {"content": content, "title": title, "metadata": {}}


def _bfs_page(url: str, content: str, *, depth: int = 1, parent: str | None = None) -> dict:
    """A page in the shape Crawl4AI's /crawl/stream returns."""
    return {
        "url": url,
        "metadata": {"depth": depth, "parent_url": parent},
        "markdown": {"raw_markdown": content},
    }


@pytest.mark.asyncio
async def test_crawl_validates_max_urls():
    with pytest.raises(ValueError, match="max_urls must be <= "):
        await server_module.crawl.fn("https://docs.example.com", max_urls=1000)


@pytest.mark.asyncio
async def test_crawl_happy_path_returns_content_for_root_and_depth1():
    """Every returned result has non-empty content, correctly depth-labelled."""
    discovery_mock = AsyncMock(return_value=_discovery("Docs", [
        {"url": "https://docs.example.com/guide", "title": "Guide", "text": "Guide", "link_type": "internal"},
        {"url": "https://docs.example.com/api", "title": "API", "text": "API", "link_type": "internal"},
    ]))
    scrape_map = {
        "https://docs.example.com": _scrape("# Root\n\nroot page body paragraph one paragraph two.", "Docs Home"),
        "https://docs.example.com/guide": _scrape("# Guide\n\nguide body with enough chars.", "Guide"),
        "https://docs.example.com/api": _scrape("# API\n\nAPI body with enough chars.", "API"),
    }
    scrape_mock = AsyncMock(side_effect=lambda url, cache: scrape_map[url])
    bfs_mock = AsyncMock(return_value=[])  # no depth-2 for this case

    with (
        patch(PATCH_DISCOVER, discovery_mock),
        patch(PATCH_SCRAPE_CACHED, scrape_mock),
        patch(PATCH_DEEP_CRAWL, bfs_mock),
    ):
        payload = await server_module.crawl_impl("https://docs.example.com", max_urls=5)

    assert len(payload["results"]) == 3
    for r in payload["results"]:
        assert r["status"] == "ok", f"{r['url']} has no content"
        assert len(r["content"]) > 10
    depths = [r["depth"] for r in payload["results"]]
    assert depths[0] == 0  # root
    assert 1 in depths     # at least one depth-1


@pytest.mark.asyncio
async def test_crawl_honors_max_urls():
    """Never return more pages than requested."""
    discovery_mock = AsyncMock(return_value=_discovery("Big", [
        {"url": f"https://docs.example.com/p{i}", "title": f"P{i}", "text": None, "link_type": "internal"}
        for i in range(20)
    ]))
    scrape_mock = AsyncMock(return_value=_scrape("# Page\n\nfiller filler filler."))
    bfs_mock = AsyncMock(return_value=[])

    with (
        patch(PATCH_DISCOVER, discovery_mock),
        patch(PATCH_SCRAPE_CACHED, scrape_mock),
        patch(PATCH_DEEP_CRAWL, bfs_mock),
    ):
        payload = await server_module.crawl_impl("https://docs.example.com", max_urls=4)

    assert payload["meta"]["urls_returned"] == 4
    assert len(payload["results"]) == 4


@pytest.mark.asyncio
async def test_crawl_filters_cross_domain_links():
    """Links outside the registrable domain are dropped, not scraped."""
    discovery_mock = AsyncMock(return_value=_discovery("Home", [
        {"url": "https://docs.example.com/in", "title": "In", "text": None, "link_type": "internal"},
        {"url": "https://evil.org/out", "title": "Out", "text": None, "link_type": "external"},
    ]))
    scrape_map = {
        "https://docs.example.com": _scrape("# Home\n\nroot body."),
        "https://docs.example.com/in": _scrape("# In\n\nin body."),
    }

    async def _scrape_side(url, cache):
        assert url in scrape_map, f"unexpected cross-domain scrape: {url!r}"
        return scrape_map[url]

    scrape_mock = AsyncMock(side_effect=_scrape_side)
    bfs_mock = AsyncMock(return_value=[])

    with (
        patch(PATCH_DISCOVER, discovery_mock),
        patch(PATCH_SCRAPE_CACHED, scrape_mock),
        patch(PATCH_DEEP_CRAWL, bfs_mock),
    ):
        payload = await server_module.crawl_impl("https://docs.example.com", max_urls=10)

    urls = [r["url"] for r in payload["results"]]
    assert "https://evil.org/out" not in urls


@pytest.mark.asyncio
async def test_crawl_applies_include_patterns():
    """Discovery results filtered by include_patterns before scraping."""
    discovery_mock = AsyncMock(return_value=_discovery("Home", [
        {"url": "https://docs.example.com/api/auth", "title": "Auth", "text": None, "link_type": "internal"},
        {"url": "https://docs.example.com/blog/post", "title": "Blog", "text": None, "link_type": "internal"},
    ]))
    scrape_map = {
        "https://docs.example.com": _scrape("# Home"),
        "https://docs.example.com/api/auth": _scrape("# Auth"),
    }

    async def _scrape_side(url, cache):
        assert url in scrape_map, f"blog path should not be scraped: {url!r}"
        return scrape_map[url]

    scrape_mock = AsyncMock(side_effect=_scrape_side)
    bfs_mock = AsyncMock(return_value=[])

    with (
        patch(PATCH_DISCOVER, discovery_mock),
        patch(PATCH_SCRAPE_CACHED, scrape_mock),
        patch(PATCH_DEEP_CRAWL, bfs_mock),
    ):
        payload = await server_module.crawl_impl(
            "https://docs.example.com",
            max_urls=5,
            include_patterns=["https://docs.example.com/api/*"],
        )

    urls = [r["url"] for r in payload["results"]]
    assert "https://docs.example.com/api/auth" in urls
    assert "https://docs.example.com/blog/post" not in urls
    # The call into _deep_crawl must also carry the include_patterns.
    assert bfs_mock.call_args.kwargs["include_patterns"] == ["https://docs.example.com/api/*"]


@pytest.mark.asyncio
async def test_crawl_deduplicates_url_variants():
    """Trailing-slash / www. variants must collapse to one result."""
    discovery_mock = AsyncMock(return_value=_discovery("Home", [
        {"url": "https://docs.example.com/guide/", "title": "G1", "text": None, "link_type": "internal"},
        {"url": "https://docs.example.com/guide", "title": "G2", "text": None, "link_type": "internal"},
    ]))
    scrape_mock = AsyncMock(return_value=_scrape("# Page\n\nfiller."))
    bfs_mock = AsyncMock(return_value=[])

    with (
        patch(PATCH_DISCOVER, discovery_mock),
        patch(PATCH_SCRAPE_CACHED, scrape_mock),
        patch(PATCH_DEEP_CRAWL, bfs_mock),
    ):
        payload = await server_module.crawl_impl("https://docs.example.com", max_urls=10)

    guide_entries = [r for r in payload["results"] if "/guide" in r["url"]]
    assert len(guide_entries) == 1


@pytest.mark.asyncio
async def test_crawl_sparse_discovery_returns_root_plus_seeds():
    """Regression guard: small discovery (3 links) yields root + 3 pages,
    each with content. This is the old BFS-silent-fail failure mode."""
    discovery_mock = AsyncMock(return_value=_discovery("Home", [
        {"url": "https://docs.example.com/a", "title": "A", "text": None, "link_type": "internal"},
        {"url": "https://docs.example.com/b", "title": "B", "text": None, "link_type": "internal"},
        {"url": "https://docs.example.com/c", "title": "C", "text": None, "link_type": "internal"},
    ]))
    scrape_mock = AsyncMock(return_value=_scrape("# Page\n\nbody content here."))
    bfs_mock = AsyncMock(return_value=[])  # BFS silently empty — the bug we were papering over

    with (
        patch(PATCH_DISCOVER, discovery_mock),
        patch(PATCH_SCRAPE_CACHED, scrape_mock),
        patch(PATCH_DEEP_CRAWL, bfs_mock),
    ):
        payload = await server_module.crawl_impl("https://docs.example.com", max_urls=10)

    # 1 root + 3 seeds = 4 results, all with content.
    assert len(payload["results"]) == 4
    for r in payload["results"]:
        assert r["status"] == "ok"
        assert len(r["content"]) > 0


@pytest.mark.asyncio
async def test_crawl_preserves_depth2_from_bfs():
    """When BFS returns depth-2 children, they appear in results with depth=2."""
    discovery_mock = AsyncMock(return_value=_discovery("Root", [
        {"url": "https://docs.example.com/section", "title": "Section", "text": None, "link_type": "internal"},
    ]))
    scrape_map = {
        "https://docs.example.com": _scrape("# Root\n\nroot."),
        "https://docs.example.com/section": _scrape("# Section\n\nsection body."),
    }
    scrape_mock = AsyncMock(side_effect=lambda url, cache: scrape_map[url])

    # BFS returns seed + one depth-2 child.
    bfs_mock = AsyncMock(return_value=[
        _bfs_page("https://docs.example.com/section", "section from bfs", depth=0),
        _bfs_page("https://docs.example.com/section/nested", "nested page body",
                  depth=1, parent="https://docs.example.com/section"),
    ])

    with (
        patch(PATCH_DISCOVER, discovery_mock),
        patch(PATCH_SCRAPE_CACHED, scrape_mock),
        patch(PATCH_DEEP_CRAWL, bfs_mock),
    ):
        payload = await server_module.crawl_impl("https://docs.example.com", max_urls=10)

    by_url = {r["url"]: r for r in payload["results"]}
    assert "https://docs.example.com/section/nested" in by_url
    depth2 = by_url["https://docs.example.com/section/nested"]
    assert depth2["depth"] == 2
    assert depth2["discovered_from"] == "https://docs.example.com/section"
    assert len(depth2["content"]) > 0


@pytest.mark.asyncio
async def test_crawl_bfs_failure_still_returns_root_and_seeds():
    """BFS raising an exception must not drop root+seed results."""
    discovery_mock = AsyncMock(return_value=_discovery("Home", [
        {"url": "https://docs.example.com/a", "title": "A", "text": None, "link_type": "internal"},
    ]))
    scrape_mock = AsyncMock(return_value=_scrape("# Page\n\nbody."))
    bfs_mock = AsyncMock(side_effect=RuntimeError("crawl4ai BFS offline"))

    with (
        patch(PATCH_DISCOVER, discovery_mock),
        patch(PATCH_SCRAPE_CACHED, scrape_mock),
        patch(PATCH_DEEP_CRAWL, bfs_mock),
    ):
        payload = await server_module.crawl_impl("https://docs.example.com", max_urls=5)

    urls = [r["url"] for r in payload["results"]]
    assert "https://docs.example.com" in urls
    assert "https://docs.example.com/a" in urls
    assert any(w["type"] == "crawl_failed" for w in payload["meta"]["warnings"])


@pytest.mark.asyncio
async def test_crawl_collapses_same_body_at_different_urls():
    """Near-identical bodies at sibling URLs get MinHash-deduped."""
    body = (
        "# Intro\n\nThe Model Context Protocol standardizes how applications "
        "expose tools and data to large language models. It gives developers a "
        "common integration surface so that any MCP-aware client can connect to "
        "any MCP-aware server without bespoke glue code each time."
    )
    discovery_mock = AsyncMock(return_value=_discovery("Intro", [
        {"url": "https://example.com/docs/getting-started", "title": "A", "text": None, "link_type": "internal"},
        {"url": "https://example.com/docs/getting-started/intro", "title": "B", "text": None, "link_type": "internal"},
    ]))
    scrape_map = {
        "https://example.com": _scrape(body),
        "https://example.com/docs/getting-started": _scrape(body + "\n\n_mirror_"),
        "https://example.com/docs/getting-started/intro": _scrape(body + "\n\n_home_"),
    }
    scrape_mock = AsyncMock(side_effect=lambda url, cache: scrape_map[url])
    bfs_mock = AsyncMock(return_value=[])

    with (
        patch(PATCH_DISCOVER, discovery_mock),
        patch(PATCH_SCRAPE_CACHED, scrape_mock),
        patch(PATCH_DEEP_CRAWL, bfs_mock),
    ):
        payload = await server_module.crawl_impl("https://example.com", max_urls=5)

    assert payload["meta"]["urls_deduplicated"] == 2
    assert payload["meta"]["urls_returned"] == 1


@pytest.mark.asyncio
async def test_crawl_max_urls_exceeds_available_returns_all():
    """Graceful: fewer scrapes than requested doesn't error."""
    discovery_mock = AsyncMock(return_value=_discovery("Small", [
        {"url": "https://docs.example.com/a", "title": "A", "text": None, "link_type": "internal"},
    ]))
    scrape_mock = AsyncMock(return_value=_scrape("# Page\n\nbody."))
    bfs_mock = AsyncMock(return_value=[])

    with (
        patch(PATCH_DISCOVER, discovery_mock),
        patch(PATCH_SCRAPE_CACHED, scrape_mock),
        patch(PATCH_DEEP_CRAWL, bfs_mock),
    ):
        payload = await server_module.crawl_impl("https://docs.example.com", max_urls=20)

    # root + 1 = 2 entries, no error
    assert payload["meta"]["urls_returned"] == 2
