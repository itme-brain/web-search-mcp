from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import server_module

PATCH_MAP_IMPL = "impls.map_impl"
PATCH_EXTRACT_IMPL = "impls.extract_impl"


def _map_result(
    url: str,
    *,
    depth: int,
    title: str | None = None,
    discovered_from: str | None = None,
    link_type: str = "internal",
    rank: int = 1,
) -> dict:
    return {
        "rank": rank,
        "url": url,
        "domain": server_module._domain_from_url(url),
        "title": title,
        "link_text": None,
        "depth": depth,
        "discovered_from": discovered_from,
        "link_type": link_type,
    }


def _extract_result(
    url: str,
    *,
    status: str = "ok",
    title: str | None = None,
    content: str = "",
    error: str | None = None,
    cached: bool = False,
) -> dict:
    return {
        "url": url,
        "domain": server_module._domain_from_url(url),
        "status": status,
        "content_type": "text/html",
        "file_type": "html",
        "title": title,
        "content": content,
        "chars_shown": len(content),
        "total_chars": len(content),
        "top_chunks": [],
        "chunks": [],
        "cached": cached,
        "error": error,
    }


@pytest.mark.asyncio
async def test_crawl_validates_max_urls():
    with pytest.raises(ValueError, match="max_urls must be <= "):
        await server_module.crawl.fn("https://docs.example.com", max_urls=1000)


@pytest.mark.asyncio
async def test_crawl_composes_map_and_extract_results():
    map_payload = {
        "url": "https://docs.example.com",
        "results": [
            _map_result("https://docs.example.com", depth=0, title="Root", link_type="seed", rank=1),
            _map_result(
                "https://docs.example.com/guide",
                depth=1,
                title="Guide",
                discovered_from="https://docs.example.com",
                rank=2,
            ),
            _map_result(
                "https://docs.example.com/guide/install",
                depth=2,
                title="Install",
                discovered_from="https://docs.example.com/guide",
                rank=3,
            ),
        ],
        "meta": {"warnings": [], "urls_returned": 3, "pages_visited": 3},
    }
    extract_payload = {
        "results": [
            _extract_result("https://docs.example.com", title="Root", content="# Root\n\nbody"),
            _extract_result("https://docs.example.com/guide", title="Guide", content="# Guide\n\nguide body"),
            _extract_result("https://docs.example.com/guide/install", title="Install", content="# Install\n\ninstall body"),
        ],
        "meta": {"urls_requested": 3, "urls_succeeded": 3, "urls_failed": 0},
    }

    with (
        patch(PATCH_MAP_IMPL, AsyncMock(return_value=map_payload)) as map_mock,
        patch(PATCH_EXTRACT_IMPL, AsyncMock(return_value=extract_payload)) as extract_mock,
    ):
        payload = await server_module.crawl_impl("https://docs.example.com", max_urls=10)

    map_mock.assert_awaited_once()
    extract_mock.assert_awaited_once()
    assert [r["depth"] for r in payload["results"]] == [0, 1, 2]
    assert payload["results"][2]["discovered_from"] == "https://docs.example.com/guide"
    assert payload["meta"]["urls_discovered"] == 3
    assert payload["meta"]["urls_succeeded"] == 3
    assert payload["meta"]["urls_failed"] == 0


@pytest.mark.asyncio
async def test_crawl_honors_max_urls():
    map_payload = {
        "url": "https://docs.example.com",
        "results": [_map_result(f"https://docs.example.com/p{i}", depth=0 if i == 0 else 1, rank=i + 1) for i in range(4)],
        "meta": {"warnings": [], "urls_returned": 4, "pages_visited": 4},
    }
    extract_payload = {
        "results": [_extract_result(f"https://docs.example.com/p{i}", content="body") for i in range(4)],
        "meta": {"urls_requested": 4, "urls_succeeded": 4, "urls_failed": 0},
    }

    with (
        patch(PATCH_MAP_IMPL, AsyncMock(return_value=map_payload)),
        patch(PATCH_EXTRACT_IMPL, AsyncMock(return_value=extract_payload)),
    ):
        payload = await server_module.crawl_impl("https://docs.example.com", max_urls=4)

    assert payload["meta"]["urls_returned"] == 4
    assert len(payload["results"]) == 4


@pytest.mark.asyncio
async def test_crawl_passes_include_patterns_to_map():
    map_mock = AsyncMock(return_value={
        "url": "https://docs.example.com",
        "results": [_map_result("https://docs.example.com", depth=0, link_type="seed")],
        "meta": {"warnings": [], "urls_returned": 1, "pages_visited": 1},
    })
    extract_mock = AsyncMock(return_value={
        "results": [_extract_result("https://docs.example.com", content="body")],
        "meta": {"urls_requested": 1, "urls_succeeded": 1, "urls_failed": 0},
    })

    with (
        patch(PATCH_MAP_IMPL, map_mock),
        patch(PATCH_EXTRACT_IMPL, extract_mock),
    ):
        await server_module.crawl_impl(
            "https://docs.example.com",
            max_urls=5,
            include_patterns=["https://docs.example.com/api/*"],
        )

    assert map_mock.await_args.kwargs["include_patterns"] == ["https://docs.example.com/api/*"]


@pytest.mark.asyncio
async def test_crawl_surfaces_map_warnings_and_extract_failures():
    map_payload = {
        "url": "https://docs.example.com",
        "results": [
            _map_result("https://docs.example.com", depth=0, title="Root", link_type="seed", rank=1),
            _map_result("https://docs.example.com/a", depth=1, title="A", discovered_from="https://docs.example.com", rank=2),
        ],
        "meta": {
            "warnings": [server_module._warning("link_discovery_failed", "crawl4ai", "partial tree")],
            "urls_returned": 2,
            "pages_visited": 1,
        },
    }
    extract_payload = {
        "results": [
            _extract_result("https://docs.example.com", title="Root", content="root body"),
            _extract_result("https://docs.example.com/a", status="error", error="403 forbidden"),
        ],
        "meta": {"urls_requested": 2, "urls_succeeded": 1, "urls_failed": 1},
    }

    with (
        patch(PATCH_MAP_IMPL, AsyncMock(return_value=map_payload)),
        patch(PATCH_EXTRACT_IMPL, AsyncMock(return_value=extract_payload)),
    ):
        payload = await server_module.crawl_impl("https://docs.example.com", max_urls=5)

    assert any(w["type"] == "link_discovery_failed" for w in payload["meta"]["warnings"])
    failed = next(r for r in payload["results"] if r["url"].endswith("/a"))
    assert failed["status"] == "error"
    assert failed["error"] == "403 forbidden"
    assert payload["meta"]["urls_failed"] == 1


@pytest.mark.asyncio
async def test_crawl_query_reorders_by_relevance():
    map_payload = {
        "url": "https://docs.example.com",
        "results": [
            _map_result("https://docs.example.com", depth=0, title="Root", link_type="seed", rank=1),
            _map_result(
                "https://docs.example.com/intro",
                depth=1, title="Intro",
                discovered_from="https://docs.example.com", rank=2,
            ),
            _map_result(
                "https://docs.example.com/rate-limits",
                depth=1, title="Rate Limits",
                discovered_from="https://docs.example.com", rank=3,
            ),
            _map_result(
                "https://docs.example.com/faq",
                depth=1, title="FAQ",
                discovered_from="https://docs.example.com", rank=4,
            ),
        ],
        "meta": {"warnings": [], "urls_returned": 4, "pages_visited": 4},
    }

    # Per-URL extract docs with controlled best-chunk scores so the test
    # is deterministic regardless of the FlashRank stub. The rate-limits
    # page scores highest and must surface as rank 1.
    score_by_url = {
        "https://docs.example.com": 0.10,
        "https://docs.example.com/intro": 0.20,
        "https://docs.example.com/rate-limits": 0.95,
        "https://docs.example.com/faq": 0.30,
    }

    def _doc(url: str) -> dict:
        score = score_by_url[url]
        return {
            "status": "ok",
            "url": url,
            "content_type": "text/html",
            "file_type": "html",
            "title": url.rsplit("/", 1)[-1] or "root",
            "content": f"top chunk text for {url}",
            "total_chars": 200,
            "metadata": {},
            "top_chunks": [{"id": 0, "text": f"top chunk text for {url}", "score": score}],
            "chunks": [{"id": 0, "text": "..."}],
            "shown_chunk_ids": [0],
            "total_chunks": 1,
            "chunk_mode": "relevant",
            "cached": False,
        }

    async def _fake_extract_url_document(url, query, cache, chunk_ids=None):
        assert query == "rate limits"
        return _doc(url)

    with (
        patch(PATCH_MAP_IMPL, AsyncMock(return_value=map_payload)),
        patch("core._extract_url_document", AsyncMock(side_effect=_fake_extract_url_document)),
    ):
        payload = await server_module.crawl_impl(
            "https://docs.example.com",
            max_urls=4,
            query="rate limits",
        )

    assert payload["query"] == "rate limits"
    assert [r["url"] for r in payload["results"]][0] == "https://docs.example.com/rate-limits"
    assert payload["results"][0]["rank"] == 1
    assert payload["results"][0]["top_chunks"] == ["top chunk text for https://docs.example.com/rate-limits"]
    # The seed page (lowest score) should land last.
    assert payload["results"][-1]["url"] == "https://docs.example.com"


@pytest.mark.asyncio
async def test_crawl_without_query_preserves_bfs_order():
    map_payload = {
        "url": "https://docs.example.com",
        "results": [
            _map_result("https://docs.example.com", depth=0, title="Root", link_type="seed", rank=1),
            _map_result(
                "https://docs.example.com/a",
                depth=1, title="A",
                discovered_from="https://docs.example.com", rank=2,
            ),
            _map_result(
                "https://docs.example.com/b",
                depth=1, title="B",
                discovered_from="https://docs.example.com", rank=3,
            ),
        ],
        "meta": {"warnings": [], "urls_returned": 3, "pages_visited": 3},
    }
    extract_payload = {
        "results": [
            _extract_result("https://docs.example.com", title="Root", content="root body"),
            _extract_result("https://docs.example.com/a", title="A", content="a body"),
            _extract_result("https://docs.example.com/b", title="B", content="b body"),
        ],
        "meta": {"urls_requested": 3, "urls_succeeded": 3, "urls_failed": 0},
    }

    with (
        patch(PATCH_MAP_IMPL, AsyncMock(return_value=map_payload)),
        patch(PATCH_EXTRACT_IMPL, AsyncMock(return_value=extract_payload)),
    ):
        payload = await server_module.crawl_impl(
            "https://docs.example.com", max_urls=3,
        )

    # No query → BFS discovery order preserved (matches map order).
    assert [r["url"] for r in payload["results"]] == [
        "https://docs.example.com",
        "https://docs.example.com/a",
        "https://docs.example.com/b",
    ]
    assert payload.get("query") in (None, "")


@pytest.mark.asyncio
async def test_crawl_markdown_renders_tree_and_error_state():
    map_payload = {
        "url": "https://docs.example.com",
        "results": [
            _map_result("https://docs.example.com", depth=0, title="Root", link_type="seed", rank=1),
            _map_result("https://docs.example.com/guide", depth=1, title="Guide", discovered_from="https://docs.example.com", rank=2),
        ],
        "meta": {"warnings": [], "urls_returned": 2, "pages_visited": 2},
    }
    extract_payload = {
        "results": [
            _extract_result("https://docs.example.com", title="Root", content="# Root\n\nbody"),
            _extract_result("https://docs.example.com/guide", status="error", title="Guide", error="extraction failed"),
        ],
        "meta": {"urls_requested": 2, "urls_succeeded": 1, "urls_failed": 1},
    }

    with (
        patch(PATCH_MAP_IMPL, AsyncMock(return_value=map_payload)),
        patch(PATCH_EXTRACT_IMPL, AsyncMock(return_value=extract_payload)),
    ):
        markdown = await server_module.crawl.fn("https://docs.example.com", max_urls=5)

    text = markdown.content[0].text
    assert "failed: 1" in text
    assert "[Root](https://docs.example.com)" in text
    assert "[Guide](https://docs.example.com/guide)" in text
    assert "**Error:** extraction failed" in text
