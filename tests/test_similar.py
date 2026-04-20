from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Client

from tests.conftest import server_app, server_module, FakeContext

PATCH_SEARCH = "web_search_server._search"
PATCH_SCRAPE = "web_search_server._scrape"
PATCH_RERANK = "web_search_server._rerank_scored"


class TestExtractKeywords:
    def test_returns_expected_terms(self):
        text = "python asyncio python tutorial python programming asyncio event loop"
        keywords = server_module._extract_keywords(text)
        assert "python" in keywords
        assert "asyncio" in keywords

    def test_excludes_stopwords(self):
        text = "the and is are python python python"
        keywords = server_module._extract_keywords(text)
        assert "the" not in keywords
        assert "and" not in keywords
        assert "python" in keywords

    def test_excludes_short_words(self):
        text = "go to my python app at the end"
        keywords = server_module._extract_keywords(text)
        for kw in keywords:
            assert len(kw) >= 3

    def test_max_keywords_limit(self):
        text = " ".join(f"word{i} " * (20 - i) for i in range(20))
        keywords = server_module._extract_keywords(text, max_keywords=5)
        assert len(keywords) <= 5

    def test_empty_content(self):
        assert server_module._extract_keywords("") == []


class TestFindSimilarImpl:
    @pytest.mark.asyncio
    async def test_generates_queries_and_searches(self):
        scrape_mock = AsyncMock(return_value={
            "content": "Python asyncio tutorial. Learn about event loops and coroutines in Python programming.",
            "title": "Python Asyncio Tutorial",
            "screenshot": None,
        })

        similar_results = [
            {"title": "Async Python Guide", "url": "https://example.com/async", "content": "async guide"},
            {"title": "Coroutines Explained", "url": "https://example.com/coro", "content": "coroutines"},
        ]
        search_mock = AsyncMock(return_value=similar_results)

        def fake_rerank(_query, docs):
            return [(i, 1.0 - i * 0.1) for i in range(len(docs))]

        rerank_mock = MagicMock(side_effect=fake_rerank)

        with (
            patch(PATCH_SCRAPE, scrape_mock),
            patch(PATCH_SEARCH, search_mock),
            patch(PATCH_RERANK, rerank_mock),
        ):
            result = await server_module._find_similar_impl(
                url="https://example.com/source",
                num_results=5,
            )

        assert result["source_url"] == "https://example.com/source"
        assert result["source_title"] == "Python Asyncio Tutorial"
        assert len(result["results"]) > 0
        assert search_mock.call_count >= 2  # at least title + keyword queries

    @pytest.mark.asyncio
    async def test_excludes_source_url(self):
        scrape_mock = AsyncMock(return_value={
            "content": "Content about testing. Testing is important for software quality and reliability.",
            "title": "Testing Guide",
            "screenshot": None,
        })

        search_results = [
            {"title": "Source Page", "url": "https://example.com/source", "content": "source"},
            {"title": "Other Page", "url": "https://example.com/other", "content": "other"},
        ]
        search_mock = AsyncMock(return_value=search_results)
        rerank_mock = MagicMock(side_effect=lambda q, d: [(i, 0.5) for i in range(len(d))])

        with (
            patch(PATCH_SCRAPE, scrape_mock),
            patch(PATCH_SEARCH, search_mock),
            patch(PATCH_RERANK, rerank_mock),
        ):
            result = await server_module._find_similar_impl(
                url="https://example.com/source",
                num_results=5,
            )

        result_urls = [r["url"] for r in result["results"]]
        assert "https://example.com/source" not in result_urls

    @pytest.mark.asyncio
    async def test_handles_scrape_failure(self):
        scrape_mock = AsyncMock(return_value={
            "content": None,
            "title": None,
            "screenshot": None,
        })

        with patch(PATCH_SCRAPE, scrape_mock):
            result = await server_module._find_similar_impl(
                url="https://example.com/source",
                num_results=5,
            )

        assert result["results"] == []
        assert "error" in result["meta"]

    @pytest.mark.asyncio
    async def test_deduplicates_across_queries(self):
        scrape_mock = AsyncMock(return_value={
            "content": "Python tutorial content about async programming patterns and best practices.",
            "title": "Python Tutorial",
            "screenshot": None,
        })

        # All queries return the same result
        dup_result = [
            {"title": "Duplicate", "url": "https://example.com/dup", "content": "same page"},
            {"title": "Unique", "url": "https://example.com/unique", "content": "unique page"},
        ]
        search_mock = AsyncMock(return_value=dup_result)
        rerank_mock = MagicMock(side_effect=lambda q, d: [(i, 0.5) for i in range(len(d))])

        with (
            patch(PATCH_SCRAPE, scrape_mock),
            patch(PATCH_SEARCH, search_mock),
            patch(PATCH_RERANK, rerank_mock),
        ):
            result = await server_module._find_similar_impl(
                url="https://example.com/source",
                num_results=10,
            )

        urls = [r["url"] for r in result["results"]]
        assert len(urls) == len(set(urls))


class TestFindSimilarTool:
    @pytest.mark.asyncio
    async def test_tool_validates_url(self):
        async with Client(server_app) as client:
            with pytest.raises(Exception):
                await client.call_tool(
                    "find_similar",
                    {"url": "not-a-url"},
                )

    @pytest.mark.asyncio
    async def test_tool_returns_structured_response(self):
        scrape_mock = AsyncMock(return_value={
            "content": "Testing content with enough words for keyword extraction to work properly.",
            "title": "Test Page",
            "screenshot": None,
        })
        search_mock = AsyncMock(return_value=[
            {"title": "Similar", "url": "https://example.com/similar", "content": "similar content"},
        ])
        rerank_mock = MagicMock(side_effect=lambda q, d: [(i, 0.5) for i in range(len(d))])

        with (
            patch(PATCH_SCRAPE, scrape_mock),
            patch(PATCH_SEARCH, search_mock),
            patch(PATCH_RERANK, rerank_mock),
        ):
            async with Client(server_app) as client:
                result = await client.call_tool(
                    "find_similar",
                    {"url": "https://example.com/source", "num_results": 5},
                )
                payload = result.data

        assert "https://example.com/source" in payload
        assert "Similar" in payload
