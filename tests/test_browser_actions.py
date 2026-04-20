from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Client

from tests.conftest import server_app, server_module, FakeContext

PATCH_SCRAPE_IMPL = "web_search_server._scrape_impl"
PATCH_SCRAPE = "web_search_server._scrape"
PATCH_DETECT_FILE_TYPE = "web_search_server._detect_file_type"


class TestBuildCrawlConfig:
    def test_default_matches_original_shape(self):
        config = server_module._build_crawl_config()
        assert config["type"] == "CrawlerRunConfig"
        params = config["params"]
        assert "nav" in params["excluded_tags"]
        assert params["markdown_generator"]["type"] == "DefaultMarkdownGenerator"

    def test_js_code_included(self):
        config = server_module._build_crawl_config(js_code=["document.querySelector('.btn').click()"])
        assert config["params"]["js_code"] == ["document.querySelector('.btn').click()"]

    def test_wait_for_included(self):
        config = server_module._build_crawl_config(wait_for=".loaded")
        assert config["params"]["wait_for"] == ".loaded"

    def test_page_timeout_included(self):
        config = server_module._build_crawl_config(page_timeout=60000)
        assert config["params"]["page_timeout"] == 60000

    def test_screenshot_flag(self):
        config = server_module._build_crawl_config(screenshot=True)
        assert config["params"]["screenshot"] is True

    def test_remove_overlays_default_true(self):
        config = server_module._build_crawl_config()
        assert config["params"]["remove_overlay_elements"] is True

    def test_remove_overlays_false(self):
        config = server_module._build_crawl_config(remove_overlays=False)
        assert "remove_overlay_elements" not in config["params"]

    def test_scroll_full_page(self):
        config = server_module._build_crawl_config(scroll_full_page=True)
        assert config["params"]["scan_full_page"] is True

    def test_combined_params(self):
        config = server_module._build_crawl_config(
            js_code=["click()"],
            wait_for=".el",
            screenshot=True,
            scroll_full_page=True,
        )
        p = config["params"]
        assert p["js_code"] == ["click()"]
        assert p["wait_for"] == ".el"
        assert p["screenshot"] is True
        assert p["scan_full_page"] is True


class TestMaybeBuildCrawlConfig:
    def test_returns_none_for_all_defaults(self):
        result = server_module._maybe_build_crawl_config(
            js_code=None, wait_for=None, page_timeout=None,
            screenshot=False, remove_overlays=True, scroll_full_page=False,
        )
        assert result is None

    def test_returns_config_when_js_code_set(self):
        result = server_module._maybe_build_crawl_config(
            js_code=["x"], wait_for=None, page_timeout=None,
            screenshot=False, remove_overlays=True, scroll_full_page=False,
        )
        assert result is not None
        assert result["params"]["js_code"] == ["x"]

    def test_returns_config_when_screenshot_true(self):
        result = server_module._maybe_build_crawl_config(
            js_code=None, wait_for=None, page_timeout=None,
            screenshot=True, remove_overlays=True, scroll_full_page=False,
        )
        assert result is not None


class TestScrapeCustomConfig:
    @pytest.mark.asyncio
    async def test_scrape_impl_uses_custom_config(self):
        custom_config = server_module._build_crawl_config(js_code=["click()"])
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [{
                "markdown": "content",
                "html": "<p>content</p>",
                "metadata": {"title": "Test"},
            }],
        }
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("web_search_server.httpx.AsyncClient", return_value=mock_client):
            result = await server_module._scrape_impl("https://example.com", crawl_config=custom_config)

        post_call = mock_client.post.call_args
        payload = post_call[1]["json"] if "json" in post_call[1] else post_call[0][1]
        assert payload["crawler_config"]["params"]["js_code"] == ["click()"]


class TestScrapeCacheBypass:
    @pytest.mark.asyncio
    async def test_cache_hit_with_default_config(self):
        cache = {"https://example.com": "cached content"}
        result = await server_module._scrape_cached("https://example.com", cache)
        assert result == "cached content"

    @pytest.mark.asyncio
    async def test_cache_bypassed_with_custom_config(self):
        cache = {"https://example.com": "cached content"}
        custom_config = server_module._build_crawl_config(js_code=["x"])

        scrape_mock = AsyncMock(return_value={
            "content": "fresh content", "title": None, "screenshot": None,
        })
        with patch(PATCH_SCRAPE, scrape_mock):
            result = await server_module._scrape_cached(
                "https://example.com", cache, crawl_config=custom_config,
            )
        assert result == "fresh content"
        scrape_mock.assert_called_once()


class TestExtractUrlScreenshot:
    @pytest.mark.asyncio
    async def test_screenshot_in_response(self):
        scrape_mock = AsyncMock(return_value={
            "content": "page content here with enough text",
            "title": "Test Page",
            "screenshot": "base64encodedpng",
        })
        detect_mock = AsyncMock(return_value=("html", "text/html"))

        with (
            patch(PATCH_SCRAPE, scrape_mock),
            patch(PATCH_DETECT_FILE_TYPE, detect_mock),
        ):
            async with Client(server_app) as client:
                result = await client.call_tool(
                    "extract_url",
                    {"url": "https://example.com", "screenshot": True},
                )
                payload = result.data

        assert payload["result"]["screenshot"] == "base64encodedpng"
