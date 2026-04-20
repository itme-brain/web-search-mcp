"""Retry behavior for Crawl4AI transient errors."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from tenacity import wait_none

from tests.conftest import server_module


@pytest.fixture(autouse=True)
def _no_retry_backoff():
    """Skip exponential backoff sleeps during tests."""
    original = server_module._crawl_post.retry.wait
    server_module._crawl_post.retry.wait = wait_none()
    yield
    server_module._crawl_post.retry.wait = original


def _fake_5xx_response() -> httpx.Response:
    req = httpx.Request("POST", "http://crawl4ai:11235/crawl")
    return httpx.Response(503, request=req)


def _fake_success_response() -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"results": [{"markdown": "content"}]})
    return resp


class TestIsRetryableCrawlError:
    def test_5xx_is_retryable(self):
        exc = httpx.HTTPStatusError("bad", request=MagicMock(), response=_fake_5xx_response())
        assert server_module._is_retryable_crawl_error(exc) is True

    def test_4xx_is_not_retryable(self):
        req = httpx.Request("POST", "http://crawl4ai:11235/crawl")
        exc = httpx.HTTPStatusError(
            "bad", request=req, response=httpx.Response(404, request=req),
        )
        assert server_module._is_retryable_crawl_error(exc) is False

    def test_transport_error_is_retryable(self):
        exc = httpx.ConnectError("refused")
        assert server_module._is_retryable_crawl_error(exc) is True

    def test_timeout_is_retryable(self):
        exc = httpx.ReadTimeout("slow")
        assert server_module._is_retryable_crawl_error(exc) is True

    def test_unrelated_error_is_not_retryable(self):
        assert server_module._is_retryable_crawl_error(ValueError("nope")) is False


@pytest.mark.asyncio
async def test_crawl_post_retries_on_5xx_then_succeeds():
    """A transient 5xx should trigger retry, not kill the scrape."""
    fail_resp = MagicMock()
    fail_resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "boom", request=MagicMock(), response=_fake_5xx_response(),
        ),
    )
    success_resp = _fake_success_response()

    mock_client = MagicMock()
    mock_client.post = AsyncMock(side_effect=[fail_resp, success_resp])

    data = await server_module._crawl_post(mock_client, "https://example.com", priority=8)

    assert data == {"results": [{"markdown": "content"}]}
    assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_poll_crawl_task_stops_after_max_iterations():
    """Broken status endpoint returning 'running' forever must not hang."""
    stuck_resp = MagicMock()
    stuck_resp.raise_for_status = MagicMock()
    stuck_resp.json = MagicMock(return_value={"status": "running"})

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=stuck_resp)

    # Swap asyncio.sleep for a no-op so the test doesn't wait real seconds.
    import web_search_server as server
    original_sleep = server.asyncio.sleep

    async def _fake_sleep(_secs):
        return None

    server.asyncio.sleep = _fake_sleep
    try:
        result = await server._poll_crawl_task(mock_client, "task-123")
    finally:
        server.asyncio.sleep = original_sleep

    assert result is None
    assert mock_client.get.call_count == server._MAX_TASK_POLLS


@pytest.mark.asyncio
async def test_crawl_post_does_not_retry_on_4xx():
    """4xx errors are caller bugs, not transient — fail fast."""
    req = httpx.Request("POST", "http://crawl4ai:11235/crawl")
    fail_resp = MagicMock()
    fail_resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "bad input", request=req, response=httpx.Response(400, request=req),
        ),
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=fail_resp)

    with pytest.raises(httpx.HTTPStatusError):
        await server_module._crawl_post(mock_client, "https://example.com", priority=8)

    assert mock_client.post.call_count == 1
