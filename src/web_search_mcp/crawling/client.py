"""Crawl4AI request/config helpers."""

import json

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

DEFAULT_CRAWL_CONFIG = {
    "type": "CrawlerRunConfig",
    "params": {
        "excluded_tags": ["nav", "footer", "header", "aside"],
        "markdown_generator": {
            "type": "DefaultMarkdownGenerator",
            "params": {
                "content_filter": {
                    "type": "PruningContentFilter",
                    "params": {"threshold": 0.48, "threshold_type": "fixed", "min_word_threshold": 0},
                },
            },
        },
    },
}

MAP_CRAWL_CONFIG = {"type": "CrawlerRunConfig", "params": {}}


def is_retryable_crawl_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, (httpx.TransportError, httpx.TimeoutException))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(is_retryable_crawl_error),
    reraise=True,
)
async def crawl_post(
    client: httpx.AsyncClient,
    *,
    crawl4ai_url: str,
    urls: list[str],
    priority: int,
    api_token: str = "",
    crawler_config: dict | None = None,
) -> dict:
    headers = {"Authorization": f"Bearer {api_token}"} if api_token else None
    resp = await client.post(
        f"{crawl4ai_url}/crawl/stream",
        headers=headers,
        json={"urls": urls, "priority": priority, "crawler_config": crawler_config or DEFAULT_CRAWL_CONFIG},
    )
    resp.raise_for_status()
    body = (await resp.aread()).decode("utf-8", errors="replace")
    results: list[dict] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        if status == "completed":
            break
        if status == "failed":
            detail = item.get("error") or item.get("error_message") or "crawl stream failed"
            raise ValueError(detail)
        results.append(item)
    return {"results": results}
