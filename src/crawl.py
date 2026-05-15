"""Crawl4AI request/config helpers."""

import copy
import json

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

import urls as url_utils

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


def domain_filter_patterns(root_url: str, same_domain_only: bool) -> list[str]:
    if not same_domain_only:
        return []
    root_domain = url_utils.domain_from_url(root_url).lower()
    registrable = url_utils.registrable_domain(root_domain)
    patterns: list[str] = []
    for scheme in ("http", "https"):
        patterns.append(f"{scheme}://{registrable}/*")
        patterns.append(f"{scheme}://*.{registrable}/*")
        patterns.append(f"{scheme}://{root_domain}/*")
    return patterns


def filter_chain(*, root_url: str, same_domain_only: bool, include_patterns: list[str] | None) -> list[dict]:
    filters: list[dict] = []
    domain_patterns = domain_filter_patterns(root_url, same_domain_only)
    if domain_patterns:
        filters.append({"type": "URLPatternFilter", "params": {"patterns": domain_patterns}})
    if include_patterns:
        filters.append({"type": "URLPatternFilter", "params": {"patterns": include_patterns}})
    filters.append({"type": "ContentTypeFilter", "params": {"allowed_types": ["text/html"]}})
    return filters


def deep_crawl_config(
    *, root_url: str, max_depth: int, max_pages: int,
    same_domain_only: bool, include_patterns: list[str] | None = None,
) -> dict:
    params = copy.deepcopy(MAP_CRAWL_CONFIG["params"])
    strategy_params: dict = {
        "max_depth": max_depth,
        "include_external": not same_domain_only,
        "max_pages": max_pages,
    }
    filters = filter_chain(
        root_url=root_url,
        same_domain_only=same_domain_only,
        include_patterns=include_patterns,
    )
    if filters:
        strategy_params["filter_chain"] = {"type": "FilterChain", "params": {"filters": filters}}
    params["deep_crawl_strategy"] = {"type": "BFSDeepCrawlStrategy", "params": strategy_params}
    return {"type": "CrawlerRunConfig", "params": params}


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
    crawler_config: dict | None = None,
) -> dict:
    resp = await client.post(
        f"{crawl4ai_url}/crawl/stream",
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
