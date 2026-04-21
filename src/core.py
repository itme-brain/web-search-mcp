"""Private infrastructure the impls call into.

Everything that isn't a public impl or a markdown formatter lives here:
settings + constants, HTTP + retry + polling, rerank (FlashRank model
load and inference), text/URL utilities, cache helpers for ctx-backed
session state, validators, and the per-file-type extractors.
"""

import asyncio
import fnmatch
import json
import logging
import mimetypes
import os
import re
from collections import defaultdict
from inspect import isawaitable
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import httpx
from datasketch import MinHash, MinHashLSH
from langchain_text_splitters import MarkdownTextSplitter
import magic
from rapidfuzz import fuzz
import tldextract
import trafilatura
from cachetools import TTLCache
from fastmcp import Context
from flashrank import Ranker, RerankRequest
from pydantic_settings import BaseSettings
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from url_normalize import url_normalize


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("web-search-mcp")


# ---------------------------------------------------------------------------
# Settings + public env-backed config
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    searxng_url: str = "http://searxng:8080"
    crawl4ai_url: str = "http://crawl4ai:11235"
    rerank_model: str = "ms-marco-MiniLM-L-12-v2"
    request_timeout: int = 30
    max_results: int = 10
    max_scrape: int = 5


settings = Settings()

SEARXNG_URL = settings.searxng_url
CRAWL4AI_URL = settings.crawl4ai_url
RERANK_MODEL = settings.rerank_model
REQUEST_TIMEOUT = settings.request_timeout
MAX_RESULTS = settings.max_results
MAX_SCRAPE = settings.max_scrape


# ---------------------------------------------------------------------------
# Internal constants — not user-configurable
# ---------------------------------------------------------------------------
_RERANK_MAX_LENGTH = 512
_HTTP_TIMEOUT = max(REQUEST_TIMEOUT // 2, 10)
_MAX_CONTENT_CHARS = 8000
_DEDUP_SIMILARITY = 0.75
_DEDUP_NUM_PERM = 128
_TITLE_DEDUP_THRESHOLD = 97.0
_TOP_CHUNKS = 3
_MAX_CHUNKS_PER_PAGE = 10
# Joiner for reranked chunks from the same page — standard editorial "elided
# material" mark, signals that chunks are discontinuous excerpts rather than
# continuous prose so the LLM doesn't mis-read e.g. paragraph 2 + paragraph 15
# as one logical flow.
_CHUNK_GAP = "\n\n[…]\n\n"
_MAX_EXTRACT_URLS = 20
_MAX_MAP_URLS = 50
_MAX_MAP_DEPTH = 3
# Minimum FlashRank relevance score for a result's best chunk.  Entries
# scoring below this are CAPTCHA walls, wrong-language pages, or
# auto-generated spam — noise the reranker confidently identifies as
# irrelevant.  0.05 is conservative: real content almost always exceeds
# it, while garbage rarely reaches it.
_MIN_RELEVANCE_SCORE = 0.05

# Session cache bounds — prevent unbounded growth in long-lived sessions.
_CACHE_MAXSIZE = 1000
_CACHE_TTL_S = 3600

STATE_SCRAPE_CACHE = "scrape_cache"
STATE_QUERY_CACHE = "query_cache"
STATE_SEEN_URLS = "seen_urls"
STATE_EXTRACT_CACHE = "extract_cache"


def _new_cache() -> TTLCache:
    return TTLCache(maxsize=_CACHE_MAXSIZE, ttl=_CACHE_TTL_S)


VALID_TIME_RANGES = frozenset({"day", "week", "month", "year"})

_TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "ref", "fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid",
})
_WORD_SPLIT = re.compile(r"\W+")
_WHITESPACE = re.compile(r"\s+")
_MARKDOWN_SPLITTER = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=0)


# ---------------------------------------------------------------------------
# Warnings + diagnostics
# ---------------------------------------------------------------------------
def _warning(error_type: str, source: str, detail: str) -> dict:
    """Create a structured warning dict for programmatic error handling."""
    return {"type": error_type, "source": source, "detail": detail}


def _dedup_unresponsive_engines(entries: list) -> list[tuple[str, str]]:
    """Normalize SearXNG's unresponsive_engines shapes and dedup by (engine, reason).

    SearXNG ships entries as `[engine_name, error_string]` lists; some
    integrations wrap them as dicts. Return a deduped list of
    `(engine, reason)` tuples in insertion order, dropping empty engine names.
    """
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for entry in entries:
        if isinstance(entry, (list, tuple)) and entry:
            engine = str(entry[0])
            reason = str(entry[1]) if len(entry) > 1 else ""
        elif isinstance(entry, dict):
            engine = str(entry.get("name") or entry.get("engine") or "")
            reason = str(entry.get("error") or entry.get("reason") or "")
        else:
            continue
        if not engine:
            continue
        key = (engine, reason)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


# ---------------------------------------------------------------------------
# URL + text utilities
# ---------------------------------------------------------------------------
def _normalize_url(url: str) -> str:
    normalized = url_normalize(url)
    parsed = urlparse(normalized)
    host = parsed.hostname or ""
    if host.startswith("www."):
        host = host[4:]
    params = {k: v for k, v in parse_qs(parsed.query).items() if k not in _TRACKING_PARAMS}
    return urlunparse((parsed.scheme, host, parsed.path.rstrip("/"), "", urlencode(params, doseq=True), ""))


def _dedup_results(results: list[dict]) -> list[dict]:
    seen_urls: set[str] = set()
    seen_titles: dict[str, list[str]] = defaultdict(list)
    deduped: list[dict] = []
    for r in results:
        norm = _normalize_url(r.get("url", ""))
        domain = _domain_from_url(r.get("url", "")).lower()
        title = _normalize_title(r.get("title", ""))
        if norm in seen_urls:
            continue
        if title and any(fuzz.ratio(title, seen) >= _TITLE_DEDUP_THRESHOLD for seen in seen_titles[domain]):
            continue
        seen_urls.add(norm)
        if title:
            seen_titles[domain].append(title)
        deduped.append(r)
    return deduped


def _normalize_title(title: str) -> str:
    normalized = _WHITESPACE.sub(" ", title.strip().lower())
    normalized = re.sub(r"[^a-z0-9 ]+", "", normalized)
    normalized = _WHITESPACE.sub(" ", normalized)
    return normalized.strip()


def _word_set(text: str) -> set[str]:
    return {word for word in _WORD_SPLIT.split(text.lower()) if word}


def _chunk_minhash(words: set[str]) -> MinHash:
    sketch = MinHash(num_perm=_DEDUP_NUM_PERM)
    for word in sorted(words):
        if word:
            sketch.update(word.encode("utf-8"))
    return sketch


def _dedup_chunks(chunks: list[str], entry_map: list[int]) -> tuple[list[str], list[int]]:
    kept_chunks: list[str] = []
    kept_entries: list[int] = []
    lsh = MinHashLSH(threshold=_DEDUP_SIMILARITY, num_perm=_DEDUP_NUM_PERM)
    for chunk, eidx in zip(chunks, entry_map):
        words = _word_set(chunk)
        if not words:
            continue
        sketch = _chunk_minhash(words)
        if lsh.query(sketch):
            continue
        key = len(kept_chunks)
        lsh.insert(key, sketch)
        kept_chunks.append(chunk)
        kept_entries.append(eidx)
    return kept_chunks, kept_entries


def _domain_from_url(url: str) -> str:
    return urlparse(url).hostname or ""


def _canonical_hostname(host: str) -> str:
    normalized = host.strip().lower().rstrip(".")
    if normalized.startswith("www."):
        normalized = normalized[4:]
    return normalized


def _registrable_domain(domain: str) -> str:
    """Return the PSL-aware registrable domain for host/domain matching."""
    bare = _canonical_hostname(domain)
    extracted = tldextract.extract(bare)
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}"
    return bare


def _chunk_text(text: str) -> list[str]:
    """Split extracted markdown into bounded chunks for reranking.

    Preserve existing paragraph boundaries for short blocks so reranked
    excerpts remain visibly discontinuous, and delegate only oversized
    blocks to LangChain's maintained markdown-aware splitter.
    """
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    chunks: list[str] = []
    for block in blocks:
        if len(block) <= 1000:
            chunks.append(block)
            continue
        chunks.extend(
            chunk.strip()
            for chunk in _MARKDOWN_SPLITTER.split_text(block)
            if chunk.strip()
        )
    return chunks[:_MAX_CHUNKS_PER_PAGE]


def _diversify_ranked_entries(ranked_entry_idxs: list[int], entries: list[dict]) -> list[int]:
    """Interleave domains so the top results are not monopolized by one source."""
    by_domain: dict[str, list[int]] = defaultdict(list)
    domain_order: list[str] = []
    for eidx in ranked_entry_idxs:
        domain = _domain_from_url(entries[eidx]["url"]).lower() or entries[eidx]["url"]
        if domain not in by_domain:
            domain_order.append(domain)
        by_domain[domain].append(eidx)

    diversified: list[int] = []
    while by_domain:
        next_round: list[str] = []
        for domain in domain_order:
            queue = by_domain.get(domain)
            if not queue:
                continue
            diversified.append(queue.pop(0))
            if queue:
                next_round.append(domain)
            else:
                by_domain.pop(domain, None)
        domain_order = next_round
    return diversified


# ---------------------------------------------------------------------------
# Validators + normalizers
# ---------------------------------------------------------------------------
def _normalize_time_range(time_range: str | None) -> str | None:
    if time_range in (None, "null", "none", "None", ""):
        return None
    normalized = time_range.strip().strip('"').lower()
    if normalized not in VALID_TIME_RANGES:
        raise ValueError(f"invalid time_range: {time_range!r}. Expected one of {sorted(VALID_TIME_RANGES)}")
    return normalized


def _validate_query(query: str) -> str:
    normalized = query.strip()
    if not normalized:
        raise ValueError("query must not be empty")
    return normalized


def _validate_positive_int(name: str, value: int, *, maximum: int) -> int:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    if value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _validate_urls(urls: list[str], *, maximum: int) -> list[str]:
    if not urls:
        raise ValueError("urls must not be empty")
    if len(urls) > maximum:
        raise ValueError(f"urls must contain at most {maximum} entries")
    normalized: list[str] = []
    for url in urls:
        value = url.strip()
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"invalid URL: {url!r}")
        normalized.append(value)
    return normalized


def _normalize_domains(domains: list[str] | None, *, field_name: str) -> list[str]:
    if not domains:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for domain in domains:
        value = _canonical_hostname(domain)
        if not value:
            continue
        if "/" in value:
            raise ValueError(f"{field_name} entries must be bare domains, got {domain!r}")
        if value not in seen:
            seen.add(value)
            normalized.append(value)
    return normalized


def _normalize_glob_patterns(patterns: list[str] | None, *, field_name: str) -> list[str]:
    if not patterns:
        return []
    normalized: list[str] = []
    for pattern in patterns:
        value = pattern.strip()
        if not value:
            continue
        normalized.append(value)
    if not normalized:
        raise ValueError(f"{field_name} must not be empty when provided")
    return normalized


def _match_domain(domain: str, patterns: list[str]) -> bool:
    host = _canonical_hostname(domain)
    host_registrable = _registrable_domain(host)
    for pattern in patterns:
        normalized = _canonical_hostname(pattern)
        if host == normalized or host.endswith(f".{normalized}"):
            return True
        if "." not in normalized and host_registrable == normalized:
            return True
    return False


def _filter_results_by_domain(
    results: list[dict],
    include_domains: list[str],
    exclude_domains: list[str],
) -> list[dict]:
    filtered: list[dict] = []
    for result in results:
        host = _canonical_hostname(_domain_from_url(result.get("url", "")))
        if include_domains and not _match_domain(host, include_domains):
            continue
        if exclude_domains and _match_domain(host, exclude_domains):
            continue
        filtered.append(result)
    return filtered


def _url_matches_patterns(url: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(url, pattern) for pattern in patterns)


# ---------------------------------------------------------------------------
# Context state helpers (cache round-trip + awaiting mixed sync/async APIs)
# ---------------------------------------------------------------------------
async def _maybe_await(value):
    if isawaitable(value):
        return await value
    return value


async def _ctx_get_state(ctx: Context, key: str):
    return await _maybe_await(ctx.get_state(key))


async def _ctx_set_state(ctx: Context, key: str, value) -> None:
    await _maybe_await(ctx.set_state(key, value))


async def _load_cache(ctx: Context | None, key: str) -> TTLCache:
    """Load a cache from ctx state as a TTLCache.

    FastMCP's ctx state requires values to be JSON-serializable, so we
    persist as plain dicts and instantiate a TTLCache wrapper on read to
    regain bounded-size + TTL semantics within the request.
    """
    cache = _new_cache()
    if ctx:
        stored = await _ctx_get_state(ctx, key)
        if stored:
            cache.update(stored)
    return cache


async def _save_cache(ctx: Context | None, key: str, cache: TTLCache) -> None:
    """Persist a cache to ctx state as a plain dict."""
    if ctx:
        await _ctx_set_state(ctx, key, dict(cache))


# ---------------------------------------------------------------------------
# SearXNG + Crawl4AI response unwrapping
# ---------------------------------------------------------------------------
async def _search(
    query: str,
    num_results: int = 10,
    time_range: str | None = None,
    pageno: int = 1,
    language: str | None = "en",
) -> dict:
    """Query SearXNG and return {"results": [...], "unresponsive_engines": [...]}.

    `unresponsive_engines` is SearXNG's per-engine failure report — list of
    [engine_name, error_string] pairs for engines that didn't contribute to
    this response (CAPTCHA'd, rate-limited, timed out, etc).
    """
    params: dict = {
        "q": query,
        "format": "json",
        "number_of_results": num_results,
    }
    if time_range:
        params["time_range"] = time_range.strip('"')
    if pageno > 1:
        params["pageno"] = pageno
    if language:
        params["language"] = language

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        resp = await client.get(f"{SEARXNG_URL}/search", params=params)
        resp.raise_for_status()
        data = resp.json()
        return {
            "results": data.get("results", [])[:num_results],
            "unresponsive_engines": data.get("unresponsive_engines", []),
        }


async def _probe_dependency(url: str) -> dict[str, str]:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        return {"status": "ok"}
    except httpx.HTTPError as exc:
        return {"status": "error", "detail": str(exc)}


def _extract_markdown(result: dict) -> str | None:
    html = result.get("html")
    if html:
        try:
            extracted = trafilatura.extract(
                html, output_format="txt", include_links=True, include_tables=True,
            )
            if extracted and len(extracted.strip()) >= 50:
                return extracted
        except Exception:
            pass

    md = result.get("markdown")
    if isinstance(md, dict):
        return md.get("fit_markdown") or md.get("raw_markdown")
    if isinstance(md, str):
        return md
    return result.get("cleaned_html")


def _content_type_without_charset(content_type: str | None) -> str | None:
    if not content_type:
        return None
    return content_type.split(";", 1)[0].strip().lower() or None


def _guess_file_type(url: str, content_type: str | None) -> str:
    normalized_content_type = _content_type_without_charset(content_type)
    suffix = (urlparse(url).path.rsplit(".", 1)[-1].lower() if "." in urlparse(url).path else "")

    if normalized_content_type == "application/pdf" or suffix == "pdf":
        return "pdf"
    if (
        normalized_content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or suffix == "docx"
    ):
        return "docx"
    if (
        normalized_content_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        or suffix == "xlsx"
    ):
        return "xlsx"
    if (
        normalized_content_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        or suffix == "pptx"
    ):
        return "pptx"
    if normalized_content_type in {"text/html", "application/xhtml+xml"} or suffix in {"html", "htm", "xhtml"}:
        return "html"
    if normalized_content_type in {"text/markdown", "text/x-markdown"} or suffix == "md":
        return "markdown"
    if normalized_content_type == "application/json" or suffix == "json":
        return "json"
    if normalized_content_type in {"application/yaml", "application/x-yaml", "text/yaml", "text/x-yaml"} or suffix in {"yaml", "yml"}:
        return "yaml"
    if normalized_content_type in {"application/xml", "text/xml"} or suffix in {"xml", "rss", "atom"}:
        return "xml"
    if normalized_content_type in {"text/csv", "application/csv", "application/vnd.ms-excel"} or suffix == "csv":
        return "csv"
    if normalized_content_type and normalized_content_type.startswith("text/"):
        return "text"
    guessed_type, _ = mimetypes.guess_type(url)
    guessed_type = _content_type_without_charset(guessed_type)
    if guessed_type and guessed_type.startswith("text/"):
        return "text"
    return "unknown"


def _extract_crawl_result(data: dict) -> dict:
    results = data.get("results")
    if results and isinstance(results, list):
        return results[0]
    return data.get("result", data)


def _extract_crawl_results(data: dict) -> list[dict]:
    results = data.get("results")
    if isinstance(results, list):
        return [result for result in results if isinstance(result, dict)]
    result = data.get("result", data)
    return [result] if isinstance(result, dict) else []


def _extract_crawl_title(result: dict) -> str | None:
    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        title = metadata.get("title")
        if title:
            return title
    title = result.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return None


def _extract_crawl_links(result: dict, base_url: str) -> list[dict]:
    link_groups = result.get("links")
    if not isinstance(link_groups, dict):
        return []

    links: list[dict] = []
    for link_type in ("internal", "external"):
        bucket = link_groups.get(link_type) or []
        if not isinstance(bucket, list):
            continue
        for link in bucket:
            if not isinstance(link, dict):
                continue
            href = link.get("href")
            if not isinstance(href, str) or not href.strip():
                continue
            absolute_url = urljoin(base_url, href.strip())
            parsed = urlparse(absolute_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                continue
            links.append({
                "url": absolute_url,
                "title": (link.get("title") or "").strip() or None,
                "text": (link.get("text") or "").strip() or None,
                "link_type": link_type,
            })
    return links


# ---------------------------------------------------------------------------
# Crawl4AI configs + HTTP layer (retry, poll, scrape, discover)
# ---------------------------------------------------------------------------
_DEFAULT_CRAWL_CONFIG = {
    "type": "CrawlerRunConfig",
    "params": {
        "excluded_tags": ["nav", "footer", "header", "aside"],
        "remove_overlay_elements": True,
        "markdown_generator": {
            "type": "DefaultMarkdownGenerator",
            "params": {
                "content_filter": {
                    "type": "PruningContentFilter",
                    "params": {
                        "threshold": 0.48,
                        "threshold_type": "fixed",
                        "min_word_threshold": 0,
                    },
                },
            },
        },
    },
}

# Separate config for link discovery (map/crawl). The default config
# strips nav/footer/header/aside to get clean content — but that's
# exactly where a docs site's link graph lives. Using the default here
# was making `map` return ~0 URLs on real sites. Keep the full page
# when we're only interested in hrefs.
_MAP_CRAWL_CONFIG = {
    "type": "CrawlerRunConfig",
    "params": {
        "remove_overlay_elements": True,
    },
}


def _domain_filter_patterns(root_url: str, same_domain_only: bool) -> list[str]:
    if not same_domain_only:
        return []
    root_domain = _domain_from_url(root_url).lower()
    registrable = _registrable_domain(root_domain)
    schemes = ("http", "https")
    patterns: list[str] = []
    for scheme in schemes:
        patterns.append(f"{scheme}://{registrable}/*")
        patterns.append(f"{scheme}://*.{registrable}/*")
        patterns.append(f"{scheme}://{root_domain}/*")
    return patterns


def _crawl_filter_chain(
    *,
    root_url: str,
    same_domain_only: bool,
    include_patterns: list[str] | None,
) -> list[dict]:
    filters: list[dict] = []
    domain_patterns = _domain_filter_patterns(root_url, same_domain_only)
    if domain_patterns:
        filters.append({
            "type": "URLPatternFilter",
            "params": {"patterns": domain_patterns},
        })
    if include_patterns:
        filters.append({
            "type": "URLPatternFilter",
            "params": {"patterns": include_patterns},
        })
    filters.append({
        "type": "ContentTypeFilter",
        "params": {"allowed_types": ["text/html"]},
    })
    # NOTE: We intentionally do NOT add a ContentRelevanceFilter here.
    # Filtering by query relevance during BFS exploration prematurely
    # rejects pages that link to relevant content but don't themselves
    # contain the query terms.  Relevance filtering happens post-crawl
    # via the FlashRank chunk reranker in crawl_impl instead.
    return filters


def _deep_crawl_config(
    *,
    root_url: str,
    max_depth: int,
    max_pages: int,
    same_domain_only: bool,
    include_patterns: list[str] | None = None,
    prefetch: bool = False,
) -> dict:
    base_config = _MAP_CRAWL_CONFIG if prefetch else _DEFAULT_CRAWL_CONFIG
    params = dict(base_config["params"])
    strategy_params: dict = {
        "max_depth": max_depth,
        "include_external": not same_domain_only,
        "max_pages": max_pages,
    }
    filter_chain = _crawl_filter_chain(
        root_url=root_url,
        same_domain_only=same_domain_only,
        include_patterns=include_patterns,
    )
    if filter_chain:
        strategy_params["filter_chain"] = filter_chain

    params["deep_crawl_strategy"] = {
        "type": "BFSDeepCrawlStrategy",
        "params": strategy_params,
    }
    if prefetch:
        params["prefetch"] = True
    return {
        "type": "CrawlerRunConfig",
        "params": params,
    }


def _is_retryable_crawl_error(exc: BaseException) -> bool:
    """Retry only on Crawl4AI transient failures: network issues and 5xx.

    Crawl4AI 0.8.x has documented browser-pool flakiness (memory leaks,
    'target page context closed' after N requests). A single transient 5xx or
    reset connection shouldn't kill the scrape.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, (httpx.TransportError, httpx.TimeoutException))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(_is_retryable_crawl_error),
    reraise=True,
)
async def _crawl_post(
    client: httpx.AsyncClient,
    url: str,
    priority: int,
    crawler_config: dict | None = None,
) -> dict:
    """POST to Crawl4AI's stream endpoint and collect NDJSON results."""
    resp = await client.post(
        f"{CRAWL4AI_URL}/crawl/stream",
        json={
            "urls": [url],
            "priority": priority,
            "crawler_config": crawler_config or _DEFAULT_CRAWL_CONFIG,
        },
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


async def _scrape_impl(url: str) -> dict:
    """Scrape a URL via Crawl4AI. Returns {content, title}."""
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        data = await _crawl_post(client, url, priority=8)

        result = _extract_crawl_result(data)
        return {
            "content": _extract_markdown(result),
            "title": _extract_crawl_title(result),
        }


async def _scrape(url: str) -> dict:
    """Scrape a URL via Crawl4AI, bounded by REQUEST_TIMEOUT seconds end-to-end.

    Returns {content, title}. On failure content is None.
    """
    empty = {"content": None, "title": None}
    try:
        return await asyncio.wait_for(_scrape_impl(url), timeout=REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        log.warning("scrape timed out url=%s budget=%ss", url, REQUEST_TIMEOUT)
    except httpx.HTTPError as e:
        log.warning("scrape http error url=%s err=%s", url, e)
    except (ValueError, KeyError) as e:
        log.warning("scrape payload error url=%s err=%s", url, e)
    return empty


async def _scrape_cached(url: str, cache: TTLCache) -> str | None:
    """Scrape with per-session cache. Returns cached content on hit, scrapes on miss."""
    if url in cache:
        log.debug("scrape cache hit url=%s", url)
        return cache[url]
    result = await _scrape(url)
    content = result["content"]
    cache[url] = content
    return content


async def _head_content_type(url: str) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
            resp = await client.head(url)
            resp.raise_for_status()
            return resp.headers.get("content-type")
    except httpx.HTTPError:
        return None


async def _sniff_content_type(url: str) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url, headers={"Range": "bytes=0-8191"})
            resp.raise_for_status()
            if not resp.content:
                return None
            detected = magic.from_buffer(resp.content, mime=True)
            return _content_type_without_charset(detected)
    except (httpx.HTTPError, OSError, ValueError):
        return None


async def _detect_file_type(url: str) -> tuple[str, str | None]:
    sniffed_content_type = await _sniff_content_type(url)
    header_content_type = await _head_content_type(url)
    content_type = sniffed_content_type or _content_type_without_charset(header_content_type)
    return _guess_file_type(url, content_type), content_type


# ---------------------------------------------------------------------------
# Per-file-type extractors / handoff
# ---------------------------------------------------------------------------
_LOCAL_EXTRACT_TYPES = {"text", "markdown", "json", "yaml", "xml", "csv"}


def _handoff_file_document(url: str, file_type: str, content_type: str | None) -> dict:
    return {
        "status": "handoff",
        "url": url,
        "content_type": content_type,
        "file_type": file_type,
        "title": None,
        "content": "",
        "total_chars": 0,
        "handoff": {
            "handler": "files",
            "reason": f"{file_type} extraction is delegated to the files MCP",
        },
    }


async def _extract_web_document(url: str) -> dict:
    result = await _scrape(url)
    content = result["content"]
    if not content:
        return {
            "status": "error",
            "url": url,
            "content_type": "text/html",
            "file_type": "html",
            "title": None,
            "content": "",
            "total_chars": 0,
            "error": "extraction failed",
        }
    return {
        "status": "ok",
        "url": url,
        "content_type": "text/html",
        "file_type": "html",
        "title": result.get("title"),
        "content": content,
        "total_chars": len(content),
    }


async def _extract_text_document(url: str, file_type: str) -> dict:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return {
            "status": "ok",
            "url": url,
            "content_type": _content_type_without_charset(resp.headers.get("content-type"))
            or mimetypes.guess_type(url)[0]
            or "text/plain",
            "file_type": file_type,
            "title": None,
            "content": resp.text,
            "total_chars": len(resp.text),
        }


async def _discover_page_links(url: str) -> dict:
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        data = await _crawl_post(client, url, priority=6, crawler_config=_MAP_CRAWL_CONFIG)

        result = _extract_crawl_result(data)
        return {
            "status": "ok",
            "url": url,
            "title": _extract_crawl_title(result),
            "links": _extract_crawl_links(result, url),
        }


async def _deep_crawl(
    url: str,
    *,
    max_depth: int,
    max_pages: int,
    same_domain_only: bool,
    include_patterns: list[str] | None = None,
    prefetch: bool = False,
) -> list[dict]:
    crawler_config = _deep_crawl_config(
        root_url=url,
        max_depth=max_depth,
        max_pages=max_pages,
        same_domain_only=same_domain_only,
        include_patterns=include_patterns,
        prefetch=prefetch,
    )
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        data = await _crawl_post(client, url, priority=7, crawler_config=crawler_config)
        return _extract_crawl_results(data)


# ---------------------------------------------------------------------------
# Ranking + central extract orchestrator
# ---------------------------------------------------------------------------
async def _rank_document_content(
    query: str | None, content: str, offset: int = 0,
) -> tuple[str, list[dict]]:
    """Return (display, top_chunks) for one document's raw content.

    offset>0 bypasses rerank and returns the next [offset, offset+MAX]
    slice — caller wants raw continuation after a prior truncated view.
    query+offset=0 returns top-K reranked chunks.
    No query, no offset returns the first-MAX slice.
    """
    if offset > 0:
        return content[offset:offset + _MAX_CONTENT_CHARS], []
    if not query or not content:
        return content[:_MAX_CONTENT_CHARS], []
    chunks = _chunk_text(content[:_MAX_CONTENT_CHARS])
    if not chunks:
        return content[:_MAX_CONTENT_CHARS], []
    scored = await _rerank_scored(query, chunks)
    top = [{"text": chunks[idx], "score": score} for idx, score in scored[:_TOP_CHUNKS]]
    if not top:
        return content[:_MAX_CONTENT_CHARS], []
    return _CHUNK_GAP.join(item["text"] for item in top), top


async def _extract_url_document(
    url: str,
    query: str | None,
    cache: TTLCache,
    offset: int = 0,
) -> dict:
    if url in cache:
        cached = cache[url]
        raw = cached.get("content", "")
        content, top_chunks = await _rank_document_content(query, raw, offset=offset)
        return {
            **cached,
            "content": content,
            "top_chunks": top_chunks,
            "cached": True,
        }

    file_type = "unknown"
    content_type = None
    try:
        file_type, content_type = await _detect_file_type(url)
        if file_type == "html":
            extracted = await _extract_web_document(url)
        elif file_type in _LOCAL_EXTRACT_TYPES:
            extracted = await _extract_text_document(url, file_type)
        else:
            extracted = _handoff_file_document(url, file_type, content_type)
    except Exception as exc:
        extracted = {
            "status": "error",
            "url": url,
            "content_type": content_type,
            "file_type": file_type,
            "title": None,
            "content": "",
            "total_chars": 0,
            "error": str(exc),
        }

    if extracted["status"] in {"ok", "handoff"}:
        # Cache successful local extracts and file handoffs so repeated
        # calls do not re-sniff/reclassify the same resource.
        raw = extracted.get("content", "")
        total_chars = extracted.get("total_chars", len(raw))
        cached_entry = {
            "status": extracted["status"],
            "url": url,
            "content_type": extracted.get("content_type"),
            "file_type": extracted.get("file_type"),
            "title": extracted.get("title"),
            "content": raw,
            "total_chars": total_chars,
            "handoff": extracted.get("handoff"),
        }
        cache[url] = cached_entry
        content, top_chunks = await _rank_document_content(query, raw, offset=offset)
        extracted["content"] = content
        extracted["total_chars"] = total_chars
        extracted["top_chunks"] = top_chunks
        extracted["cached"] = False
        return extracted

    extracted.setdefault("total_chars", 0)
    extracted["top_chunks"] = []
    extracted["cached"] = False
    return extracted


# ---------------------------------------------------------------------------
# Reranker — loaded once at import
# ---------------------------------------------------------------------------
log.info("loading reranker model=%s max_length=%d", RERANK_MODEL, _RERANK_MAX_LENGTH)
_ranker = Ranker(model_name=RERANK_MODEL, max_length=_RERANK_MAX_LENGTH)
log.info("reranker ready")


def _rerank_sync(query: str, documents: list[str]) -> list[tuple[int, float]]:
    """Synchronous rerank — runs the FlashRank ONNX model. Call via _rerank_scored."""
    if not documents:
        return []
    passages = [{"id": i, "text": doc, "meta": {}} for i, doc in enumerate(documents)]
    request = RerankRequest(query=query, passages=passages)
    results = _ranker.rerank(request)
    return [(r["id"], float(r["score"])) for r in results]


async def _rerank_scored(query: str, documents: list[str]) -> list[tuple[int, float]]:
    """Rerank documents via FlashRank off the event loop.

    ONNX inference on a CPU model takes tens of ms per call. Offloading to a
    thread keeps concurrent requests (scraping, searching) responsive.
    """
    return await asyncio.to_thread(_rerank_sync, query, documents)
