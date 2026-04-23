"""Private infrastructure the impls call into.

Everything that isn't a public impl or a markdown formatter lives here:
settings + constants, HTTP + retry + polling, rerank (FlashRank model
load and inference), text/URL utilities, validators, and the
per-file-type extractors. Cache adapters live in cache.py.
"""

import asyncio
import copy
import fnmatch
import ipaddress
import json
import logging
import mimetypes
import os
import re
import socket
from collections import defaultdict
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import httpx
from datasketch import MinHash, MinHashLSH
from langchain_text_splitters import MarkdownTextSplitter
import magic
# Kept available even though no current consumer uses it — markdown-it-py
# is already installed as a transitive dep through mdformat, and any
# future feature that wants structured access to our markdown (section
# extraction, AST-aware formatting) should use this parser rather than
# hand-rolled regex.
from markdown_it import MarkdownIt  # noqa: F401
from rapidfuzz import fuzz
import tldextract
import trafilatura
from flashrank import Ranker, RerankRequest
from pydantic_settings import BaseSettings
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from url_normalize import url_normalize

import cache as cache_module
from cache import KVCache


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
_SNIFF_MAX_BYTES = 8192
_DISPLAY_CHUNK_COUNT = 3
# Minimum FlashRank relevance score for a result's best chunk.  Entries
# scoring below this are CAPTCHA walls, wrong-language pages, or
# auto-generated spam — noise the reranker confidently identifies as
# irrelevant.  0.05 is conservative: real content almost always exceeds
# it, while garbage rarely reaches it.
_MIN_RELEVANCE_SCORE = 0.05

VALID_TIME_RANGES = frozenset({"day", "week", "month", "year"})

_TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "ref", "fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid",
})
_WORD_SPLIT = re.compile(r"\W+")
_WHITESPACE = re.compile(r"\s+")
_MARKDOWN_SPLITTER = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=0)
_PILCROW_LINK = re.compile(r"\[¶\]\([^)]*\)")
_MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\([^)]+\)")


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


def _dedup_pages(entries: list[dict], *, min_chars: int = 200) -> tuple[list[dict], int]:
    """Collapse pages with near-identical body content; keep first-seen entry.

    Catches the case where a site serves the same rendered page at several URLs
    (e.g. `/`, `/docs/getting-started`, `/docs/getting-started/intro` all rendering
    one intro page). Pages shorter than `min_chars` bypass the check since short
    bodies false-match easily against shared boilerplate.
    """
    kept: list[dict] = []
    lsh = MinHashLSH(threshold=_DEDUP_SIMILARITY, num_perm=_DEDUP_NUM_PERM)
    dropped = 0
    for entry in entries:
        content = entry.get("content") or ""
        if len(content) < min_chars:
            kept.append(entry)
            continue
        words = _word_set(content)
        if not words:
            kept.append(entry)
            continue
        sketch = _chunk_minhash(words)
        if lsh.query(sketch):
            dropped += 1
            continue
        lsh.insert(len(kept), sketch)
        kept.append(entry)
    return kept, dropped


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
    return chunks


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
        _reject_non_public_target(parsed.hostname, url=value)
        normalized.append(value)
    return normalized


def _is_blocked_ip(ip: ipaddress._BaseAddress) -> bool:
    return any((
        ip.is_private,
        ip.is_loopback,
        ip.is_link_local,
        ip.is_multicast,
        ip.is_reserved,
        ip.is_unspecified,
        not ip.is_global,
    ))


def _reject_non_public_target(hostname: str | None, *, url: str) -> None:
    """Reject localhost and DNS targets that resolve to non-public IP space."""
    if not hostname:
        raise ValueError(f"invalid URL: {url!r}")

    host = hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError(f"URL resolves to a private or reserved target: {url!r}")

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None

    if ip is not None:
        if _is_blocked_ip(ip):
            raise ValueError(f"URL resolves to a private or reserved target: {url!r}")
        return

    try:
        resolved = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return

    for family, _, _, _, sockaddr in resolved:
        if family == socket.AF_INET:
            candidate = ipaddress.ip_address(sockaddr[0])
        elif family == socket.AF_INET6:
            candidate = ipaddress.ip_address(sockaddr[0])
        else:
            continue
        if _is_blocked_ip(candidate):
            raise ValueError(f"URL resolves to a private or reserved target: {url!r}")


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


# Matches markdown table separator rows like "| --- | --- |" or "| :---: | ---: |".
# Alignment markers (`:`) and dashes only — any other content disqualifies.
_TABLE_SEPARATOR_ROW = re.compile(r"^\s*\|?\s*(:?-{3,}:?\s*\|\s*)+:?-{3,}:?\s*\|?\s*$")


def _strip_table_separator_rows(text: str) -> str:
    """Drop `|---|---|` noise rows while leaving the surrounding table intact."""
    if "|" not in text or "---" not in text:
        return text
    return "\n".join(
        line for line in text.splitlines() if not _TABLE_SEPARATOR_ROW.match(line)
    )


def _is_link_soup_line(line: str) -> bool:
    """Detect dense nav/TOC lines that are mostly markdown links."""
    matches = list(_MARKDOWN_LINK.finditer(line))
    if len(matches) < 4:
        return False
    residue = _MARKDOWN_LINK.sub("", line)
    residue = residue.replace("`", "").replace("*", "").replace("_", "")
    residue = _WHITESPACE.sub("", residue)
    return len(residue) <= 12


def _clean_extracted_markdown(text: str | None) -> str | None:
    """Light cleanup for site chrome that leaks through extraction."""
    if not text:
        return text
    cleaned = _PILCROW_LINK.sub("", text)
    lines = [line.rstrip() for line in cleaned.splitlines()]
    kept: list[str] = []
    blank_streak = 0
    for line in lines:
        stripped = line.strip()
        if stripped and _is_link_soup_line(stripped):
            continue
        if not stripped:
            blank_streak += 1
            if blank_streak > 1:
                continue
        else:
            blank_streak = 0
        kept.append(line)
    cleaned = "\n".join(kept).strip("\n")
    return cleaned or None


def _extract_markdown(result: dict) -> str | None:
    html = result.get("html")
    if html:
        try:
            # output_format="markdown" preserves structure the txt format
            # throws away: code gets properly fenced with ```, headings
            # carry their '#' prefix, bold/italic survive. This is what
            # the LLM reads AND what _extract_structure walks for the
            # structural metadata (headings, code_blocks, outgoing_links).
            extracted = trafilatura.extract(
                html, output_format="markdown", include_links=True, include_tables=True,
            )
            if extracted and len(extracted.strip()) >= 50:
                return _clean_extracted_markdown(_strip_table_separator_rows(extracted))
        except Exception:
            pass

    md = result.get("markdown")
    content: str | None
    if isinstance(md, dict):
        content = md.get("fit_markdown") or md.get("raw_markdown")
    elif isinstance(md, str):
        content = md
    else:
        content = result.get("cleaned_html")
    return _clean_extracted_markdown(_strip_table_separator_rows(content)) if content else content


def _extract_html_metadata(html: str | None) -> dict:
    """Pull author/date/site_name/description from raw HTML via trafilatura."""
    if not html:
        return {}
    try:
        doc = trafilatura.extract_metadata(html)
    except Exception:
        return {}
    if doc is None:
        return {}
    return {
        "author": doc.author or None,
        "date": doc.date or None,
        "site_name": doc.sitename or None,
        "description": doc.description or None,
    }


def _content_hash(content: str) -> str:
    """Stable fingerprint for exact-duplicate detection at write time."""
    import hashlib
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


def _build_document_metadata(html: str | None, content: str | None) -> dict:
    """Citation metadata for the response. Structural info (headings,
    code blocks, links) is intentionally not duplicated here — it's
    already present inline in the markdown body the LLM receives."""
    metadata = _extract_html_metadata(html)
    if content:
        metadata["word_count"] = len(content.split())
    return {k: v for k, v in metadata.items() if v is not None}


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


# Visible link text used for accessibility shortcuts — useless as a map snippet
# since it describes the *mechanism* of the link, not its destination.
_A11Y_LINK_TEXTS = frozenset({
    "skip to main content",
    "skip to content",
    "skip to main",
    "skip navigation",
    "skip to navigation",
    "jump to content",
    "jump to main content",
    "jump to navigation",
    "main content",
})


def _clean_link_label(value: str | None) -> str | None:
    if not value:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.lower() in _A11Y_LINK_TEXTS:
        return None
    return stripped


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
                "title": _clean_link_label(link.get("title")),
                "text": _clean_link_label(link.get("text")),
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
) -> dict:
    # Discovery should keep the full page chrome where site topology
    # often lives. Starting from the content-pruned default config can
    # collapse map/crawl to the root page on real sites.
    params = copy.deepcopy(_MAP_CRAWL_CONFIG["params"])
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
    urls: list[str],
    priority: int,
    crawler_config: dict | None = None,
) -> dict:
    """POST to Crawl4AI's stream endpoint and collect NDJSON results.

    `urls` is the seed list. With a BFS deep_crawl_strategy in the
    config, Crawl4AI expands each seed independently.
    """
    resp = await client.post(
        f"{CRAWL4AI_URL}/crawl/stream",
        json={
            "urls": urls,
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
    """Scrape a URL via Crawl4AI. Returns {content, title, metadata}."""
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        data = await _crawl_post(client, [url], priority=8)

        result = _extract_crawl_result(data)
        content = _extract_markdown(result)
        return {
            "content": content,
            "title": _extract_crawl_title(result),
            "metadata": _build_document_metadata(result.get("html"), content),
        }


async def _scrape(url: str) -> dict:
    """Scrape a URL via Crawl4AI, bounded by REQUEST_TIMEOUT seconds end-to-end.

    Returns {content, title, metadata}. On failure content is None.
    """
    empty = {"content": None, "title": None, "metadata": {}}
    try:
        return await asyncio.wait_for(_scrape_impl(url), timeout=REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        log.warning("scrape timed out url=%s budget=%ss", url, REQUEST_TIMEOUT)
    except httpx.HTTPError as e:
        log.warning("scrape http error url=%s err=%s", url, e)
    except (ValueError, KeyError) as e:
        log.warning("scrape payload error url=%s err=%s", url, e)
    return empty


# Minimum word count for a speculative scrape (search / crawl paths) to
# be admitted to the cache. Calibrated to exclude CAPTCHA walls and
# 404 shells (typically <15 words) without rejecting short-but-useful
# doc snippets. User-directed extract bypasses this gate.
_MIN_CACHE_WORDS = 20


async def _scrape_cached(url: str, cache: KVCache) -> dict:
    """Scrape with shared page cache. Returns the full envelope.

    The envelope is the same shape a fresh extract would cache, so the
    search pipeline and user-facing extract calls share a single entry
    per normalized URL. Callers that only need {content, title,
    metadata} read those fields; callers that need status / file_type /
    etc. read those too.

    Speculative fetches (search/crawl) apply a minimum-length gate:
    scraped pages with fewer than _MIN_CACHE_WORDS cache as failures.
    Blocks CAPTCHA walls / error shells from crowding the cache.
    """
    key = _normalize_url(url)
    existing = await _page_get(url, cache)
    if existing is not None:
        log.debug("page cache hit url=%s", url)
        return existing
    result = await _scrape(url)
    content = result.get("content")
    # Length floor — only applied at speculative write time, not on user-
    # directed extract (which wants whatever it asked for).
    if content and len(content.split()) < _MIN_CACHE_WORDS:
        content = None
    entry = _page_entry(
        url=url,
        content=content,
        title=result.get("title"),
        metadata=result.get("metadata") or {},
    )
    await _page_set(url, entry, cache)
    return entry


async def _page_set(url: str, entry: dict, cache: KVCache) -> None:
    """Write a page entry, aliasing when content_hash already seen.

    If another URL has already cached a page with this exact content,
    write a lightweight alias entry instead of a duplicate full entry.
    Dangling detection happens on read via _page_get.

    Failed/rejected entries (status != "ok") get FAILURE_TTL_S so
    transient upstream issues recover quickly instead of being cached
    as broken for the full TTL.
    """
    key = _normalize_url(url)
    content_hash = entry.get("_content_hash")
    if not content_hash or entry.get("status") != "ok":
        # No hash (no content) or rejected entry → plain write, no dedup.
        ttl = cache_module.FAILURE_TTL_S if entry.get("status") != "ok" else None
        await cache.set(key, entry, ttl=ttl)
        return

    canonical_key = await cache_module.content_alias.get(content_hash)
    if canonical_key and canonical_key != key:
        canonical = await cache.get(canonical_key)
        canonical_hash = (canonical or {}).get("_content_hash")
        if canonical and canonical_hash == content_hash:
            # Existing canonical confirmed — alias through it.
            await cache.set(key, {
                "_schema_version": 1,
                "alias": canonical_key,
                "content_hash": content_hash,
            })
            return
        # Canonical missing or drifted — fall through and reclaim the hash.

    await cache.set(key, entry)
    await cache_module.content_alias.set(content_hash, key)


async def _page_get(url: str, cache: KVCache) -> dict | None:
    """Read a page entry, dereferencing aliases and detecting dangling.

    An alias entry (`{alias, content_hash}`) is resolved against the
    canonical URL. If the canonical is missing or its content_hash
    drifted (stale alias), return None so the caller re-scrapes.
    """
    key = _normalize_url(url)
    entry = await cache.get(key)
    if not entry:
        return None
    if "alias" not in entry:
        return entry
    canonical_key = entry["alias"]
    expected_hash = entry.get("content_hash")
    canonical = await cache.get(canonical_key)
    if not canonical:
        return None
    canonical_hash = canonical.get("_content_hash")
    if canonical_hash != expected_hash:
        return None
    # Preserve the caller's requested URL in the returned view.
    return {**canonical, "url": url}


def _page_entry(
    *,
    url: str,
    content: str | None,
    title: str | None,
    metadata: dict,
    content_type: str = "text/html",
    file_type: str = "html",
    handoff: dict | None = None,
    status: str | None = None,
) -> dict:
    """Construct a unified page-cache envelope.

    Callers supply what they know; everything else is defaulted. `status`
    defaults to 'ok' when content is non-empty and 'error' otherwise.
    """
    if status is None:
        status = "ok" if content else "error"
    envelope: dict = {
        "_schema_version": 1,
        "status": status,
        "url": url,
        "content_type": content_type,
        "file_type": file_type,
        "title": title,
        "content": content,
        "total_chars": len(content) if content else 0,
        "metadata": metadata,
        "handoff": handoff,
    }
    # Internal-only field: content fingerprint used by _page_set /
    # _page_get for exact-dupe aliasing. Prefixed with '_' so it never
    # gets confused with caller-facing metadata fields. Stripped from
    # the response in impls.*_impl when building the structured_content
    # payload.
    if content and status == "ok":
        envelope["_content_hash"] = _content_hash(content)
    return envelope


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
            async with client.stream(
                "GET",
                url,
                headers={
                    "Range": f"bytes=0-{_SNIFF_MAX_BYTES - 1}",
                    "Accept-Encoding": "identity",
                },
            ) as resp:
                resp.raise_for_status()
                if resp.status_code != httpx.codes.PARTIAL_CONTENT:
                    return None
                length = resp.headers.get("content-length")
                if length is not None and int(length) > _SNIFF_MAX_BYTES:
                    return None
                chunks: list[bytes] = []
                total = 0
                async for chunk in resp.aiter_bytes():
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > _SNIFF_MAX_BYTES:
                        return None
                    chunks.append(chunk)
            if not chunks:
                return None
            detected = magic.from_buffer(b"".join(chunks), mime=True)
            return _content_type_without_charset(detected)
    except (httpx.HTTPError, OSError, ValueError):
        return None


async def _detect_file_type(url: str) -> tuple[str, str | None]:
    header_content_type = await _head_content_type(url)
    content_type = _content_type_without_charset(header_content_type)
    guessed_content_type = _content_type_without_charset(mimetypes.guess_type(url)[0])
    if content_type in {None, "application/octet-stream"}:
        content_type = guessed_content_type or content_type
    if content_type in {None, "application/octet-stream"}:
        content_type = await _sniff_content_type(url) or content_type
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
        "metadata": {},
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
            "metadata": {},
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
        "metadata": result.get("metadata") or {},
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
            "metadata": {"word_count": len(resp.text.split())} if resp.text else {},
        }


async def _discover_page_links(url: str) -> dict:
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        data = await _crawl_post(client, [url], priority=6, crawler_config=_MAP_CRAWL_CONFIG)

        result = _extract_crawl_result(data)
        return {
            "status": "ok",
            "url": url,
            "title": _extract_crawl_title(result),
            "links": _extract_crawl_links(result, url),
        }


async def _deep_crawl(
    seeds: list[str],
    *,
    max_depth: int,
    max_pages: int,
    same_domain_only: bool,
    include_patterns: list[str] | None = None,
) -> list[dict]:
    """BFS deep crawl starting from every seed URL.

    The BFS strategy's same_domain and include_pattern filters are keyed
    off the first seed (as the "root") for domain-scope purposes, since
    all seeds should already be in-scope at the call site.
    """
    if not seeds:
        return []
    crawler_config = _deep_crawl_config(
        root_url=seeds[0],
        max_depth=max_depth,
        max_pages=max_pages,
        same_domain_only=same_domain_only,
        include_patterns=include_patterns,
    )
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        data = await _crawl_post(client, seeds, priority=7, crawler_config=crawler_config)
        return _extract_crawl_results(data)


# ---------------------------------------------------------------------------
# Ranking + central extract orchestrator
# ---------------------------------------------------------------------------
async def _rank_document_content(
    query: str | None,
    content: str,
    chunk_ids: list[int] | None = None,
) -> tuple[str, list[dict], list[dict], list[int], str]:
    """Return (display, top_chunks, chunks) for one document's raw content.

    `chunks` is the full chunk list for the first _MAX_CONTENT_CHARS of
    content with stable ids (derived fresh each call — chunking is
    deterministic, so ids are stable across calls for the same raw
    content). Empty when there is no content to chunk.

    `top_chunks` is the reranked top-K — populated only when a query is
    given and no explicit chunk selection override applies.

    `display` is what the caller reads:
      - chunk_ids provided: joined text of the requested ids
      - query provided:     joined text of top-K reranked chunks
      - otherwise:          first display-sized chunk window in document order
    """
    chunks = [
        {"id": i, "text": text} for i, text in enumerate(_chunk_text(content))
    ]

    if chunk_ids is not None:
        wanted = set(chunk_ids)
        selected = [c for c in chunks if c["id"] in wanted]
        display = _CHUNK_GAP.join(c["text"] for c in selected)
        return display, [], chunks, [c["id"] for c in selected], "selected"

    if not query or not content:
        selected = chunks[:_DISPLAY_CHUNK_COUNT]
        display = _CHUNK_GAP.join(c["text"] for c in selected)
        return display, [], chunks, [c["id"] for c in selected], "document"

    if not chunks:
        return content[:_MAX_CONTENT_CHARS], [], chunks, [], "relevant"

    chunk_texts = [c["text"] for c in chunks]
    scored = await _rerank_scored(query, chunk_texts)
    top = [
        {"id": idx, "text": chunk_texts[idx], "score": score}
        for idx, score in scored[:_TOP_CHUNKS]
    ]
    if not top:
        selected = chunks[:_DISPLAY_CHUNK_COUNT]
        display = _CHUNK_GAP.join(c["text"] for c in selected)
        return display, [], chunks, [c["id"] for c in selected], "document"
    return (
        _CHUNK_GAP.join(item["text"] for item in top),
        top,
        chunks,
        [item["id"] for item in top],
        "relevant",
    )


async def _extract_url_document(
    url: str,
    query: str | None,
    cache: KVCache,
    chunk_ids: list[int] | None = None,
) -> dict:
    # Normalized URL is the cache key so www./trailing-slash variants
    # collapse. The stored entry keeps the caller's original URL for
    # display (see cached_entry["url"] below).
    cached = await _page_get(url, cache)
    if cached is not None:
        raw = cached.get("content") or ""
        content, top_chunks, chunks, shown_chunk_ids, chunk_mode = await _rank_document_content(
            query, raw, chunk_ids=chunk_ids,
        )
        return {
            **cached,
            "content": content,
            "top_chunks": top_chunks,
            "chunks": chunks,
            "shown_chunk_ids": shown_chunk_ids,
            "total_chunks": len(chunks),
            "chunk_mode": chunk_mode,
            "cached": True,
        }
    key = _normalize_url(url)

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
            "metadata": {},
            "error": str(exc),
        }

    if extracted["status"] in {"ok", "handoff"}:
        # Cache successful local extracts and file handoffs so repeated
        # calls do not re-sniff/reclassify the same resource.
        raw = extracted.get("content", "")
        total_chars = extracted.get("total_chars", len(raw))
        cached_entry = _page_entry(
            url=url,
            content=raw,
            title=extracted.get("title"),
            metadata=extracted.get("metadata") or {},
            content_type=extracted.get("content_type") or "text/html",
            file_type=extracted.get("file_type") or "html",
            handoff=extracted.get("handoff"),
            status=extracted["status"],
        )
        # Preserve the upstream's total_chars (e.g. local text documents
        # that know their own length) rather than deriving from content.
        cached_entry["total_chars"] = total_chars
        await _page_set(url, cached_entry, cache)
        content, top_chunks, chunks, shown_chunk_ids, chunk_mode = await _rank_document_content(
            query, raw, chunk_ids=chunk_ids,
        )
        extracted["content"] = content
        extracted["total_chars"] = total_chars
        extracted["top_chunks"] = top_chunks
        extracted["chunks"] = chunks
        extracted["shown_chunk_ids"] = shown_chunk_ids
        extracted["total_chunks"] = len(chunks)
        extracted["chunk_mode"] = chunk_mode
        extracted["cached"] = False
        return extracted

    extracted.setdefault("total_chars", 0)
    extracted["top_chunks"] = []
    extracted["chunks"] = []
    extracted["shown_chunk_ids"] = []
    extracted["total_chunks"] = 0
    extracted["chunk_mode"] = None
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
