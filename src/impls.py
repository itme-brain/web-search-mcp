"""The four public Python-API impls: search, extract, map, crawl.

Each returns a structured dict. `@mcp.tool` wrappers in server.py call
these, then push the result through a formatter in formatters.py to
produce the LLM-facing markdown string.
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import Counter, defaultdict

from fastmcp import Context

# Module-qualified import so `unittest.mock.patch("core.X")` intercepts
# every call site — both from core itself and from here. (`from core
# import X` would bind X into impls's namespace, creating a second patch
# target we'd have to mock separately.) Constants are safe to import
# by-name since they're not patched.
import cache as cache_module
import core
import models
import semantic
from core import (
    MAX_RESULTS,
    MAX_SCRAPE,
    RERANK_MODEL,
    _CHUNK_GAP,
    _MAX_CONTENT_CHARS,
    _MAX_EXTRACT_URLS,
    _MAX_MAP_URLS,
    _MIN_RELEVANCE_SCORE,
)

log = logging.getLogger("web-search-mcp")


# In-flight SearXNG requests, keyed on the searxng_cache key. Two
# concurrent searches for the same (query, num_results, time_range,
# language, pageno) share a single upstream call instead of both
# cache-missing and both hitting the SearXNG → brave/google/etc. chain.
# Cleared on completion (success or failure); awaiters of a failed
# request re-raise and the next caller will retry.
_searxng_inflight: dict[str, asyncio.Future] = {}


def _validated_response(model_cls, response: dict) -> dict:
    """Return a model-validated payload without inventing unset keys."""
    return models.dump_response(model_cls, response)


def _search_mode_budget(mode: str, num_results: int) -> tuple[int, int, int]:
    """Return (searx_pages, scrape_budget, default_passages) for a search mode."""
    if mode == "fast":
        return 1, min(num_results, 5, MAX_SCRAPE), 1
    if mode == "research":
        return 3, min(max(num_results * 2, 10), MAX_SCRAPE), 4
    return 2, min(num_results, MAX_SCRAPE), 2


def _validate_optional_positive_int(name: str, value: int | None, *, maximum: int) -> int | None:
    if value is None:
        return None
    return core._validate_positive_int(name, value, maximum=maximum)


def _query_terms(query: str) -> list[str]:
    stop = {
        "a", "an", "and", "are", "as", "at", "by", "for", "from", "how", "in", "is",
        "it", "of", "on", "or", "the", "to", "used", "using", "what", "when", "where",
        "which", "who", "why", "with",
    }
    terms = [term for term in core.re.findall(r"[A-Za-z0-9_.+-]{3,}", query) if term.lower() not in stop]
    return terms[:8]


def _keyphrase(query: str) -> str:
    terms = _query_terms(query)
    return " ".join(terms[:5]) or query


def _search_queries(query: str, mode: str) -> list[str]:
    """Small, deterministic research expansion without an LLM/API."""
    if mode != "research":
        return [query]
    keyphrase = _keyphrase(query)
    quoted = f'"{keyphrase}"' if keyphrase != query or len(query) <= 80 else query
    variants = [
        query,
        quoted,
        f'{keyphrase} documentation docs official',
        f'{keyphrase} github gitlab repository',
        f'{keyphrase} issue discussion mailing list',
        f'{keyphrase} example case study',
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        key = variant.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(variant)
    return deduped


_SOURCE_TYPE_ALIASES = {
    "docs": "official_docs",
    "official_docs": "official_docs",
    "repo": "repo",
    "github": "repo",
    "gitlab": "repo",
    "issue": "issue_tracker",
    "issues": "issue_tracker",
    "issue_tracker": "issue_tracker",
    "mail": "mailing_list",
    "mailing_list": "mailing_list",
    "qa": "qa",
    "stackoverflow": "qa",
    "blog": "blog",
    "pdf": "pdf",
    "paper": "pdf",
    "web": "web",
}


def _normalize_source_types(source_types: list[str] | None) -> list[str] | None:
    if not source_types:
        return None
    normalized: list[str] = []
    for value in source_types:
        key = value.strip().lower()
        if not key:
            continue
        mapped = _SOURCE_TYPE_ALIASES.get(key)
        if mapped is None:
            raise ValueError(f"invalid source_type: {value!r}. Expected one of {sorted(_SOURCE_TYPE_ALIASES)}")
        if mapped not in normalized:
            normalized.append(mapped)
    return normalized or None


def _source_boost(source_type: str) -> float:
    return {
        "official_docs": 0.06,
        "repo": 0.05,
        "issue_tracker": 0.04,
        "mailing_list": 0.035,
        "qa": 0.015,
        "pdf": 0.01,
        "web": 0.0,
        "blog": -0.015,
    }.get(source_type, 0.0)


def _matches_source_types(url: str, source_types: list[str] | None) -> bool:
    return not source_types or _source_type(url) in source_types


def _entry_sort_score(eidx: int, entries: list[dict], entry_best: dict[int, float | None]) -> float:
    return (entry_best.get(eidx) or 0.0) + _source_boost(_source_type(entries[eidx]["url"]))


def _diversify_by_source_type(ranked_entry_idxs: list[int], entries: list[dict]) -> list[int]:
    """Avoid giving small models several same-kind sources before variety."""
    by_type: dict[str, list[int]] = defaultdict(list)
    order: list[str] = []
    for eidx in ranked_entry_idxs:
        source_type = _source_type(entries[eidx]["url"])
        if source_type not in by_type:
            order.append(source_type)
        by_type[source_type].append(eidx)
    diversified: list[int] = []
    while by_type:
        next_order: list[str] = []
        for source_type in order:
            queue = by_type.get(source_type)
            if not queue:
                continue
            diversified.append(queue.pop(0))
            if queue:
                next_order.append(source_type)
            else:
                by_type.pop(source_type, None)
        order = next_order
    return diversified


def _brief_from_results(results: list[dict]) -> list[str]:
    brief: list[str] = []
    for result in results[:3]:
        passages = result.get("passages") or []
        if not passages:
            continue
        text = passages[0].get("text", "").replace("\n", " ").strip()
        if len(text) > 220:
            text = text[:219].rstrip() + "…"
        if text:
            brief.append(f"[{result['rank']}] {text}")
    return brief


def _next_actions(mode: str, degraded: bool, results: list[dict], warnings: list[dict]) -> list[str]:
    actions: list[str] = []
    if degraded or any(w.get("type") in {"search_failed", "scrape_failed", "rerank_failed"} for w in warnings):
        actions.append("Results are degraded; retry or use mode=research for more coverage.")
    if results:
        actions.append("Use extract on the best URL if more context is needed.")
    if mode != "research" and len(results) < 3:
        actions.append("Use mode=research or broaden the query if coverage is too thin.")
    return actions[:3]


def _limit_passages(passages: list[tuple[str, float]], max_passages: int, max_chars: int) -> list[tuple[str, float]]:
    kept: list[tuple[str, float]] = []
    used = 0
    for text, score in passages:
        remaining = max_chars - used
        if remaining <= 0 or len(kept) >= max_passages:
            break
        clipped = text if len(text) <= remaining else text[: max(0, remaining - 1)].rstrip() + "…"
        if clipped:
            kept.append((clipped, score))
            used += len(clipped)
    return kept


def _source_type(url: str) -> str:
    domain = core._domain_from_url(url).lower()
    path = core.urlparse(url).path.lower()
    if domain in {"github.com", "gitlab.com", "bitbucket.org", "sourceforge.net"}:
        if "/issues" in path or "/-/issues" in path:
            return "issue_tracker"
        return "repo"
    if path.endswith(".pdf"):
        return "pdf"
    if any(part in domain for part in ("docs.", "readthedocs", "documentation", "developer.")) or any(part in path for part in ("/docs", "/documentation", "/reference", "/api/")):
        return "official_docs"
    if any(part in domain for part in ("lists.", "mail.", "mailman", "groups.google")):
        return "mailing_list"
    if "stackoverflow.com" in domain or "stackexchange.com" in domain:
        return "qa"
    if any(part in domain for part in ("medium.com", "substack.com", "blog")) or "/blog" in path:
        return "blog"
    return "web"


async def search_impl(
    query: str,
    num_results: int = 5,
    mode: str = "deep",
    max_passages: int | None = None,
    max_chars_per_result: int | None = None,
    include_raw_content: bool = False,
    source_types: list[str] | None = None,
    time_range: str | None = None,
    language: str | None = "en",
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    ctx: Context | None = None,
) -> dict:
    """Search the web, scrape top results, and return structured JSON ranked by relevance.

    Pipeline: SearXNG search -> Crawl4AI scrape -> chunk -> FlashRank reranker -> formatted output.
    Scraped pages are split into paragraphs and reranked at the chunk level, so only
    the most query-relevant excerpts from each page are returned.

    Fetches page 2 from SearXNG only if page 1 after dedup/filter is short of
    `num_results`. Always scrapes `min(num_results, MAX_SCRAPE)` top candidates.
    Results are backed by shared Valkey caches across requests.
    """
    query = core._validate_query(query)
    num_results = core._validate_positive_int("num_results", num_results, maximum=MAX_RESULTS)
    mode = core._normalize_search_mode(mode)
    time_range = core._normalize_time_range(time_range)
    language = core._coerce_optional_str(language)
    include_domains = core._normalize_domains(include_domains, field_name="include_domains")
    exclude_domains = core._normalize_domains(exclude_domains, field_name="exclude_domains")
    source_types = _normalize_source_types(source_types)
    searx_pages, scrape_budget, default_passages = _search_mode_budget(mode, num_results)
    max_passages = _validate_optional_positive_int("max_passages", max_passages, maximum=8) or default_passages
    default_chars = 900 if mode == "fast" else 1800 if mode == "deep" else 3200
    max_chars_per_result = _validate_optional_positive_int(
        "max_chars_per_result", max_chars_per_result, maximum=8000
    ) or default_chars
    search_queries = _search_queries(query, mode)
    started = time.monotonic()
    warnings: list[dict] = []
    degraded = False
    timings_ms = {"search": 0, "scrape": 0, "rerank": 0, "total": 0}

    # --- shared cache (Valkey-backed, cross-session + cross-process) ---
    page_cache = cache_module.page_cache
    searxng_cache = cache_module.searxng_cache
    seen_urls = cache_module.seen_urls
    semantic_cache = cache_module.semantic_cache

    async def _searxng_cached(search_query: str, pageno: int) -> dict:
        """SearXNG call cached on (query, num_results, time_range, language, pageno).

        Filters (include_domains / exclude_domains) are NOT part of the
        key — the cache holds raw SearXNG output and filters apply at
        response-shaping time.

        Single-flighted: concurrent callers with the same key share one
        upstream SearXNG request instead of amplifying load on brave /
        google / etc.
        """
        key = hashlib.sha256(
            json.dumps(
                [search_query.lower().strip(), num_results, time_range, language, pageno],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        cached = await searxng_cache.get(key)
        if cached is not None:
            return cached
        inflight = _searxng_inflight.get(key)
        if inflight is not None:
            return await inflight
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        _searxng_inflight[key] = fut
        try:
            result = await core._search(
                search_query, num_results=num_results, time_range=time_range,
                language=language, pageno=pageno,
            )
            await searxng_cache.set(key, result)
            fut.set_result(result)
            return result
        except Exception as exc:
            fut.set_exception(exc)
            raise
        finally:
            _searxng_inflight.pop(key, None)

    # --- search (mode-aware multi-page retrieval) ---
    search_started = time.monotonic()
    unresponsive_engines: list = []
    raw_results: list[dict] = []
    for search_query in search_queries:
        for pageno in range(1, searx_pages + 1):
            try:
                page = await _searxng_cached(search_query, pageno=pageno)
            except Exception as exc:
                if not raw_results and pageno == 1:
                    degraded = True
                    warnings.append(core._warning("search_failed", "searxng", str(exc)))
                else:
                    warnings.append(core._warning("search_failed", "searxng", f"{search_query} page {pageno}: {exc}"))
                break
            raw_results.extend(page["results"])
            unresponsive_engines.extend(page.get("unresponsive_engines", []))
            filtered_so_far = core._dedup_results(core._filter_results_by_domain(raw_results, include_domains, exclude_domains))
            if mode == "fast" or (mode == "deep" and len(filtered_so_far) >= num_results):
                break
        if mode != "research":
            break

    results = core._dedup_results(core._filter_results_by_domain(raw_results, include_domains, exclude_domains))
    results = [r for r in results if _matches_source_types(r.get("url", ""), source_types)]
    results = results[:max(num_results, scrape_budget)]

    # Surface per-engine failures from SearXNG (e.g. "google: CAPTCHA").
    # Does NOT flip `degraded` — the multi-engine hedge means a single
    # upstream going down is expected, not a pipeline failure.
    for engine, reason in core._dedup_unresponsive_engines(unresponsive_engines):
        detail = f"{engine}: {reason}" if reason else engine
        warnings.append(core._warning("engine_unresponsive", "searxng", detail))

    timings_ms["search"] = int((time.monotonic() - search_started) * 1000)
    if not results:
        response = {
            "query": query,
            "time_range": time_range,
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
            "results": [],
            "meta": {
                "mode": mode,
                "brief": [],
                "next_actions": ["No results found; broaden the query or remove filters."],
                "num_results_requested": num_results,
                "num_results_returned": 0,
                "scrape_top": scrape_budget,
                "max_passages": max_passages,
                "max_chars_per_result": max_chars_per_result,
                "search_queries": search_queries,
                "source_types": source_types,
                "include_raw_content": include_raw_content,
                "search_backend": "searxng",
                "reranker": {"name": "flashrank", "model": RERANK_MODEL},
                "degraded": degraded,
                "warnings": warnings or [core._warning("no_results", "searxng", query)],
                "timings_ms": {
                    **timings_ms,
                    "total": int((time.monotonic() - started) * 1000),
                },
            },
        }
        return _validated_response(models.SearchResponseModel, response)

    # --- scrape (cache-aware) ---
    to_scrape = min(scrape_budget, len(results))
    scrape_started = time.monotonic()
    scrape_tasks = [core._scrape_cached(r["url"], page_cache) for r in results[:to_scrape]]
    scraped = await asyncio.gather(*scrape_tasks)
    timings_ms["scrape"] = int((time.monotonic() - scrape_started) * 1000)
    scrape_failures = sum(1 for s in scraped if s.get("content") is None)
    if scrape_failures:
        degraded = True
        warnings.append(core._warning("scrape_failed", "crawl4ai", f"{scrape_failures} of {to_scrape} pages failed"))

    # --- build entries ---
    entries: list[dict] = []
    for i, result in enumerate(results[:to_scrape]):
        scrape = scraped[i]
        content = scrape.get("content")
        metadata = scrape.get("metadata") or {}
        raw = content[:_MAX_CONTENT_CHARS] if content else None
        entries.append({
            "title": result.get("title", "Untitled"),
            "url": result.get("url", ""),
            "content": raw or result.get("content", ""),
            "scraped": raw is not None,
            "metadata": metadata,
        })

    for result in results[to_scrape:]:
        entries.append({
            "title": result.get("title", "Untitled"),
            "url": result.get("url", ""),
            "content": result.get("content", ""),
            "scraped": False,
            "metadata": {},
        })

    # Add recently cached pages for local-memory retrieval. This uses
    # Valkey as a lightweight semantic cache when enabled, plus page-level
    # cached entries, reducing repeat web fetches and giving small/local
    # models better continuity.
    cached_entries = await asyncio.gather(
        *(semantic_cache.get(core._normalize_url(r.get("url", ""))) for r in results)
    )
    semantic_hits = await semantic.search(query)
    by_cached_url: dict[str, list[dict]] = defaultdict(list)
    for hit in semantic_hits:
        by_cached_url[core._normalize_url(hit["url"])].append(hit)
    existing_urls = {core._normalize_url(entry["url"]) for entry in entries if entry.get("url")}
    for normalized, hits in by_cached_url.items():
        if normalized in existing_urls:
            continue
        first = hits[0]
        url = first["url"]
        if not _matches_source_types(url, source_types):
            continue
        text = _CHUNK_GAP.join(hit["text"] for hit in hits[:max_passages])
        entries.append({
            "title": first.get("title", "Untitled"),
            "url": url,
            "content": text,
            "scraped": True,
            "metadata": first.get("metadata") or {},
        })
        existing_urls.add(normalized)
    for cached in cached_entries:
        if not cached or not cached.get("content"):
            continue
        normalized = core._normalize_url(cached.get("url", ""))
        if normalized in existing_urls or not _matches_source_types(cached.get("url", ""), source_types):
            continue
        entries.append({
            "title": cached.get("title", "Untitled"),
            "url": cached.get("url", ""),
            "content": cached.get("content", ""),
            "scraped": True,
            "metadata": cached.get("metadata") or {},
        })
        existing_urls.add(normalized)

    # --- chunk scraped pages, keep snippets as single chunks ---
    all_chunks: list[str] = []
    chunk_to_entry: list[int] = []
    for i, entry in enumerate(entries):
        if entry["scraped"] and entry["content"]:
            chunks = core._chunk_text(entry["content"])
        else:
            chunks = [entry["content"]] if entry["content"] else []
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_to_entry.append(i)

    # --- deduplicate near-identical chunks across pages ---
    all_chunks, chunk_to_entry = core._dedup_chunks(all_chunks, chunk_to_entry)

    # --- rerank at the chunk level ---
    rerank_started = time.monotonic()
    rerank_failed = False
    try:
        scored = await core._rerank_scored(query, all_chunks)
    except Exception as exc:
        log.warning("rerank failed query=%r err=%s", query, exc)
        warnings.append(core._warning("rerank_failed", "flashrank", str(exc)))
        degraded = True
        rerank_failed = True
        scored = []
    timings_ms["rerank"] = int((time.monotonic() - rerank_started) * 1000)

    # --- group scores by entry, select top-K chunks per page ---
    entry_chunks: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for chunk_idx, score in scored:
        eidx = chunk_to_entry[chunk_idx]
        entry_chunks[eidx].append((all_chunks[chunk_idx], score))

    for eidx in entry_chunks:
        entry_chunks[eidx].sort(key=lambda x: x[1], reverse=True)
        entry_chunks[eidx] = _limit_passages(entry_chunks[eidx], max_passages, max_chars_per_result)

    # --- rank pages by best chunk score, drop noise below threshold ---
    entry_best: dict[int, float | None] = {
        eidx: chunks[0][1] for eidx, chunks in entry_chunks.items() if chunks
    }
    if rerank_failed:
        ranked_entry_idxs = list(range(len(entries)))
    else:
        ranked_entry_idxs = sorted(
            entry_best,
            key=lambda eidx: _entry_sort_score(eidx, entries, entry_best),
            reverse=True,
        )
        for i in range(len(entries)):
            if i not in entry_best:
                ranked_entry_idxs.append(i)
        # Filter out entries whose best chunk scored below the noise
        # threshold — CAPTCHA walls, wrong-language pages, auto-generated
        # spam.  Only applied when reranking succeeded (scores are meaningful).
        noise_count = 0
        filtered_idxs = []
        for eidx in ranked_entry_idxs:
            score = entry_best.get(eidx)
            if score is not None and score < _MIN_RELEVANCE_SCORE:
                noise_count += 1
                continue
            filtered_idxs.append(eidx)
        ranked_entry_idxs = filtered_idxs
        if noise_count:
            warnings.append(core._warning(
                "low_relevance_filtered", "flashrank",
                f"{noise_count} result(s) dropped below relevance threshold",
            ))
    ranked_entry_idxs = _diversify_by_source_type(
        core._diversify_ranked_entries(ranked_entry_idxs, entries), entries
    )[:num_results]

    # --- format structured output ---
    structured_results: list[dict] = []
    new_urls: list[str] = []
    # Pre-fetch "previously_seen" flags in one batch to avoid N sequential
    # awaits inside the per-result loop.
    ranked_normalized = [
        core._normalize_url(entries[eidx]["url"]) for eidx in ranked_entry_idxs
    ]
    seen_flags = await asyncio.gather(
        *(seen_urls.contains(u) for u in ranked_normalized)
    )
    for rank, (eidx, normalized_url, previously_seen) in enumerate(
        zip(ranked_entry_idxs, ranked_normalized, seen_flags), 1
    ):
        entry = entries[eidx]
        url = entry["url"]
        top = entry_chunks.get(eidx, [])
        if top:
            content = _CHUNK_GAP.join(chunk for chunk, _ in top)
        else:
            content = entry["content"]

        source_type = _source_type(url)
        passages = [{"text": chunk, "score": score} for chunk, score in top]
        structured = {
            "rank": rank,
            "title": entry["title"],
            "url": url,
            "domain": core._domain_from_url(url),
            "source_type": source_type,
            "snippet": results[eidx].get("content", "") if eidx < len(results) else "",
            "content": content,
            "passages": passages,
            "highlights": passages,
            "top_chunks": [chunk for chunk, _ in top],
            "scraped": entry["scraped"],
            "previously_seen": previously_seen,
        }
        if include_raw_content:
            structured["raw_content"] = entry["content"][:max_chars_per_result]
        metadata = entry.get("metadata") or {}
        if metadata:
            structured["metadata"] = metadata
        structured_results.append(structured)
        new_urls.append(normalized_url)

    brief = _brief_from_results(structured_results)
    next_actions = _next_actions(mode, degraded, structured_results, warnings)

    response = {
        "query": query,
        "time_range": time_range,
        "include_domains": include_domains,
        "exclude_domains": exclude_domains,
        "results": structured_results,
        "meta": {
            "mode": mode,
            "brief": brief,
            "next_actions": next_actions,
            "num_results_requested": num_results,
            "num_results_returned": len(structured_results),
            "scrape_top": to_scrape,
            "max_passages": max_passages,
            "max_chars_per_result": max_chars_per_result,
            "search_queries": search_queries,
            "source_types": source_types,
            "include_raw_content": include_raw_content,
            "search_backend": "searxng",
            "reranker": {"name": "flashrank", "model": RERANK_MODEL},
            "degraded": degraded,
            "warnings": warnings,
            "timings_ms": {
                **timings_ms,
                "total": int((time.monotonic() - started) * 1000),
            },
        },
    }

    # --- persist to shared cache ---
    if new_urls:
        semantic_writes = []
        for result in structured_results:
            if result.get("scraped") and result.get("content"):
                content_for_cache = result.get("raw_content") or result.get("content")
                semantic_writes.append(semantic_cache.set(core._normalize_url(result["url"]), {
                    "url": result["url"],
                    "title": result["title"],
                    "domain": result["domain"],
                    "source_type": result.get("source_type"),
                    "content": content_for_cache,
                    "metadata": result.get("metadata") or {},
                    "updated_at": int(time.time()),
                }))
                semantic_writes.append(semantic.index_page(
                    result["url"], result["title"], content_for_cache, result.get("metadata") or {}
                ))
        await asyncio.gather(
            *(seen_urls.set(url, 1) for url in new_urls),
            *semantic_writes,
        )

    log.info("query=%r chunks=%d pages=%d", query, len(all_chunks), len(entries))

    return _validated_response(models.SearchResponseModel, response)


async def extract_impl(
    urls: list[str],
    query: str | None = None,
    chunk_ids: list[int] | None = None,
    ctx: Context | None = None,
) -> dict:
    """Extract a batch of URLs with per-URL status reporting.

    Uses Crawl4AI for web pages and local fetch for text-like resources.
    Binary document formats are classified here and handed off to the
    `files` MCP via structured metadata rather than parsed locally.

    `chunk_ids` cherry-picks specific chunks from the full cached
    document by stable id (see the `chunks` field on the response).
    """
    urls = core._validate_urls(urls, maximum=_MAX_EXTRACT_URLS)
    if chunk_ids is not None and any(i < 0 for i in chunk_ids):
        raise ValueError("chunk_ids entries must be >= 0")
    normalized_query = core._coerce_optional_str(query)
    started = time.monotonic()

    page_cache = cache_module.page_cache

    documents = await asyncio.gather(*[
        core._extract_url_document(
            url, normalized_query, page_cache,
            chunk_ids=chunk_ids,
        )
        for url in urls
    ])

    results: list[dict] = []
    urls_succeeded = 0
    urls_failed = 0
    for document in documents:
        if document["status"] in {"ok", "handoff"}:
            urls_succeeded += 1
        else:
            urls_failed += 1
        url = document["url"]
        content = document.get("content", "")
        total_chars = document.get("total_chars", len(content))
        entry = {
            "url": url,
            "domain": core._domain_from_url(url),
            "status": document["status"],
            "content_type": document.get("content_type"),
            "file_type": document.get("file_type"),
            "title": document.get("title"),
            "content": content,
            "chars_shown": len(content),
            "total_chars": total_chars,
            "total_chunks": document.get("total_chunks"),
            "shown_chunk_ids": document.get("shown_chunk_ids", []),
            "chunk_mode": document.get("chunk_mode"),
            "top_chunks": [
                c["text"] if isinstance(c, dict) else c
                for c in document.get("top_chunks", [])
            ],
            "chunks": document.get("chunks", []),
            "cached": document.get("cached", False),
            "error": document.get("error"),
            "handoff": document.get("handoff"),
        }
        metadata = document.get("metadata") or {}
        if metadata:
            entry["metadata"] = metadata
        results.append(entry)

    response = {
        "query": normalized_query,
        "results": results,
        "meta": {
            "urls_requested": len(urls),
            "urls_succeeded": urls_succeeded,
            "urls_failed": urls_failed,
            "timings_ms": {
                "total": int((time.monotonic() - started) * 1000),
            },
        },
    }
    log.info("extract requested=%d succeeded=%d failed=%d", len(urls), urls_succeeded, urls_failed)
    return _validated_response(models.ExtractResponseModel, response)


async def map_impl(
    url: str,
    max_urls: int = 25,
    include_patterns: list[str] | None = None,
) -> dict:
    """Discover an in-scope site tree rooted at one URL.

    Discovery is link-only: Crawl4AI walks the site graph without this
    tool returning page bodies. The result is a bounded tree the caller
    can use as a planning surface before spending crawl budget on
    selected nodes.
    """
    root_url = core._validate_urls([url], maximum=1)[0]
    max_urls = core._validate_positive_int("max_urls", max_urls, maximum=_MAX_MAP_URLS)
    include_patterns = core._normalize_glob_patterns(include_patterns, field_name="include_patterns")

    started = time.monotonic()
    warnings: list[dict] = []
    pages_visited = 0
    try:
        discovered_pages = await core._deep_crawl(
            [root_url],
            max_depth=max_urls,
            max_pages=max_urls,
            same_domain_only=True,
            include_patterns=include_patterns,
        )
        pages_visited = len({
            core._normalize_url(page.get("url", ""))
            for page in discovered_pages
            if isinstance(page, dict) and page.get("url")
        }) or 1
    except Exception as exc:
        warnings.append(core._warning("link_discovery_failed", "crawl4ai", str(exc)))
        discovered_pages = []

    results: list[dict] = []
    visited: set[str] = set()

    root_normalized = core._normalize_url(root_url)
    visited.add(root_normalized)
    root_entry = {
        "url": root_url,
        "domain": core._domain_from_url(root_url),
        "title": None,
        "link_text": None,
        "depth": 0,
        "discovered_from": None,
        "link_type": "seed",
    }
    results.append(root_entry)

    for page in discovered_pages:
        if len(results) >= max_urls:
            break
        if not isinstance(page, dict):
            continue
        page_url = page.get("url")
        if not isinstance(page_url, str) or not page_url:
            continue
        normalized_url = core._normalize_url(page_url)
        metadata = page.get("metadata") if isinstance(page.get("metadata"), dict) else {}
        if normalized_url == root_normalized:
            root_entry["title"] = core._extract_crawl_title(page)
            continue
        if normalized_url in visited:
            continue
        visited.add(normalized_url)
        depth = metadata.get("depth")
        if not isinstance(depth, int) or depth < 1:
            depth = 1
        parent_url = metadata.get("parent_url")
        if not isinstance(parent_url, str) or not parent_url:
            parent_url = root_url
        entry = {
            "url": page_url,
            "domain": core._domain_from_url(page_url),
            "title": core._extract_crawl_title(page),
            "link_text": None,
            "depth": depth,
            "discovered_from": parent_url,
            "link_type": "internal",
        }
        results.append(entry)

    for rank, entry in enumerate(results, start=1):
        entry["rank"] = rank

    response = {
        "url": root_url,
        "results": results,
        "meta": {
            "max_urls_requested": max_urls,
            "urls_returned": len(results),
            "pages_visited": pages_visited,
            "warnings": warnings,
            "timings_ms": {
                "total": int((time.monotonic() - started) * 1000),
            },
        },
    }
    log.info(
        "map url=%s returned=%d warnings=%d",
        root_url, len(results), len(warnings),
    )
    return _validated_response(models.MapResponseModel, response)


async def crawl_impl(
    url: str,
    max_urls: int = 10,
    include_patterns: list[str] | None = None,
    query: str | None = None,
) -> dict:
    """Discover a site tree, then extract content for each discovered node.

    When `query` is set, results are reordered by per-page best-chunk
    relevance score (FlashRank cross-encoder) instead of BFS discovery
    order, and each result's `content` carries the joined top chunks
    rather than the document head.
    """
    effective_max_urls = core._validate_positive_int(
        "max_urls",
        max_urls,
        maximum=min(_MAX_MAP_URLS, _MAX_EXTRACT_URLS),
    )
    normalized_query = core._coerce_optional_str(query)
    started = time.monotonic()
    tree = await map_impl(
        url=url,
        max_urls=effective_max_urls,
        include_patterns=include_patterns,
    )
    root_url = tree["url"]
    urls = [entry["url"] for entry in tree["results"]]

    # Query-driven path bypasses extract_impl to retain per-chunk scores
    # for cross-page ranking; non-query path keeps using extract_impl so
    # existing call-sites and test mocks remain unchanged.
    score_by_url: dict[str, float | None] = {}
    if normalized_query:
        page_cache = cache_module.page_cache
        documents = await asyncio.gather(*[
            core._extract_url_document(u, normalized_query, page_cache, chunk_ids=None)
            for u in urls
        ])
        urls_succeeded = sum(1 for d in documents if d["status"] in {"ok", "handoff"})
        urls_failed = len(documents) - urls_succeeded
        doc_by_url: dict[str, dict] = dict(zip(urls, documents))
        for u, doc in doc_by_url.items():
            top = doc.get("top_chunks") or []
            score_by_url[u] = (
                top[0].get("score") if top and isinstance(top[0], dict) else None
            )
    else:
        extracted = await extract_impl(
            urls=urls,
            query=None,
            chunk_ids=None,
        )
        urls_succeeded = extracted["meta"]["urls_succeeded"]
        urls_failed = extracted["meta"]["urls_failed"]
        doc_by_url = {entry["url"]: entry for entry in extracted["results"]}

    results: list[dict] = []
    for node in tree["results"]:
        document = doc_by_url.get(node["url"], {})
        top_chunks_raw = document.get("top_chunks", []) or []
        top_chunks = [
            c["text"] if isinstance(c, dict) else c
            for c in top_chunks_raw
        ]
        content = document.get("content", "")
        merged = {
            "url": node["url"],
            "domain": node["domain"],
            "title": document.get("title") or node.get("title"),
            "link_text": node.get("link_text"),
            "depth": node["depth"],
            "discovered_from": node.get("discovered_from"),
            "link_type": node["link_type"],
            "status": document.get("status", "error"),
            "content_type": document.get("content_type"),
            "content": content,
            "chars_shown": document.get("chars_shown", len(content)),
            "total_chars": document.get("total_chars", 0),
            "top_chunks": top_chunks,
            "cached": document.get("cached", False),
            "error": document.get("error"),
        }
        metadata = document.get("metadata") or {}
        if metadata:
            merged["metadata"] = metadata
        results.append(merged)

    if normalized_query:
        # Pages with a real score sort by score desc; pages without one
        # (extract failures, no chunks) trail in stable order.
        results.sort(key=lambda r: (
            score_by_url.get(r["url"]) is None,
            -(score_by_url.get(r["url"]) or 0.0),
        ))

    for rank, entry in enumerate(results, start=1):
        entry["rank"] = rank

    warnings = list(tree["meta"].get("warnings", []))
    response = {
        "url": root_url,
        "query": normalized_query,
        "results": results,
        "meta": {
            "max_urls_requested": effective_max_urls,
            "urls_discovered": len(tree["results"]),
            "urls_returned": len(results),
            "urls_truncated_by_limit": 0,
            "urls_deduplicated": 0,
            "urls_succeeded": urls_succeeded,
            "urls_failed": urls_failed,
            "warnings": warnings,
            "timings_ms": {
                "total": int((time.monotonic() - started) * 1000),
            },
        },
    }
    log.info(
        "crawl url=%s query=%r discovered=%d returned=%d dedup=%d succeeded=%d failed=%d",
        root_url,
        normalized_query,
        response["meta"]["urls_discovered"],
        len(results),
        response["meta"]["urls_deduplicated"],
        urls_succeeded,
        urls_failed,
    )
    return _validated_response(models.CrawlResponseModel, response)
