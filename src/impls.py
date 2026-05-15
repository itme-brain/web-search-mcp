"""Public Python-API impls: search, research, extract, map, crawl.

Each returns a structured dict. `@mcp.tool` wrappers in server.py call
these, then push the result through a formatter in formatters.py to
produce the LLM-facing markdown string.
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict

# Module-qualified import so `unittest.mock.patch("core.X")` intercepts
# every call site — both from core itself and from here. (`from core
# import X` would bind X into impls's namespace, creating a second patch
# target we'd have to mock separately.) Constants are safe to import
# by-name since they're not patched.
import cache as cache_module
import core
import models
import evidence
import query_expansion
import search_config
import semantic
import observability
import source_quality
from core import (
    MAX_RESULTS,
    RERANK_NAME,
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


async def _searxng_cached(
    search_query: str,
    *,
    num_results: int,
    time_range: str | None,
    language: str | None,
    pageno: int,
) -> dict:
    """Cached, single-flighted SearXNG call."""
    key = hashlib.sha256(
        json.dumps(
            [search_query.lower().strip(), num_results, time_range, language, pageno],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    cached = await cache_module.searxng_cache.get(key)
    if cached is not None:
        return cached
    inflight = _searxng_inflight.get(key)
    if inflight is not None:
        return await inflight
    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    _searxng_inflight[key] = fut
    try:
        result = await core._search(
            search_query,
            num_results=num_results,
            time_range=time_range,
            language=language,
            pageno=pageno,
        )
        await cache_module.searxng_cache.set(key, result)
        fut.set_result(result)
        return result
    except Exception as exc:
        fut.set_exception(exc)
        raise
    finally:
        _searxng_inflight.pop(key, None)


async def _collect_search_candidates(
    *,
    search_queries: list[str],
    profile: str,
    num_results: int,
    searx_pages: int,
    time_range: str | None,
    language: str | None,
    include_domains: list[str] | None,
    exclude_domains: list[str] | None,
    source_types: list[str] | None,
) -> tuple[list[dict], list, list[dict], bool]:
    """Search SearXNG pages and return filtered candidates plus warnings."""
    warnings: list[dict] = []
    degraded = False
    unresponsive_engines: list = []
    raw_results: list[dict] = []
    for search_query in search_queries:
        for pageno in range(1, searx_pages + 1):
            try:
                page = await _searxng_cached(
                    search_query,
                    num_results=num_results,
                    time_range=time_range,
                    language=language,
                    pageno=pageno,
                )
            except Exception as exc:
                if not raw_results and pageno == 1:
                    degraded = True
                    warnings.append(core._warning("search_failed", "searxng", str(exc)))
                else:
                    warnings.append(core._warning("search_failed", "searxng", f"{search_query} page {pageno}: {exc}"))
                break
            raw_results.extend(page["results"])
            unresponsive_engines.extend(page.get("unresponsive_engines", []))
            filtered_so_far = core._dedup_results(
                core._filter_results_by_domain(raw_results, include_domains, exclude_domains)
            )
            if profile == "search" and len(filtered_so_far) >= num_results:
                break
        if profile != "research":
            break

    results = core._dedup_results(core._filter_results_by_domain(raw_results, include_domains, exclude_domains))
    results = [r for r in results if source_quality.matches_source_types(r.get("url", ""), source_types)]
    return results, unresponsive_engines, warnings, degraded


def _validated_response(model_cls, response: dict) -> dict:
    """Return a model-validated payload without inventing unset keys."""
    return models.dump_response(model_cls, response)


def _empty_search_response(
    *, request_id: str, query: str, profile: str, time_range: str | None,
    include_domains: list[str] | None, exclude_domains: list[str] | None,
    num_results: int, scrape_budget: int, max_passages: int,
    max_chars_per_result: int, search_queries: list[str],
    source_types: list[str] | None, degraded: bool, warnings: list[dict],
    timings_ms: dict, started: float,
) -> dict:
    return {
        "query": query,
        "time_range": time_range,
        "include_domains": include_domains,
        "exclude_domains": exclude_domains,
        "results": [],
        "meta": {
            "request_id": request_id,
            "profile": profile,
            "brief": [],
            "findings": [],
            "answer": [],
            "key_evidence": [],
            "gaps": ["No supporting sources found."] if profile == "research" else [],
            "next_actions": ["No results found; broaden the query or remove filters."],
            "num_results_requested": num_results,
            "num_results_returned": 0,
            "scrape_top": scrape_budget,
            "max_passages": max_passages,
            "max_chars_per_result": max_chars_per_result,
            "search_queries": search_queries,
            "source_types": source_types,
            "search_backend": "searxng",
            "reranker": {"name": RERANK_NAME, "model": RERANK_MODEL},
            "semantic_hits": 0,
            "degraded": degraded,
            "warnings": warnings or [core._warning("no_results", "searxng", query)],
            "timings_ms": {**timings_ms, "total": int((time.monotonic() - started) * 1000)},
        },
    }


async def _scrape_search_entries(results: list[dict], scrape_budget: int) -> tuple[list[dict], int, list[dict]]:
    """Scrape top candidates and build page/snippet entries."""
    to_scrape = min(scrape_budget, len(results))
    scraped = await asyncio.gather(*[
        core._scrape_cached(r["url"], cache_module.page_cache) for r in results[:to_scrape]
    ])
    warnings: list[dict] = []
    scrape_failures = sum(1 for s in scraped if s.get("content") is None)
    if scrape_failures:
        warnings.append(core._warning("scrape_failed", "crawl4ai", f"{scrape_failures} of {to_scrape} pages failed"))
    for scrape in scraped:
        diagnostic = (scrape.get("metadata") or {}).get("diagnostic")
        if diagnostic:
            warnings.append(core._warning("content_rejected", "crawl4ai", diagnostic))

    entries: list[dict] = []
    for i, result in enumerate(results[:to_scrape]):
        scrape = scraped[i]
        content = scrape.get("content")
        metadata = {k: v for k, v in (scrape.get("metadata") or {}).items() if k != "diagnostic"}
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
    return entries, to_scrape, warnings


async def _append_vector_memory_entries(
    *, entries: list[dict], query: str, max_passages: int,
    source_types: list[str] | None, existing_urls: set[str],
) -> int:
    """Append chunk-level semantic/vector hits from semantic.py."""
    semantic_hits = await semantic.search(query)
    by_cached_url: dict[str, list[dict]] = defaultdict(list)
    for hit in semantic_hits:
        by_cached_url[core._normalize_url(hit["url"])].append(hit)
    added = 0
    for normalized, hits in by_cached_url.items():
        if normalized in existing_urls:
            continue
        first = hits[0]
        url = first["url"]
        if not source_quality.matches_source_types(url, source_types):
            continue
        entries.append({
            "title": first.get("title", "Untitled"),
            "url": url,
            "content": _CHUNK_GAP.join(hit["text"] for hit in hits[:max_passages]),
            "scraped": True,
            "metadata": first.get("metadata") or {},
            "retrieval_source": "semantic_memory",
        })
        existing_urls.add(normalized)
        added += 1
    return added


async def _append_page_memory_entries(
    *, entries: list[dict], results: list[dict], source_types: list[str] | None,
    existing_urls: set[str],
) -> int:
    """Append page-level retrieval-memory hits from cache.py."""
    keys = [core._normalize_url(r.get("url", "")) for r in results]
    cached_entries = await asyncio.gather(*(cache_module.page_memory_cache.get(key) for key in keys))
    added = 0
    for cached in cached_entries:
        if not cached or not cached.get("content"):
            continue
        normalized = core._normalize_url(cached.get("url", ""))
        if normalized in existing_urls or not source_quality.matches_source_types(cached.get("url", ""), source_types):
            continue
        entries.append({
            "title": cached.get("title", "Untitled"),
            "url": cached.get("url", ""),
            "content": cached.get("content", ""),
            "scraped": True,
            "metadata": cached.get("metadata") or {},
            "retrieval_source": "page_memory",
        })
        existing_urls.add(normalized)
        added += 1
    return added


async def _merge_memory_entries(
    *, entries: list[dict], results: list[dict], query: str,
    max_passages: int, source_types: list[str] | None,
) -> int:
    """Append memory evidence not already present in entries."""
    existing_urls = {core._normalize_url(entry["url"]) for entry in entries if entry.get("url")}
    added = await _append_vector_memory_entries(
        entries=entries, query=query, max_passages=max_passages,
        source_types=source_types, existing_urls=existing_urls,
    )
    added += await _append_page_memory_entries(
        entries=entries, results=results, source_types=source_types,
        existing_urls=existing_urls,
    )
    return added


async def _rank_search_entries(
    *, query: str, entries: list[dict], max_passages: int,
    max_chars_per_result: int, num_results: int, warnings: list[dict],
) -> tuple[list[int], dict[int, list[tuple[str, float]]], int, bool, bool]:
    """Chunk, rerank, filter, and diversify entries."""
    all_chunks: list[str] = []
    chunk_to_entry: list[int] = []
    for i, entry in enumerate(entries):
        chunks = core._chunk_text(entry["content"]) if entry["scraped"] and entry["content"] else ([entry["content"]] if entry["content"] else [])
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_to_entry.append(i)
    all_chunks, chunk_to_entry = core._dedup_chunks(all_chunks, chunk_to_entry)

    rerank_failed = False
    degraded = False
    try:
        scored = await core._rerank_scored(query, all_chunks)
    except Exception as exc:
        log.warning("rerank failed query=%r err=%s", query, exc)
        warnings.append(core._warning("rerank_failed", RERANK_NAME, str(exc)))
        degraded = True
        rerank_failed = True
        scored = []

    entry_chunks: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for chunk_idx, score in scored:
        eidx = chunk_to_entry[chunk_idx]
        entry_chunks[eidx].append((all_chunks[chunk_idx], score))
    for eidx in entry_chunks:
        entry_chunks[eidx].sort(key=lambda x: x[1], reverse=True)
        entry_chunks[eidx] = evidence.limit_passages(entry_chunks[eidx], max_passages, max_chars_per_result)

    entry_best = {eidx: chunks[0][1] for eidx, chunks in entry_chunks.items() if chunks}
    if rerank_failed:
        ranked_entry_idxs = list(range(len(entries)))
    else:
        ranked_entry_idxs = sorted(entry_best, key=lambda eidx: source_quality.entry_sort_score(eidx, entries, entry_best), reverse=True)
        ranked_entry_idxs.extend(i for i in range(len(entries)) if i not in entry_best)
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
            warnings.append(core._warning("low_relevance_filtered", RERANK_NAME, f"{noise_count} result(s) dropped below relevance threshold"))
    ranked_entry_idxs = source_quality.diversify_by_source_type(
        core._diversify_ranked_entries(ranked_entry_idxs, entries), entries
    )[:num_results]
    return ranked_entry_idxs, entry_chunks, len(all_chunks), degraded, rerank_failed


async def _build_structured_search_results(
    *, ranked_entry_idxs: list[int], entries: list[dict], entry_chunks: dict[int, list[tuple[str, float]]],
    results: list[dict], profile: str = "search",
) -> tuple[list[dict], list[str]]:
    ranked_normalized = [core._normalize_url(entries[eidx]["url"]) for eidx in ranked_entry_idxs]
    seen_flags = await asyncio.gather(*(cache_module.seen_urls.contains(u) for u in ranked_normalized))
    structured_results: list[dict] = []
    for rank, (eidx, normalized_url, seen_recently) in enumerate(zip(ranked_entry_idxs, ranked_normalized, seen_flags), 1):
        top = entry_chunks.get(eidx, [])
        if profile == "research" and not top:
            continue
        if profile == "research" and entries[eidx].get("scraped") is False and rank > 3:
            continue
        entry = entries[eidx]
        url = entry["url"]
        content = _CHUNK_GAP.join(chunk for chunk, _ in top) if top else entry["content"]
        structured = {
            "rank": rank,
            "title": entry["title"],
            "url": url,
            "domain": core._domain_from_url(url),
            "source_type": source_quality.source_type(url),
            "snippet": results[eidx].get("content", "") if eidx < len(results) else "",
            "content": content,
            "passages": [
                {"citation": f"{rank}.{idx}", "text": chunk, "score": score}
                for idx, (chunk, score) in enumerate(top, 1)
            ],
            "scraped": entry["scraped"],
            "seen_recently": seen_recently,
            "retrieval_source": entry.get("retrieval_source", "live_search"),
        }
        if top:
            structured["best_score"] = top[0][1]
        metadata = entry.get("metadata") or {}
        latest_date = metadata.get("date") if isinstance(metadata, dict) else None
        if latest_date:
            structured["latest_date"] = latest_date
        if metadata:
            structured["metadata"] = metadata
        structured_results.append(structured)
    return structured_results, ranked_normalized


async def _persist_search_memory(structured_results: list[dict], normalized_urls: list[str]) -> None:
    if not normalized_urls:
        return
    writes = [cache_module.seen_urls.set(url, 1) for url in normalized_urls]
    for result in structured_results:
        if result.get("scraped") and result.get("content"):
            content = result.get("content")
            writes.append(cache_module.page_memory_cache.set(core._normalize_url(result["url"]), {
                "url": result["url"], "title": result["title"], "domain": result["domain"],
                "source_type": result.get("source_type"), "content": content,
                "metadata": result.get("metadata") or {}, "updated_at": int(time.time()),
            }))
            writes.append(semantic.index_page(result["url"], result["title"], content, result.get("metadata") or {}))
    await asyncio.gather(*writes)


async def search_impl(
    query: str,
    num_results: int = 5,
    profile: str = "search",
    max_passages: int | None = None,
    max_chars_per_result: int | None = None,
    source_types: list[str] | None = None,
    time_range: str | None = None,
    language: str | None = "en",
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> dict:
    """Search the web, scrape top results, and return structured JSON ranked by relevance.

    Pipeline: SearXNG search -> Crawl4AI scrape -> chunk -> local reranker -> formatted output.
    Scraped pages are split into paragraphs and reranked at the chunk level, so only
    the most query-relevant excerpts from each page are returned.

    Fetches page 2 from SearXNG only if page 1 after dedup/filter is short of
    `num_results`. Always scrapes `min(num_results, MAX_SCRAPE)` top candidates.
    Results are backed by shared Valkey caches across requests.
    """
    query = core._validate_query(query)
    request_id = uuid.uuid4().hex
    num_results = core._validate_positive_int("num_results", num_results, maximum=MAX_RESULTS)
    profile = search_config.normalize_profile(profile)
    time_range = core._normalize_time_range(time_range)
    language = core._coerce_optional_str(language)
    include_domains = core._normalize_domains(include_domains, field_name="include_domains")
    exclude_domains = core._normalize_domains(exclude_domains, field_name="exclude_domains")
    source_types = source_quality.normalize_source_types(source_types)
    searx_pages, scrape_budget, default_passages = search_config.profile_budget(profile, num_results)
    max_passages = search_config.validate_optional_positive_int("max_passages", max_passages, maximum=8) or default_passages
    default_chars = search_config.default_chars_per_result(profile)
    max_chars_per_result = search_config.validate_optional_positive_int(
        "max_chars_per_result", max_chars_per_result, maximum=8000
    ) or default_chars
    search_queries = query_expansion.search_queries(query, profile)
    started = time.monotonic()
    warnings: list[dict] = []
    degraded = False
    timings_ms = {"search": 0, "scrape": 0, "semantic": 0, "rerank": 0, "total": 0}
    semantic_hits = 0

    # --- search (profile-aware multi-page retrieval) ---
    search_started = time.monotonic()
    results, unresponsive_engines, search_warnings, search_degraded = await _collect_search_candidates(
        search_queries=search_queries,
        profile=profile,
        num_results=num_results,
        searx_pages=searx_pages,
        time_range=time_range,
        language=language,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        source_types=source_types,
    )
    warnings.extend(search_warnings)
    degraded = degraded or search_degraded
    results = results[:max(num_results, scrape_budget)]

    # Surface per-engine failures from SearXNG (e.g. "google: CAPTCHA").
    # Does NOT flip `degraded` — the multi-engine hedge means a single
    # upstream going down is expected, not a pipeline failure.
    for engine, reason in core._dedup_unresponsive_engines(unresponsive_engines):
        detail = f"{engine}: {reason}" if reason else engine
        warnings.append(core._warning("engine_unresponsive", "searxng", detail))

    timings_ms["search"] = int((time.monotonic() - search_started) * 1000)
    if not results:
        response = _empty_search_response(
            request_id=request_id, query=query, profile=profile, time_range=time_range,
            include_domains=include_domains, exclude_domains=exclude_domains,
            num_results=num_results, scrape_budget=scrape_budget,
            max_passages=max_passages, max_chars_per_result=max_chars_per_result,
            search_queries=search_queries, source_types=source_types,
            degraded=degraded, warnings=warnings, timings_ms=timings_ms,
            started=started,
        )
        observability.observe_tool_response(profile, response)
        log.info("request_id=%s query=%r profile=%s results=0 degraded=%s", request_id, query, profile, degraded)
        return _validated_response(models.SearchResponseModel, response)

    scrape_started = time.monotonic()
    entries, to_scrape, scrape_warnings = await _scrape_search_entries(results, scrape_budget)
    timings_ms["scrape"] = int((time.monotonic() - scrape_started) * 1000)
    warnings.extend(scrape_warnings)
    degraded = degraded or any(w.get("type") == "scrape_failed" for w in scrape_warnings)

    semantic_started = time.monotonic()
    semantic_hits = await _merge_memory_entries(
        entries=entries, results=results, query=query,
        max_passages=max_passages, source_types=source_types,
    )
    timings_ms["semantic"] = int((time.monotonic() - semantic_started) * 1000)

    rerank_started = time.monotonic()
    ranked_entry_idxs, entry_chunks, chunk_count, rank_degraded, _ = await _rank_search_entries(
        query=query, entries=entries, max_passages=max_passages,
        max_chars_per_result=max_chars_per_result, num_results=num_results,
        warnings=warnings,
    )
    timings_ms["rerank"] = int((time.monotonic() - rerank_started) * 1000)
    degraded = degraded or rank_degraded

    structured_results, new_urls = await _build_structured_search_results(
        ranked_entry_idxs=ranked_entry_idxs,
        entries=entries,
        entry_chunks=entry_chunks,
        results=results,
        profile=profile,
    )

    brief = evidence.brief_from_results(structured_results)
    findings = evidence.research_findings(structured_results) if profile == "research" else []
    answer = findings
    summary = evidence.research_summary(structured_results, warnings) if profile == "research" else []
    key_evidence = evidence.research_key_evidence(structured_results) if profile == "research" else []
    gaps = evidence.research_gaps(structured_results, warnings) if profile == "research" else []
    next_actions = evidence.next_actions(profile, degraded, structured_results, warnings)

    response = {
        "query": query,
        "time_range": time_range,
        "include_domains": include_domains,
        "exclude_domains": exclude_domains,
        "results": structured_results,
        "meta": {
            "request_id": request_id,
            "profile": profile,
            "brief": brief,
            "findings": findings,
            "answer": answer,
            "summary": summary,
            "key_evidence": key_evidence,
            "gaps": gaps,
            "next_actions": next_actions,
            "num_results_requested": num_results,
            "num_results_returned": len(structured_results),
            "scrape_top": to_scrape,
            "max_passages": max_passages,
            "max_chars_per_result": max_chars_per_result,
            "search_queries": search_queries,
            "source_types": source_types,
            "search_backend": "searxng",
            "reranker": {"name": RERANK_NAME, "model": RERANK_MODEL},
            "semantic_hits": semantic_hits,
            "degraded": degraded,
            "warnings": warnings,
            "timings_ms": {
                **timings_ms,
                "total": int((time.monotonic() - started) * 1000),
            },
        },
    }

    # --- persist to shared cache ---
    await _persist_search_memory(structured_results, new_urls)

    observability.observe_tool_response(profile, response)
    log.info(
        "request_id=%s query=%r profile=%s chunks=%d pages=%d semantic_hits=%d degraded=%s",
        request_id, query, profile, chunk_count, len(entries), semantic_hits, degraded,
    )

    return _validated_response(models.SearchResponseModel, response)


async def research_impl(
    query: str,
    num_results: int = 8,
    time_range: str | None = None,
    source_types: list[str] | None = None,
) -> dict:
    """Broader/slower search profile for hard questions."""
    return await search_impl(
        query=query,
        num_results=num_results,
        profile="research",
        time_range=time_range,
        source_types=source_types,
    )


async def extract_impl(
    urls: list[str],
    chunk_ids: list[int] | None = None,
    observe: bool = True,
) -> dict:
    """Extract full cleaned documents with per-URL status reporting.

    Uses Crawl4AI for web pages and local fetch for text-like resources.
    Binary document formats are classified here and handed off via
    structured metadata rather than parsed locally.

    `chunk_ids` is an internal escape hatch for tests/debugging; public
    MCP extract always reads the document body.
    """
    urls = core._validate_urls(urls, maximum=_MAX_EXTRACT_URLS)
    request_id = uuid.uuid4().hex
    if chunk_ids is not None and any(i < 0 for i in chunk_ids):
        raise ValueError("chunk_ids entries must be >= 0")
    started = time.monotonic()

    page_cache = cache_module.page_cache

    documents = await asyncio.gather(*[
        core._extract_url_document(
            url, page_cache,
            chunk_ids=chunk_ids,
        )
        for url in urls
    ])

    results: list[dict] = []
    urls_succeeded = 0
    urls_failed = 0
    for document in documents:
        if document["status"] == "ok":
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
            "truncated": document.get("truncated", len(content) < total_chars),
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
        }
        metadata = document.get("metadata") or {}
        if metadata:
            entry["metadata"] = metadata
        results.append(entry)

    response = {
        "query": None,
        "results": results,
        "meta": {
            "request_id": request_id,
            "urls_requested": len(urls),
            "urls_succeeded": urls_succeeded,
            "urls_failed": urls_failed,
            "timings_ms": {
                "total": int((time.monotonic() - started) * 1000),
            },
        },
    }
    if observe:
        observability.observe_tool_response("extract", response)
    log.info(
        "request_id=%s extract requested=%d succeeded=%d failed=%d",
        request_id, len(urls), urls_succeeded, urls_failed,
    )
    return _validated_response(models.ExtractResponseModel, response)


async def map_impl(
    url: str,
    max_urls: int = 25,
    include_patterns: list[str] | None = None,
    observe: bool = True,
) -> dict:
    """Discover an in-scope site tree rooted at one URL.

    Discovery is link-only: Crawl4AI walks the site graph without this
    tool returning page bodies. The result is a bounded tree the caller
    can use as a planning surface before spending crawl budget on
    selected nodes.
    """
    root_url = core._validate_urls([url], maximum=1)[0]
    request_id = uuid.uuid4().hex
    max_urls = core._validate_positive_int("max_urls", max_urls, maximum=_MAX_MAP_URLS)
    include_patterns = core._normalize_glob_patterns(include_patterns, field_name="include_patterns")

    started = time.monotonic()
    warnings: list[dict] = []
    pages_visited = 0
    try:
        discovered_pages = await core._deep_crawl(
            [root_url],
            max_depth=2,
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
            "request_id": request_id,
            "max_urls_requested": max_urls,
            "urls_returned": len(results),
            "pages_visited": pages_visited,
            "warnings": warnings,
            "timings_ms": {
                "total": int((time.monotonic() - started) * 1000),
            },
        },
    }
    if observe:
        observability.observe_tool_response("map", response)
    log.info(
        "request_id=%s map url=%s returned=%d warnings=%d",
        request_id, root_url, len(results), len(warnings),
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
    relevance score (configured cross-encoder) instead of BFS discovery
    order, and each result's `content` carries the joined top chunks
    rather than the document head.
    """
    effective_max_urls = core._validate_positive_int(
        "max_urls",
        max_urls,
        maximum=min(_MAX_MAP_URLS, _MAX_EXTRACT_URLS),
    )
    request_id = uuid.uuid4().hex
    normalized_query = core._coerce_optional_str(query)
    started = time.monotonic()
    tree = await map_impl(
        url=url,
        max_urls=effective_max_urls,
        include_patterns=include_patterns,
        observe=False,
    )
    root_url = tree["url"]
    urls = [entry["url"] for entry in tree["results"]]

    extracted = await extract_impl(
        urls=urls,
        chunk_ids=None,
        observe=False,
    )
    urls_succeeded = extracted["meta"]["urls_succeeded"]
    urls_failed = extracted["meta"]["urls_failed"]
    doc_by_url = {entry["url"]: entry for entry in extracted["results"]}

    score_by_url: dict[str, float | None] = {}
    if normalized_query:
        chunk_docs: list[str] = []
        chunk_to_url: list[str] = []
        for u, doc in doc_by_url.items():
            if doc.get("status") != "ok":
                continue
            chunks = doc.get("chunks") or []
            for chunk in chunks:
                text = chunk.get("text") if isinstance(chunk, dict) else None
                if text:
                    chunk_docs.append(text)
                    chunk_to_url.append(u)
        if chunk_docs:
            try:
                scored = await core._rerank_scored(normalized_query, chunk_docs)
            except Exception as exc:
                log.warning("crawl rerank failed query=%r err=%s", normalized_query, exc)
                scored = []
            top_by_url: dict[str, list[dict]] = defaultdict(list)
            for chunk_idx, score in scored:
                u = chunk_to_url[chunk_idx]
                if score_by_url.get(u) is None or score > (score_by_url[u] or 0):
                    score_by_url[u] = score
                if len(top_by_url[u]) < 3:
                    top_by_url[u].append({"text": chunk_docs[chunk_idx], "score": score})
            for u, top in top_by_url.items():
                doc_by_url[u]["top_chunks"] = top

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
    sparse = len(results) < min(effective_max_urls, 3) or urls_succeeded == 0
    sparsity_reason = None
    if urls_succeeded == 0:
        sparsity_reason = "No pages could be extracted."
    elif len(results) < min(effective_max_urls, 3):
        sparsity_reason = f"Only {len(results)} in-scope page(s) were discovered under this root."
    if sparsity_reason:
        warnings.append(core._warning("crawl_sparse", "crawl", sparsity_reason))
    response = {
        "url": root_url,
        "query": normalized_query,
        "results": results,
        "meta": {
            "request_id": request_id,
            "max_urls_requested": effective_max_urls,
            "urls_discovered": len(tree["results"]),
            "urls_returned": len(results),
            "urls_truncated_by_limit": 0,
            "urls_deduplicated": 0,
            "sparse": sparse,
            "sparsity_reason": sparsity_reason,
            "urls_succeeded": urls_succeeded,
            "urls_failed": urls_failed,
            "warnings": warnings,
            "timings_ms": {
                "total": int((time.monotonic() - started) * 1000),
            },
        },
    }
    observability.observe_tool_response("crawl", response)
    log.info(
        "request_id=%s crawl url=%s query=%r discovered=%d returned=%d dedup=%d succeeded=%d failed=%d",
        request_id,
        root_url,
        normalized_query,
        response["meta"]["urls_discovered"],
        len(results),
        response["meta"]["urls_deduplicated"],
        urls_succeeded,
        urls_failed,
    )
    return _validated_response(models.CrawlResponseModel, response)
