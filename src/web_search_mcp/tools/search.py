"""Search and research tool implementations.

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

from web_search_mcp.storage import cache as cache_module
from web_search_mcp.common import (
    _chunk_text,
    _coerce_optional_str,
    _dedup_chunks,
    _dedup_results,
    _dedup_unresponsive_engines,
    _diversify_ranked_entries,
    _domain_from_url,
    _filter_results_by_domain,
    _normalize_domains,
    _normalize_time_range,
    _normalize_url,
    _validate_positive_int,
    _validate_query,
    _warning,
)
from web_search_mcp.presentation import models
from web_search_mcp.preprocessing import lfm
from web_search_mcp.ranking import evidence
from web_search_mcp.ranking import intent as intent_module
from web_search_mcp.ranking import query_expansion
from web_search_mcp.config import search as search_config
from web_search_mcp.config import freshness as freshness_config
from web_search_mcp.storage import semantic
from web_search_mcp.storage import evidence as evidence_store
from web_search_mcp import observability
from web_search_mcp.ranking import source_quality
from web_search_mcp.config.settings import (
    MAX_RESULTS,
    RERANK_MODEL,
    _CHUNK_GAP,
    _MAX_CONTENT_CHARS,
    _MIN_RELEVANCE_SCORE,
)
from web_search_mcp.ranking.service import RERANK_NAME, _rerank_scored
from web_search_mcp.search_client import _search
from web_search_mcp.storage.pages import _scrape_cached

log = logging.getLogger("web-search-mcp")


# In-flight SearXNG requests, keyed on the searxng_cache key. Two
# concurrent searches for the same (query, num_results, time_range,
# language, pageno) share a single upstream call instead of both
# cache-missing and both hitting the SearXNG → brave/google/etc. chain.
# Cleared on completion (success or failure); awaiters of a failed
# request re-raise and the next caller will retry.
_searxng_inflight: dict[str, asyncio.Future] = {}


def _interleave_result_groups(groups: list[list[dict]]) -> list[dict]:
    """Merge query variants fairly before applying the candidate budget."""
    if not groups:
        return []
    return [
        group[position]
        for position in range(max(map(len, groups), default=0))
        for group in groups
        if position < len(group)
    ]


async def _searxng_cached(
    search_query: str,
    *,
    num_results: int,
    time_range: str | None,
    language: str | None,
    pageno: int,
    max_age_seconds: int | None = None,
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
        if "payload" not in cached:
            if max_age_seconds is None:
                return cached
        else:
            cached_at = cached.get("cached_at", 0)
            if max_age_seconds is None or time.time() - cached_at <= max_age_seconds:
                return cached["payload"]
    inflight = _searxng_inflight.get(key)
    if inflight is not None:
        return await inflight
    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    _searxng_inflight[key] = fut
    try:
        result = await _search(
            search_query,
            num_results=num_results,
            time_range=time_range,
            language=language,
            pageno=pageno,
        )
        await cache_module.searxng_cache.set(key, {
            "cached_at": int(time.time()),
            "payload": result,
        })
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
    candidate_budget: int,
    searx_pages: int,
    time_range: str | None,
    language: str | None,
    include_domains: list[str] | None,
    exclude_domains: list[str] | None,
    source_types: list[str] | None,
    freshness_policy: freshness_config.FreshnessPolicy,
) -> tuple[list[dict], list, list[dict], bool]:
    """Search SearXNG pages and return filtered candidates plus warnings."""
    warnings: list[dict] = []
    degraded = False
    unresponsive_engines: list = []
    result_groups: list[list[dict]] = []
    for search_query in search_queries:
        query_results: list[dict] = []
        for pageno in range(1, searx_pages + 1):
            try:
                page = await _searxng_cached(
                    search_query,
                    num_results=candidate_budget,
                    time_range=time_range,
                    language=language,
                    pageno=pageno,
                    max_age_seconds=freshness_policy.search_max_age_seconds,
                )
            except Exception as exc:
                if not any(result_groups) and not query_results and pageno == 1:
                    degraded = True
                    warnings.append(_warning("search_failed", "searxng", str(exc)))
                else:
                    warnings.append(_warning("search_failed", "searxng", f"{search_query} page {pageno}: {exc}"))
                break
            query_results.extend(page["results"])
            unresponsive_engines.extend(page.get("unresponsive_engines", []))
            filtered_so_far = _dedup_results(
                _filter_results_by_domain(query_results, include_domains, exclude_domains)
            )
            if profile == "search" and len(filtered_so_far) >= candidate_budget:
                break
        result_groups.append(query_results)
        if profile != "research":
            break

    # Research variants are intentionally diverse. Round-robin their results so
    # one broad or poorly interpreted query cannot consume the entire bounded
    # candidate pool before title/snippet reranking.
    raw_results = (
        _interleave_result_groups(result_groups)
        if profile == "research"
        else [result for group in result_groups for result in group]
    )
    results = _dedup_results(_filter_results_by_domain(raw_results, include_domains, exclude_domains))
    results = [r for r in results if source_quality.matches_source_types(r.get("url", ""), source_types)]
    return results, unresponsive_engines, warnings, degraded


async def _rank_search_candidates(
    query: str,
    results: list[dict],
    intent_profile: intent_module.IntentProfile,
    warnings: list[dict],
) -> tuple[list[dict], bool]:
    """Rerank title/snippet candidates before selecting pages to scrape."""
    if not results:
        return [], False
    documents = [
        "\n".join(filter(None, [
            result.get("title", ""), result.get("content", ""), result.get("url", ""),
        ]))
        for result in results
    ]
    try:
        scored = await _rerank_scored(query, documents)
    except Exception as exc:
        log.warning("candidate rerank failed query=%r err=%s", query, exc)
        warnings.append(_warning("candidate_rerank_failed", RERANK_NAME, str(exc)))
        return results, True
    scores = {idx: score for idx, score in scored}
    entries = [{"url": result.get("url", ""), "title": result.get("title", "")} for result in results]
    ranked = sorted(
        range(len(results)),
        key=lambda idx: source_quality.entry_sort_score(
            idx, entries, scores, intent_profile, query=query,
        ),
        reverse=True,
    )
    ranked = source_quality.diversify_by_source_type(
        _diversify_ranked_entries(ranked, entries), entries,
    )
    return [results[idx] for idx in ranked], False


def _validated_response(model_cls, response: dict) -> dict:
    """Return a model-validated payload without inventing unset keys."""
    return models.dump_response(model_cls, response)


def _empty_search_response(
    *, request_id: str, query: str, profile: str, time_range: str | None,
    include_domains: list[str] | None, exclude_domains: list[str] | None,
    num_results: int, scrape_budget: int, max_passages: int,
    max_chars_per_result: int, search_queries: list[str],
    source_types: list[str] | None, degraded: bool, warnings: list[dict],
    timings_ms: dict, started: float, intent_name: str,
    preprocessing: dict,
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
            "intent": intent_name,
            "overview": [],
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
            "preprocessing": preprocessing,
            "semantic_hits": 0,
            "degraded": degraded,
            "warnings": warnings or [_warning("no_results", "searxng", query)],
            "timings_ms": {**timings_ms, "total": int((time.monotonic() - started) * 1000)},
        },
    }


async def _scrape_search_entries(
    results: list[dict],
    scrape_budget: int,
    freshness_policy: freshness_config.FreshnessPolicy,
) -> tuple[list[dict], int, list[dict]]:
    """Scrape top candidates and build page/snippet entries."""
    to_scrape = min(scrape_budget, len(results))
    scraped = await asyncio.gather(*[
        _scrape_cached(
            r["url"], cache_module.page_cache,
            max_age_seconds=freshness_policy.page_max_age_seconds,
        ) for r in results[:to_scrape]
    ])
    warnings: list[dict] = []
    scrape_failures = sum(1 for s in scraped if s.get("content") is None)
    if scrape_failures:
        warnings.append(_warning("scrape_failed", "crawl4ai", f"{scrape_failures} of {to_scrape} pages failed"))
    for scrape in scraped:
        diagnostic = (scrape.get("metadata") or {}).get("diagnostic")
        if diagnostic:
            warnings.append(_warning("content_rejected", "crawl4ai", diagnostic))

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
            "full_content": content,
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
    persisted = await asyncio.gather(*[
        evidence_store.persist_document(
            url=entry["url"], title=entry.get("title"),
            content=entry["full_content"], metadata=entry.get("metadata"),
        )
        for entry in entries if entry.get("scraped") and entry.get("full_content")
    ])
    persisted_by_url = {_normalize_url(item["url"]): item for item in persisted}
    for entry in entries:
        manifest = persisted_by_url.get(_normalize_url(entry.get("url", "")))
        if not manifest:
            continue
        entry["document_id"] = manifest["id"]
        entry["resource_uri"] = manifest["uri"]
        chunks = _chunk_text(manifest["content"])
        entry["chunk_refs"] = {
            text: manifest["chunks"][idx] for idx, text in enumerate(chunks)
        }
    return entries, to_scrape, warnings


async def _append_vector_memory_entries(
    *, entries: list[dict], query: str, max_passages: int,
    source_types: list[str] | None, include_domains: list[str],
    exclude_domains: list[str], existing_urls: set[str],
) -> int:
    """Append chunk-level semantic/vector hits from semantic.py."""
    semantic_hits = await semantic.search(query)
    by_cached_url: dict[str, list[dict]] = defaultdict(list)
    for hit in semantic_hits:
        by_cached_url[_normalize_url(hit["url"])].append(hit)
    added = 0
    for normalized, hits in by_cached_url.items():
        if normalized in existing_urls:
            continue
        first = hits[0]
        url = first["url"]
        if (
            not _filter_results_by_domain([first], include_domains, exclude_domains)
            or not source_quality.matches_source_types(url, source_types)
        ):
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
    include_domains: list[str], exclude_domains: list[str],
    existing_urls: set[str],
) -> int:
    """Append page-level retrieval-memory hits from cache.py."""
    keys = [_normalize_url(r.get("url", "")) for r in results]
    cached_entries = await asyncio.gather(*(cache_module.page_memory_cache.get(key) for key in keys))
    added = 0
    for cached in cached_entries:
        if not cached or not cached.get("content"):
            continue
        normalized = _normalize_url(cached.get("url", ""))
        if (
            normalized in existing_urls
            or not _filter_results_by_domain(
                [cached], include_domains, exclude_domains,
            )
            or not source_quality.matches_source_types(
                cached.get("url", ""), source_types,
            )
        ):
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
    include_domains: list[str], exclude_domains: list[str],
    freshness_policy: freshness_config.FreshnessPolicy,
) -> int:
    """Append memory evidence not already present in entries."""
    if not freshness_policy.allow_unverified_memory:
        return 0
    existing_urls = {_normalize_url(entry["url"]) for entry in entries if entry.get("url")}
    added = await _append_vector_memory_entries(
        entries=entries, query=query, max_passages=max_passages,
        source_types=source_types, include_domains=include_domains,
        exclude_domains=exclude_domains, existing_urls=existing_urls,
    )
    added += await _append_page_memory_entries(
        entries=entries, results=results, source_types=source_types,
        include_domains=include_domains, exclude_domains=exclude_domains,
        existing_urls=existing_urls,
    )
    return added


async def _rank_search_entries(
    *, query: str, entries: list[dict], max_passages: int,
    max_chars_per_result: int, num_results: int, warnings: list[dict],
    intent_profile: intent_module.IntentProfile,
) -> tuple[list[int], dict[int, list[tuple[str, float]]], int, bool, bool]:
    """Chunk, rerank, filter, and diversify entries."""
    all_chunks: list[str] = []
    chunk_to_entry: list[int] = []
    for i, entry in enumerate(entries):
        chunks = _chunk_text(entry["content"]) if entry["scraped"] and entry["content"] else ([entry["content"]] if entry["content"] else [])
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_to_entry.append(i)
    all_chunks, chunk_to_entry = _dedup_chunks(all_chunks, chunk_to_entry)

    rerank_failed = False
    degraded = False
    try:
        scored = await _rerank_scored(query, all_chunks)
    except Exception as exc:
        log.warning("rerank failed query=%r err=%s", query, exc)
        warnings.append(_warning("rerank_failed", RERANK_NAME, str(exc)))
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
        ranked_entry_idxs = sorted(
            entry_best,
            key=lambda eidx: source_quality.entry_sort_score(
                eidx, entries, entry_best, intent_profile, query=query,
            ),
            reverse=True,
        )
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
            warnings.append(_warning("low_relevance_filtered", RERANK_NAME, f"{noise_count} result(s) dropped below relevance threshold"))
    ranked_entry_idxs = source_quality.diversify_by_source_type(
        _diversify_ranked_entries(ranked_entry_idxs, entries), entries
    )[:num_results]
    return ranked_entry_idxs, entry_chunks, len(all_chunks), degraded, rerank_failed


async def _build_structured_search_results(
    *, ranked_entry_idxs: list[int], entries: list[dict], entry_chunks: dict[int, list[tuple[str, float]]],
    results: list[dict], profile: str = "search",
) -> tuple[list[dict], list[str]]:
    ranked_normalized = [_normalize_url(entries[eidx]["url"]) for eidx in ranked_entry_idxs]
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
        structured = {
            "rank": rank,
            "title": entry["title"],
            "url": url,
            "domain": _domain_from_url(url),
            "source_type": source_quality.source_type(url),
            "passages": [],
            "scraped": entry["scraped"],
            "seen_recently": seen_recently,
            "retrieval_source": entry.get("retrieval_source", "live_search"),
        }
        if not top:
            structured["content"] = entry["content"]
        if entry.get("scraped") is False:
            structured["snippet"] = entry["content"]
        for idx, (chunk, score) in enumerate(top, 1):
            passage = {"citation": f"{rank}.{idx}", "text": chunk, "score": score}
            chunk_ref = (entry.get("chunk_refs") or {}).get(chunk)
            if chunk_ref:
                passage["chunk_id"] = chunk_ref["id"]
                passage["resource_uri"] = chunk_ref["uri"]
            structured["passages"].append(passage)
        if entry.get("document_id"):
            structured["document_id"] = entry["document_id"]
            structured["resource_uri"] = entry["resource_uri"]
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


async def _persist_search_memory(
    structured_results: list[dict], normalized_urls: list[str], entries: list[dict]
) -> None:
    if not normalized_urls:
        return
    writes = [cache_module.seen_urls.set(url, 1) for url in normalized_urls]
    entries_by_url = {
        _normalize_url(entry.get("url", "")): entry for entry in entries
    }
    for result in structured_results:
        entry = entries_by_url.get(_normalize_url(result["url"]), {})
        content = entry.get("full_content") or entry.get("content")
        if result.get("scraped") and content:
            metadata = entry.get("metadata") or result.get("metadata") or {}
            writes.append(cache_module.page_memory_cache.set(_normalize_url(result["url"]), {
                "url": result["url"], "title": result["title"], "domain": result["domain"],
                "source_type": result.get("source_type"), "content": content,
                "metadata": metadata, "updated_at": int(time.time()),
            }))
            writes.append(semantic.index_page(result["url"], result["title"], content, metadata))
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

    Pipeline: broad SearXNG search -> candidate rerank -> selective scrape ->
    chunk rerank -> formatted evidence. Scraped pages are split into paragraphs
    and reranked at the chunk level, so only query-relevant excerpts are returned.

    Candidate retrieval is intentionally broader than `num_results`; scraping
    remains bounded by the profile and MAX_SCRAPE. Results are backed by shared
    Valkey caches across requests.
    """
    query = _validate_query(query)
    request_id = uuid.uuid4().hex
    num_results = _validate_positive_int("num_results", num_results, maximum=MAX_RESULTS)
    profile = search_config.normalize_profile(profile)
    time_range = _normalize_time_range(time_range)
    freshness_policy = freshness_config.policy_for(time_range)
    language = _coerce_optional_str(language)
    include_domains = _normalize_domains(include_domains, field_name="include_domains")
    exclude_domains = _normalize_domains(exclude_domains, field_name="exclude_domains")
    source_types = source_quality.normalize_source_types(source_types)
    searx_pages, scrape_budget, default_passages = search_config.profile_budget(profile, num_results)
    max_passages = search_config.validate_optional_positive_int("max_passages", max_passages, maximum=8) or default_passages
    default_chars = search_config.default_chars_per_result(profile)
    max_chars_per_result = search_config.validate_optional_positive_int(
        "max_chars_per_result", max_chars_per_result, maximum=8000
    ) or default_chars
    started = time.monotonic()
    warnings: list[dict] = []
    degraded = False
    timings_ms = {
        "preprocessing": 0, "search": 0, "candidate_rerank": 0,
        "scrape": 0, "semantic": 0, "rerank": 0, "total": 0,
    }
    semantic_hits = 0
    intent_profile = intent_module.classify(query, time_range=time_range)
    search_queries = query_expansion.search_queries(query, profile, intent_profile.name)
    preprocessing = {**lfm.status(), "planning_used": False, "digest_used": False}
    if profile == "research":
        preprocessing_started = time.monotonic()
        plan = await lfm.plan_query(query, intent_profile, search_queries)
        timings_ms["preprocessing"] = int((time.monotonic() - preprocessing_started) * 1000)
        intent_profile = plan.intent
        search_queries = plan.queries
        preprocessing["planning_used"] = plan.used
        if plan.error:
            warnings.append(_warning("lfm_preprocessing_failed", "lfm", plan.error))
    candidate_budget = search_config.candidate_budget(profile, num_results)

    # --- search (profile-aware multi-page retrieval) ---
    search_started = time.monotonic()
    results, unresponsive_engines, search_warnings, search_degraded = await _collect_search_candidates(
        search_queries=search_queries,
        profile=profile,
        candidate_budget=candidate_budget,
        searx_pages=searx_pages,
        time_range=time_range,
        language=language,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        source_types=source_types,
        freshness_policy=freshness_policy,
    )
    warnings.extend(search_warnings)
    degraded = degraded or search_degraded
    results = results[:candidate_budget]

    # Surface per-engine failures from SearXNG (e.g. "google: CAPTCHA").
    # Does NOT flip `degraded` — the multi-engine hedge means a single
    # upstream going down is expected, not a pipeline failure.
    for engine, reason in _dedup_unresponsive_engines(unresponsive_engines):
        detail = f"{engine}: {reason}" if reason else engine
        warnings.append(_warning("engine_unresponsive", "searxng", detail))

    timings_ms["search"] = int((time.monotonic() - search_started) * 1000)
    if not results:
        response = _empty_search_response(
            request_id=request_id, query=query, profile=profile, time_range=time_range,
            include_domains=include_domains, exclude_domains=exclude_domains,
            num_results=num_results, scrape_budget=scrape_budget,
            max_passages=max_passages, max_chars_per_result=max_chars_per_result,
            search_queries=search_queries, source_types=source_types,
            degraded=degraded, warnings=warnings, timings_ms=timings_ms,
            started=started, intent_name=intent_profile.name,
            preprocessing=preprocessing,
        )
        observability.observe_tool_response(profile, response)
        log.info("request_id=%s query=%r profile=%s results=0 degraded=%s", request_id, query, profile, degraded)
        return _validated_response(models.SearchResponseModel, response)

    candidate_rerank_started = time.monotonic()
    results, candidate_rank_degraded = await _rank_search_candidates(
        query, results, intent_profile, warnings,
    )
    timings_ms["candidate_rerank"] = int((time.monotonic() - candidate_rerank_started) * 1000)
    degraded = degraded or candidate_rank_degraded

    scrape_started = time.monotonic()
    entries, to_scrape, scrape_warnings = await _scrape_search_entries(
        results, scrape_budget, freshness_policy,
    )
    timings_ms["scrape"] = int((time.monotonic() - scrape_started) * 1000)
    warnings.extend(scrape_warnings)
    degraded = degraded or any(w.get("type") == "scrape_failed" for w in scrape_warnings)

    semantic_started = time.monotonic()
    semantic_hits = await _merge_memory_entries(
        entries=entries, results=results, query=query,
        max_passages=max_passages, source_types=source_types,
        include_domains=include_domains, exclude_domains=exclude_domains,
        freshness_policy=freshness_policy,
    )
    timings_ms["semantic"] = int((time.monotonic() - semantic_started) * 1000)

    rerank_started = time.monotonic()
    ranked_entry_idxs, entry_chunks, chunk_count, rank_degraded, _ = await _rank_search_entries(
        query=query, entries=entries, max_passages=max_passages,
        max_chars_per_result=max_chars_per_result, num_results=num_results,
        warnings=warnings, intent_profile=intent_profile,
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

    synthesis_results = [
        result for result in structured_results
        if source_quality.synthesis_eligible(result)
    ]
    if profile == "research" and len(synthesis_results) < len(structured_results):
        warnings.append(_warning(
            "weak_sources_excluded_from_overview",
            "ranking",
            f"{len(structured_results) - len(synthesis_results)} source(s) remain visible but do not support the overview",
        ))
    overview = evidence.research_summary(synthesis_results, warnings) if profile == "research" else []
    if profile == "research":
        preprocessing_started = time.monotonic()
        digest = await lfm.digest_evidence(query, synthesis_results, overview)
        timings_ms["preprocessing"] += int((time.monotonic() - preprocessing_started) * 1000)
        overview = digest.overview
        preprocessing["digest_used"] = digest.used
        if digest.error:
            warnings.append(_warning("lfm_preprocessing_failed", "lfm", digest.error))
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
            "intent": intent_profile.name,
            "candidate_pool_size": len(results),
            "overview": overview,
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
            "preprocessing": preprocessing,
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
    await _persist_search_memory(structured_results, new_urls, entries)

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
