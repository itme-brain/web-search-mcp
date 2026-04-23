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
from collections import defaultdict

from fastmcp import Context

# Module-qualified import so `unittest.mock.patch("core.X")` intercepts
# every call site — both from core itself and from here. (`from core
# import X` would bind X into impls's namespace, creating a second patch
# target we'd have to mock separately.) Constants are safe to import
# by-name since they're not patched.
import cache as cache_module
import core
import models
from core import (
    MAX_RESULTS,
    MAX_SCRAPE,
    RERANK_MODEL,
    _CHUNK_GAP,
    _MAX_CONTENT_CHARS,
    _MAX_EXTRACT_URLS,
    _MAX_MAP_URLS,
    _MIN_RELEVANCE_SCORE,
    _TOP_CHUNKS,
)

log = logging.getLogger("web-search-mcp")




def _validated_response(model_cls, response: dict) -> dict:
    """Return a model-validated payload without inventing unset keys."""
    return models.dump_response(model_cls, response)


async def search_impl(
    query: str,
    num_results: int = 10,
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
    Results are cached within the session.
    """
    query = core._validate_query(query)
    num_results = core._validate_positive_int("num_results", num_results, maximum=MAX_RESULTS)
    time_range = core._normalize_time_range(time_range)
    include_domains = core._normalize_domains(include_domains, field_name="include_domains")
    exclude_domains = core._normalize_domains(exclude_domains, field_name="exclude_domains")
    scrape_budget = min(num_results, MAX_SCRAPE)
    started = time.monotonic()
    warnings: list[dict] = []
    degraded = False
    timings_ms = {"search": 0, "scrape": 0, "rerank": 0, "total": 0}

    # --- shared cache (Valkey-backed, cross-session + cross-process) ---
    page_cache = cache_module.page_cache
    query_cache = cache_module.query_cache
    seen_urls = cache_module.seen_urls

    # --- exact query cache ---
    qkey = hashlib.sha256(
        json.dumps(
            [
                query.lower().strip(),
                num_results,
                time_range,
                language,
                include_domains,
                exclude_domains,
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    cached_response = await query_cache.get(qkey)
    if cached_response is not None:
        log.info("query cache hit query=%r", query)
        return cached_response

    # --- search (page 1, with reactive page-2 fallback on underflow) ---
    search_started = time.monotonic()
    unresponsive_engines: list = []
    try:
        page1 = await core._search(query, num_results=num_results, time_range=time_range, language=language)
    except Exception as exc:
        degraded = True
        warnings.append(core._warning("search_failed", "searxng", str(exc)))
        raw_results: list[dict] = []
    else:
        raw_results = page1["results"]
        unresponsive_engines.extend(page1.get("unresponsive_engines", []))

    results = core._dedup_results(core._filter_results_by_domain(raw_results, include_domains, exclude_domains))
    if raw_results and len(results) < num_results:
        try:
            page2 = await core._search(query, num_results=num_results, time_range=time_range, language=language, pageno=2)
        except Exception as exc:
            warnings.append(core._warning("search_failed", "searxng", f"page 2: {exc}"))
        else:
            raw_results = raw_results + page2["results"]
            unresponsive_engines.extend(page2.get("unresponsive_engines", []))
            results = core._dedup_results(core._filter_results_by_domain(raw_results, include_domains, exclude_domains))
    results = results[:num_results]

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
                "num_results_requested": num_results,
                "num_results_returned": 0,
                "scrape_top": scrape_budget,
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
        response = _validated_response(models.SearchResponseModel, response)
        await query_cache.set(qkey, response)
        return response

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
        entry_chunks[eidx] = entry_chunks[eidx][:_TOP_CHUNKS]

    # --- rank pages by best chunk score, drop noise below threshold ---
    entry_best: dict[int, float | None] = {
        eidx: chunks[0][1] for eidx, chunks in entry_chunks.items() if chunks
    }
    if rerank_failed:
        ranked_entry_idxs = list(range(len(entries)))
    else:
        ranked_entry_idxs = sorted(entry_best, key=entry_best.get, reverse=True)
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
    ranked_entry_idxs = core._diversify_ranked_entries(ranked_entry_idxs, entries)

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

        structured = {
            "rank": rank,
            "search_rank": eidx + 1,
            "title": entry["title"],
            "url": url,
            "normalized_url": normalized_url,
            "domain": core._domain_from_url(url),
            "snippet": results[eidx].get("content", ""),
            "content": content,
            "top_chunks": [{"text": chunk, "score": score} for chunk, score in top],
            "score": entry_best.get(eidx),
            "scraped": entry["scraped"],
            "previously_seen": previously_seen,
        }
        metadata = entry.get("metadata") or {}
        if metadata:
            structured["metadata"] = metadata
        structured_results.append(structured)
        new_urls.append(normalized_url)

    response = {
        "query": query,
        "time_range": time_range,
        "include_domains": include_domains,
        "exclude_domains": exclude_domains,
        "results": structured_results,
        "meta": {
            "num_results_requested": num_results,
            "num_results_returned": len(structured_results),
            "scrape_top": to_scrape,
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
    await asyncio.gather(
        query_cache.set(qkey, response),
        *(seen_urls.set(url, 1) for url in new_urls),
    )

    log.info("query=%r chunks=%d pages=%d", query, len(all_chunks), len(entries))

    return _validated_response(models.SearchResponseModel, response)


async def extract_impl(
    urls: list[str],
    query: str | None = None,
    offset: int = 0,
    chunk_ids: list[int] | None = None,
    ctx: Context | None = None,
) -> dict:
    """Extract a batch of URLs with per-URL status reporting.

    Uses Crawl4AI for web pages and local fetch for text-like resources.
    Binary document formats are classified here and handed off to the
    `files` MCP via structured metadata rather than parsed locally.

    `offset` slides the return window N chars into the full extracted
    content — pair with the per-result `total_chars` / `chars_shown`
    metadata to paginate through long documents. offset>0 bypasses
    query-based rerank in favor of raw continuation.

    `chunk_ids` cherry-picks specific chunks from the document by their
    stable id (see the `chunks` field on the response). Mutually
    exclusive with offset>0 — raise if both are set.
    """
    urls = core._validate_urls(urls, maximum=_MAX_EXTRACT_URLS)
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if chunk_ids is not None and offset > 0:
        raise ValueError("chunk_ids and offset>0 are mutually exclusive")
    if chunk_ids is not None and any(i < 0 for i in chunk_ids):
        raise ValueError("chunk_ids entries must be >= 0")
    normalized_query = query.strip() if query else None
    started = time.monotonic()

    page_cache = cache_module.page_cache

    documents = await asyncio.gather(*[
        core._extract_url_document(
            url, normalized_query, page_cache,
            offset=offset, chunk_ids=chunk_ids,
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
            "normalized_url": core._normalize_url(url),
            "domain": core._domain_from_url(url),
            "status": document["status"],
            "content_type": document.get("content_type"),
            "file_type": document.get("file_type"),
            "title": document.get("title"),
            "content": content,
            "chars_shown": len(content),
            "offset": offset,
            "total_chars": total_chars,
            "top_chunks": document.get("top_chunks", []),
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
    """Discover in-scope URLs from a site via single-page link extraction.

    One authoritative request: pull the rendered link graph of the root
    page from Crawl4AI and filter it to the registrable domain + any
    caller-supplied include_patterns. Depth-1 only; multi-hop discovery
    is out of scope for map (callers who want deeper structure should
    crawl one step at a time).
    """
    root_url = core._validate_urls([url], maximum=1)[0]
    max_urls = core._validate_positive_int("max_urls", max_urls, maximum=_MAX_MAP_URLS)
    include_patterns = core._normalize_glob_patterns(include_patterns, field_name="include_patterns")

    started = time.monotonic()
    warnings: list[dict] = []
    try:
        discovery = await core._discover_page_links(root_url)
    except Exception as exc:
        warnings.append(core._warning("link_discovery_failed", "crawl4ai", str(exc)))
        discovery = {"links": [], "title": None}

    results: list[dict] = []
    visited: set[str] = set()

    # Seed: the root itself.
    root_normalized = core._normalize_url(root_url)
    visited.add(root_normalized)
    results.append({
        "url": root_url,
        "normalized_url": root_normalized,
        "domain": core._domain_from_url(root_url),
        "title": discovery.get("title"),
        "link_text": None,
        "depth": 0,
        "discovered_from": None,
        "link_type": "seed",
    })

    # Discovered in-scope links at depth 1.
    domain_patterns = core._domain_filter_patterns(root_url, True)
    for link in discovery.get("links", []):
        if len(results) >= max_urls:
            break
        link_url = link.get("url")
        if not isinstance(link_url, str) or not link_url:
            continue
        normalized_url = core._normalize_url(link_url)
        if normalized_url in visited:
            continue
        if domain_patterns and not core._url_matches_patterns(link_url, domain_patterns):
            continue
        if include_patterns and not core._url_matches_patterns(link_url, include_patterns):
            continue
        visited.add(normalized_url)
        results.append({
            "url": link_url,
            "normalized_url": normalized_url,
            "domain": core._domain_from_url(link_url),
            "title": link.get("title"),
            "link_text": link.get("text"),
            "depth": 1,
            "discovered_from": root_url,
            "link_type": link.get("link_type", "internal"),
        })

    for rank, entry in enumerate(results, start=1):
        entry["rank"] = rank

    response = {
        "url": root_url,
        "results": results,
        "meta": {
            "max_urls_requested": max_urls,
            "urls_returned": len(results),
            "pages_visited": len(visited),
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
) -> dict:
    """Deep-crawl a site through Crawl4AI and return a structured dict.

    Pure discovery + extraction — no query-based reranking.  For relevance
    ranking use search_impl with include_domains instead.
    """
    effective_max_urls = core._validate_positive_int(
        "max_urls",
        max_urls,
        maximum=min(_MAX_MAP_URLS, _MAX_EXTRACT_URLS),
    )
    started = time.monotonic()
    root_url = core._validate_urls([url], maximum=1)[0]
    include_patterns = core._normalize_glob_patterns(include_patterns, field_name="include_patterns")
    warnings: list[dict] = []

    def _entry(
        *,
        url: str,
        normalized_url: str,
        title: str | None,
        content: str,
        metadata: dict,
        depth: int,
        discovered_from: str | None,
        link_type: str,
        link_text: str | None = None,
    ) -> dict:
        raw_content = (content or "").strip()
        status = "ok" if raw_content else "error"
        e = {
            "rank": 0,  # filled in after final dedup
            "url": url,
            "normalized_url": normalized_url,
            "domain": core._domain_from_url(url),
            "title": title,
            "link_text": link_text,
            "depth": depth,
            "discovered_from": discovered_from,
            "link_type": link_type,
            "status": status,
            "content_type": "text/html",
            "content": raw_content,
            "chars_shown": len(raw_content),
            "offset": 0,
            "total_chars": len(raw_content),
            "cached": False,
            "error": None if status == "ok" else "extraction failed",
        }
        if metadata:
            e["metadata"] = metadata
        return e

    # Phase 1: root scrape + root link-graph discovery in parallel.
    # Two Crawl4AI calls because they use different configs (scrape uses
    # content-filter pruning; discovery keeps nav/footer so the link graph
    # survives).
    root_scrape, discovery = await asyncio.gather(
        core._scrape_cached(root_url, cache_module.page_cache),
        core._discover_page_links(root_url),
        return_exceptions=True,
    )
    if isinstance(root_scrape, BaseException):
        warnings.append(core._warning("scrape_failed", "crawl4ai", f"root: {root_scrape}"))
        root_scrape = {"content": None, "title": None, "metadata": {}}
    if isinstance(discovery, BaseException):
        warnings.append(core._warning("link_discovery_failed", "crawl4ai", f"root: {discovery}"))
        discovery = {"links": [], "title": None}

    entries: list[dict] = []
    seen: set[str] = set()

    # Phase 2: root entry at depth 0.
    root_normalized = core._normalize_url(root_url)
    seen.add(root_normalized)
    entries.append(_entry(
        url=root_url,
        normalized_url=root_normalized,
        title=root_scrape.get("title") or discovery.get("title"),
        content=root_scrape.get("content") or "",
        metadata=root_scrape.get("metadata") or {},
        depth=0,
        discovered_from=None,
        link_type="seed",
    ))

    # Phase 3: in-scope discovered links become depth-1 entries via
    # parallel cache-aware scrape. Always fetched (not relying on BFS to
    # cover them) — BFS is additive for depth-2 only.
    domain_patterns = core._domain_filter_patterns(root_url, True)
    seed_candidates: list[tuple[str, str, dict]] = []
    for link in discovery.get("links", []):
        link_url = link.get("url")
        if not isinstance(link_url, str) or not link_url:
            continue
        normalized = core._normalize_url(link_url)
        if normalized in seen:
            continue
        if domain_patterns and not core._url_matches_patterns(link_url, domain_patterns):
            continue
        if include_patterns and not core._url_matches_patterns(link_url, include_patterns):
            continue
        seed_candidates.append((link_url, normalized, link))

    seed_budget = max(0, effective_max_urls - len(entries))
    seeds_to_fetch = seed_candidates[:seed_budget]

    if seeds_to_fetch:
        seed_scrapes = await asyncio.gather(
            *(core._scrape_cached(su[0], cache_module.page_cache) for su in seeds_to_fetch),
            return_exceptions=True,
        )
        for (seed_url, normalized, link_info), scrape in zip(seeds_to_fetch, seed_scrapes):
            if isinstance(scrape, BaseException):
                warnings.append(core._warning("scrape_failed", "crawl4ai", f"{seed_url}: {scrape}"))
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            entries.append(_entry(
                url=seed_url,
                normalized_url=normalized,
                title=scrape.get("title") or link_info.get("title"),
                content=scrape.get("content") or "",
                metadata=scrape.get("metadata") or {},
                depth=1,
                discovered_from=root_url,
                link_type=link_info.get("link_type", "internal"),
                link_text=link_info.get("text"),
            ))

    # Phase 4: seeded BFS for depth-2 children, if budget remains.
    # BFS is additive — we've already covered the seeds; we only add
    # pages BFS discovered that we haven't already seen.
    remaining_budget = effective_max_urls - len(entries)
    if remaining_budget > 0 and seeds_to_fetch:
        try:
            bfs_pages = await core._deep_crawl(
                [s[0] for s in seeds_to_fetch],
                max_depth=1,
                max_pages=remaining_budget + len(seeds_to_fetch),
                same_domain_only=True,
                include_patterns=include_patterns,
            )
        except Exception as exc:
            warnings.append(core._warning("crawl_failed", "crawl4ai", f"bfs: {exc}"))
            bfs_pages = []

        for page in bfs_pages:
            if len(entries) >= effective_max_urls:
                break
            page_url = page.get("url")
            if not isinstance(page_url, str) or not page_url:
                continue
            normalized = core._normalize_url(page_url)
            if normalized in seen:
                continue
            seen.add(normalized)
            raw_content = core._extract_markdown(page) or ""
            crawl_md = page.get("metadata") if isinstance(page.get("metadata"), dict) else {}
            entries.append(_entry(
                url=page_url,
                normalized_url=normalized,
                title=core._extract_crawl_title(page),
                content=raw_content,
                metadata=core._build_document_metadata(page.get("html"), raw_content),
                depth=2,
                discovered_from=crawl_md.get("parent_url"),
                link_type="internal",
            ))

    # Phase 5: near-duplicate body dedup (pages served at multiple URLs).
    entries, urls_deduplicated = core._dedup_pages(entries)
    for idx, entry in enumerate(entries, start=1):
        entry["rank"] = idx
    results = entries[:effective_max_urls]

    urls_truncated = len(entries) - len(results)
    response = {
        "url": root_url,
        "results": results,
        "meta": {
            "max_urls_requested": effective_max_urls,
            "urls_discovered": len(entries) + urls_deduplicated,
            "urls_returned": len(results),
            "urls_truncated_by_limit": urls_truncated,
            "urls_deduplicated": urls_deduplicated,
            "urls_succeeded": sum(1 for result in results if result["status"] == "ok"),
            "urls_failed": sum(1 for result in results if result["status"] != "ok"),
            "warnings": warnings,
            "timings_ms": {
                "total": int((time.monotonic() - started) * 1000),
            },
        },
    }
    log.info(
        "crawl url=%s discovered=%d returned=%d dedup=%d succeeded=%d failed=%d",
        root_url,
        response["meta"]["urls_discovered"],
        len(results),
        urls_deduplicated,
        response["meta"]["urls_succeeded"],
        response["meta"]["urls_failed"],
    )
    return _validated_response(models.CrawlResponseModel, response)
