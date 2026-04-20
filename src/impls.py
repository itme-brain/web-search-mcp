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
from collections import defaultdict, deque

from fastmcp import Context

# Module-qualified import so `unittest.mock.patch("core.X")` intercepts
# every call site — both from core itself and from here. (`from core
# import X` would bind X into impls's namespace, creating a second patch
# target we'd have to mock separately.) Constants are safe to import
# by-name since they're not patched.
import core
from core import (
    MAX_RESULTS,
    MAX_SCRAPE,
    RERANK_MODEL,
    STATE_EXTRACT_CACHE,
    STATE_QUERY_CACHE,
    STATE_SCRAPE_CACHE,
    STATE_SEEN_URLS,
    _CHUNK_GAP,
    _MAX_CONTENT_CHARS,
    _MAX_EXTRACT_URLS,
    _MAX_MAP_DEPTH,
    _MAX_MAP_URLS,
    _TOP_CHUNKS,
)

log = logging.getLogger("web-search-mcp")


async def search_impl(
    query: str,
    num_results: int = 10,
    time_range: str | None = None,
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

    # --- session state (TTL-capped to prevent unbounded growth) ---
    scrape_cache = await core._load_cache(ctx, STATE_SCRAPE_CACHE)
    query_cache = await core._load_cache(ctx, STATE_QUERY_CACHE)
    seen_urls = await core._load_cache(ctx, STATE_SEEN_URLS)

    # --- exact query cache ---
    qkey = hashlib.sha256(
        json.dumps(
            [
                query.lower().strip(),
                num_results,
                time_range,
                include_domains,
                exclude_domains,
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if qkey in query_cache:
        log.info("query cache hit query=%r", query)
        return query_cache[qkey]

    # --- search (page 1, with reactive page-2 fallback on underflow) ---
    search_started = time.monotonic()
    unresponsive_engines: list = []
    try:
        page1 = await core._search(query, num_results=num_results, time_range=time_range)
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
            page2 = await core._search(query, num_results=num_results, time_range=time_range, pageno=2)
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
        query_cache[qkey] = response
        await core._save_cache(ctx, STATE_QUERY_CACHE, query_cache)
        return response

    # --- scrape (cache-aware) ---
    to_scrape = min(scrape_budget, len(results))
    scrape_started = time.monotonic()
    scrape_tasks = [core._scrape_cached(r["url"], scrape_cache) for r in results[:to_scrape]]
    scraped = await asyncio.gather(*scrape_tasks)
    timings_ms["scrape"] = int((time.monotonic() - scrape_started) * 1000)
    scrape_failures = sum(1 for content in scraped if content is None)
    if scrape_failures:
        degraded = True
        warnings.append(core._warning("scrape_failed", "crawl4ai", f"{scrape_failures} of {to_scrape} pages failed"))

    # --- build entries ---
    entries: list[dict] = []
    for i, result in enumerate(results[:to_scrape]):
        raw = scraped[i][:_MAX_CONTENT_CHARS] if scraped[i] else None
        entries.append({
            "title": result.get("title", "Untitled"),
            "url": result.get("url", ""),
            "content": raw or result.get("content", ""),
            "scraped": raw is not None,
        })

    for result in results[to_scrape:]:
        entries.append({
            "title": result.get("title", "Untitled"),
            "url": result.get("url", ""),
            "content": result.get("content", ""),
            "scraped": False,
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

    # --- rank pages by best chunk score ---
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
    ranked_entry_idxs = core._diversify_ranked_entries(ranked_entry_idxs, entries)

    # --- format structured output ---
    structured_results: list[dict] = []
    new_urls: list[str] = []
    for rank, eidx in enumerate(ranked_entry_idxs, 1):
        entry = entries[eidx]
        url = entry["url"]
        normalized_url = core._normalize_url(url)
        top = entry_chunks.get(eidx, [])
        if top:
            content = _CHUNK_GAP.join(chunk for chunk, _ in top)
        else:
            content = entry["content"]

        structured_results.append({
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
            "previously_seen": normalized_url in seen_urls,
        })
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

    # --- persist session state ---
    for url in new_urls:
        seen_urls[url] = True
    query_cache[qkey] = response
    await core._save_cache(ctx, STATE_SCRAPE_CACHE, scrape_cache)
    await core._save_cache(ctx, STATE_QUERY_CACHE, query_cache)
    await core._save_cache(ctx, STATE_SEEN_URLS, seen_urls)

    log.info(
        "query=%r chunks=%d pages=%d scrape_cache=%d query_cache=%d seen=%d",
        query, len(all_chunks), len(entries), len(scrape_cache), len(query_cache), len(seen_urls),
    )

    return response


async def extract_impl(
    urls: list[str],
    query: str | None = None,
    offset: int = 0,
    ctx: Context | None = None,
) -> dict:
    """Extract a batch of URLs with per-URL status reporting.

    Uses Crawl4AI for web pages and pymupdf4llm for PDF documents.
    PDFs are fully extracted with per-page chunking and optional
    query-based reranking.

    `offset` slides the return window N chars into the full extracted
    content — pair with the per-result `total_chars` / `chars_shown`
    metadata to paginate through long documents. offset>0 bypasses
    query-based rerank in favor of raw continuation.
    """
    urls = core._validate_urls(urls, maximum=_MAX_EXTRACT_URLS)
    if offset < 0:
        raise ValueError("offset must be >= 0")
    normalized_query = query.strip() if query else None
    started = time.monotonic()

    extract_cache = await core._load_cache(ctx, STATE_EXTRACT_CACHE)

    documents = await asyncio.gather(*[
        core._extract_url_document(url, normalized_query, extract_cache, offset=offset)
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
            "cached": document.get("cached", False),
            "error": document.get("error"),
        }
        if document.get("total_pages") is not None:
            entry["total_pages"] = document["total_pages"]
        results.append(entry)

    await core._save_cache(ctx, STATE_EXTRACT_CACHE, extract_cache)

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
    return response


async def map_impl(
    url: str,
    max_urls: int = 25,
    max_depth: int = 1,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    same_domain_only: bool = True,
) -> dict:
    """Discover in-scope URLs from a site using Crawl4AI link extraction."""
    root_url = core._validate_urls([url], maximum=1)[0]
    max_urls = core._validate_positive_int("max_urls", max_urls, maximum=_MAX_MAP_URLS)
    max_depth = core._validate_positive_int("max_depth", max_depth, maximum=_MAX_MAP_DEPTH)
    include_patterns = core._normalize_glob_patterns(include_patterns, field_name="include_patterns")
    exclude_patterns = core._normalize_glob_patterns(exclude_patterns, field_name="exclude_patterns")

    started = time.monotonic()
    root_domain = core._domain_from_url(root_url).lower()
    root_registrable = core._registrable_domain(root_domain)

    queue = deque([(root_url, 0, None)])
    visited_pages: set[str] = set()
    discovered: dict[str, dict] = {}
    warnings: list[dict] = []

    while queue and len(discovered) < max_urls:
        current_url, depth, discovered_from = queue.popleft()
        normalized_current = core._normalize_url(current_url)
        if normalized_current in visited_pages:
            continue
        visited_pages.add(normalized_current)

        current_entry = discovered.setdefault(normalized_current, {
            "url": current_url,
            "normalized_url": normalized_current,
            "domain": core._domain_from_url(current_url),
            "title": None,
            "link_text": None,
            "depth": depth,
            "discovered_from": discovered_from,
            "link_type": "seed" if depth == 0 else "internal",
        })

        if depth >= max_depth:
            continue

        try:
            page = await asyncio.wait_for(core._discover_page_links(current_url), timeout=core.REQUEST_TIMEOUT)
        except asyncio.TimeoutError:
            warnings.append(core._warning("link_discovery_timeout", "crawl4ai", current_url))
            continue
        except core.httpx.HTTPError as exc:
            warnings.append(core._warning("link_discovery_failed", "crawl4ai", f"{current_url}: {exc}"))
            continue

        if page["status"] != "ok":
            warnings.append(core._warning("link_discovery_failed", "crawl4ai", page.get("error") or current_url))
            continue

        current_entry["title"] = page.get("title") or current_entry["title"]

        for link in page.get("links", []):
            link_url = link["url"]
            normalized_link = core._normalize_url(link_url)
            domain = core._domain_from_url(link_url).lower()

            if same_domain_only and core._registrable_domain(domain) != root_registrable:
                continue
            if include_patterns and not core._url_matches_patterns(link_url, include_patterns):
                continue
            if exclude_patterns and core._url_matches_patterns(link_url, exclude_patterns):
                continue

            if normalized_link not in discovered:
                if len(discovered) >= max_urls:
                    break
                discovered[normalized_link] = {
                    "url": link_url,
                    "normalized_url": normalized_link,
                    "domain": core._domain_from_url(link_url),
                    "title": link.get("title"),
                    "link_text": link.get("text"),
                    "depth": depth + 1,
                    "discovered_from": current_url,
                    "link_type": link.get("link_type", "internal"),
                }

            if depth + 1 <= max_depth and normalized_link not in visited_pages:
                queue.append((link_url, depth + 1, current_url))

    results = list(discovered.values())
    for rank, entry in enumerate(results, start=1):
        entry["rank"] = rank

    response = {
        "url": root_url,
        "results": results,
        "meta": {
            "max_urls_requested": max_urls,
            "max_depth": max_depth,
            "urls_returned": len(results),
            "pages_visited": len(visited_pages),
            "same_domain_only": same_domain_only,
            "warnings": warnings,
            "timings_ms": {
                "total": int((time.monotonic() - started) * 1000),
            },
        },
    }
    log.info(
        "map url=%s returned=%d visited=%d warnings=%d",
        root_url,
        len(results),
        len(visited_pages),
        len(warnings),
    )
    return response


async def crawl_impl(
    url: str,
    query: str | None = None,
    max_urls: int = 10,
    max_depth: int = 1,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    same_domain_only: bool = True,
    ctx: Context | None = None,
) -> dict:
    """Map a site then extract each discovered page. Returns a structured dict."""
    effective_max_urls = core._validate_positive_int(
        "max_urls",
        max_urls,
        maximum=min(_MAX_MAP_URLS, _MAX_EXTRACT_URLS),
    )
    started = time.monotonic()

    mapped = await map_impl(
        url=url,
        max_urls=effective_max_urls,
        max_depth=max_depth,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        same_domain_only=same_domain_only,
    )
    mapped_urls = [result["url"] for result in mapped["results"]]
    extracted = await extract_impl(urls=mapped_urls, query=query, ctx=ctx)

    extract_by_normalized_url = {
        result["normalized_url"]: result
        for result in extracted["results"]
    }

    results: list[dict] = []
    for mapped_result in mapped["results"]:
        extracted_result = extract_by_normalized_url.get(mapped_result["normalized_url"])
        if not extracted_result:
            continue
        combined = {
            **mapped_result,
            "status": extracted_result["status"],
            "content_type": extracted_result.get("content_type"),
            "content": extracted_result.get("content", ""),
            "chars_shown": extracted_result.get("chars_shown"),
            "offset": extracted_result.get("offset", 0),
            "total_chars": extracted_result.get("total_chars"),
            "top_chunks": extracted_result.get("top_chunks", []),
            "cached": extracted_result.get("cached", False),
            "error": extracted_result.get("error"),
        }
        if extracted_result.get("total_pages") is not None:
            combined["total_pages"] = extracted_result["total_pages"]
        if extracted_result.get("title"):
            combined["title"] = extracted_result["title"]
        results.append(combined)

    if query:
        results.sort(
            key=lambda result: max(
                (chunk.get("score") or 0.0) for chunk in result.get("top_chunks", [])
            ) if result.get("top_chunks") else float("-inf"),
            reverse=True,
        )
        for rank, result in enumerate(results, start=1):
            result["rank"] = rank

    response = {
        "url": mapped["url"],
        "query": query.strip() if query else None,
        "results": results,
        "meta": {
            "max_urls_requested": effective_max_urls,
            "max_depth": max_depth,
            "urls_discovered": mapped["meta"]["urls_returned"],
            "urls_succeeded": extracted["meta"]["urls_succeeded"],
            "urls_failed": extracted["meta"]["urls_failed"],
            "same_domain_only": same_domain_only,
            "warnings": [
                *mapped["meta"].get("warnings", []),
            ],
            "timings_ms": {
                "map": mapped["meta"]["timings_ms"]["total"],
                "extract": extracted["meta"]["timings_ms"]["total"],
                "total": int((time.monotonic() - started) * 1000),
            },
        },
    }
    log.info(
        "crawl url=%s discovered=%d succeeded=%d failed=%d",
        url,
        mapped["meta"]["urls_returned"],
        extracted["meta"]["urls_succeeded"],
        extracted["meta"]["urls_failed"],
    )
    return response
