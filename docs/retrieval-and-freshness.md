# Retrieval and freshness

Search retrieves a candidate pool larger than the requested result count. A
cheap title/snippet/URL rerank orders that pool before the bounded scrape budget
is spent. The second rerank operates on cleaned document chunks and returns only
the highest-value passages.

The lexical intent classifier selects source preferences for technical docs,
current events, academic research, products, factual lookups, comparisons, and
general research. Optional LFM planning may choose among those same profiles;
it cannot invent new ranking policy.

A time-bounded request (`day`, `week`, `month`, or `year`) uses age-limited
SearXNG/page cache records and does not merge page or semantic retrieval memory.
This prevents an old cached page from silently satisfying a freshness-sensitive
query. Unbounded searches may reuse shared memory and report its source in each
result.

## Staged search and page fetching

Ordinary searches explicitly select two primary SearXNG engines. Technical
queries start with Bing/Yandex and can broaden to GitHub, Stack Overflow and
Fynd. Current events start with Bing News/Google News; academic queries start
with Bing/arXiv; factual lookups start with Bing/Wikipedia. General queries
start with Bing/Yandex and can broaden to Fynd/Wiby. The engine groups live in
`config/routing.py` and must match the deployed SearXNG engine configuration.
SearXNG continues to manage engine suspension after upstream errors.

Fallback runs when the primary pool lacks at least `max(4, 2*num_results)`
relevant candidates across two hosts. A request restricted to one domain only
requires one host. Relevance uses the existing local reranker and its minimum
score; this is a retrieval heuristic, not a calibrated confidence estimate.
Fallback also runs when the first engine request fails. Each ordinary stage
requests page one; research retains query variants and up to three pages with
a broader engine pool. Primary and fallback candidates are interleaved before
the pool limit so fallback sources can reach the final reranker.

Fetching probes up to two ranked pages first, then fills the remaining target.
Rejected pages (including rate limits, access denials and login walls) retain
search snippets while the next candidates are fetched. The attempt limit is
`min(MAX_SCRAPE, max(2*target, target+2), candidate_count)`; successful fetches
stop at the target. If `MAX_SCRAPE` already equals the target, replacement has
no extra request allowance. Page and evidence caches remain shared.

Structured search metadata includes `retrieval_steps` (query, engines, page,
result count, failure flag), including cache-served retrievals. `scrape_top`
reports pages attempted, including failed attempts. These fields make it
possible to compare request budgets, source coverage and timing without
adding another model call to search planning.

A `time_range` constraint makes retrieval freshness-sensitive without replacing
an explicit documentation, academic, product or comparison intent with news.
