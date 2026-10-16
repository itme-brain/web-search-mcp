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

A `time_range` constraint stays strict without replacing a documentation,
academic, product or comparison topic with news. General engines are queried
without an unsupported upstream date parameter, then sources are filtered
locally. News routes still pass the filter to Bing News when querying it alone;
other routes can contribute dated evidence even when Bing News is unavailable.
Date-filtered ordinary searches consult both retrieval stages because snippet
relevance alone cannot establish date coverage.

Only sources with a parseable date within the requested rolling window are
returned: 1, 7, 30 or 365 days for day/week/month/year. Unknown, invalid, old and
future dates are excluded. Search-index publication dates take precedence over
extracted page dates. These are detected source dates, not independently
verified publication claims. Retrieval/cache timestamps never qualify a source.
`meta.date_filter` reports the cutoff, window end and exclusion counts;
`retrieval_steps[].upstream_time_range` records native engine filtering.
Empty results can mean dates were missing or out of range; they do not establish
that no recent sources exist on the web. Date-eligible fetches count toward the
fetch target, with the same hard attempt cap as unfiltered searches.

`degraded` indicates an unrecovered retrieval failure reducing the requested
coverage, or a failed ranking stage. A rejected page or unavailable engine stays
visible as a warning; when other sources fill the response, that warning alone
does not recommend a retry. Inspect `scraped`, snippets and warning details to
decide whether an individual source needs direct extraction.

Sparse crawl warnings describe discovery from the supplied starting URL. For a
possible leaf page they suggest `extract` for that document or a parent/index
URL for multi-page crawling. They do not infer that the whole site is sparse,
and the crawler never broadens its scope automatically.

Recognizable Cloudflare/challenge shells are rejected as access errors, including
HTTP-200 shells and previously cached shells. Failed cached extracts carry a
non-empty error and no challenge body. Articles discussing Cloudflare remain
valid documents.
