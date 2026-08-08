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
