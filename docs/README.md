# Architecture notes

The retrieval pipeline is intentionally deterministic at its core:

1. classify intent and generate bounded query variants;
1. collect a broad SearXNG candidate pool;
1. rerank title/snippet candidates before spending scrape budget;
1. scrape selected pages and merge eligible retrieval memory;
1. rerank page chunks and persist complete evidence in Valkey;
1. return compact passages plus stable MCP document/chunk resource handles.

Time-bounded searches bypass semantic/page memory and accept only cache entries
within the configured freshness window. Any MCP replica can resolve persisted
evidence because handles contain stable content hashes and payloads live in
shared Valkey storage.

Optional LFM preprocessing is described in [preprocessing-lfm.md](preprocessing-lfm.md).
It never replaces extraction, retrieval, reranking, or stored evidence.

- [Retrieval and freshness](retrieval-and-freshness.md)
- [Evidence resources](evidence-resources.md)
- [Production canary](production-canary.md)
