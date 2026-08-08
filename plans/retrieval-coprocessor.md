# Retrieval Coprocessor Implementation Plan

## Scope

Implement the focused improvement plan except for the end-to-end evaluation
harness. Preserve `extract(url)` for arbitrary URLs and keep every generative
preprocessing feature optional with deterministic fallback behavior.

## Interface contracts

- `RetrievalPlan` is the single source of truth for intent, query variants,
  source preferences, and candidate/scrape/result budgets.
- `FreshnessPolicy` is passed to every live/cache/memory retrieval boundary.
  Semantic memory is disabled for time-bounded requests until its publication
  time can be verified.
- Scraped documents are persisted under content-addressed document and chunk
  IDs. Any MCP instance can resolve those IDs from Valkey.
- Search returns compact passages once. Each scraped passage may carry a
  document ID, chunk ID, and MCP resource URI.
- `extract(url)` remains unchanged. `read_evidence(reference)` is the tool-only
  compatibility path for clients without MCP Resources support.
- MCP HTTP is stateless. Long-running tools support optional MCP Tasks backed
  by Valkey, while synchronous calls continue to work.
- LFM preprocessing uses an OpenAI-compatible endpoint, defaults to
  `LiquidAI/LFM2.5-2.6B`, is disabled by default, validates structured output,
  and falls back to deterministic planning/evidence shaping on any failure.

## Implementation waves

1. Intent-aware retrieval plan and pre-scrape candidate reranking.
2. Strict freshness and cache/memory timestamp semantics.
3. Durable evidence manifests, resources, links, and targeted expansion.
4. Stateless MCP, constrained schemas, tasks, and compact output contracts.
5. Optional LFM query planning and evidence digest generation.
6. Configuration, Nix/just workflows, module documentation, and smoke repair.
7. Targeted tests, full suite, local MCP protocol tests, and bounded read-only
   integration tests against `192.168.0.23`.

## Acceptance criteria

1. Candidate ranking occurs before scrape selection and uses a larger pool
   than the final result count.
2. A request with `time_range` cannot receive semantic-memory evidence and
   cannot reuse an unverifiably old page snapshot.
3. Stable document/chunk references resolve after a new MCP process handles
   the request.
4. Research query variants and source preferences differ by detected intent.
5. Tool schemas expose enum/range/list constraints and read-only annotations.
6. Research and crawl support optional durable tasks without requiring them.
7. Search/research no longer duplicate passages across content, brief,
   findings, answer, summary, and key-evidence fields.
8. LFM preprocessing can be enabled against a compatible endpoint and fails
   open to deterministic behavior.
9. Existing extraction, map, crawl, cache, and reranker behaviors remain
   covered; all tests pass.
10. No evaluation-harness implementation is added or merged.

## Risks

- `breaking-change`: compact structured search output removes redundant fields.
- `external-api`: SearXNG, Crawl4AI, and optional LFM endpoint calls.
- `security`: untrusted web content and resource identifiers cross boundaries.
- `concurrent`: in-flight retrieval and background tasks may run concurrently.
- `data-mutation`: Valkey gains durable evidence and task records.

## Out of scope

- Deploying to or changing `192.168.0.23`.
- Switching the running llama.cpp model.
- Adding or merging the full benchmark suite.
- Paid search or model APIs.
