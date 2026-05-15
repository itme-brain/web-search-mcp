# Evaluation

This directory gives the project a repeatable way to measure retrieval changes.

## Files

- `queries.json`: benchmark query set
- `run_eval.py`: executes the local search implementation and writes JSONL results
- `score.py`: summarizes a saved run into simple metrics
- `benchmark_rerankers.py`: runs the query set once per reranker backend/model and compares scores

## Workflow

Run the benchmark:

```sh
just eval
```

That writes a JSONL file under `eval/runs/`.

Score a run:

```sh
just eval-score eval/runs/<timestamp>.jsonl
```

Compare rerankers:

```sh
just eval-rerankers
```

The default comparison runs:

- `flashrank=flashrank:ms-marco-MiniLM-L-12-v2`
- `minilm-l6=sentence-transformers:cross-encoder/ms-marco-MiniLM-L6-v2`

The current compose default is `sentence-transformers:cross-encoder/ms-marco-MiniLM-L4-v2`.
It matched MiniLM-L6/L12 usefulness on the bundled eval set while reducing
CPU rerank latency in local benchmarks.
Each benchmarked reranker is loaded in its own subprocess. Normal deploys
download/load only the single backend/model selected by `RERANK_BACKEND`
and `RERANK_MODEL`.

Pass explicit specs to compare other English rerankers:

```sh
just eval-rerankers \
  --spec flashrank=flashrank:ms-marco-MiniLM-L-12-v2 \
  --spec minilm-l12=sentence-transformers:cross-encoder/ms-marco-MiniLM-L12-v2
```

Each spec runs in a fresh Python process because the configured reranker is loaded at import time. Results are written under `eval/runs/rerankers/<timestamp>/`.

## What it measures today

- total latency from the MCP response metadata
- number of returned results
- number of scraped results
- degraded responses
- top-domain concentration in the top 3 and top 5
- expected-domain hits in the top 3 and top 5
- simple top-3 and top-5 usefulness scores against query judgments
- how often usefulness targets are met

## Judgment fields

Each query entry in `queries.json` can include:

- `expected_domains`: domains you would consider strong evidence of a good result set
- `freshness_sensitive`: marks queries where recency matters
- `top3_usefulness_target`: desired top-3 usefulness score
- `top5_usefulness_target`: desired top-5 usefulness score

Current usefulness scoring is intentionally simple:

- `0`: weak result set
- `1`: partially useful
- `2`: clearly useful

If `expected_domains` are present, usefulness is based on how many of those domains appear near the top.
If they are absent, usefulness falls back to whether the result set contains scraped content in the top results.

This is still a lightweight proxy, but it is enough to compare retrieval changes.

## What this gives you

This first pass is not a full semantic relevance benchmark. It gives a baseline for:

- duplicate pressure
- source diversity
- scrape coverage
- latency regressions
- expected-source coverage
- rough usefulness target tracking
- reranker latency/quality tradeoffs when run through `just eval-rerankers`

## How to extend it

Add optional manual labels to `queries.json` or to a separate judgments file later:

- expected domains
- freshness-sensitive queries
- relevant result count
- top-3 usefulness

The next upgrade would be per-query human judgments over actual runs, for example:

- which top-3 results were truly relevant
- whether freshness was acceptable
- whether excerpts were useful enough to answer the query

Once that exists, `score.py` can grow from a proxy benchmark into a real retrieval benchmark.
