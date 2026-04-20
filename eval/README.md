# Evaluation

This directory gives the project a repeatable way to measure retrieval changes.

## Files

- `queries.json`: benchmark query set
- `run_eval.py`: executes the local search implementation and writes JSONL results
- `score.py`: summarizes a saved run into simple metrics

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
