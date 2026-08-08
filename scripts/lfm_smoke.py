"""Bounded compatibility smoke for an LFM/OpenAI-compatible endpoint."""

import argparse
import asyncio
import json
import os


async def _run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="LFM2.5-2.6B-Q8_0.gguf")
    args = parser.parse_args()

    os.environ["ENABLE_LFM_PREPROCESSING"] = "1"
    os.environ["LFM_BASE_URL"] = args.base_url.rstrip("/")
    os.environ["LFM_MODEL"] = args.model

    from web_search_mcp.preprocessing.lfm import digest_evidence, plan_query
    from web_search_mcp.ranking.intent import classify

    query = "Compare FastMCP 4 background tasks with FastMCP 3"
    plan = await plan_query(query, classify(query), [query])
    if plan.error:
        print(json.dumps({"status": "error", "stage": "planning", "detail": plan.error}))
        return 1

    digest = await digest_evidence(
        query,
        [{
            "title": "Synthetic protocol evidence",
            "url": "https://example.invalid/protocol",
            "passages": [{
                "citation": "1.1",
                "text": "FastMCP 4 exposes background execution through a task extension.",
            }],
        }],
        ["fallback"],
    )
    if digest.error:
        print(json.dumps({"status": "error", "stage": "digest", "detail": digest.error}))
        return 1
    print(json.dumps({
        "status": "ok",
        "planning_used": plan.used,
        "query_count": len(plan.queries),
        "digest_used": digest.used,
        "overview_count": len(digest.overview),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
