import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _load_server_module():
    """Import impls from the src/ package path.

    Project layout is flat files in src/ that cross-import each other;
    adding src/ to sys.path lets us import them as top-level modules.
    """
    import impls  # noqa: F401  (triggers core + ranker init)
    import server  # noqa: F401
    return server


def _load_queries(path: Path) -> list[dict]:
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("queries file must contain a JSON array")
    return data


async def _run_query(server_module, query_spec: dict) -> dict:
    started_at = datetime.now(UTC).isoformat()
    payload = await server_module.search_impl(
        query=query_spec["query"],
        num_results=query_spec.get("num_results", 5),
        time_range=query_spec.get("time_range"),
        include_domains=query_spec.get("include_domains"),
        exclude_domains=query_spec.get("exclude_domains"),
        ctx=None,
    )
    return {
        "id": query_spec["id"],
        "query": query_spec["query"],
        "started_at": started_at,
        "request": {
            "num_results": query_spec.get("num_results", 5),
            "time_range": query_spec.get("time_range"),
            "include_domains": query_spec.get("include_domains", []),
            "exclude_domains": query_spec.get("exclude_domains", []),
        },
        "notes": query_spec.get("notes", ""),
        "judgments": query_spec.get("judgments", {}),
        "response": payload,
    }


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark queries against the local web-search implementation.")
    parser.add_argument(
        "--queries",
        default=str(ROOT / "eval" / "queries.json"),
        help="Path to benchmark query JSON file.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "eval" / "runs" / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.jsonl"),
        help="Path to write JSONL results.",
    )
    args = parser.parse_args()

    queries_path = Path(args.queries)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    server_module = _load_server_module()
    queries = _load_queries(queries_path)

    with output_path.open("w") as fh:
        for query_spec in queries:
            result = await _run_query(server_module, query_spec)
            fh.write(json.dumps(result) + "\n")
            print(f"ran {query_spec['id']}: {query_spec['query']}")

    print(f"wrote {len(queries)} results to {output_path}")


if __name__ == "__main__":
    asyncio.run(_main())
