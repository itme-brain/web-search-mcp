"""Small-model retrieval smoke evaluator.

Runs representative queries directly against search_impl and reports rough
answerability signals: output size, source types, warnings, and whether any
brief/highlights were produced. It is intentionally lightweight and does not
require an LLM judge.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from web_search_mcp.tools.search import search_impl  # noqa: E402

QUERIES = [
    "current Python pathlib Path.walk docs",
    "which projects used reposurgeon for SVN to Git migration",
    "find FastMCP streamable HTTP client configuration docs",
    "what changed in Crawl4AI 0.8 release",
    "site:docs.python.org asyncio TaskGroup cancellation behavior",
]


async def main() -> None:
    rows = []
    for query in QUERIES:
        result = await search_impl(query, num_results=5)
        rendered_chars = sum(len(r.get("content", "")) for r in result.get("results", []))
        rows.append({
            "query": query,
            "results": len(result.get("results", [])),
            "brief_items": len(result.get("meta", {}).get("brief", [])),
            "source_types": sorted({r.get("source_type") for r in result.get("results", []) if r.get("source_type")}),
            "warnings": [w.get("type") for w in result.get("meta", {}).get("warnings", [])],
            "content_chars": rendered_chars,
        })
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
