"""Live smoke test: exercise each MCP tool against a running stack.

Run after `nix run .#deploy` (or `just up`). Hits the real SearXNG +
Crawl4AI so you can see exactly what markdown an LLM client will get.

Usage
-----
    nix develop -c python eval/live_smoke.py              # fast default suite
    nix develop -c python eval/live_smoke.py --full       # adds PDF + degraded spike
    nix develop -c python eval/live_smoke.py --only map   # run a single tool
    nix develop -c python eval/live_smoke.py --url http://localhost:8002/mcp

The script prints per-call latency, response size, any warnings/degraded
status parsed from the markdown header, and a preview of the output.
"""

import argparse
import asyncio
import time

from fastmcp import Client


DEFAULT_URL = "http://localhost:8002/mcp"
MAX_PREVIEW_CHARS = 1500
SEPARATOR = "=" * 72


TESTS = [
    {
        "name": "search — docs-scoped with include_domains",
        "tool": "search",
        "args": {
            "query": "python asyncio taskgroup",
            "include_domains": ["docs.python.org"],
            "num_results": 3,
        },
    },
    {
        "name": "search — site: prefix + time_range",
        "tool": "search",
        "args": {
            "query": "site:github.com model context protocol",
            "time_range": "month",
            "num_results": 3,
        },
    },
    {
        "name": "extract — html page",
        "tool": "extract",
        "args": {
            "urls": ["https://docs.python.org/3/library/asyncio-task.html"],
        },
    },
    {
        "name": "map — bounded discovery",
        "tool": "map",
        "args": {
            "url": "https://docs.pydantic.dev/latest",
            "max_urls": 5,
        },
    },
    {
        "name": "crawl — discover + extract",
        "tool": "crawl",
        "args": {
            "url": "https://docs.pydantic.dev/latest",
            "max_urls": 3,
        },
    },
]


FULL_EXTRAS = [
    {
        "name": "extract — PDF with per-page rerank",
        "tool": "extract",
        "args": {
            "urls": ["https://arxiv.org/pdf/2310.06825.pdf"],
            "query": "attention mechanism",
        },
    },
    {
        "name": "cache — cold vs warm extract latency",
        "tool": "__cache_warmup",
        "args": None,
    },
    {
        "name": "chunks — cherry-pick by id",
        "tool": "__chunk_ids",
        "args": None,
    },
    {
        "name": "degraded-mode spike",
        "tool": "__burst_search",
        "args": None,
    },
]


def _preview(text: str, *, max_chars: int = MAX_PREVIEW_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n... [truncated, total {len(text)} chars]"


def _parse_header(markdown: str) -> list[str]:
    """Lift the key: value lines from the first section (before ---)."""
    first_section, _, _ = markdown.partition("\n---\n")
    return [
        line.strip()
        for line in first_section.splitlines()
        if line.strip() and ":" in line
    ]


def _markdown_text(result) -> str:
    """Pull the markdown payload off a CallToolResult.

    `result.data` is a validated Pydantic model (output_schema path).
    The markdown lives in the first content block.
    """
    blocks = getattr(result, "content", None) or []
    if not blocks:
        return ""
    text = getattr(blocks[0], "text", None)
    return text or ""


async def _run_tool(client: Client, name: str, tool: str, args: dict) -> None:
    print(f"\n{SEPARATOR}\n[{name}]\n  {tool}({args})\n{SEPARATOR}")
    started = time.monotonic()
    try:
        result = await client.call_tool(tool, args)
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return
    elapsed_ms = int((time.monotonic() - started) * 1000)
    output = _markdown_text(result)
    print(f"  latency: {elapsed_ms} ms   bytes: {len(output)}")
    for line in _parse_header(output):
        print(f"  {line}")
    print()
    print(_preview(output))


async def _burst_search(client: Client) -> None:
    """Fire 5 searches in quick succession and collect engine warnings.

    Google tends to CAPTCHA under load — we expect engine_unresponsive
    warnings to show up in at least some of the responses. If none do,
    the system is either lightly loaded or your local IP is trusted.
    """
    print(f"\n{SEPARATOR}\n[degraded-mode spike: 5 consecutive searches]\n{SEPARATOR}")
    queries = [
        "global markets today",
        "breaking tech news",
        "trending python libraries",
        "latest ai announcements",
        "current llm context window records",
    ]
    engines_seen: set[str] = set()
    scrape_failures = 0
    for q in queries:
        try:
            result = await client.call_tool("search", {"query": q, "num_results": 3})
        except Exception as exc:
            print(f"  {q!r}: FAILED {exc}")
            continue
        output = _markdown_text(result)
        warnings_line = next(
            (line for line in _parse_header(output) if line.startswith("warnings:")),
            None,
        )
        if warnings_line:
            print(f"  {q!r}: {warnings_line}")
            for chunk in warnings_line.split(":", 1)[1].split(";"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if _looks_like_engine_warning(chunk):
                    engines_seen.add(chunk)
                else:
                    scrape_failures += 1
        else:
            print(f"  {q!r}: clean")
    print()
    if engines_seen:
        print(f"unique engine warnings across burst: {sorted(engines_seen)}")
    else:
        print("no engine warnings surfaced — all upstreams responsive this run")
    if scrape_failures:
        print(f"non-engine warnings across burst (scrape/rerank): {scrape_failures}")


async def _cache_warmup(client: Client) -> None:
    """Call extract twice on the same URL and compare latencies.

    The second call must hit Valkey and skip Crawl4AI — a drop of 3×
    or more is the signal that the shared cache is actually in play.
    URL is deliberately outside the other smoke cases so the first
    call is genuinely cold.
    """
    print(f"\n{SEPARATOR}\n[cache warmup: extract the same URL twice]\n{SEPARATOR}")
    url = "https://docs.python.org/3/library/itertools.html"

    started = time.monotonic()
    try:
        await client.call_tool("extract", {"urls": [url]})
    except Exception as exc:
        print(f"  cold call FAILED: {exc}")
        return
    cold_ms = int((time.monotonic() - started) * 1000)

    started = time.monotonic()
    try:
        result = await client.call_tool("extract", {"urls": [url]})
    except Exception as exc:
        print(f"  warm call FAILED: {exc}")
        return
    warm_ms = int((time.monotonic() - started) * 1000)

    cached_flag = False
    payload = (result.structured_content or {}).get("results") or []
    if payload:
        cached_flag = bool(payload[0].get("cached"))

    ratio = cold_ms / warm_ms if warm_ms else float("inf")
    print(f"  cold: {cold_ms} ms   warm: {warm_ms} ms   speedup: {ratio:.1f}x   cached={cached_flag}")
    if not cached_flag:
        print("  !! warm call did not report cached=true — Valkey may be bypassed")
    elif ratio < 3:
        print("  !! speedup < 3× — warm call is not reading from cache cleanly")


async def _chunk_ids(client: Client) -> None:
    """Extract a page, then re-request specific chunks by id."""
    print(f"\n{SEPARATOR}\n[chunks: cherry-pick by id]\n{SEPARATOR}")
    url = "https://docs.python.org/3/library/asyncio-task.html"
    try:
        first = await client.call_tool("extract", {"urls": [url], "query": "task group"})
    except Exception as exc:
        print(f"  initial extract FAILED: {exc}")
        return

    payload = (first.structured_content or {}).get("results") or []
    if not payload:
        print("  !! no results from initial extract")
        return
    chunks = payload[0].get("chunks") or []
    print(f"  initial extract returned {len(chunks)} chunks")
    if len(chunks) < 2:
        print("  !! need at least 2 chunks to exercise chunk_ids; skipping")
        return

    wanted = [chunks[0]["id"], chunks[-1]["id"]]
    started = time.monotonic()
    try:
        second = await client.call_tool("extract", {"urls": [url], "chunk_ids": wanted})
    except Exception as exc:
        print(f"  chunk_ids call FAILED: {exc}")
        return
    elapsed_ms = int((time.monotonic() - started) * 1000)

    second_payload = (second.structured_content or {}).get("results") or []
    if not second_payload:
        print("  !! no results from chunk_ids extract")
        return
    got = second_payload[0]
    cached_flag = bool(got.get("cached"))
    content = got.get("content") or ""
    print(f"  chunk_ids={wanted}   latency: {elapsed_ms} ms   cached={cached_flag}   chars: {len(content)}")
    if not cached_flag:
        print("  !! chunk_ids call did not hit cache — rerank/scrape was re-done")


def _looks_like_engine_warning(chunk: str) -> bool:
    """Engine warnings render as `<engine>: <reason>`; other warning types
    (scrape_failed = 'N of M pages failed') have no colon, or a multi-word
    prefix like 'page 2: ...'. Heuristic: first `:` must be preceded by a
    single non-whitespace token that looks like an engine identifier."""
    head, sep, _ = chunk.partition(":")
    if not sep:
        return False
    return bool(head) and " " not in head


async def _main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", default=DEFAULT_URL, help=f"MCP URL (default {DEFAULT_URL})")
    parser.add_argument("--only", metavar="TOOL",
                        help="Run only tests whose tool or name matches this substring")
    parser.add_argument("--full", action="store_true",
                        help="Run the extended suite (adds PDF + degraded spike)")
    args = parser.parse_args()

    tests = TESTS + (FULL_EXTRAS if args.full else [])
    if args.only:
        tests = [t for t in tests if args.only in t["tool"] or args.only in t["name"]]
        if not tests:
            parser.error(f"no tests matched --only={args.only!r}")

    print(f"live smoke: connecting to {args.url}")
    try:
        async with Client(args.url) as client:
            tools = await client.list_tools()
            print(f"tools exposed: {[t.name for t in tools]}")
            for test in tests:
                if test["tool"] == "__burst_search":
                    await _burst_search(client)
                elif test["tool"] == "__cache_warmup":
                    await _cache_warmup(client)
                elif test["tool"] == "__chunk_ids":
                    await _chunk_ids(client)
                else:
                    await _run_tool(client, test["name"], test["tool"], test["args"])
    except Exception as exc:
        print(f"\nFATAL: could not connect to MCP at {args.url}")
        print(f"  {type(exc).__name__}: {exc}")
        print("  is the stack up? try `just health` or `nix run .#deploy`.")
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(_main())
