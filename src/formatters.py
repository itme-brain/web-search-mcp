"""Markdown rendering for tool responses.

Every `@mcp.tool` wrapper calls one of these to convert its impl's
structured dict into the string the LLM consumes. Kept separate from
the impls so the dict shape is independently testable and the Python
scripting layer (`from impls import search_impl`) gets untouched dicts.
"""

import mdformat


def _render_markdown(sections: list[str]) -> str:
    raw = "\n\n".join(section for section in sections if section)
    return mdformat.text(raw).rstrip()


def _format_search_results(response: dict) -> str:
    """Format search response as markdown for LLM consumption."""
    parts = [f"query: {response['query']}"]
    if response.get("time_range"):
        parts.append(f"time_range: {response['time_range']}")
    meta = response.get("meta", {})
    parts.append(f"results: {meta.get('num_results_returned', len(response.get('results', [])))}")
    if meta.get("degraded"):
        parts.append("status: degraded")
    warnings = meta.get("warnings", [])
    if warnings:
        parts.append("warnings: " + "; ".join(w.get("detail", str(w)) for w in warnings))
    header = "\n".join(parts)

    sections = [header, "---"]
    for r in response.get("results", []):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        content = r.get("content", "")
        section = f"## [{title}]({url})\n\n{content}" if content else f"## [{title}]({url})"
        sections.append(section)

    return _render_markdown(sections)


def _format_extract_results(response: dict) -> str:
    """Format extract response as markdown."""
    parts = []
    if response.get("query"):
        parts.append(f"query: {response['query']}")
    results = response.get("results", [])
    meta = response.get("meta", {})
    succeeded = meta.get("urls_succeeded", sum(1 for r in results if r.get("status") == "ok"))
    failed = meta.get("urls_failed", sum(1 for r in results if r.get("status") != "ok"))
    parts.append(f"succeeded: {succeeded}")
    if failed:
        parts.append(f"failed: {failed}")
    header = "\n".join(parts)

    sections = [header, "---"] if parts else ["---"]
    for r in results:
        sections.append(_format_document_section(r))

    return _render_markdown(sections)


def _format_document_section(r: dict) -> str:
    """Render one extracted-document section as markdown.

    Shared by the extract + crawl formatters. Emits the truncation signal
    when the caller would benefit from a follow-up `offset=N` call."""
    title = r.get("title") or "Untitled"
    url = r.get("url", "")
    content = r.get("content", "")
    status = r.get("status", "")
    if status == "error":
        error = r.get("error", "extraction failed")
        section = f"## [{title}]({url})\n\n**Error:** {error}"
    elif status == "handoff":
        handoff = r.get("handoff") or {}
        handler = handoff.get("handler", "files")
        reason = handoff.get("reason", "delegated to another MCP")
        file_type = r.get("file_type") or "file"
        section = (
            f"## [{title}]({url})\n\n"
            f"**Handoff:** `{file_type}` content should be handled by the `{handler}` MCP.\n\n"
            f"{reason}"
        )
    elif content:
        section = f"## [{title}]({url})\n\n{content}"
    else:
        section = f"## [{title}]({url})"
    # Truncation signal: tell the LLM exactly where to resume via offset.
    offset = r.get("offset", 0) or 0
    total_chars = r.get("total_chars")
    chars_shown = r.get("chars_shown", len(content))
    if total_chars and offset + chars_shown < total_chars:
        next_offset = offset + chars_shown
        section += (
            f"\n\n_{offset:,}–{next_offset:,} of {total_chars:,} chars shown — "
            f"pass `offset={next_offset}` to continue._"
        )
    elif total_chars and offset > 0:
        section += f"\n\n_end of document ({total_chars:,} chars total)._"
    return section


def _format_map_results(response: dict) -> str:
    """Format map response as markdown."""
    parts = [f"url: {response['url']}"]
    meta = response.get("meta", {})
    parts.append(f"urls_found: {meta.get('urls_returned', len(response.get('results', [])))}")
    warnings = meta.get("warnings", [])
    if warnings:
        parts.append("warnings: " + "; ".join(w.get("detail", str(w)) for w in warnings))
    header = "\n".join(parts)

    sections = [header, "---"]
    for r in response.get("results", []):
        title = r.get("title") or r.get("link_text") or r.get("url", "")
        url = r.get("url", "")
        depth = r.get("depth", 0)
        indent = "  " * depth
        sections.append(f"{indent}- [{title}]({url})")

    return _render_markdown(sections)


def _format_crawl_results(response: dict) -> str:
    """Format crawl response as markdown."""
    parts = [f"url: {response['url']}"]
    meta = response.get("meta", {})
    discovered = meta.get("urls_discovered", 0)
    returned = meta.get("urls_returned", meta.get("urls_succeeded", 0))
    parts.append(f"pages: {returned} returned of {discovered} discovered")
    if meta.get("urls_truncated_by_limit"):
        parts.append(f"note: {meta['urls_truncated_by_limit']} additional page(s) available — increase max_urls to see more")
    if meta.get("urls_failed"):
        parts.append(f"failed: {meta['urls_failed']}")
    warnings = meta.get("warnings", [])
    if warnings:
        parts.append("warnings: " + "; ".join(w.get("detail", str(w)) for w in warnings))
    header = "\n".join(parts)

    sections = [header, "---"]
    for r in response.get("results", []):
        sections.append(_format_document_section(r))

    return _render_markdown(sections)
