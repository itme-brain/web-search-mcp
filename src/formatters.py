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


def _chunk_range(ids: list[int]) -> str | None:
    if not ids:
        return None
    ordered = sorted(set(ids))
    if len(ordered) == 1:
        return str(ordered[0])
    return f"{ordered[0]}..{ordered[-1]}"


def _rank_band(rank: int) -> str:
    if rank == 1:
        return "top match"
    if rank <= 3:
        return "strong match"
    return "possible match"


def _render_kv_block(items: list[tuple[str, str | None]]) -> str:
    lines = [f"{key}: {value}" for key, value in items if value]
    return "\n".join(lines)


def _tree_line(entry: dict) -> str:
    title = entry.get("title") or entry.get("link_text") or entry.get("url", "")
    url = entry.get("url", "")
    depth = entry.get("depth", 0)
    indent = "  " * depth
    suffix = " (ext)" if entry.get("link_type") == "external" else ""
    return f"{indent}- [{title}]({url}){suffix}"


def _tree_block(entries: list[dict]) -> str:
    return "\n".join(_tree_line(entry) for entry in entries)


def _format_search_results(response: dict) -> str:
    """Format search response as markdown for LLM consumption."""
    parts = [("query", response["query"])]
    if response.get("time_range"):
        parts.append(("time_range", response["time_range"]))
    meta = response.get("meta", {})
    parts.append(("results", str(meta.get("num_results_returned", len(response.get("results", []))))))
    if meta.get("degraded"):
        parts.append(("status", "degraded"))
    warnings = meta.get("warnings", [])
    if warnings:
        parts.append(("warnings", "; ".join(w.get("detail", str(w)) for w in warnings)))
    header = _render_kv_block(parts)

    sections = [header, "---"]
    for r in response.get("results", []):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        content = r.get("content", "")
        meta_parts: list[str] = []
        rank = r.get("rank")
        title_prefix = f"{rank}. " if isinstance(rank, int) else ""
        domain = r.get("domain")
        if domain:
            meta_parts.append(domain)
        metadata = r.get("metadata") or {}
        if isinstance(metadata, dict) and metadata.get("date"):
            meta_parts.append(str(metadata["date"]))
        if isinstance(rank, int):
            meta_parts.append(_rank_band(rank))
        meta_line = "_{}_".format(" | ".join(meta_parts)) if meta_parts else ""
        if content:
            section = f"## {title_prefix}[{title}]({url})"
            if meta_line:
                section += f"\n\n{meta_line}"
            section += f"\n\n{content}"
        else:
            section = f"## {title_prefix}[{title}]({url})"
            if meta_line:
                section += f"\n\n{meta_line}"
        sections.append(section)

    return _render_markdown(sections)


def _format_extract_results(response: dict) -> str:
    """Format extract response as markdown."""
    parts: list[tuple[str, str | None]] = []
    if response.get("query"):
        parts.append(("query", response["query"]))
    results = response.get("results", [])
    meta = response.get("meta", {})
    succeeded = meta.get("urls_succeeded", sum(1 for r in results if r.get("status") == "ok"))
    failed = meta.get("urls_failed", sum(1 for r in results if r.get("status") != "ok"))
    if len(results) != 1 or failed:
        parts.append(("succeeded", str(succeeded)))
    if failed:
        parts.append(("failed", str(failed)))
    header = _render_kv_block(parts) if parts else ""

    sections = [header, "---"] if header else []
    for r in results:
        sections.append(_format_document_section(r))

    return _render_markdown(sections)


def _format_document_section(r: dict) -> str:
    """Render one extracted-document section as markdown.

    Shared by the extract + crawl formatters. Surfaces compact metadata
    about content type, chunk coverage, and content length."""
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

    meta_parts: list[str] = []
    file_type = r.get("file_type")
    content_type = r.get("content_type")
    if file_type and file_type != "html":
        meta_parts.append(f"type: {file_type}")
    elif content_type and content_type != "text/html":
        meta_parts.append(f"content_type: {content_type}")

    shown_chunk_ids = r.get("shown_chunk_ids", []) or []
    total_chunks = r.get("total_chunks")
    chunk_mode = r.get("chunk_mode")
    chunk_span = _chunk_range(shown_chunk_ids)
    if total_chunks:
        if chunk_span:
            meta_parts.append(f"chunks: {chunk_span} of 0..{total_chunks - 1}")
        else:
            meta_parts.append(f"chunks: 0..{total_chunks - 1}")
    if chunk_mode == "relevant":
        meta_parts.append("mode: relevant")
    elif chunk_mode == "document":
        meta_parts.append("mode: document")
    elif chunk_mode == "selected":
        meta_parts.append("mode: selected")

    total_chars = r.get("total_chars")
    chars_shown = r.get("chars_shown", len(content))
    if total_chars and chars_shown:
        meta_parts.append(f"{chars_shown:,} of {total_chars:,} chars")
    if meta_parts:
        meta_line = " | ".join(meta_parts)
        if content or status in {"error", "handoff"}:
            section = section.replace("\n\n", f"\n\n_{meta_line}_\n\n", 1)
        else:
            section += f"\n\n_{meta_line}_"
    return section


def _format_map_results(response: dict) -> str:
    """Format map response as a site tree."""
    sections: list[str] = []
    meta = response.get("meta", {})
    warnings = meta.get("warnings", [])
    if warnings:
        sections.append("warnings: " + "; ".join(w.get("detail", str(w)) for w in warnings))
        sections.append("---")
    sections.append(_tree_block(response.get("results", [])))

    return _render_markdown(sections)


def _format_crawl_results(response: dict) -> str:
    """Format crawl response as summary + map block + flat document sections."""
    meta = response.get("meta", {})
    warnings = meta.get("warnings", [])
    sections: list[str] = []
    if warnings:
        sections.append("warnings: " + "; ".join(w.get("detail", str(w)) for w in warnings))

    summary = _render_kv_block([
        ("discovered", str(meta.get("urls_discovered", 0))),
        ("extracted", str(meta.get("urls_succeeded", 0))),
        ("failed", str(meta.get("urls_failed", 0)) if meta.get("urls_failed") else None),
    ])
    if summary:
        sections.append(summary)

    tree = _tree_block(response.get("results", []))
    if tree:
        sections.append("map:\n" + tree)

    if sections:
        sections.append("---")

    for r in response.get("results", []):
        sections.append(_format_document_section(r))

    return _render_markdown(sections)
