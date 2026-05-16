"""HTML-to-markdown extraction and document metadata helpers."""

import hashlib
import re
from urllib.parse import urljoin

import trafilatura

from web_search_mcp.config.settings import _MARKDOWN_LINK, _PILCROW_LINK, _WHITESPACE
from web_search_mcp.common import _normalize_url
_TABLE_SEPARATOR_ROW = re.compile(r"^\s*\|?\s*(:?-{3,}:?\s*\|\s*)+:?-{3,}:?\s*\|?\s*$")


def _strip_table_separator_rows(text: str) -> str:
    """Drop `|---|---|` noise rows while leaving the surrounding table intact."""
    if "|" not in text or "---" not in text:
        return text
    return "\n".join(
        line for line in text.splitlines() if not _TABLE_SEPARATOR_ROW.match(line)
    )


def _is_link_soup_line(line: str) -> bool:
    """Detect dense nav/TOC lines that are mostly markdown links."""
    matches = list(_MARKDOWN_LINK.finditer(line))
    if len(matches) < 4:
        return False
    residue = _MARKDOWN_LINK.sub("", line)
    residue = residue.replace("`", "").replace("*", "").replace("_", "")
    residue = _WHITESPACE.sub("", residue)
    return len(residue) <= 12


def _clean_extracted_markdown(text: str | None) -> str | None:
    """Light cleanup for site chrome that leaks through extraction."""
    if not text:
        return text
    cleaned = _PILCROW_LINK.sub("", text)
    lines = [line.rstrip() for line in cleaned.splitlines()]
    kept: list[str] = []
    blank_streak = 0
    for line in lines:
        stripped = line.strip()
        if stripped and _is_link_soup_line(stripped):
            continue
        if not stripped:
            blank_streak += 1
            if blank_streak > 1:
                continue
        else:
            blank_streak = 0
        kept.append(line)
    cleaned = "\n".join(kept).strip("\n")
    return cleaned or None


def _extract_markdown(result: dict) -> str | None:
    html = result.get("html")
    if html:
        try:
            # output_format="markdown" preserves structure the txt format
            # throws away: code gets properly fenced with ```, headings
            # carry their '#' prefix, bold/italic survive. This is what
            # the LLM reads AND what _extract_structure walks for the
            # structural metadata (headings, code_blocks, outgoing_links).
            extracted = trafilatura.extract(
                html, output_format="markdown", include_links=True, include_tables=True,
            )
            if extracted and len(extracted.strip()) >= 50:
                return _clean_extracted_markdown(_strip_table_separator_rows(extracted))
        except Exception:
            pass

    md = result.get("markdown")
    content: str | None
    if isinstance(md, dict):
        content = md.get("fit_markdown") or md.get("raw_markdown")
    elif isinstance(md, str):
        content = md
    else:
        content = result.get("cleaned_html")
    return _clean_extracted_markdown(_strip_table_separator_rows(content)) if content else content


def _extract_html_metadata(html: str | None) -> dict:
    """Pull author/date/site_name/description from raw HTML via trafilatura."""
    if not html:
        return {}
    try:
        doc = trafilatura.extract_metadata(html)
    except Exception:
        return {}
    if doc is None:
        return {}
    return {
        "author": doc.author or None,
        "date": doc.date or None,
        "site_name": doc.sitename or None,
        "description": doc.description or None,
    }


def _content_hash(content: str) -> str:
    """Stable fingerprint for exact-duplicate detection at write time."""
    import hashlib
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


def _extract_canonical_url(html: str | None) -> str | None:
    if not html:
        return None
    match = re.search(r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)["\']', html, re.IGNORECASE)
    return match.group(1).strip() if match else None


def _detect_language(content: str | None) -> str | None:
    """Tiny no-dependency language hint for diagnostics, not ranking."""
    if not content:
        return None
    sample = content[:4000]
    ascii_letters = sum(1 for ch in sample if ch.isascii() and ch.isalpha())
    letters = sum(1 for ch in sample if ch.isalpha())
    if letters and ascii_letters / letters > 0.85:
        return "en"
    return None


def _build_document_metadata(
    html: str | None,
    content: str | None,
    *,
    requested_url: str | None = None,
    final_url: str | None = None,
) -> dict:
    """Citation metadata for the response. Structural info (headings,
    code blocks, links) is intentionally not duplicated here — it's
    already present inline in the markdown body the LLM receives."""
    metadata = _extract_html_metadata(html)
    canonical_url = _extract_canonical_url(html)
    if canonical_url:
        metadata["canonical_url"] = urljoin(final_url or requested_url or "", canonical_url)
    if final_url and requested_url and _normalize_url(final_url) != _normalize_url(requested_url):
        metadata["final_url"] = final_url
    if content:
        metadata["word_count"] = len(content.split())
        language = _detect_language(content)
        if language:
            metadata["language"] = language
    return {k: v for k, v in metadata.items() if v is not None}


