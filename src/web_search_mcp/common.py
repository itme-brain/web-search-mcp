"""Shared URL, validation, deduplication, and warning helpers."""

from collections import defaultdict
import fnmatch
from urllib.parse import urlparse

from rapidfuzz import fuzz

from web_search_mcp.config.settings import _TITLE_DEDUP_THRESHOLD
from web_search_mcp.extraction import text_utils
from web_search_mcp.http import urls, validators
def _warning(error_type: str, source: str, detail: str) -> dict:
    """Create a structured warning dict for programmatic error handling."""
    return {"type": error_type, "source": source, "detail": detail}


def _dedup_unresponsive_engines(entries: list) -> list[tuple[str, str]]:
    """Normalize SearXNG's unresponsive_engines shapes and dedup by (engine, reason).

    SearXNG ships entries as `[engine_name, error_string]` lists; some
    integrations wrap them as dicts. Return a deduped list of
    `(engine, reason)` tuples in insertion order, dropping empty engine names.
    """
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for entry in entries:
        if isinstance(entry, (list, tuple)) and entry:
            engine = str(entry[0])
            reason = str(entry[1]) if len(entry) > 1 else ""
        elif isinstance(entry, dict):
            engine = str(entry.get("name") or entry.get("engine") or "")
            reason = str(entry.get("error") or entry.get("reason") or "")
        else:
            continue
        if not engine:
            continue
        key = (engine, reason)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


# ---------------------------------------------------------------------------
# URL + text utilities
# ---------------------------------------------------------------------------
def _normalize_url(url: str) -> str:
    return urls.normalize_url(url)


def _dedup_results(results: list[dict]) -> list[dict]:
    seen_urls: set[str] = set()
    seen_titles: dict[str, list[str]] = defaultdict(list)
    deduped: list[dict] = []
    for r in results:
        norm = _normalize_url(r.get("url", ""))
        domain = _domain_from_url(r.get("url", "")).lower()
        title = urls.normalize_title(r.get("title", ""))
        if norm in seen_urls:
            continue
        if title and any(fuzz.ratio(title, seen) >= _TITLE_DEDUP_THRESHOLD for seen in seen_titles[domain]):
            continue
        seen_urls.add(norm)
        if title:
            seen_titles[domain].append(title)
        deduped.append(r)
    return deduped


def _dedup_chunks(chunks: list[str], entry_map: list[int]) -> tuple[list[str], list[int]]:
    return text_utils.dedup_chunks(chunks, entry_map)


def _dedup_pages(entries: list[dict], *, min_chars: int = 200) -> tuple[list[dict], int]:
    return text_utils.dedup_pages(entries, min_chars=min_chars)


def _domain_from_url(url: str) -> str:
    return urls.domain_from_url(url)


def _canonical_hostname(host: str) -> str:
    return urls.canonical_hostname(host)


def _registrable_domain(domain: str) -> str:
    return urls.registrable_domain(domain)


def _chunk_text(text: str) -> list[str]:
    return text_utils.chunk_text(text)


def _diversify_ranked_entries(ranked_entry_idxs: list[int], entries: list[dict]) -> list[int]:
    """Interleave domains so the top results are not monopolized by one source."""
    by_domain: dict[str, list[int]] = defaultdict(list)
    domain_order: list[str] = []
    for eidx in ranked_entry_idxs:
        domain = _domain_from_url(entries[eidx]["url"]).lower() or entries[eidx]["url"]
        if domain not in by_domain:
            domain_order.append(domain)
        by_domain[domain].append(eidx)

    diversified: list[int] = []
    while by_domain:
        next_round: list[str] = []
        for domain in domain_order:
            queue = by_domain.get(domain)
            if not queue:
                continue
            diversified.append(queue.pop(0))
            if queue:
                next_round.append(domain)
            else:
                by_domain.pop(domain, None)
        domain_order = next_round
    return diversified


# ---------------------------------------------------------------------------
# Validators + normalizers
# ---------------------------------------------------------------------------
def _coerce_optional_str(value: str | None) -> str | None:
    return validators.coerce_optional_str(value)


def _normalize_time_range(time_range: str | None) -> str | None:
    return validators.normalize_time_range(time_range)


def _validate_query(query: str) -> str:
    return validators.validate_query(query)


def _validate_positive_int(name: str, value: int, *, maximum: int) -> int:
    return validators.validate_positive_int(name, value, maximum=maximum)


def _validate_urls(urls: list[str], *, maximum: int) -> list[str]:
    return validators.validate_urls(urls, maximum=maximum)


def _is_blocked_ip(ip) -> bool:
    return validators.is_blocked_ip(ip)


def _reject_non_public_target(hostname: str | None, *, url: str) -> None:
    return validators.reject_non_public_target(hostname, url=url)


def _normalize_domains(domains: list[str] | None, *, field_name: str) -> list[str]:
    return validators.normalize_domains(domains, field_name=field_name)


def _normalize_glob_patterns(patterns: list[str] | None, *, field_name: str) -> list[str]:
    return validators.normalize_glob_patterns(patterns, field_name=field_name)


def _match_domain(domain: str, patterns: list[str]) -> bool:
    return validators.match_domain(domain, patterns)


def _filter_results_by_domain(
    results: list[dict],
    include_domains: list[str],
    exclude_domains: list[str],
) -> list[dict]:
    filtered: list[dict] = []
    for result in results:
        host = _canonical_hostname(_domain_from_url(result.get("url", "")))
        if include_domains and not _match_domain(host, include_domains):
            continue
        if exclude_domains and _match_domain(host, exclude_domains):
            continue
        filtered.append(result)
    return filtered


def _url_matches_patterns(url: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(url, pattern) for pattern in patterns)
