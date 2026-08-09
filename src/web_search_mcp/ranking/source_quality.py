"""Source classification, filtering, and lightweight ranking boosts."""

from collections import defaultdict
import re
from urllib.parse import urlparse

from web_search_mcp.common import _domain_from_url

_PRIMARY_DOMAINS = {
    "developer.mozilla.org", "docs.python.org", "go.dev", "doc.rust-lang.org",
    "nodejs.org", "react.dev", "docs.github.com", "kubernetes.io",
    "www.w3.org", "ietf.org", "www.rfc-editor.org", "docs.docker.com",
    "docs.rs", "tokio.rs", "async.rs",
}

_LOW_QUALITY_DOMAINS = {
    "geeksforgeeks.org", "tutorialspoint.com", "w3schools.com",
    "copyprogramming.com", "programmersought.com", "issueantenna.com",
}

_SOCIAL_DOMAINS = {
    "facebook.com", "linkedin.com", "medium.com", "quora.com", "x.com",
    "www.facebook.com", "www.linkedin.com", "www.quora.com", "www.x.com",
}

_QUERY_ENTITY_STOPWORDS = frozenset({
    "about", "and", "async", "between", "compare", "comparison", "differences",
    "documentation", "for", "from", "latest", "library", "runtime", "rust",
    "the", "their", "versus", "what", "which", "with",
})

SOURCE_TYPE_ALIASES = {
    "docs": "official_docs",
    "official_docs": "official_docs",
    "repo": "repo",
    "github": "repo",
    "gitlab": "repo",
    "issue": "issue_tracker",
    "issues": "issue_tracker",
    "issue_tracker": "issue_tracker",
    "mail": "mailing_list",
    "mailing_list": "mailing_list",
    "qa": "qa",
    "stackoverflow": "qa",
    "blog": "blog",
    "pdf": "pdf",
    "paper": "pdf",
    "web": "web",
}


def source_type(url: str) -> str:
    domain = _domain_from_url(url).lower()
    path = urlparse(url).path.lower()
    if domain in _PRIMARY_DOMAINS:
        return "official_docs"
    if domain in {"github.com", "gitlab.com", "bitbucket.org", "sourceforge.net"}:
        if "/issues" in path or "/-/issues" in path:
            return "issue_tracker"
        return "repo"
    if path.endswith(".pdf"):
        return "pdf"
    if domain in _SOCIAL_DOMAINS:
        return "web"
    if any(part in domain for part in ("medium.com", "substack.com", "blog")) or "/blog" in path:
        return "blog"
    if (
        domain.startswith(("docs.", "developer."))
        or domain.endswith((".readthedocs.io", ".readthedocs.org"))
        or "documentation" in domain
    ):
        return "official_docs"
    if any(part in domain for part in ("lists.", "mail.", "mailman", "groups.google")):
        return "mailing_list"
    if "stackoverflow.com" in domain or "stackexchange.com" in domain:
        return "qa"
    return "web"


def normalize_source_types(source_types: list[str] | None) -> list[str] | None:
    if not source_types:
        return None
    normalized: list[str] = []
    for value in source_types:
        key = value.strip().lower()
        if not key:
            continue
        mapped = SOURCE_TYPE_ALIASES.get(key)
        if mapped is None:
            raise ValueError(f"invalid source_type: {value!r}. Expected one of {sorted(SOURCE_TYPE_ALIASES)}")
        if mapped not in normalized:
            normalized.append(mapped)
    return normalized or None


def matches_source_types(url: str, source_types: list[str] | None) -> bool:
    return not source_types or source_type(url) in source_types


def source_boost(kind: str) -> float:
    return {
        "official_docs": 0.075,
        "repo": 0.05,
        "issue_tracker": -0.01,
        "mailing_list": 0.035,
        "qa": 0.015,
        "pdf": 0.012,
        "web": 0.0,
        "blog": -0.02,
    }.get(kind, 0.0)


def domain_quality_boost(url: str) -> float:
    domain = _domain_from_url(url).lower()
    if domain in _PRIMARY_DOMAINS:
        return 0.035
    if domain in _SOCIAL_DOMAINS:
        return -0.20
    if domain in _LOW_QUALITY_DOMAINS:
        return -0.06
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return 0.02
    return 0.0


def query_entity_boost(query: str, entry: dict) -> float:
    """Reward URLs/titles that name distinctive entities from the query."""
    terms = {
        term for term in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", query.lower())
        if len(term) >= 4 and term not in _QUERY_ENTITY_STOPWORDS
    }
    if not terms:
        return 0.0
    haystack = f"{entry.get('title', '')} {entry.get('url', '')}".lower()
    matches = sum(term in haystack for term in terms)
    return min(0.08, matches * 0.04)


def synthesis_eligible(result: dict) -> bool:
    """Keep weak/social evidence visible without letting it drive summaries."""
    url = result.get("url") or ""
    domain = _domain_from_url(url).lower()
    kind = source_type(url)
    if domain in _SOCIAL_DOMAINS or domain in _LOW_QUALITY_DOMAINS or domain.endswith(".github.io"):
        return False
    if kind in {"issue_tracker", "qa"}:
        return False
    return kind in {"official_docs", "repo", "pdf", "blog"} or domain.endswith((".gov", ".edu"))


def content_quality_boost(entry: dict) -> float:
    title = (entry.get("title") or "").lower()
    url = (entry.get("url") or "").lower()
    if any(token in title for token in ("official", "documentation", "reference", "manual", "specification")):
        return 0.015
    if any(token in url for token in ("/docs", "/reference", "/manual", "/spec", "/releases", "/changelog")):
        return 0.015
    if any(token in title for token in ("top ", "best ", "ultimate guide", "click here")):
        return -0.02
    return 0.0


def entry_sort_score(
    eidx: int,
    entries: list[dict],
    entry_best: dict[int, float | None],
    intent_profile=None,
    query: str = "",
) -> float:
    entry = entries[eidx]
    url = entry["url"]
    intent_boost = 0.0
    if intent_profile is not None:
        from web_search_mcp.ranking.intent import preference_boost
        intent_boost = preference_boost(source_type(url), intent_profile)
    return (
        (entry_best.get(eidx) or 0.0)
        + source_boost(source_type(url))
        + domain_quality_boost(url)
        + content_quality_boost(entry)
        + intent_boost
        + query_entity_boost(query, entry)
    )


def diversify_by_source_type(ranked_entry_idxs: list[int], entries: list[dict]) -> list[int]:
    """Diversify the tail without displacing the strongest leading evidence."""
    protected = ranked_entry_idxs[:3]
    ranked_entry_idxs = ranked_entry_idxs[3:]
    by_type: dict[str, list[int]] = defaultdict(list)
    order: list[str] = []
    for eidx in ranked_entry_idxs:
        kind = source_type(entries[eidx]["url"])
        if kind not in by_type:
            order.append(kind)
        by_type[kind].append(eidx)
    diversified: list[int] = list(protected)
    while by_type:
        next_order: list[str] = []
        for kind in order:
            queue = by_type.get(kind)
            if not queue:
                continue
            diversified.append(queue.pop(0))
            if queue:
                next_order.append(kind)
            else:
                by_type.pop(kind, None)
        order = next_order
    return diversified
