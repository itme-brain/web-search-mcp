"""Deterministic query expansion for the research tool."""

import re

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "how", "in", "is",
    "it", "of", "on", "or", "the", "to", "used", "using", "what", "when", "where",
    "which", "who", "why", "with",
    # Query-shaping modifiers are useful in the original request, but make poor
    # leading terms for subject-focused expansion queries (especially on Bing).
    "best", "current", "latest", "practice", "practices", "recent",
}


def query_terms(query: str) -> list[str]:
    terms = [term for term in re.findall(r"[A-Za-z0-9_.+-]{3,}", query) if term.lower() not in _STOP_WORDS]
    return terms[:8]


def keyphrase(query: str) -> str:
    terms = query_terms(query)
    return " ".join(terms[:5]) or query


def search_queries(query: str, profile: str, intent: str = "general_web_research") -> list[str]:
    """Return intent-aware query variants for the research profile."""
    if profile != "research":
        return [query]
    phrase = keyphrase(query)
    quoted = f'"{phrase}"' if phrase != query or len(query) <= 80 else query
    suffixes = {
        "technical_documentation": ["official documentation reference", "github gitlab repository", "issue discussion mailing list", "example implementation"],
        "current_events": ["latest announcement", "official statement", "recent reporting"],
        "academic_research": ["paper arxiv doi", "systematic review", "study results methodology"],
        "product_research": ["official specifications", "independent review", "pricing comparison"],
        "comparison": ["official documentation comparison", "independent analysis", "differences advantages disadvantages"],
        "factual_lookup": ["official source", "reference"],
        "general_web_research": ["official source", "analysis", "case study"],
    }.get(intent, ["official source", "analysis"])
    variants = [query, quoted, *(f"{phrase} {suffix}" for suffix in suffixes)]
    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        key = variant.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(variant)
    return deduped
