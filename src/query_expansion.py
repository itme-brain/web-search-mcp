"""Deterministic query expansion for the research tool."""

import core

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "how", "in", "is",
    "it", "of", "on", "or", "the", "to", "used", "using", "what", "when", "where",
    "which", "who", "why", "with",
}


def query_terms(query: str) -> list[str]:
    terms = [term for term in core.re.findall(r"[A-Za-z0-9_.+-]{3,}", query) if term.lower() not in _STOP_WORDS]
    return terms[:8]


def keyphrase(query: str) -> str:
    terms = query_terms(query)
    return " ".join(terms[:5]) or query


def search_queries(query: str, profile: str) -> list[str]:
    """Return one query for normal search, expanded variants for research."""
    if profile != "research":
        return [query]
    phrase = keyphrase(query)
    quoted = f'"{phrase}"' if phrase != query or len(query) <= 80 else query
    variants = [
        query,
        quoted,
        f"{phrase} documentation docs official",
        f"{phrase} github gitlab repository",
        f"{phrase} issue discussion mailing list",
        f"{phrase} example case study",
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        key = variant.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(variant)
    return deduped
