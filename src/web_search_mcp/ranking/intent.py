"""Deterministic research-intent classification and source preferences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

IntentName = Literal[
    "technical_documentation", "current_events", "academic_research",
    "product_research", "factual_lookup", "comparison", "general_web_research",
]


@dataclass(frozen=True)
class IntentProfile:
    """Retrieval preferences derived from a query without model inference."""

    name: IntentName
    preferred_source_types: tuple[str, ...]
    freshness_sensitive: bool = False


_CURRENT_TERMS = frozenset({
    "breaking", "current", "latest", "news", "recent", "today", "updated",
    "yesterday", "announced", "announcement",
})
_ACADEMIC_TERMS = frozenset({
    "academic", "arxiv", "citation", "doi", "journal", "paper", "papers",
    "peer-reviewed", "research", "study", "studies",
})
_PRODUCT_TERMS = frozenset({
    "buy", "buyer", "deal", "price", "pricing", "product", "review",
    "reviews", "specifications", "warranty",
})
_TECHNICAL_TERMS = frozenset({
    "api", "code", "configuration", "docs", "documentation", "error",
    "example", "framework", "library", "manual", "reference", "sdk",
    "specification", "tutorial",
})
_FACTUAL_PREFIXES = ("define ", "what is ", "when did ", "where is ", "which ", "who ")


def classify(query: str, *, time_range: str | None = None) -> IntentProfile:
    """Classify a query using conservative, explainable lexical rules."""
    normalized = " ".join(query.lower().split())
    terms = set(normalized.replace("/", " ").split())
    if time_range or terms & _CURRENT_TERMS:
        return IntentProfile("current_events", ("web", "blog"), freshness_sensitive=True)
    if " vs " in f" {normalized} " or " versus " in f" {normalized} " or terms & {
        "compare", "comparison", "difference", "differences",
    }:
        return IntentProfile("comparison", ("official_docs", "web", "blog"))
    if terms & _ACADEMIC_TERMS:
        return IntentProfile("academic_research", ("pdf", "official_docs", "web"))
    if terms & _PRODUCT_TERMS:
        return IntentProfile("product_research", ("official_docs", "web", "blog"))
    if terms & _TECHNICAL_TERMS or terms & {"github", "gitlab"}:
        return IntentProfile("technical_documentation", ("official_docs", "repo", "issue_tracker", "qa"))
    if normalized.startswith(_FACTUAL_PREFIXES):
        return IntentProfile("factual_lookup", ("official_docs", "web"))
    return IntentProfile("general_web_research", ("web", "official_docs", "blog"))


def preference_boost(source_type: str, profile: IntentProfile) -> float:
    """Return a small rank boost based on preference order for an intent."""
    try:
        position = profile.preferred_source_types.index(source_type)
    except ValueError:
        return -0.01
    return max(0.0, 0.045 - (position * 0.012))
