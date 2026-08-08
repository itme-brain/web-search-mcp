"""Freshness policy shared by live retrieval and durable memory boundaries."""

from __future__ import annotations

from dataclasses import dataclass


_PAGE_MAX_AGE_SECONDS = {
    "day": 15 * 60,
    "week": 60 * 60,
    "month": 6 * 60 * 60,
    "year": 24 * 60 * 60,
}
_SEARCH_MAX_AGE_SECONDS = {
    "day": 5 * 60,
    "week": 10 * 60,
    "month": 15 * 60,
    "year": 15 * 60,
}


@dataclass(frozen=True)
class FreshnessPolicy:
    """Cache and memory rules for a normalized optional time range."""

    time_range: str | None
    page_max_age_seconds: int | None
    search_max_age_seconds: int | None
    allow_unverified_memory: bool


def policy_for(time_range: str | None) -> FreshnessPolicy:
    """Build strict policy; bounded requests exclude unverifiable memory."""
    if time_range is None:
        return FreshnessPolicy(None, None, None, True)
    return FreshnessPolicy(
        time_range=time_range,
        page_max_age_seconds=_PAGE_MAX_AGE_SECONDS[time_range],
        search_max_age_seconds=_SEARCH_MAX_AGE_SECONDS[time_range],
        allow_unverified_memory=False,
    )
