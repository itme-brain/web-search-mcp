"""Internal search profiles and lightweight validators."""

import core
from core import MAX_SCRAPE

VALID_PROFILES = frozenset({"search", "research"})


def normalize_profile(profile: str | None) -> str:
    if profile is None:
        return "search"
    normalized = profile.strip().lower()
    if normalized not in VALID_PROFILES:
        raise ValueError(f"invalid profile: {profile!r}. Expected one of {sorted(VALID_PROFILES)}")
    return normalized


def profile_budget(profile: str, num_results: int) -> tuple[int, int, int]:
    """Return (searx_pages, scrape_budget, default_passages)."""
    if profile == "research":
        return 3, min(max(num_results * 2, 10), MAX_SCRAPE), 4
    return 2, min(num_results, MAX_SCRAPE), 2


def default_chars_per_result(profile: str) -> int:
    return 3200 if profile == "research" else 1800


def validate_optional_positive_int(name: str, value: int | None, *, maximum: int) -> int | None:
    if value is None:
        return None
    return core._validate_positive_int(name, value, maximum=maximum)
