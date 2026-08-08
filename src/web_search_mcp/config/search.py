"""Internal search profiles and lightweight validators."""

from web_search_mcp.common import _validate_positive_int
from web_search_mcp.config.settings import MAX_CANDIDATES, MAX_SCRAPE

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


def candidate_budget(profile: str, num_results: int) -> int:
    """Return a broad pre-scrape pool independent of the output budget."""
    multiplier = 6 if profile == "research" else 4
    floor = 20 if profile == "research" else 12
    return min(max(num_results * multiplier, floor), MAX_CANDIDATES)


def default_chars_per_result(profile: str) -> int:
    return 3200 if profile == "research" else 1800


def validate_optional_positive_int(name: str, value: int | None, *, maximum: int) -> int | None:
    if value is None:
        return None
    return _validate_positive_int(name, value, maximum=maximum)
