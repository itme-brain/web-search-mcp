"""Freshness policy shared by live retrieval and durable memory boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime


def parse_source_date(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (ValueError, TypeError, OverflowError):
            return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def filter_dated_entries(entries: list[dict], time_range: str, *, now: datetime | None = None) -> tuple[list[dict], dict]:
    """Strict rolling date windows: exclude unknown, old, and future dates."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days={"day": 1, "week": 7, "month": 30, "year": 365}[time_range])
    report = {"policy": "strict", "cutoff": cutoff.isoformat(), "window_end": now.isoformat(),
              "excluded_undated": 0, "excluded_outside_window": 0, "eligible": 0}
    eligible = []
    for entry in entries:
        date = parse_source_date((entry.get("metadata") or {}).get("date"))
        if date is None:
            report["excluded_undated"] += 1
        elif not cutoff <= date <= now:
            report["excluded_outside_window"] += 1
        else:
            eligible.append(entry)
    report["eligible"] = len(eligible)
    return eligible, report


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
