"""Input validation and normalization helpers."""

import fnmatch
import ipaddress
import socket
from urllib.parse import urlparse

import urls as url_utils

VALID_TIME_RANGES = frozenset({"day", "week", "month", "year"})


def coerce_optional_str(value: str | None) -> str | None:
    """Undo accidental JSON-quoting from buggy MCP clients."""
    if value is None:
        return None
    stripped = value.strip().strip('"').strip("'").strip()
    if stripped.lower() in ("", "null", "none"):
        return None
    return stripped


def normalize_time_range(time_range: str | None) -> str | None:
    coerced = coerce_optional_str(time_range)
    if coerced is None:
        return None
    normalized = coerced.lower()
    if normalized not in VALID_TIME_RANGES:
        raise ValueError(f"invalid time_range: {time_range!r}. Expected one of {sorted(VALID_TIME_RANGES)}")
    return normalized


def validate_query(query: str) -> str:
    normalized = query.strip()
    if not normalized:
        raise ValueError("query must not be empty")
    return normalized


def validate_positive_int(name: str, value: int, *, maximum: int) -> int:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    if value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def validate_urls(values: list[str], *, maximum: int) -> list[str]:
    if not values:
        raise ValueError("urls must not be empty")
    if len(values) > maximum:
        raise ValueError(f"urls must contain at most {maximum} entries")
    normalized: list[str] = []
    for raw_url in values:
        value = raw_url.strip()
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"invalid URL: {raw_url!r}")
        reject_non_public_target(parsed.hostname, url=value)
        normalized.append(value)
    return normalized


def is_blocked_ip(ip: ipaddress._BaseAddress) -> bool:
    return any((
        ip.is_private,
        ip.is_loopback,
        ip.is_link_local,
        ip.is_multicast,
        ip.is_reserved,
        ip.is_unspecified,
        not ip.is_global,
    ))


def reject_non_public_target(hostname: str | None, *, url: str) -> None:
    """Reject localhost and DNS targets that resolve to non-public IP space."""
    if not hostname:
        raise ValueError(f"invalid URL: {url!r}")

    host = hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError(f"URL resolves to a private or reserved target: {url!r}")

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None

    if ip is not None:
        if is_blocked_ip(ip):
            raise ValueError(f"URL resolves to a private or reserved target: {url!r}")
        return

    try:
        resolved = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except OSError:
        return

    for family, _, _, _, sockaddr in resolved:
        if family in {socket.AF_INET, socket.AF_INET6}:
            candidate = ipaddress.ip_address(sockaddr[0])
        else:
            continue
        if is_blocked_ip(candidate):
            raise ValueError(f"URL resolves to a private or reserved target: {url!r}")


def normalize_domains(domains: list[str] | None, *, field_name: str) -> list[str]:
    if not domains:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for domain in domains:
        value = url_utils.canonical_hostname(domain)
        if not value:
            continue
        if "/" in value:
            raise ValueError(f"{field_name} entries must be bare domains, got {domain!r}")
        if value not in seen:
            seen.add(value)
            normalized.append(value)
    return normalized


def normalize_glob_patterns(patterns: list[str] | None, *, field_name: str) -> list[str]:
    if not patterns:
        return []
    normalized = [pattern.strip() for pattern in patterns if pattern.strip()]
    if not normalized:
        raise ValueError(f"{field_name} must not be empty when provided")
    return normalized


def match_domain(domain: str, patterns: list[str]) -> bool:
    host = url_utils.canonical_hostname(domain)
    host_registrable = url_utils.registrable_domain(host)
    for pattern in patterns:
        pat = url_utils.canonical_hostname(pattern)
        if not pat:
            continue
        if fnmatch.fnmatch(host, pat) or fnmatch.fnmatch(host_registrable, pat):
            return True
    return False
