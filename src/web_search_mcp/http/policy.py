"""Shared HTTP request policy for direct origin fetches.

These headers make direct requests behave like a competent web client without
adding anti-bot evasion mechanics such as fingerprint rotation, proxy churn, or
session fakery. Provider modules can extend the base headers when an API has
additional requirements.
"""

import os


DEFAULT_USER_AGENT = (
    "Mozilla/5.0"
    "(X11; Linux x86_64; rv:150.0) Gecko/20100101 Firefox/150.0"
)
DEFAULT_ACCEPT_LANGUAGE = "en-US,en;q=0.8"


def user_agent() -> str:
    """Return the configured product user agent."""
    return os.environ.get("WEB_SEARCH_MCP_USER_AGENT", DEFAULT_USER_AGENT)


def accept_language() -> str:
    """Return the configured language negotiation header."""
    return os.environ.get("WEB_SEARCH_MCP_ACCEPT_LANGUAGE", DEFAULT_ACCEPT_LANGUAGE)


def browser_compatible_headers(*, accept: str | None = None) -> dict[str, str]:
    """Headers for public web document retrieval.

    The defaults are stable and reproducible. Callers may override Accept for
    specific document/API types while retaining User-Agent and language policy.
    """
    return {
        "User-Agent": user_agent(),
        "Accept": accept or "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
        "Accept-Language": accept_language(),
        "Accept-Encoding": "gzip, deflate, br",
    }


def identity_headers(*, accept: str | None = None) -> dict[str, str]:
    """Headers for bounded byte-range/binary reads where compression is unsafe."""
    headers = browser_compatible_headers(accept=accept)
    headers["Accept-Encoding"] = "identity"
    return headers


def wikimedia_api_headers() -> dict[str, str]:
    """Headers required for Wikimedia API calls."""
    ua = user_agent()
    return {
        "User-Agent": ua,
        "Api-User-Agent": ua,
        "Accept": "application/json",
        "Accept-Language": accept_language(),
        "Accept-Encoding": "gzip, deflate, br",
    }
