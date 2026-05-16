"""Shared HTTP request policy for direct origin fetches.

These headers make direct requests behave like a competent web client without
adding anti-bot evasion mechanics such as fingerprint rotation, proxy churn, or
session fakery. Provider modules can extend the base headers when an API has
additional requirements.
"""

import asyncio
import logging
import os

import httpx

from web_search_mcp.config.settings import _HTTP_TIMEOUT

log = logging.getLogger("web-search-mcp")

# ---------------------------------------------------------------------------
# Dynamic Firefox version (fetched at startup, with safe fallback)
# ---------------------------------------------------------------------------

_FIREFOX_VERSIONS_URL = "https://product-details.mozilla.org/1.0/firefox_versions.json"
_FIREFOX_FALLBACK_VERSION = "145.0"

_cache: dict = {
    "version": None,          # resolved version string or None
    "_lock": None,            # asyncio.Lock for safe init
    "_fetched": False,        # whether we attempted the fetch
}


def _build_user_agent(version: str) -> str:
    """Build a realistic Windows 10/11 Firefox UA from a version."""
    return f"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:{version}) Gecko/20100101 Firefox/{version}"


async def _ensure_firefox_version() -> str:
    """Ensure we have a Firefox version (fetched or fallback)."""
    if _cache["version"] is not None:
        return _cache["version"]

    if _cache["_lock"] is None:
        _cache["_lock"] = asyncio.Lock()

    async with _cache["_lock"]:
        # Double-check after acquiring lock
        if _cache["version"] is not None:
            return _cache["version"]

        if _cache["_fetched"]:
            # Already tried and failed; keep fallback
            return _cache["version"] or _FIREFOX_FALLBACK_VERSION

        _cache["_fetched"] = True
        version = _FIREFOX_FALLBACK_VERSION
        try:
            async with httpx.AsyncClient(timeout=min(_HTTP_TIMEOUT, 2)) as client:
                resp = await client.get(_FIREFOX_VERSIONS_URL)
                resp.raise_for_status()
                data = resp.json()
                version = str(data.get("LATEST_FIREFOX_VERSION") or version)
        except Exception as exc:
            log.debug(
                "failed to fetch latest Firefox version, using fallback=%s err=%s",
                _FIREFOX_FALLBACK_VERSION,
                exc,
            )

        _cache["version"] = version
        return version


def user_agent() -> str:
    """Return the configured product user agent.

    Uses a dynamically fetched latest Firefox version (with fallback) and
    a Windows 10/11-style UA that is the most common and least suspicious.
    """
    env_ua = os.environ.get("WEB_SEARCH_MCP_USER_AGENT")
    if env_ua:
        return env_ua

    version = _cache.get("version")
    if version is not None:
        return _build_user_agent(version)

    # Before async init completes, fall back to a safe default.
    # After ensure_user_agent_initialized() runs, _cache["version"] is set
    # and all subsequent calls use the real latest version.
    return _build_user_agent(_FIREFOX_FALLBACK_VERSION)


async def ensure_user_agent_initialized() -> str:
    """Call this once at application startup to fetch the latest version."""
    version = await _ensure_firefox_version()
    log.debug("resolved Firefox user-agent version: %s", version)
    return _build_user_agent(version)


DEFAULT_ACCEPT_LANGUAGE = "en-US,en;q=0.8"


def accept_language() -> str:
    """Return the configured language negotiation header."""
    return os.environ.get("WEB_SEARCH_MCP_ACCEPT_LANGUAGE", DEFAULT_ACCEPT_LANGUAGE)


def browser_compatible_headers(*, accept: str | None = None) -> dict[str, str]:
    """Headers for public web document retrieval.

    Callers may override Accept for specific document/API types while retaining
    User-Agent and language policy.
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
