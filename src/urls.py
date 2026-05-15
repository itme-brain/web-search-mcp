"""URL normalization and domain helpers."""

import re
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import tldextract
from url_normalize import url_normalize

_TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "ref", "fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid",
})
_WHITESPACE = re.compile(r"\s+")


def normalize_url(url: str) -> str:
    normalized = url_normalize(url)
    parsed = urlparse(normalized)
    host = parsed.hostname or ""
    if host.startswith("www."):
        host = host[4:]
    params = {k: v for k, v in parse_qs(parsed.query).items() if k not in _TRACKING_PARAMS}
    return urlunparse((parsed.scheme, host, parsed.path.rstrip("/"), "", urlencode(params, doseq=True), ""))


def domain_from_url(url: str) -> str:
    return urlparse(url).hostname or ""


def canonical_hostname(host: str) -> str:
    normalized = host.strip().lower().rstrip(".")
    if normalized.startswith("www."):
        normalized = normalized[4:]
    return normalized


def registrable_domain(domain: str) -> str:
    """Return the PSL-aware registrable domain for host/domain matching."""
    bare = canonical_hostname(domain)
    extracted = tldextract.extract(bare)
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}"
    return bare


def normalize_title(title: str) -> str:
    normalized = _WHITESPACE.sub(" ", title.strip().lower())
    normalized = re.sub(r"[^a-z0-9 ]+", "", normalized)
    normalized = _WHITESPACE.sub(" ", normalized)
    return normalized.strip()
