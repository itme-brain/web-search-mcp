"""Wikimedia API-backed extraction.

This module keeps provider-specific behavior out of the generic extraction
pipeline. Wikipedia article URLs are public web pages, but Wikimedia exposes a
stable Action API that returns cleaner article text than scraping page chrome.
"""

from dataclasses import dataclass
from urllib.parse import unquote, urlparse

import httpx

import http_policy

REQUEST_TIMEOUT = 30


@dataclass(frozen=True)
class WikimediaArticle:
    """Normalized article target extracted from a Wikimedia URL."""

    api_url: str
    article_url: str
    host: str
    title: str
    language: str | None


def parse_wikipedia_article_url(url: str) -> WikimediaArticle | None:
    """Return a Wikimedia article target for standard Wikipedia page URLs."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return None

    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host.startswith("m."):
        host = host[2:]
    if not host.endswith(".wikipedia.org"):
        return None

    path_prefix = "/wiki/"
    if not parsed.path.startswith(path_prefix):
        return None

    raw_title = parsed.path[len(path_prefix):]
    if not raw_title:
        return None

    title = unquote(raw_title).replace("_", " ")
    language = host.removesuffix(".wikipedia.org") or None
    return WikimediaArticle(
        api_url=f"https://{host}/w/api.php",
        article_url=f"https://{host}/wiki/{raw_title}",
        host=host,
        title=title,
        language=language,
    )


def _page_from_response(data: dict) -> dict | None:
    query = data.get("query")
    if not isinstance(query, dict):
        return None
    pages = query.get("pages")
    if not isinstance(pages, list) or not pages:
        return None
    page = pages[0]
    return page if isinstance(page, dict) else None


def _metadata(article: WikimediaArticle, page: dict, content: str) -> dict:
    metadata = {
        "source": "wikimedia_api",
        "api_url": article.api_url,
        "canonical_url": page.get("fullurl") or article.article_url,
        "pageid": page.get("pageid"),
        "language": article.language,
        "word_count": len(content.split()),
    }
    return {k: v for k, v in metadata.items() if v is not None}


async def extract_document(url: str) -> dict | None:
    """Extract a Wikipedia article through the MediaWiki Action API.

    Returns None when the URL is not a supported Wikipedia article URL. For
    supported URLs, returns the same document envelope shape as core extractors.
    """
    article = parse_wikipedia_article_url(url)
    if article is None:
        return None

    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
        "prop": "extracts|info",
        "explaintext": "1",
        "exsectionformat": "wiki",
        "inprop": "url",
        "titles": article.title,
    }

    try:
        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
            headers=http_policy.wikimedia_api_headers(),
        ) as client:
            resp = await client.get(article.api_url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        return {
            "status": "error",
            "url": url,
            "content_type": "application/json",
            "file_type": "html",
            "title": article.title,
            "content": "",
            "total_chars": 0,
            "metadata": {"source": "wikimedia_api", "api_url": article.api_url},
            "error": f"wikimedia api extraction failed: {exc}",
        }

    page = _page_from_response(data)
    if page is None or page.get("missing"):
        return {
            "status": "error",
            "url": url,
            "content_type": "application/json",
            "file_type": "html",
            "title": article.title,
            "content": "",
            "total_chars": 0,
            "metadata": {"source": "wikimedia_api", "api_url": article.api_url},
            "error": "wikimedia page not found",
        }

    extract = str(page.get("extract") or "").strip()
    if not extract:
        return {
            "status": "error",
            "url": url,
            "content_type": "application/json",
            "file_type": "html",
            "title": page.get("title") or article.title,
            "content": "",
            "total_chars": 0,
            "metadata": _metadata(article, page, ""),
            "error": "wikimedia api returned no extractable text",
        }

    title = page.get("title") or article.title
    content = f"# {title}\n\n{extract}"
    return {
        "status": "ok",
        "url": url,
        "content_type": "text/html",
        "file_type": "html",
        "title": title,
        "content": content,
        "total_chars": len(content),
        "metadata": _metadata(article, page, content),
    }
