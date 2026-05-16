"""SearXNG and dependency-probe HTTP clients."""

import httpx

from web_search_mcp.config.settings import SEARXNG_URL, _HTTP_TIMEOUT
async def _search(
    query: str,
    num_results: int = 10,
    time_range: str | None = None,
    pageno: int = 1,
    language: str | None = "en",
) -> dict:
    """Query SearXNG and return {"results": [...], "unresponsive_engines": [...]}.

    `unresponsive_engines` is SearXNG's per-engine failure report — list of
    [engine_name, error_string] pairs for engines that didn't contribute to
    this response (CAPTCHA'd, rate-limited, timed out, etc).
    """
    params: dict = {
        "q": query,
        "format": "json",
        "number_of_results": num_results,
    }
    if time_range:
        params["time_range"] = time_range.strip('"')
    if pageno > 1:
        params["pageno"] = pageno
    if language:
        params["language"] = language

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        resp = await client.get(f"{SEARXNG_URL}/search", params=params)
        resp.raise_for_status()
        data = resp.json()
        return {
            "results": data.get("results", [])[:num_results],
            "unresponsive_engines": data.get("unresponsive_engines", []),
        }


async def _probe_dependency(url: str) -> dict[str, str]:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        return {"status": "ok"}
    except httpx.HTTPError as exc:
        return {"status": "error", "detail": str(exc)}
