"""Small deterministic engine groups, ordered from primary to fallback."""

_GENERAL = (("bing", "yandex"), ("fynd", "wiby"))
_ROUTES = {
    "technical_documentation": (("bing", "yandex"), ("github", "stackoverflow", "fynd")),
    "current_events": (("bing news", "google news"), ("bing", "yandex"), ("fynd",)),
    "academic_research": (("bing", "arxiv"), ("yandex", "wikipedia", "fynd")),
    "factual_lookup": (("bing", "wikipedia"), ("yandex", "fynd", "wiby")),
}


def engine_stages(intent: str, profile: str, *, time_range: str | None = None) -> tuple[tuple[str, ...], ...]:
    """Research keeps a broad engine pool for each query variant."""
    # All other enabled engines in the deployed SearXNG release declare
    # time_range_support=False and are skipped when a date filter is sent.
    if time_range:
        return (("bing news",),)
    stages = _ROUTES.get(intent, _GENERAL)
    if profile == "research":
        return (tuple(engine for stage in stages for engine in stage),)
    return (stages[0], tuple(engine for stage in stages[1:] for engine in stage))
