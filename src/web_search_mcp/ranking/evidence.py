"""Small-model evidence shaping helpers."""


def research_summary(results: list[dict], warnings: list[dict]) -> list[str]:
    """Return a conservative answer-first assessment for research output."""
    if not results:
        return ["No reliable assessment is possible because no supporting sources were found."]
    summary: list[str] = []
    top_titles = [r.get("title") or "Untitled" for r in results[:3]]
    summary.append(f"Top retrieved evidence centers on: {', '.join(top_titles)}.")
    dated = [r.get("latest_date") for r in results if r.get("latest_date")]
    if dated:
        summary.append(f"Newest dated source in the retrieved set: {max(str(d) for d in dated)}.")
    else:
        summary.append("Retrieved sources are undated or dates were not detected; treat freshness as uncertain.")
    if len(results) < 3:
        summary.append("Evidence coverage is thin, so conclusions should remain tentative.")
    if any(w.get("type") in {"search_failed", "scrape_failed", "rerank_failed"} for w in warnings):
        summary.append("Some retrieval steps were degraded; verify important claims against extracted sources.")
    return summary[:4]


def research_gaps(results: list[dict], warnings: list[dict]) -> list[str]:
    gaps: list[str] = []
    if not results:
        gaps.append("No supporting sources found.")
    if any(w.get("type") in {"search_failed", "scrape_failed", "rerank_failed"} for w in warnings):
        gaps.append("Some retrieval steps were degraded; verify with extract or retry.")
    if len(results) < 3:
        gaps.append("Coverage is thin; broaden the query or remove filters.")
    return gaps[:3]


def next_actions(profile: str, degraded: bool, results: list[dict], warnings: list[dict]) -> list[str]:
    actions: list[str] = []
    has_retrieval_warning = any(w.get("type") in {"search_failed", "scrape_failed", "rerank_failed"} for w in warnings)
    if degraded or has_retrieval_warning:
        actions.append("Results are degraded; retry or use research for more coverage.")
    if profile == "research" and results:
        actions.append("Use extract on the best URL if more context is needed.")
    if profile != "research" and len(results) < 3:
        actions.append("Use research or broaden the query if coverage is too thin.")
    return actions[:3]


def limit_passages(passages: list[tuple[str, float]], max_passages: int, max_chars: int) -> list[tuple[str, float]]:
    kept: list[tuple[str, float]] = []
    used = 0
    for text, score in passages:
        remaining = max_chars - used
        if remaining <= 0 or len(kept) >= max_passages:
            break
        clipped = text if len(text) <= remaining else text[: max(0, remaining - 1)].rstrip() + "…"
        if clipped:
            kept.append((clipped, score))
            used += len(clipped)
    return kept
