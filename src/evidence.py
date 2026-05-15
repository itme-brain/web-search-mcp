"""Small-model evidence shaping helpers."""


def brief_from_results(results: list[dict]) -> list[str]:
    brief: list[str] = []
    for result in results[:3]:
        passages = result.get("passages") or []
        if not passages:
            continue
        text = passages[0].get("text", "").replace("\n", " ").strip()
        if len(text) > 220:
            text = text[:219].rstrip() + "…"
        citation = passages[0].get("citation") or str(result["rank"])
        if text:
            brief.append(f"[{citation}] {text}")
    return brief


def research_findings(results: list[dict]) -> list[str]:
    """Return conservative extractive findings with citations.

    This deliberately does not synthesize beyond retrieved text; it gives
    small models Tavily-like cited findings while keeping this server API-less.
    """
    findings: list[str] = []
    seen: set[str] = set()
    for result in results[:4]:
        passages = result.get("passages") or []
        if not passages:
            continue
        text = passages[0].get("text", "").replace("\n", " ").strip()
        if len(text) > 260:
            text = text[:259].rstrip() + "…"
        key = text[:120].lower()
        citation = passages[0].get("citation") or str(result["rank"])
        if text and key not in seen:
            seen.add(key)
            findings.append(f"[{citation}] {text}")
    return findings


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


def research_key_evidence(results: list[dict]) -> list[str]:
    evidence: list[str] = []
    for result in results[:5]:
        passages = result.get("passages") or []
        if not passages:
            continue
        text = passages[0].get("text", "").replace("\n", " ").strip()
        if len(text) > 180:
            text = text[:179].rstrip() + "…"
        citation = passages[0].get("citation") or str(result["rank"])
        if text:
            evidence.append(f"[{citation}] {result.get('title', 'Untitled')}: {text}")
    return evidence


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
