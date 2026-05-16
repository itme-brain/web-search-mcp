"""Reranker runtime/lifecycle wrapper."""

import logging

from web_search_mcp.ranking import rerankers

log = logging.getLogger("web-search-mcp")

_reranker = None


def load_reranker(
    *,
    backend: str,
    model: str,
    max_length: int,
    batch_size: int,
    device: str | None,
):
    global _reranker
    if _reranker is None:
        log.info(
            "loading reranker backend=%s model=%s max_length=%d device=%s batch_size=%d",
            backend,
            model,
            max_length,
            device or "auto",
            batch_size,
        )
        _reranker = rerankers.build_reranker(
            backend=backend,
            model=model,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
        log.info("reranker ready backend=%s model=%s", _reranker.name, _reranker.model)
    return _reranker


async def rerank_scored(query: str, documents: list[str]) -> list[tuple[int, float]]:
    if _reranker is None:
        raise RuntimeError("reranker is not loaded")
    return await _reranker.rerank(query, documents)
