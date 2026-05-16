"""Reranker lifecycle and scoring facade."""

from web_search_mcp.config.settings import (
    RERANK_BACKEND,
    RERANK_BATCH_SIZE,
    RERANK_DEVICE,
    RERANK_MAX_LENGTH,
    RERANK_MODEL,
)
from web_search_mcp.ranking import rerank
_reranker = rerank.load_reranker(
    backend=RERANK_BACKEND,
    model=RERANK_MODEL,
    max_length=RERANK_MAX_LENGTH,
    batch_size=RERANK_BATCH_SIZE,
    device=RERANK_DEVICE,
)
RERANK_NAME = _reranker.name


async def _rerank_scored(query: str, documents: list[str]) -> list[tuple[int, float]]:
    return await rerank.rerank_scored(query, documents)
