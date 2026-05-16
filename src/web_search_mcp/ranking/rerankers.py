"""Local reranker backends used by the search pipeline."""

from __future__ import annotations

import asyncio
from typing import Protocol


class Reranker(Protocol):
    """Common interface for query/document cross-encoder rerankers."""

    name: str
    model: str

    async def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        """Return ``(document_index, score)`` pairs sorted by score descending."""


class FlashRankReranker:
    """FlashRank ONNX reranker, kept as the CPU-friendly default."""

    name = "flashrank"

    def __init__(self, model: str, *, max_length: int) -> None:
        from flashrank import Ranker

        self.model = model
        self._ranker = Ranker(model_name=model, max_length=max_length)

    def _rerank_sync(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        from flashrank import RerankRequest

        if not documents:
            return []
        passages = [{"id": i, "text": doc, "meta": {}} for i, doc in enumerate(documents)]
        request = RerankRequest(query=query, passages=passages)
        results = self._ranker.rerank(request)
        return [(r["id"], float(r["score"])) for r in results]

    async def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        return await asyncio.to_thread(self._rerank_sync, query, documents)


class SentenceTransformersReranker:
    """Sentence Transformers CrossEncoder backend for English rerankers."""

    name = "sentence-transformers"

    def __init__(
        self,
        model: str,
        *,
        max_length: int,
        batch_size: int,
        device: str | None,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model = model
        self._batch_size = batch_size
        self._model = CrossEncoder(
            model,
            max_length=max_length,
            device=device,
        )

    def _rerank_sync(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        if not documents:
            return []
        scores = self._model.predict(
            [(query, document) for document in documents],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return sorted(
            [(idx, float(score)) for idx, score in enumerate(scores)],
            key=lambda item: item[1],
            reverse=True,
        )

    async def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        return await asyncio.to_thread(self._rerank_sync, query, documents)


class NoopReranker:
    """Deterministic fallback useful for degraded mode and tests."""

    name = "none"
    model = "none"

    async def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        return [(idx, 0.0) for idx, _ in enumerate(documents)]


def build_reranker(
    *,
    backend: str,
    model: str,
    max_length: int,
    batch_size: int,
    device: str | None,
) -> Reranker:
    """Build a configured reranker backend."""
    if max_length < 1:
        raise ValueError("RERANK_MAX_LENGTH must be >= 1")
    if batch_size < 1:
        raise ValueError("RERANK_BATCH_SIZE must be >= 1")
    normalized = backend.strip().lower().replace("_", "-")
    if normalized == "flashrank":
        return FlashRankReranker(model, max_length=max_length)
    if normalized in {"sentence-transformers", "sentence-transformer", "cross-encoder"}:
        return SentenceTransformersReranker(
            model,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
    if normalized in {"none", "noop", "disabled"}:
        return NoopReranker()
    raise ValueError(
        "invalid RERANK_BACKEND: "
        f"{backend!r}. Expected one of flashrank, sentence-transformers, none"
    )
