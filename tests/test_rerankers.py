import sys
import types

import pytest

import core
import rerankers


@pytest.mark.asyncio
async def test_flashrank_backend_preserves_ranker_scores():
    backend = rerankers.build_reranker(
        backend="flashrank",
        model="ms-marco-MiniLM-L-12-v2",
        max_length=128,
        batch_size=4,
        device=None,
    )

    scores = await backend.rerank("query", ["first", "second"])

    assert backend.name == "flashrank"
    assert backend.model == "ms-marco-MiniLM-L-12-v2"
    assert scores == [(0, 0.5), (1, 0.5)]


@pytest.mark.asyncio
async def test_sentence_transformers_backend_sorts_cross_encoder_scores(monkeypatch):
    fake_module = types.ModuleType("sentence_transformers")
    captured = {}

    class FakeCrossEncoder:
        def __init__(self, model, *, max_length, device):
            captured["init"] = {
                "model": model,
                "max_length": max_length,
                "device": device,
            }

        def predict(self, pairs, *, batch_size, show_progress_bar):
            captured["predict"] = {
                "pairs": pairs,
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
            }
            return [0.2, 0.9, -0.1]

    fake_module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    backend = rerankers.build_reranker(
        backend="cross_encoder",
        model="cross-encoder/ms-marco-MiniLM-L6-v2",
        max_length=256,
        batch_size=8,
        device="cpu",
    )

    scores = await backend.rerank("query", ["first", "second", "third"])

    assert backend.name == "sentence-transformers"
    assert captured["init"] == {
        "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "max_length": 256,
        "device": "cpu",
    }
    assert captured["predict"] == {
        "pairs": [("query", "first"), ("query", "second"), ("query", "third")],
        "batch_size": 8,
        "show_progress_bar": False,
    }
    assert scores == [(1, 0.9), (0, 0.2), (2, -0.1)]


@pytest.mark.asyncio
async def test_noop_backend_preserves_input_order():
    backend = rerankers.build_reranker(
        backend="none",
        model="ignored",
        max_length=128,
        batch_size=4,
        device=None,
    )

    assert await backend.rerank("query", ["first", "second"]) == [(0, 0.0), (1, 0.0)]


def test_invalid_backend_raises_clear_error():
    with pytest.raises(ValueError, match="invalid RERANK_BACKEND"):
        rerankers.build_reranker(
            backend="qwen",
            model="ignored",
            max_length=128,
            batch_size=4,
            device=None,
        )


def test_invalid_backend_settings_raise_clear_errors():
    with pytest.raises(ValueError, match="RERANK_MAX_LENGTH"):
        rerankers.build_reranker(
            backend="none",
            model="ignored",
            max_length=0,
            batch_size=4,
            device=None,
        )
    with pytest.raises(ValueError, match="RERANK_BATCH_SIZE"):
        rerankers.build_reranker(
            backend="none",
            model="ignored",
            max_length=128,
            batch_size=0,
            device=None,
        )


def test_core_uses_flashrank_by_default():
    assert core.RERANK_NAME == "flashrank"
