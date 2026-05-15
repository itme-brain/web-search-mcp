import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "benchmark_rerankers",
    ROOT / "eval" / "benchmark_rerankers.py",
)
benchmark_rerankers = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["benchmark_rerankers"] = benchmark_rerankers
SPEC.loader.exec_module(benchmark_rerankers)


def test_parse_spec_accepts_label_backend_and_model():
    parsed = benchmark_rerankers.parse_spec(
        "minilm-l6=sentence-transformers:cross-encoder/ms-marco-MiniLM-L6-v2"
    )

    assert parsed.label == "minilm-l6"
    assert parsed.backend == "sentence-transformers"
    assert parsed.model == "cross-encoder/ms-marco-MiniLM-L6-v2"


def test_parse_spec_generates_safe_label():
    parsed = benchmark_rerankers.parse_spec("sentence-transformers:cross-encoder/ms-marco-MiniLM-L6-v2")

    assert parsed.label == "sentence-transformers-cross-encoder-ms-marco-MiniLM-L6-v2"


@pytest.mark.parametrize("raw", ["flashrank", "label=flashrank:", ":model"])
def test_parse_spec_rejects_invalid_specs(raw):
    with pytest.raises(ValueError, match="invalid reranker spec"):
        benchmark_rerankers.parse_spec(raw)


def test_comparison_table_projects_common_metrics_by_label():
    rows = benchmark_rerankers.comparison_table([
        {
            "label": "a",
            "aggregate": {
                "avg_latency_ms_total": 10,
                "top5_targets_met": 2,
            },
        },
        {
            "label": "b",
            "aggregate": {
                "avg_latency_ms_total": 20,
                "top5_targets_met": 3,
            },
        },
    ])

    assert rows[0] == {
        "metric": "avg_latency_ms_total",
        "a": 10,
        "b": 20,
    }
    assert rows[-1] == {
        "metric": "top5_targets_met",
        "a": 2,
        "b": 3,
    }
