"""In-process observability helpers for tool calls and retrieval stages."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Any


_DURATION_BUCKETS_MS = (50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000)
_lock = Lock()
_counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
_histograms: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, Any]] = {}


def _labels_key(labels: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((key, str(value)) for key, value in labels.items()))


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    rendered = ",".join(f'{key}="{_escape_label(value)}"' for key, value in labels)
    return f"{{{rendered}}}"


def counter(name: str, amount: float = 1.0, **labels: Any) -> None:
    with _lock:
        _counters[(name, _labels_key(labels))] += amount


def histogram(name: str, value: float, **labels: Any) -> None:
    key = (name, _labels_key(labels))
    with _lock:
        state = _histograms.setdefault(
            key,
            {
                "buckets": {bucket: 0 for bucket in _DURATION_BUCKETS_MS},
                "inf": 0,
                "count": 0,
                "sum": 0.0,
            },
        )
        for bucket in _DURATION_BUCKETS_MS:
            if value <= bucket:
                state["buckets"][bucket] += 1
        state["inf"] += 1
        state["count"] += 1
        state["sum"] += value


def observe_tool_response(tool: str, response: dict) -> None:
    """Record counters/histograms from a structured tool response."""
    meta = response.get("meta") or {}
    degraded = bool(meta.get("degraded", False))
    profile = str(meta.get("profile") or tool)
    status = "degraded" if degraded else "ok"
    counter("web_search_mcp_tool_requests_total", tool=tool, profile=profile, status=status)
    for warning in meta.get("warnings", []) or []:
        counter(
            "web_search_mcp_warnings_total",
            tool=tool,
            type=warning.get("type", "unknown"),
            source=warning.get("source", "unknown"),
        )
    timings = meta.get("timings_ms") or {}
    for stage, elapsed in timings.items():
        if elapsed is None:
            continue
        histogram("web_search_mcp_stage_duration_ms", float(elapsed), tool=tool, stage=stage)
    if "semantic_hits" in meta:
        counter("web_search_mcp_semantic_hits_total", float(meta.get("semantic_hits") or 0), tool=tool)


def prometheus_text(extra_gauges: dict[str, tuple[float, dict[str, Any]]] | None = None) -> str:
    """Render metrics in Prometheus text exposition format."""
    lines = [
        "# HELP web_search_mcp_tool_requests_total Tool responses by status.",
        "# TYPE web_search_mcp_tool_requests_total counter",
    ]
    with _lock:
        counters = dict(_counters)
        histograms = {key: {k: (v.copy() if isinstance(v, dict) else v) for k, v in state.items()} for key, state in _histograms.items()}

    for (name, labels), value in sorted(counters.items()):
        lines.append(f"{name}{_format_labels(labels)} {value:g}")

    if histograms:
        lines.extend([
            "# HELP web_search_mcp_stage_duration_ms Tool stage duration in milliseconds.",
            "# TYPE web_search_mcp_stage_duration_ms histogram",
        ])
    for (name, labels), state in sorted(histograms.items()):
        base_labels = dict(labels)
        for bucket in _DURATION_BUCKETS_MS:
            bucket_labels = _labels_key({**base_labels, "le": bucket})
            lines.append(f"{name}_bucket{_format_labels(bucket_labels)} {state['buckets'][bucket]:g}")
        inf_labels = _labels_key({**base_labels, "le": "+Inf"})
        lines.append(f"{name}_bucket{_format_labels(inf_labels)} {state['inf']:g}")
        lines.append(f"{name}_count{_format_labels(labels)} {state['count']:g}")
        lines.append(f"{name}_sum{_format_labels(labels)} {state['sum']:g}")

    for name, (value, labels) in sorted((extra_gauges or {}).items()):
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name}{_format_labels(_labels_key(labels))} {value:g}")
    return "\n".join(lines) + "\n"


def reset() -> None:
    """Clear in-process metrics. Used by tests."""
    with _lock:
        _counters.clear()
        _histograms.clear()
