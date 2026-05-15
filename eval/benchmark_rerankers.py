"""Compare configured local reranker backends on the benchmark query set.

Each reranker runs in a fresh subprocess because the server imports and
initializes the configured model once at process startup.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPECS = [
    "flashrank=flashrank:ms-marco-MiniLM-L-12-v2",
    "minilm-l6=sentence-transformers:cross-encoder/ms-marco-MiniLM-L6-v2",
]
SUMMARY_METRICS = [
    "avg_latency_ms_total",
    "avg_results_returned",
    "avg_scraped_results",
    "degraded_queries",
    "avg_expected_domain_hits_3",
    "avg_expected_domain_hits_5",
    "avg_top3_usefulness_score",
    "avg_top5_usefulness_score",
    "top3_targets_met",
    "top5_targets_met",
]


@dataclass(frozen=True)
class RerankerSpec:
    label: str
    backend: str
    model: str


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "reranker"


def parse_spec(raw: str) -> RerankerSpec:
    """Parse ``[label=]backend:model`` benchmark specs."""
    label: str | None = None
    body = raw.strip()
    if "=" in body:
        label, body = body.split("=", 1)
    if ":" not in body:
        raise ValueError(f"invalid reranker spec {raw!r}; expected [label=]backend:model")
    backend, model = body.split(":", 1)
    backend = backend.strip()
    model = model.strip()
    if not backend or not model:
        raise ValueError(f"invalid reranker spec {raw!r}; backend and model are required")
    return RerankerSpec(
        label=_slug(label or f"{backend}-{model}"),
        backend=backend,
        model=model,
    )


def _run_command(args: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout.rstrip())
        raise subprocess.CalledProcessError(result.returncode, args, output=result.stdout)
    return result


def run_eval_for_spec(
    spec: RerankerSpec,
    *,
    queries: Path,
    output_dir: Path,
    python: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    """Run eval + score for one reranker spec and return summary data."""
    run_file = output_dir / f"{spec.label}.jsonl"
    score_file = output_dir / f"{spec.label}.score.json"
    env = os.environ.copy()
    env.update({
        "RERANK_BACKEND": spec.backend,
        "RERANK_MODEL": spec.model,
        "RERANK_DEVICE": device,
        "RERANK_BATCH_SIZE": str(batch_size),
        "RERANK_MAX_LENGTH": str(max_length),
    })
    eval_cmd = [
        python,
        str(ROOT / "eval" / "run_eval.py"),
        "--queries",
        str(queries),
        "--output",
        str(run_file),
    ]
    score_cmd = [python, str(ROOT / "eval" / "score.py"), str(run_file)]

    print(f">> {spec.label}: {spec.backend} / {spec.model}")
    eval_result = _run_command(eval_cmd, env=env)
    if eval_result.stdout:
        print(eval_result.stdout.rstrip())
    score_result = _run_command(score_cmd, env=env)
    score_file.write_text(score_result.stdout)
    score_payload = json.loads(score_result.stdout)
    return {
        "label": spec.label,
        "backend": spec.backend,
        "model": spec.model,
        "run_file": str(run_file.relative_to(ROOT)),
        "score_file": str(score_file.relative_to(ROOT)),
        "aggregate": score_payload.get("aggregate", {}),
    }


def comparison_table(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return metric rows keyed by reranker label."""
    rows: list[dict[str, Any]] = []
    for metric in SUMMARY_METRICS:
        row: dict[str, Any] = {"metric": metric}
        for run in runs:
            row[run["label"]] = run.get("aggregate", {}).get(metric)
        rows.append(row)
    return rows


def _print_table(rows: list[dict[str, Any]], labels: list[str]) -> None:
    columns = ["metric", *labels]
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    print()
    print(" | ".join(column.ljust(widths[column]) for column in columns))
    print("-+-".join("-" * widths[column] for column in columns))
    for row in rows:
        print(" | ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark multiple local reranker backends.")
    parser.add_argument(
        "--spec",
        action="append",
        default=[],
        help="Reranker spec: [label=]backend:model. May be passed multiple times.",
    )
    parser.add_argument(
        "--queries",
        default=str(ROOT / "eval" / "queries.json"),
        help="Path to benchmark query JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "eval" / "runs" / "rerankers" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")),
        help="Directory for per-reranker JSONL runs and summary output.",
    )
    parser.add_argument("--device", default="cpu", help="Device passed to compatible reranker backends.")
    parser.add_argument("--batch-size", type=int, default=16, help="Reranker batch size.")
    parser.add_argument("--max-length", type=int, default=512, help="Reranker max sequence length.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for subprocess runs.")
    args = parser.parse_args()

    specs = [parse_spec(raw) for raw in (args.spec or DEFAULT_SPECS)]
    queries = Path(args.queries)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        run_eval_for_spec(
            spec,
            queries=queries,
            output_dir=output_dir,
            python=args.python,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        for spec in specs
    ]
    table = comparison_table(runs)
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "queries": str(queries.relative_to(ROOT)) if queries.is_relative_to(ROOT) else str(queries),
        "device": args.device,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "runs": runs,
        "comparison": table,
    }
    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    _print_table(table, [run["label"] for run in runs])
    print(f"\nwrote summary to {summary_file}")


if __name__ == "__main__":
    main()
