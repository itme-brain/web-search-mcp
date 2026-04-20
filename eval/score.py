import argparse
import json
from collections import Counter
from pathlib import Path


def _load_run(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _top_domain_concentration(results: list[dict], *, top_n: int) -> int:
    domains = [item.get("domain", "") for item in results[:top_n] if item.get("domain")]
    if not domains:
        return 0
    return Counter(domains).most_common(1)[0][1]


def _count_expected_domain_hits(results: list[dict], expected_domains: list[str], *, top_n: int) -> int:
    if not expected_domains:
        return 0
    hits = 0
    for item in results[:top_n]:
        domain = item.get("domain", "")
        if any(domain == expected or domain.endswith(f".{expected}") for expected in expected_domains):
            hits += 1
    return hits


def _usefulness_score(results: list[dict], expected_domains: list[str], *, top_n: int) -> int:
    subset = results[:top_n]
    if not subset:
        return 0
    if expected_domains:
        hits = _count_expected_domain_hits(results, expected_domains, top_n=top_n)
        if hits >= 2:
            return 2
        if hits >= 1:
            return 1
        return 0
    scraped_hits = sum(1 for item in subset if item.get("scraped") and item.get("content"))
    if scraped_hits >= min(2, top_n):
        return 2
    if scraped_hits >= 1:
        return 1
    return 0


def _query_summary(row: dict) -> dict:
    response = row["response"]
    results = response.get("results", [])
    meta = response.get("meta", {})
    judgments = row.get("judgments", {})
    expected_domains = judgments.get("expected_domains", [])
    top3_usefulness = _usefulness_score(results, expected_domains, top_n=3)
    top5_usefulness = _usefulness_score(results, expected_domains, top_n=5)
    scraped = sum(1 for item in results if item.get("scraped"))
    return {
        "id": row["id"],
        "query": row["query"],
        "results_returned": len(results),
        "scraped_results": scraped,
        "degraded": bool(meta.get("degraded")),
        "warnings": len(meta.get("warnings", [])),
        "latency_ms_total": meta.get("timings_ms", {}).get("total", 0),
        "top_domain_concentration_3": _top_domain_concentration(results, top_n=3),
        "top_domain_concentration_5": _top_domain_concentration(results, top_n=5),
        "expected_domain_hits_3": _count_expected_domain_hits(results, expected_domains, top_n=3),
        "expected_domain_hits_5": _count_expected_domain_hits(results, expected_domains, top_n=5),
        "freshness_sensitive": bool(judgments.get("freshness_sensitive", False)),
        "top3_usefulness_score": top3_usefulness,
        "top5_usefulness_score": top5_usefulness,
        "top3_usefulness_target": judgments.get("top3_usefulness_target"),
        "top5_usefulness_target": judgments.get("top5_usefulness_target"),
        "top3_target_met": (
            judgments.get("top3_usefulness_target") is not None
            and top3_usefulness >= judgments["top3_usefulness_target"]
        ),
        "top5_target_met": (
            judgments.get("top5_usefulness_target") is not None
            and top5_usefulness >= judgments["top5_usefulness_target"]
        ),
    }


def _aggregate(summaries: list[dict]) -> dict:
    count = len(summaries)
    if not count:
        return {}
    return {
        "queries": count,
        "avg_latency_ms_total": round(sum(item["latency_ms_total"] for item in summaries) / count, 2),
        "avg_results_returned": round(sum(item["results_returned"] for item in summaries) / count, 2),
        "avg_scraped_results": round(sum(item["scraped_results"] for item in summaries) / count, 2),
        "degraded_queries": sum(1 for item in summaries if item["degraded"]),
        "avg_top_domain_concentration_3": round(
            sum(item["top_domain_concentration_3"] for item in summaries) / count, 2
        ),
        "avg_top_domain_concentration_5": round(
            sum(item["top_domain_concentration_5"] for item in summaries) / count, 2
        ),
        "avg_expected_domain_hits_3": round(sum(item["expected_domain_hits_3"] for item in summaries) / count, 2),
        "avg_expected_domain_hits_5": round(sum(item["expected_domain_hits_5"] for item in summaries) / count, 2),
        "avg_top3_usefulness_score": round(sum(item["top3_usefulness_score"] for item in summaries) / count, 2),
        "avg_top5_usefulness_score": round(sum(item["top5_usefulness_score"] for item in summaries) / count, 2),
        "top3_targets_met": sum(1 for item in summaries if item["top3_target_met"]),
        "top5_targets_met": sum(1 for item in summaries if item["top5_target_met"]),
        "freshness_sensitive_queries": sum(1 for item in summaries if item["freshness_sensitive"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score an evaluation run from eval/run_eval.py output.")
    parser.add_argument("run_file", help="Path to a JSONL run file.")
    args = parser.parse_args()

    rows = _load_run(Path(args.run_file))
    summaries = [_query_summary(row) for row in rows]
    aggregate = _aggregate(summaries)

    print(json.dumps({"aggregate": aggregate, "queries": summaries}, indent=2))


if __name__ == "__main__":
    main()
