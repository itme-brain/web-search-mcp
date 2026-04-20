"""Head-to-head reranker comparison against the real failing case.

Context: our PDF extract path reranks pages using `page_content[:500]` as
input. Live smoke on the Mistral paper with query "attention mechanism"
returned bibliography pages, not the body. Two possible causes:

  1. MiniLM is too weak for technical/academic queries — model fault.
  2. Feeding only 500 chars favors reference pages (their first 500
     chars is a dense list of paper titles hitting any technical query)
     — truncation fault.

This script tests both axes independently by downloading the actual
Mistral 7B paper (arxiv 2310.06825), running our pymupdf4llm-based page
extraction, then running every combination of:

  model ∈ {ms-marco-MiniLM-L-12-v2 (current), rank-T5-flan (proposed)}
  truncation ∈ {500 chars, 2000 chars}

The output is the top-3 page numbers each combination picks, so you can
eyeball whether the fix is the model, the truncation, both, or neither.

Usage:
    nix develop -c python eval/compare_rerankers.py
"""

import sys
import time
from pathlib import Path

import httpx
import pymupdf
import pymupdf4llm
from flashrank import Ranker, RerankRequest


PDF_URL = "https://arxiv.org/pdf/2310.06825.pdf"
QUERIES = [
    "attention mechanism",
    "sliding window attention",
    "instruction tuning benchmark",
]
MODELS = ["ms-marco-MiniLM-L-12-v2", "rank-T5-flan"]
TRUNCATIONS = [500, 2000]


def fetch_pdf(url: str) -> bytes:
    cache = Path("/tmp/compare_rerankers_mistral.pdf")
    if cache.exists():
        print(f"using cached pdf: {cache}")
        return cache.read_bytes()
    print(f"downloading: {url}")
    data = httpx.get(url, timeout=60, follow_redirects=True).content
    cache.write_bytes(data)
    return data


def extract_pages(pdf_bytes: bytes) -> list[dict]:
    doc = pymupdf.Document(stream=pdf_bytes, filetype="pdf")
    chunks = pymupdf4llm.to_markdown(doc, page_chunks=True, hdr_info=False)
    pages: list[dict] = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        page_num = chunk.get("metadata", {}).get("page", 0) + 1
        pages.append({"page": page_num, "content": text})
    return pages


def rerank(model: str, query: str, passages: list[str]) -> tuple[int, list[tuple[int, float]]]:
    """Return (rerank_ms, [(passage_idx, score), ...] top 3)."""
    ranker = _model_cache.setdefault(model, Ranker(model_name=model, max_length=512))
    req = RerankRequest(
        query=query,
        passages=[{"id": i, "text": p, "meta": {}} for i, p in enumerate(passages)],
    )
    t0 = time.monotonic()
    results = ranker.rerank(req)
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    top3 = [(r["id"], round(float(r["score"]), 4)) for r in results[:3]]
    return elapsed_ms, top3


_model_cache: dict[str, Ranker] = {}


def _top3_summary(top3: list[tuple[int, float]], pages: list[dict]) -> str:
    return ", ".join(f"p{pages[i]['page']}({s})" for i, s in top3)


def main() -> int:
    pdf_bytes = fetch_pdf(PDF_URL)
    pages = extract_pages(pdf_bytes)
    if not pages:
        print("ERROR: no pages extracted", file=sys.stderr)
        return 1

    print(f"extracted {len(pages)} pages (page numbers: {[p['page'] for p in pages]})")
    print()

    # Pre-load models (first load is slow, shouldn't count against rerank latency)
    print("preloading models...")
    for model in MODELS:
        t0 = time.monotonic()
        _model_cache[model] = Ranker(model_name=model, max_length=512)
        ms = int((time.monotonic() - t0) * 1000)
        print(f"  {model}: {ms} ms")
    print()

    for query in QUERIES:
        print(f"{'=' * 72}")
        print(f"query: {query!r}")
        print(f"{'=' * 72}")
        for trunc in TRUNCATIONS:
            passages = [p["content"][:trunc] for p in pages]
            for model in MODELS:
                elapsed, top3 = rerank(model, query, passages)
                print(
                    f"  trunc={trunc:<4}  model={model:<30s}  "
                    f"{elapsed:>4}ms  top3: {_top3_summary(top3, pages)}"
                )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
