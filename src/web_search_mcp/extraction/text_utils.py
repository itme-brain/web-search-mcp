"""Text chunking and deduplication helpers."""

import re

from datasketch import MinHash, MinHashLSH
from langchain_text_splitters import MarkdownTextSplitter

DEDUP_SIMILARITY = 0.75
DEDUP_NUM_PERM = 128
WORD_SPLIT = re.compile(r"\W+")
MARKDOWN_SPLITTER = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=0)


def word_set(text: str) -> set[str]:
    return {word for word in WORD_SPLIT.split(text.lower()) if word}


def chunk_minhash(words: set[str]) -> MinHash:
    sketch = MinHash(num_perm=DEDUP_NUM_PERM)
    for word in sorted(words):
        if word:
            sketch.update(word.encode("utf-8"))
    return sketch


def dedup_chunks(chunks: list[str], entry_map: list[int]) -> tuple[list[str], list[int]]:
    kept_chunks: list[str] = []
    kept_entries: list[int] = []
    lsh = MinHashLSH(threshold=DEDUP_SIMILARITY, num_perm=DEDUP_NUM_PERM)
    for chunk, eidx in zip(chunks, entry_map):
        words = word_set(chunk)
        if not words:
            continue
        sketch = chunk_minhash(words)
        if lsh.query(sketch):
            continue
        key = len(kept_chunks)
        lsh.insert(key, sketch)
        kept_chunks.append(chunk)
        kept_entries.append(eidx)
    return kept_chunks, kept_entries


def dedup_pages(entries: list[dict], *, min_chars: int = 200) -> tuple[list[dict], int]:
    """Collapse pages with near-identical body content; keep first-seen entry."""
    kept: list[dict] = []
    lsh = MinHashLSH(threshold=DEDUP_SIMILARITY, num_perm=DEDUP_NUM_PERM)
    dropped = 0
    for entry in entries:
        content = entry.get("content") or ""
        if len(content) < min_chars:
            kept.append(entry)
            continue
        words = word_set(content)
        if not words:
            kept.append(entry)
            continue
        sketch = chunk_minhash(words)
        if lsh.query(sketch):
            dropped += 1
            continue
        lsh.insert(len(kept), sketch)
        kept.append(entry)
    return kept, dropped


def chunk_text(text: str) -> list[str]:
    """Split extracted markdown into bounded chunks for reranking."""
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    chunks: list[str] = []
    for block in blocks:
        if len(block) <= 1000:
            chunks.append(block)
            continue
        chunks.extend(
            chunk.strip()
            for chunk in MARKDOWN_SPLITTER.split_text(block)
            if chunk.strip()
        )
    return chunks
