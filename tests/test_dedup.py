from tests.conftest import server_module

_normalize_url = server_module._normalize_url
_dedup_results = server_module._dedup_results
_dedup_chunks = server_module._dedup_chunks
_diversify_ranked_entries = server_module._diversify_ranked_entries


def test_normalize_url_strips_tracking_params():
    url = "https://www.example.com/page?utm_source=google&utm_medium=cpc&q=test"
    assert _normalize_url(url) == "https://example.com/page?q=test"


def test_normalize_url_strips_www_and_trailing_slash():
    assert _normalize_url("https://www.example.com/path/") == "https://example.com/path"


def test_normalize_url_preserves_meaningful_params():
    url = "https://example.com/search?q=hello&page=2"
    assert "q=hello" in _normalize_url(url)
    assert "page=2" in _normalize_url(url)


def test_dedup_results_removes_url_duplicates():
    results = [
        {"url": "https://www.example.com/page?utm_source=google", "title": "First"},
        {"url": "https://example.com/page", "title": "Duplicate"},
        {"url": "https://other.com/page", "title": "Different"},
    ]
    deduped = _dedup_results(results)
    assert len(deduped) == 2
    assert deduped[0]["title"] == "First"
    assert deduped[1]["title"] == "Different"


def test_dedup_results_removes_same_domain_same_title_duplicates():
    results = [
        {"url": "https://example.com/post-1", "title": "Breaking News: Launch Day"},
        {"url": "https://example.com/post-2", "title": "Breaking News - Launch Day"},
        {"url": "https://other.com/post-3", "title": "Breaking News Launch Day"},
    ]
    deduped = _dedup_results(results)
    assert len(deduped) == 2
    assert deduped[0]["url"] == "https://example.com/post-1"
    assert deduped[1]["url"] == "https://other.com/post-3"


def test_dedup_chunks_removes_near_identical():
    base = "Large language models have transformed natural language processing with advanced attention mechanisms and deep learning techniques."
    chunks = [
        base,
        base + " Read more.",
        "Quantum computing uses qubits to perform calculations that classical computers cannot.",
    ]
    entry_map = [0, 1, 2]
    kept, entries = _dedup_chunks(chunks, entry_map)
    assert len(kept) == 2
    assert entries[0] == 0
    assert entries[1] == 2


def test_dedup_chunks_keeps_distinct():
    chunks = [
        "The quick brown fox jumps over the lazy dog near the riverbank.",
        "Quantum computing uses qubits to perform massively parallel calculations.",
        "Neural networks consist of layers of interconnected nodes processing data.",
    ]
    entry_map = [0, 1, 2]
    kept, entries = _dedup_chunks(chunks, entry_map)
    assert len(kept) == 3


def test_dedup_chunks_preserves_first_occurrence():
    chunks = [
        "This is an important finding about climate change and global warming effects.",
        "Something completely different about software engineering practices.",
        "This is an important finding about climate change and global warming effects.",
    ]
    entry_map = [0, 1, 2]
    kept, entries = _dedup_chunks(chunks, entry_map)
    assert len(kept) == 2
    assert entries[0] == 0
    assert entries[1] == 1


def test_diversify_ranked_entries_interleaves_domains():
    entries = [
        {"url": "https://a.com/1"},
        {"url": "https://a.com/2"},
        {"url": "https://b.com/1"},
        {"url": "https://a.com/3"},
        {"url": "https://c.com/1"},
    ]
    diversified = _diversify_ranked_entries([0, 1, 2, 3, 4], entries)
    assert diversified[:5] == [0, 2, 4, 1, 3]
