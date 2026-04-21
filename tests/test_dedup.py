from tests.conftest import server_module

_normalize_url = server_module._normalize_url
_dedup_results = server_module._dedup_results
_dedup_chunks = server_module._dedup_chunks
_diversify_ranked_entries = server_module._diversify_ranked_entries
_chunk_text = server_module._chunk_text
_match_domain = server_module._match_domain
_filter_results_by_domain = server_module._filter_results_by_domain


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


def test_dedup_results_removes_same_domain_fuzzy_title_duplicates():
    results = [
        {"url": "https://example.com/post-1", "title": "Django ORM Guide (2026 Edition)"},
        {"url": "https://example.com/post-2", "title": "Django ORM Guide 2026 Edition"},
        {"url": "https://example.com/post-3", "title": "Django ORM Guide for Beginners"},
    ]
    deduped = _dedup_results(results)
    assert len(deduped) == 2
    assert deduped[0]["url"] == "https://example.com/post-1"
    assert deduped[1]["url"] == "https://example.com/post-3"


def test_dedup_results_keeps_same_domain_distinct_titles():
    results = [
        {"url": "https://example.com/models", "title": "Django ORM Models Guide"},
        {"url": "https://example.com/querysets", "title": "Django ORM QuerySets Guide"},
    ]
    deduped = _dedup_results(results)
    assert len(deduped) == 2
    assert deduped[0]["url"] == "https://example.com/models"
    assert deduped[1]["url"] == "https://example.com/querysets"


def test_match_domain_matches_subdomains_and_strips_www():
    assert _match_domain("www.docs.python.org", ["python.org"])
    assert _match_domain("docs.python.org", ["docs.python.org"])
    assert not _match_domain("python.org", ["docs.python.org"])


def test_filter_results_by_domain_handles_public_suffix_domains():
    results = [
        {"url": "https://www.docs.service.example.co.uk/guide", "title": "Guide"},
        {"url": "https://other.example.com/page", "title": "Other"},
    ]
    filtered = _filter_results_by_domain(results, ["example.co.uk"], [])
    assert [item["url"] for item in filtered] == ["https://www.docs.service.example.co.uk/guide"]


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


def test_chunk_text_preserves_markdown_structure_and_bounds_chunks():
    text = (
        "# Title\n\n"
        "Opening paragraph.\n\n"
        "## Section\n\n"
        + ("Sentence. " * 250)
    )
    chunks = _chunk_text(text)
    assert chunks[0].startswith("# Title")
    assert any("## Section" in chunk for chunk in chunks)
    assert len(chunks) > 1
    assert max(len(chunk) for chunk in chunks) <= 1000


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
