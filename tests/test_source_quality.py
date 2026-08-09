"""Authority classification and ranking regressions."""

from web_search_mcp.ranking import intent, source_quality


def test_personal_docs_path_is_not_official_documentation():
    assert source_quality.source_type(
        "https://akhilsharma90.github.io/tutorials/docs/async-rust"
    ) == "web"


def test_verified_documentation_hosts_remain_official():
    assert source_quality.source_type("https://docs.rs/tokio/latest/tokio/") == "official_docs"
    assert source_quality.source_type("https://docs.python.org/3/library/asyncio.html") == "official_docs"


def test_social_result_cannot_outrank_equally_relevant_project_docs():
    entries = [
        {"url": "https://www.linkedin.com/posts/tokio-vs-async-std", "title": "Tokio vs async-std"},
        {"url": "https://docs.rs/tokio/latest/tokio/", "title": "Tokio runtime"},
    ]
    scores = {0: 0.8, 1: 0.8}
    profile = intent.classify("Tokio versus async-std")

    social = source_quality.entry_sort_score(0, entries, scores, profile, query="Tokio versus async-std")
    official = source_quality.entry_sort_score(1, entries, scores, profile, query="Tokio versus async-std")

    assert official > social


def test_entity_affinity_distinguishes_async_std_from_generic_async_crate():
    query = "Tokio versus async-std runtime"
    async_native = {"url": "https://docs.rs/async-native-tls", "title": "async-native-tls"}
    async_std = {"url": "https://docs.rs/async-std", "title": "async-std"}

    assert source_quality.query_entity_boost(query, async_std) > source_quality.query_entity_boost(
        query, async_native
    )


def test_weak_sources_remain_results_but_are_ineligible_for_synthesis():
    assert source_quality.synthesis_eligible({
        "url": "https://www.linkedin.com/posts/tokio-vs-async-std",
    }) is False
    assert source_quality.synthesis_eligible({
        "url": "https://github.com/tokio-rs/tokio",
    }) is True
