from web_search_mcp.config.search import candidate_budget
from web_search_mcp.ranking.intent import classify
from web_search_mcp.ranking.query_expansion import keyphrase, search_queries
from web_search_mcp.tools.search import _interleave_result_groups


def test_time_range_forces_current_events_intent():
    profile = classify("AI model announcements", time_range="month")

    assert profile.name == "current_events"
    assert profile.freshness_sensitive is True


def test_comparison_intent_precedes_product_keywords():
    profile = classify("compare laptop prices versus desktop prices")

    assert profile.name == "comparison"


def test_research_expansion_uses_intent_specific_queries():
    queries = search_queries("battery chemistry", "research", "academic_research")

    assert any("arxiv" in query for query in queries)
    assert all("github gitlab" not in query for query in queries)


def test_research_keyphrase_starts_with_subject_not_query_modifiers():
    assert keyphrase("What are current best practices for microservices architecture?") == (
        "microservices architecture"
    )


def test_research_candidate_merge_fairly_represents_query_variants():
    groups = [
        [{"url": f"https://broad.example/{i}"} for i in range(10)],
        [{"url": f"https://subject.example/{i}"} for i in range(10)],
        [{"url": f"https://official.example/{i}"} for i in range(10)],
    ]

    bounded = _interleave_result_groups(groups)[:6]

    assert [result["url"] for result in bounded] == [
        "https://broad.example/0",
        "https://subject.example/0",
        "https://official.example/0",
        "https://broad.example/1",
        "https://subject.example/1",
        "https://official.example/1",
    ]


def test_candidate_pool_is_larger_than_final_result_budget():
    assert candidate_budget("search", 3) > 3
    assert candidate_budget("research", 8) > 8
