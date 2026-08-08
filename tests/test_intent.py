from web_search_mcp.config.search import candidate_budget
from web_search_mcp.ranking.intent import classify
from web_search_mcp.ranking.query_expansion import search_queries


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


def test_candidate_pool_is_larger_than_final_result_budget():
    assert candidate_budget("search", 3) > 3
    assert candidate_budget("research", 8) > 8
