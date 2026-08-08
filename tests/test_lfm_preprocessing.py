import pytest
import httpx

from web_search_mcp.preprocessing import lfm
from web_search_mcp.ranking import intent


@pytest.mark.asyncio
async def test_chat_completion_uses_openai_compatible_contract(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"intent":"comparison"}'}}]},
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    monkeypatch.setattr(lfm, "LFM_BASE_URL", "http://llama.test/v1")
    monkeypatch.setattr(lfm, "LFM_MODEL", "LFM2.5-2.6B-Q8_0.gguf")
    monkeypatch.setattr(lfm.httpx, "AsyncClient", lambda **_kwargs: client)

    result = await lfm._chat_completion("system", "user", max_tokens=64)

    assert result == {"intent": "comparison"}
    assert captured["url"] == "http://llama.test/v1/chat/completions"
    assert '"model":"LFM2.5-2.6B-Q8_0.gguf"' in captured["body"]


@pytest.mark.asyncio
async def test_disabled_planner_preserves_deterministic_plan(monkeypatch):
    async def unexpected_call(*_args, **_kwargs):
        raise AssertionError("disabled preprocessing must not call a model")

    monkeypatch.setattr(lfm, "ENABLE_LFM_PREPROCESSING", False)
    monkeypatch.setattr(lfm, "_chat_completion", unexpected_call)
    deterministic_intent = intent.classify("python api documentation")
    deterministic_queries = ["python api documentation", "python official documentation"]

    plan = await lfm.plan_query(
        "python api documentation", deterministic_intent, deterministic_queries
    )

    assert plan.intent == deterministic_intent
    assert plan.queries == deterministic_queries
    assert plan.used is False
    assert plan.error is None


@pytest.mark.asyncio
async def test_planner_validates_intent_and_deduplicates_queries(monkeypatch):
    async def completion(*_args, **_kwargs):
        return {
            "intent": "academic_research",
            "queries": [
                "graph retrieval paper",
                "graph retrieval paper",
                "x",
                42,
            ],
        }

    monkeypatch.setattr(lfm, "ENABLE_LFM_PREPROCESSING", True)
    monkeypatch.setattr(lfm, "_chat_completion", completion)
    fallback_intent = intent.classify("graph retrieval")

    plan = await lfm.plan_query("graph retrieval", fallback_intent, ["graph retrieval"])

    assert plan.intent.name == "academic_research"
    assert plan.queries == ["graph retrieval", "graph retrieval paper"]
    assert plan.used is True


@pytest.mark.asyncio
async def test_planner_fails_open(monkeypatch):
    async def completion(*_args, **_kwargs):
        raise TimeoutError("backend timed out")

    monkeypatch.setattr(lfm, "ENABLE_LFM_PREPROCESSING", True)
    monkeypatch.setattr(lfm, "_chat_completion", completion)
    fallback_intent = intent.classify("latest runtime release")
    fallback_queries = ["latest runtime release", "runtime release news"]

    plan = await lfm.plan_query("latest runtime release", fallback_intent, fallback_queries)

    assert plan.intent == fallback_intent
    assert plan.queries == fallback_queries
    assert plan.used is False
    assert plan.error == "backend timed out"


@pytest.mark.asyncio
async def test_digest_keeps_only_known_cited_statements(monkeypatch):
    async def completion(*_args, **_kwargs):
        return {
            "overview": [
                "Supported fact [1.1]",
                "Unsupported citation [9.9]",
                "Statement without citation",
                "Supported fact [1.1]",
                "Second fact [2.1]",
            ]
        }

    monkeypatch.setattr(lfm, "ENABLE_LFM_PREPROCESSING", True)
    monkeypatch.setattr(lfm, "_chat_completion", completion)
    results = [
        {
            "title": "One",
            "url": "https://example.com/one",
            "passages": [{"citation": "1.1", "text": "First evidence."}],
        },
        {
            "title": "Two",
            "url": "https://example.com/two",
            "passages": [{"citation": "2.1", "text": "Second evidence."}],
        },
    ]

    digest = await lfm.digest_evidence("question", results, ["fallback"])

    assert digest.overview == ["Supported fact [1.1]", "Second fact [2.1]"]
    assert digest.used is True
    assert digest.error is None
