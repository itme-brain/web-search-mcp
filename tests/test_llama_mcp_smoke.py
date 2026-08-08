import sys
from types import SimpleNamespace

import httpx
import pytest

from scripts import llama_mcp_smoke


def test_arguments_default_to_bounded_eight_rounds(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_BASE_URL", "http://llama.test/v1")
    monkeypatch.setenv("AGENT_LLM_MODEL", "model")
    monkeypatch.setattr(sys, "argv", ["llama_mcp_smoke.py"])

    args = llama_mcp_smoke._arguments()

    assert args.max_rounds == 8


def test_openai_tools_exposes_only_bounded_canary_surface():
    tools = [
        SimpleNamespace(
            name="search",
            description="Search",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        ),
        SimpleNamespace(name="extract", description="Extract", input_schema={"type": "object"}),
    ]

    converted = llama_mcp_smoke._openai_tools(tools)

    assert [tool["function"]["name"] for tool in converted] == ["search"]
    assert converted[0]["function"]["parameters"]["properties"]["query"]["type"] == "string"


def test_assistant_message_drops_non_protocol_response_fields():
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call-1"}],
        "reasoning_content": "private reasoning",
    }

    assert llama_mcp_smoke._assistant_message(message) == {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call-1"}],
    }


@pytest.mark.asyncio
async def test_completion_uses_openai_tool_call_contract():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "done"}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        message = await llama_mcp_smoke._completion(
            client,
            "http://llama.test/v1",
            "model",
            [{"role": "user", "content": "question"}],
            [{"type": "function", "function": {"name": "search"}}],
        )

    assert message["content"] == "done"
    assert captured["url"] == "http://llama.test/v1/chat/completions"
    assert '"tool_choice":"auto"' in captured["body"]
