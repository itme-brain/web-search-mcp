"""Bounded production-model → canary-MCP tool-calling smoke."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

import httpx
from fastmcp import Client

_ALLOWED_TOOLS = frozenset({"search", "research", "read_evidence"})
_DEFAULT_MAX_ROUNDS = 8
_MAX_TOOL_RESULT_CHARS = 24000


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcp-url", default="http://localhost:18002/mcp")
    parser.add_argument("--llama-url", default=os.getenv("AGENT_LLM_BASE_URL", ""))
    parser.add_argument("--model", default=os.getenv("AGENT_LLM_MODEL", ""))
    parser.add_argument(
        "--question",
        default="What changed in FastMCP 4 background task support? Use web search and cite sources.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-rounds", type=int, default=_DEFAULT_MAX_ROUNDS)
    args = parser.parse_args()
    if not args.llama_url:
        parser.error("--llama-url or AGENT_LLM_BASE_URL is required")
    if not args.model:
        parser.error("--model or AGENT_LLM_MODEL is required")
    if args.max_rounds < 1:
        parser.error("--max-rounds must be at least 1")
    return args


def _openai_tools(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            },
        }
        for tool in mcp_tools
        if tool.name in _ALLOWED_TOOLS
    ]


def _assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    kept = {"role": "assistant", "content": message.get("content")}
    if message.get("tool_calls"):
        kept["tool_calls"] = message["tool_calls"]
    return kept


async def _completion(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 800,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    response = await client.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=body,
    )
    response.raise_for_status()
    body = response.json()
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("llama.cpp response omitted an assistant message") from exc
    if not isinstance(message, dict):
        raise ValueError("llama.cpp assistant message must be an object")
    return message


async def _run() -> int:
    args = _arguments()
    api_key = os.getenv("AGENT_LLM_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are an integration-test agent. Use the supplied web tools before answering. "
                "Base claims only on tool evidence and include source URLs in the final answer. "
                "After gathering sufficient evidence, stop calling tools and provide the final answer."
            ),
        },
        {"role": "user", "content": args.question},
    ]
    called_tools: list[str] = []
    final_answer = ""

    async with Client(args.mcp_url) as mcp_client:
        available = await mcp_client.list_tools()
        tools = _openai_tools(available)
        if not tools:
            raise RuntimeError("canary MCP exposed none of the allowed smoke-test tools")
        async with httpx.AsyncClient(headers=headers, timeout=args.timeout) as llm_client:
            for _ in range(args.max_rounds):
                message = await _completion(
                    llm_client, args.llama_url, args.model, messages, tools
                )
                messages.append(_assistant_message(message))
                tool_calls = message.get("tool_calls") or []
                if not tool_calls:
                    final_answer = message.get("content") or ""
                    break
                for call in tool_calls:
                    call_id = call.get("id")
                    if not isinstance(call_id, str) or not call_id:
                        raise ValueError("model tool call omitted its id")
                    function = call.get("function") or {}
                    name = function.get("name")
                    if name not in _ALLOWED_TOOLS:
                        raise ValueError(f"model requested disallowed tool: {name!r}")
                    raw_arguments = function.get("arguments") or "{}"
                    if isinstance(raw_arguments, str):
                        try:
                            arguments = json.loads(raw_arguments)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"model returned invalid arguments for {name}"
                            ) from exc
                    else:
                        arguments = raw_arguments
                    if not isinstance(arguments, dict):
                        raise ValueError(f"tool arguments for {name} must be an object")
                    result = await mcp_client.call_tool_mcp(name, arguments)
                    payload = result.structured_content
                    if payload is None:
                        payload = {"content": [getattr(item, "text", "") for item in result.content]}
                    content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": content[:_MAX_TOOL_RESULT_CHARS],
                    })
                    called_tools.append(name)

            if not final_answer.strip() and called_tools:
                messages.append({
                    "role": "system",
                    "content": (
                        "The tool-call budget is exhausted. Do not request more tools. "
                        "Now provide the final answer using the evidence already gathered."
                    ),
                })
                message = await _completion(
                    llm_client, args.llama_url, args.model, messages, []
                )
                final_answer = message.get("content") or ""

    if not called_tools:
        raise RuntimeError("production model did not call a canary MCP tool")
    if not final_answer.strip():
        raise RuntimeError(
            "production model did not produce a final answer after "
            f"{args.max_rounds} rounds; called_tools={called_tools}"
        )
    print(json.dumps({
        "status": "ok",
        "called_tools": called_tools,
        "answer_chars": len(final_answer),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
