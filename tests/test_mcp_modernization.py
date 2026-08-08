import pytest
from fastmcp import Client

from tests.conftest import server_app, server_module


@pytest.mark.asyncio
async def test_tool_contracts_publish_bounded_json_schemas_and_annotations():
    async with Client(server_app) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}

    search = tools["search"]
    query_schema = search.input_schema["properties"]["query"]
    count_schema = search.input_schema["properties"]["num_results"]
    time_schema = search.input_schema["properties"]["time_range"]["anyOf"][0]
    domains_schema = search.input_schema["properties"]["include_domains"]["anyOf"][0]

    assert query_schema["minLength"] == 1
    assert query_schema["maxLength"] == 1000
    assert count_schema["minimum"] == 1
    assert count_schema["maximum"] == 10
    assert time_schema["enum"] == ["day", "week", "month", "year"]
    assert domains_schema["maxItems"] == 20
    assert search.annotations.read_only_hint is True
    assert search.annotations.destructive_hint is False
    assert search.annotations.idempotent_hint is True
    assert search.annotations.open_world_hint is True
    assert tools["read_evidence"].annotations.open_world_hint is False


@pytest.mark.asyncio
async def test_long_running_tools_are_task_eligible():
    research = await server_module.mcp.get_tool("research")
    crawl = await server_module.mcp.get_tool("crawl")
    search = await server_module.mcp.get_tool("search")

    assert "io.modelcontextprotocol/tasks" in server_module.mcp._extensions
    assert research.task_config.mode == "optional"
    assert crawl.task_config.mode == "optional"
    assert search.task_config.mode == "forbidden"
