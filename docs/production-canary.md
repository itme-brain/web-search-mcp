# Production canary

The canary is a second MCP container in the existing Compose project. It is
disabled unless the `canary` profile is selected and publishes port 18002 by
default. Production remains on port 8002.

The candidate reuses production SearXNG and Crawl4AI because those services are
stateless upstream adapters. It uses `redis://valkey:6379/2` for retrieval and
evidence state and database 3 for FastMCP tasks. This keeps canary cache writes,
stable evidence handles, and task records out of production databases 0 and 1.

`just canary-agent-smoke` connects to the candidate MCP, converts its live input
schemas to OpenAI function tools, and offers a deliberately small allowlist to
the configured production model. It executes at most four model/tool rounds and
caps each tool result sent back to the model at 24,000 characters. A pass
requires at least one MCP tool call and a non-empty final response.

The script reads the llama.cpp bearer token only from `AGENT_LLM_API_KEY`. It
does not print the token, final answer, retrieved content, or model reasoning.
Use `just canary-stop` after validation; this stops only the candidate MCP.
