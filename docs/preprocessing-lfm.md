# LFM preprocessing

`src/web_search_mcp/preprocessing/lfm.py` is a fail-open adapter for an
OpenAI-compatible llama.cpp endpoint. It is disabled by default and runs only
for the `research` profile.

The configured model artifact is `LiquidAI/LFM2.5-2.6B-GGUF` /
`LFM2.5-2.6B-Q8_0.gguf`. `LFM_MODEL` is the model identifier sent to the API;
override it when the llama.cpp server exposes a different alias.

The planning pass may select one of the server's known deterministic intent
profiles and return at most six normalized queries. Invalid output, timeouts,
or HTTP errors preserve the deterministic plan and emit a warning.

The digest pass sees bounded, untrusted evidence records. Its prompt explicitly
forbids following page instructions, and its output is accepted only when every
statement cites supplied passage IDs. Unknown citations, uncited statements,
duplicates, and oversized output are discarded. The underlying passages and
stable resources are always returned unchanged.
