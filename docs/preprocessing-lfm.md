# LFM preprocessing

`src/web_search_mcp/preprocessing/lfm.py` is a fail-open adapter for an
OpenAI-compatible llama.cpp endpoint. It is disabled by default and runs only
for the `research` profile.

The configured model artifact is `LiquidAI/LFM2.5-2.6B-GGUF` /
`LFM2.5-2.6B-Q8_0.gguf`. `LFM_MODEL` is the model identifier sent to the API;
override it when the llama.cpp server exposes a different alias.

## Compose sidecar

The `lfm` Compose profile runs the pinned upstream llama.cpp server as a second
container. It is independent of any primary inference server on the host. The
sidecar has no published port and is reachable only as `http://lfm:8080/v1` on
the Compose network.

CPU-only execution is enforced with both `--device none` and
`--n-gpu-layers 0`. The service has one inference slot, an 8192-token total
context, an 8-CPU limit, and a 6 GB memory limit by default. The exact Q8 file is
downloaded once into the `lfm-models` volume. Run `just up-lfm`; preprocessing
is enabled for the MCP container created by that recipe. The ordinary `just up`
does not start or enable the sidecar.

The planning pass may select one of the server's known deterministic intent
profiles and return at most six normalized queries. Invalid output, timeouts,
or HTTP errors preserve the deterministic plan and emit a warning.

The digest pass sees bounded, untrusted evidence records. Its prompt explicitly
forbids following page instructions, and its output is accepted only when every
statement cites supplied passage IDs. Unknown citations, uncited statements,
duplicates, and oversized output are discarded. The underlying passages and
stable resources are always returned unchanged.
