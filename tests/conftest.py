import sys
import types
from pathlib import Path

import pytest

# Make the server's flat-layout modules importable as top-level names.
_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

# Stub flashrank BEFORE src/core.py gets imported (which instantiates Ranker
# at module-load time). Same ordering requirement as before — just applied to
# the new layout.
_flashrank = types.ModuleType("flashrank")


class _FakeRanker:
    def __init__(self, *args, **kwargs):
        pass

    def rerank(self, request):
        return [
            {"id": passage["id"], "score": 0.0}
            for passage in request.passages
        ]


class _FakeRerankRequest:
    def __init__(self, query, passages):
        self.query = query
        self.passages = passages


_flashrank.Ranker = _FakeRanker
_flashrank.RerankRequest = _FakeRerankRequest
sys.modules["flashrank"] = _flashrank

# Stub trafilatura if not installed (tests mock it per-test)
if "trafilatura" not in sys.modules:
    _trafilatura = types.ModuleType("trafilatura")
    _trafilatura.extract = lambda *args, **kwargs: None
    sys.modules["trafilatura"] = _trafilatura

# Import the four split modules. `core` pulls in Settings + Ranker on first
# import; the stub above has to be in place first.
import core  # noqa: E402
import formatters  # noqa: E402
import impls  # noqa: E402
import server  # noqa: E402

server_app = server.mcp


class _ServerModuleProxy:
    """Back-compat facade resolving attributes across the four split modules.

    Tests reference `server_module.X` — resolve X by walking the split
    modules in order. New tests can also `import core`, `import impls`,
    etc. directly. `unittest.mock.patch` targets (the "core.X" / "impls.X"
    strings) always point at the module that defines X, since mock.patch
    uses sys.modules rather than attribute lookup.
    """
    _search_order = (server, impls, core, formatters)

    def __getattr__(self, name):
        for mod in self._search_order:
            if hasattr(mod, name):
                return getattr(mod, name)
        raise AttributeError(f"no such attribute across split modules: {name!r}")


server_module = _ServerModuleProxy()


class FakeContext:
    """Minimal stub for fastmcp.Context in impl-level tests.

    Enforces the same JSON-serializability contract the real FastMCP
    Context does on `set_state`, so non-serializable values (e.g. a raw
    TTLCache) fail in pytest instead of silently passing here and
    blowing up only at runtime against a real MCP client.
    """

    def __init__(self):
        self._state = {}

    def get_state(self, key):
        return self._state.get(key)

    def set_state(self, key, value):
        try:
            import json
            json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"FakeContext.set_state({key!r}): value is not JSON-serializable "
                f"({type(value).__name__}: {exc}). Real FastMCP Context would reject "
                f"this too. Persist as a plain dict/list/primitive."
            ) from exc
        self._state[key] = value


def make_search_results(
    urls: list[str],
    prefix: str = "Result",
    unresponsive_engines: list | None = None,
) -> dict:
    """Build a fake _search() response: {results, unresponsive_engines}."""
    return {
        "results": [
            {"title": f"{prefix} {i}", "url": url, "content": f"snippet for {url}"}
            for i, url in enumerate(urls, 1)
        ],
        "unresponsive_engines": unresponsive_engines or [],
    }


URLS_A = [
    "https://example.com/a1",
    "https://example.com/a2",
    "https://example.com/a3",
]

URLS_B = [
    "https://example.com/a2",
    "https://example.com/b1",
    "https://example.com/b2",
]

SCRAPE_CONTENT = {
    "https://example.com/a1": "# Page A1\n\nThis is the full content of page A1 with enough text to pass the minimum chunk size threshold for reranking.",
    "https://example.com/a2": "# Page A2\n\nThis is the full content of page A2 with enough text to pass the minimum chunk size threshold for reranking.",
    "https://example.com/a3": "# Page A3\n\nThis is the full content of page A3 with enough text to pass the minimum chunk size threshold for reranking.",
    "https://example.com/b1": "# Page B1\n\nThis is the full content of page B1 with enough text to pass the minimum chunk size threshold for reranking.",
    "https://example.com/b2": "# Page B2\n\nThis is the full content of page B2 with enough text to pass the minimum chunk size threshold for reranking.",
}


@pytest.fixture
def fake_ctx():
    return FakeContext()
