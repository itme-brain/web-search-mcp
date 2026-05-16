import sys
import types
from pathlib import Path

import pytest
import pytest_asyncio

# Make the src package importable.
_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

# Stub flashrank BEFORE the reranker service gets imported, because the default
# reranker backend is instantiated at module-load time.
_flashrank = types.ModuleType("flashrank")


class _FakeRanker:
    def __init__(self, *args, **kwargs):
        pass

    def rerank(self, request):
        return [
            {"id": passage["id"], "score": 0.5}
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
    _trafilatura.extract_metadata = lambda *args, **kwargs: None
    sys.modules["trafilatura"] = _trafilatura

# Import the split modules after dependency stubs are in place.
from web_search_mcp.storage import cache  # noqa: E402
from web_search_mcp import common  # noqa: E402
from web_search_mcp.crawling import operations as crawl_operations  # noqa: E402
from web_search_mcp.extraction import documents, file_types, html, pdf  # noqa: E402
from web_search_mcp.storage import pages  # noqa: E402
from web_search_mcp.presentation import formatters  # noqa: E402
from web_search_mcp.ranking import service as ranking_service  # noqa: E402
from web_search_mcp import server  # noqa: E402
from web_search_mcp.tools import crawl as crawl_tool  # noqa: E402
from web_search_mcp.tools import extract as extract_tool  # noqa: E402
from web_search_mcp.tools import map as map_tool  # noqa: E402
from web_search_mcp.tools import search as search_tool  # noqa: E402

server_app = server.mcp


class _ServerModuleProxy:
    """Test facade resolving attributes across split modules.

    Tests reference `server_module.X` — resolve X by walking the split
    modules in order. `unittest.mock.patch` targets point at the module
    that defines or imports the runtime use site.
    """
    _search_order = (
        server,
        search_tool,
        extract_tool,
        map_tool,
        crawl_tool,
        documents,
        file_types,
        pdf,
        html,
        pages,
        crawl_operations,
        ranking_service,
        common,
        formatters,
    )

    def __getattr__(self, name):
        for mod in self._search_order:
            if hasattr(mod, name):
                return getattr(mod, name)
        raise AttributeError(f"no such attribute across split modules: {name!r}")


server_module = _ServerModuleProxy()



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



@pytest_asyncio.fixture(autouse=True)
async def _fake_valkey():
    """Swap the cache module's backing client with a fresh fakeredis
    instance for every test. Autouse so tests don't have to opt in."""
    import fakeredis.aioredis

    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    cache.set_client(fake)
    try:
        yield fake
    finally:
        await fake.flushall()
        await fake.aclose()
        cache.set_client(None)
