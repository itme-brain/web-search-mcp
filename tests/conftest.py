import importlib.util
import sys
import types
from pathlib import Path

import pytest

_SERVER_PATH = Path(__file__).resolve().parent.parent / "mcp" / "server.py"
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

_spec = importlib.util.spec_from_file_location("web_search_server", _SERVER_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["web_search_server"] = _mod
_spec.loader.exec_module(_mod)

server_module = _mod
server_app = _mod.mcp


class FakeContext:
    def __init__(self):
        self._state = {}

    def get_state(self, key):
        return self._state.get(key)

    def set_state(self, key, value):
        self._state[key] = value


def make_search_results(urls: list[str], prefix: str = "Result") -> list[dict]:
    return [
        {"title": f"{prefix} {i}", "url": url, "content": f"snippet for {url}"}
        for i, url in enumerate(urls, 1)
    ]


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
def search_results_a():
    return make_search_results(URLS_A, prefix="Alpha")


@pytest.fixture
def search_results_b():
    return make_search_results(URLS_B, prefix="Beta")


@pytest.fixture
def scrape_content():
    return dict(SCRAPE_CONTENT)


@pytest.fixture
def fake_ctx():
    return FakeContext()
