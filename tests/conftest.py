import importlib.util
import sys
from pathlib import Path

import pytest

_SERVER_PATH = Path(__file__).resolve().parent.parent / "mcp" / "server.py"
_spec = importlib.util.spec_from_file_location("web_search_server", _SERVER_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["web_search_server"] = _mod
_spec.loader.exec_module(_mod)

server_module = _mod
server_app = _mod.mcp


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
    "https://example.com/a1": "# Page A1\nContent of page A1",
    "https://example.com/a2": "# Page A2\nContent of page A2",
    "https://example.com/a3": "# Page A3\nContent of page A3",
    "https://example.com/b1": "# Page B1\nContent of page B1",
    "https://example.com/b2": "# Page B2\nContent of page B2",
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
