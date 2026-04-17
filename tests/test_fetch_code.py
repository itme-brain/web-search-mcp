from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import Client

from tests.conftest import server_app, server_module

_to_raw_url = server_module._to_raw_url
_lang_from_path = server_module._lang_from_path


class TestToRawUrl:
    def test_github_blob(self):
        url = "https://github.com/owner/repo/blob/main/src/file.py"
        assert _to_raw_url(url) == "https://raw.githubusercontent.com/owner/repo/main/src/file.py"

    def test_github_blob_with_ref(self):
        url = "https://github.com/owner/repo/blob/abc123/deep/path/file.rs"
        assert _to_raw_url(url) == "https://raw.githubusercontent.com/owner/repo/abc123/deep/path/file.rs"

    def test_github_raw(self):
        url = "https://github.com/owner/repo/raw/main/file.txt"
        assert _to_raw_url(url) == "https://raw.githubusercontent.com/owner/repo/main/file.txt"

    def test_gitlab_blob(self):
        url = "https://gitlab.com/owner/repo/-/blob/main/src/file.py"
        assert _to_raw_url(url) == "https://gitlab.com/owner/repo/-/raw/main/src/file.py"

    def test_codeberg_src_branch(self):
        url = "https://codeberg.org/owner/repo/src/branch/main/file.go"
        assert _to_raw_url(url) == "https://codeberg.org/owner/repo/raw/branch/main/file.go"

    def test_codeberg_src_tag(self):
        url = "https://codeberg.org/owner/repo/src/tag/v1.0/file.go"
        assert _to_raw_url(url) == "https://codeberg.org/owner/repo/raw/tag/v1.0/file.go"

    def test_github_gist(self):
        url = "https://gist.github.com/user/abc123def456"
        assert _to_raw_url(url) == "https://gist.githubusercontent.com/user/abc123def456/raw/"

    def test_unsupported_host_returns_none(self):
        assert _to_raw_url("https://example.com/file.py") is None

    def test_short_path_returns_none(self):
        assert _to_raw_url("https://github.com/owner") is None

    def test_github_non_blob_path_returns_none(self):
        assert _to_raw_url("https://github.com/owner/repo/issues/42") is None


class TestLangFromPath:
    def test_python(self):
        assert _lang_from_path("file.py") == "python"

    def test_rust(self):
        assert _lang_from_path("src/main.rs") == "rust"

    def test_nix(self):
        assert _lang_from_path("flake.nix") == "nix"

    def test_no_extension(self):
        assert _lang_from_path("Makefile") == ""


@pytest.mark.asyncio
async def test_fetch_code_github_success():
    mock_response = AsyncMock()
    mock_response.text = "def hello():\n    print('world')\n"
    mock_response.raise_for_status = lambda: None

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("web_search_server.httpx.AsyncClient", return_value=mock_client):
        async with Client(server_app) as client:
            result = await client.call_tool(
                "fetch_code", {"url": "https://github.com/owner/repo/blob/main/hello.py"}
            )
            text = result.content[0].text

    assert "# hello.py" in text
    assert "```python" in text
    assert "def hello():" in text


@pytest.mark.asyncio
async def test_fetch_code_unsupported_url():
    async with Client(server_app) as client:
        result = await client.call_tool(
            "fetch_code", {"url": "https://example.com/some/file.py"}
        )
        text = result.content[0].text

    assert "Unsupported" in text
