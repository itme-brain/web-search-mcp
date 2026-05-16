from web_search_mcp.http import policy as http_policy


class _FakeMozillaResponse:
    def __init__(self, version: str) -> None:
        self._version = version

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return {"LATEST_FIREFOX_VERSION": self._version}


class _FakeMozillaClient:
    calls = 0

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def get(self, url: str) -> _FakeMozillaResponse:
        self.__class__.calls += 1
        assert url == http_policy._FIREFOX_VERSIONS_URL
        return _FakeMozillaResponse("150.0")


class _FailingMozillaClient(_FakeMozillaClient):
    async def get(self, url: str) -> _FakeMozillaResponse:
        self.__class__.calls += 1
        raise http_policy.httpx.ConnectError("offline")


def _reset_user_agent_cache() -> None:
    http_policy._cache.clear()
    http_policy._cache.update({"version": None, "_lock": None, "_fetched": False})


def test_browser_compatible_headers_are_stable_and_configurable(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_MCP_USER_AGENT", "web-search-mcp-test/1.0")
    monkeypatch.setenv("WEB_SEARCH_MCP_ACCEPT_LANGUAGE", "fr-FR,fr;q=0.8")

    headers = http_policy.browser_compatible_headers()

    assert headers["User-Agent"] == "web-search-mcp-test/1.0"
    assert headers["Accept-Language"] == "fr-FR,fr;q=0.8"
    assert "text/html" in headers["Accept"]
    assert headers["Accept-Encoding"] == "gzip, deflate, br"


def test_identity_headers_disable_compression_for_range_reads(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_MCP_USER_AGENT", "web-search-mcp-test/1.0")

    headers = http_policy.identity_headers(accept="application/pdf")

    assert headers["User-Agent"] == "web-search-mcp-test/1.0"
    assert headers["Accept"] == "application/pdf"
    assert headers["Accept-Encoding"] == "identity"


def test_wikimedia_api_headers_include_api_user_agent(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_MCP_USER_AGENT", "web-search-mcp-test/1.0")

    headers = http_policy.wikimedia_api_headers()

    assert headers["User-Agent"] == "web-search-mcp-test/1.0"
    assert headers["Api-User-Agent"] == "web-search-mcp-test/1.0"
    assert headers["Accept"] == "application/json"


def test_default_user_agent_uses_fallback_before_startup_initialization(monkeypatch):
    monkeypatch.delenv("WEB_SEARCH_MCP_USER_AGENT", raising=False)
    _reset_user_agent_cache()

    ua = http_policy.user_agent()

    assert "Windows NT 10.0; Win64; x64" in ua
    assert f"Firefox/{http_policy._FIREFOX_FALLBACK_VERSION}" in ua


async def test_user_agent_initialization_fetches_and_caches_latest_firefox(monkeypatch):
    monkeypatch.delenv("WEB_SEARCH_MCP_USER_AGENT", raising=False)
    monkeypatch.setattr(http_policy.httpx, "AsyncClient", _FakeMozillaClient)
    _FakeMozillaClient.calls = 0
    _reset_user_agent_cache()

    initialized = await http_policy.ensure_user_agent_initialized()
    cached = await http_policy.ensure_user_agent_initialized()

    assert initialized == cached
    assert "rv:150.0" in initialized
    assert http_policy.user_agent() == initialized
    assert _FakeMozillaClient.calls == 1


async def test_user_agent_initialization_falls_back_when_version_fetch_fails(monkeypatch):
    monkeypatch.delenv("WEB_SEARCH_MCP_USER_AGENT", raising=False)
    monkeypatch.setattr(http_policy.httpx, "AsyncClient", _FailingMozillaClient)
    _FailingMozillaClient.calls = 0
    _reset_user_agent_cache()

    initialized = await http_policy.ensure_user_agent_initialized()
    cached = await http_policy.ensure_user_agent_initialized()

    assert initialized == cached
    assert f"Firefox/{http_policy._FIREFOX_FALLBACK_VERSION}" in initialized
    assert _FailingMozillaClient.calls == 1


async def test_configured_user_agent_remains_authoritative_after_initialization(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_MCP_USER_AGENT", "web-search-mcp-test/2.0")
    monkeypatch.setattr(http_policy.httpx, "AsyncClient", _FakeMozillaClient)
    _reset_user_agent_cache()

    await http_policy.ensure_user_agent_initialized()

    assert http_policy.user_agent() == "web-search-mcp-test/2.0"
