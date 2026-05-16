from web_search_mcp.http import policy as http_policy


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
