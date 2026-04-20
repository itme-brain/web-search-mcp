from unittest.mock import patch

from tests.conftest import server_module


class TestExtractMarkdown:
    def test_prefers_trafilatura_when_html_present(self):
        long_text = "This is substantial extracted content. " * 10
        result = {"html": "<html><body><p>test</p></body></html>", "markdown": "raw md"}

        with patch("web_search_server.trafilatura.extract", return_value=long_text) as mock:
            output = server_module._extract_markdown(result)
            mock.assert_called_once()
            assert output == long_text

    def test_falls_back_when_trafilatura_returns_short_text(self):
        result = {
            "html": "<html><body>x</body></html>",
            "markdown": {"fit_markdown": "fit content here", "raw_markdown": "raw content"},
        }

        with patch("web_search_server.trafilatura.extract", return_value="x"):
            output = server_module._extract_markdown(result)
            assert output == "fit content here"

    def test_falls_back_when_trafilatura_returns_none(self):
        result = {
            "html": "<html><body></body></html>",
            "markdown": {"fit_markdown": "fit md", "raw_markdown": "raw md"},
        }

        with patch("web_search_server.trafilatura.extract", return_value=None):
            output = server_module._extract_markdown(result)
            assert output == "fit md"

    def test_falls_back_when_trafilatura_raises(self):
        result = {
            "html": "<html></html>",
            "markdown": "string markdown",
        }

        with patch("web_search_server.trafilatura.extract", side_effect=Exception("boom")):
            output = server_module._extract_markdown(result)
            assert output == "string markdown"

    def test_no_html_uses_crawl4ai_markdown_dict(self):
        result = {"markdown": {"fit_markdown": "fit", "raw_markdown": "raw"}}
        output = server_module._extract_markdown(result)
        assert output == "fit"

    def test_no_html_uses_crawl4ai_markdown_string(self):
        result = {"markdown": "just a string"}
        output = server_module._extract_markdown(result)
        assert output == "just a string"

    def test_no_html_no_markdown_uses_cleaned_html(self):
        result = {"cleaned_html": "<p>cleaned</p>"}
        output = server_module._extract_markdown(result)
        assert output == "<p>cleaned</p>"

    def test_no_fields_returns_none(self):
        result = {}
        output = server_module._extract_markdown(result)
        assert output is None

    def test_fit_markdown_preferred_over_raw(self):
        result = {"markdown": {"fit_markdown": "fit", "raw_markdown": "raw"}}
        output = server_module._extract_markdown(result)
        assert output == "fit"

    def test_raw_markdown_when_fit_empty(self):
        result = {"markdown": {"fit_markdown": "", "raw_markdown": "raw"}}
        output = server_module._extract_markdown(result)
        assert output == "raw"
