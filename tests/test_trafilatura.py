from types import SimpleNamespace
from unittest.mock import patch

from tests.conftest import server_module


class TestExtractMarkdown:
    def test_prefers_trafilatura_when_html_present(self):
        long_text = "This is substantial extracted content. " * 10
        result = {"html": "<html><body><p>test</p></body></html>", "markdown": "raw md"}

        with patch("core.trafilatura.extract", return_value=long_text) as mock:
            output = server_module._extract_markdown(result)
            mock.assert_called_once()
            assert output == long_text

    def test_falls_back_when_trafilatura_returns_short_text(self):
        result = {
            "html": "<html><body>x</body></html>",
            "markdown": {"fit_markdown": "fit content here", "raw_markdown": "raw content"},
        }

        with patch("core.trafilatura.extract", return_value="x"):
            output = server_module._extract_markdown(result)
            assert output == "fit content here"

    def test_falls_back_when_trafilatura_returns_none(self):
        result = {
            "html": "<html><body></body></html>",
            "markdown": {"fit_markdown": "fit md", "raw_markdown": "raw md"},
        }

        with patch("core.trafilatura.extract", return_value=None):
            output = server_module._extract_markdown(result)
            assert output == "fit md"

    def test_falls_back_when_trafilatura_raises(self):
        result = {
            "html": "<html></html>",
            "markdown": "string markdown",
        }

        with patch("core.trafilatura.extract", side_effect=Exception("boom")):
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


class TestDocumentMetadata:
    def _doc(self, **kwargs):
        fields = {"author": None, "date": None, "sitename": None, "description": None}
        fields.update(kwargs)
        return SimpleNamespace(**fields)

    def test_extracts_author_date_site_name_description(self):
        doc = self._doc(
            author="Jane Doe",
            date="2024-01-15",
            sitename="Example Blog",
            description="A short description.",
        )
        with patch("core.trafilatura.extract_metadata", return_value=doc):
            metadata = server_module._extract_html_metadata("<html>...</html>")
        assert metadata == {
            "author": "Jane Doe",
            "date": "2024-01-15",
            "site_name": "Example Blog",
            "description": "A short description.",
        }

    def test_empty_html_returns_empty_dict(self):
        assert server_module._extract_html_metadata(None) == {}
        assert server_module._extract_html_metadata("") == {}

    def test_extract_metadata_returning_none_yields_empty_dict(self):
        with patch("core.trafilatura.extract_metadata", return_value=None):
            assert server_module._extract_html_metadata("<html/>") == {}

    def test_extract_metadata_raising_yields_empty_dict(self):
        with patch("core.trafilatura.extract_metadata", side_effect=Exception("boom")):
            assert server_module._extract_html_metadata("<html/>") == {}

    def test_build_document_metadata_adds_word_count(self):
        with patch("core.trafilatura.extract_metadata", return_value=self._doc()):
            metadata = server_module._build_document_metadata("<html/>", "one two three four five")
        assert metadata == {"word_count": 5}

    def test_build_document_metadata_strips_none_fields(self):
        metadata = server_module._build_document_metadata(None, "one two three")
        assert metadata == {"word_count": 3}

    def test_build_document_metadata_keeps_populated_fields(self):
        doc = self._doc(author="Alice", date="2024-03-01")
        with patch("core.trafilatura.extract_metadata", return_value=doc):
            metadata = server_module._build_document_metadata("<html/>", "hello world")
        assert metadata == {
            "author": "Alice",
            "date": "2024-03-01",
            "word_count": 2,
        }

    def test_build_document_metadata_does_not_duplicate_structure_already_in_markdown(self):
        """Headings and code blocks live inline in the markdown body where
        the LLM reads them. We intentionally do NOT emit parallel fields."""
        content = (
            "# Top\n\nintro body.\n\n"
            "## Sub\n\nparagraph\n\n"
            "```python\nprint('x')\n```\n"
        )
        with patch("core.trafilatura.extract_metadata", return_value=self._doc()):
            metadata = server_module._build_document_metadata("<html/>", content)
        assert "headings" not in metadata
        assert "code_blocks" not in metadata
        assert "outgoing_links" not in metadata
        assert "content_hash" not in metadata  # now envelope-level, not metadata


class TestTableSeparatorStrip:
    def test_strips_basic_separator_row(self):
        text = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        out = server_module._strip_table_separator_rows(text)
        assert out == "| A | B |\n| 1 | 2 |"

    def test_strips_alignment_variants(self):
        text = "| A | B | C |\n| :--- | :---: | ---: |\n| 1 | 2 | 3 |"
        out = server_module._strip_table_separator_rows(text)
        assert "---" not in out
        assert "| A | B | C |" in out
        assert "| 1 | 2 | 3 |" in out

    def test_preserves_data_rows_that_contain_dashes(self):
        text = "| Name | Value |\n| --- | --- |\n| foo-bar | baz-qux |"
        out = server_module._strip_table_separator_rows(text)
        assert "foo-bar" in out
        assert "baz-qux" in out

    def test_noop_when_no_table(self):
        text = "Just a paragraph with --- dashes but no table pipes."
        assert server_module._strip_table_separator_rows(text) == text

    def test_preserves_horizontal_rules(self):
        text = "heading\n\n---\n\nbody"
        assert server_module._strip_table_separator_rows(text) == text


class TestA11yLinkCleanup:
    def test_strips_skip_to_content_labels(self):
        result = {
            "links": {
                "internal": [
                    {"href": "/docs", "title": "Skip to main content", "text": "Skip to main content"},
                    {"href": "/api", "title": "API Reference", "text": "API"},
                ],
                "external": [],
            },
        }
        links = server_module._extract_crawl_links(result, "https://example.com")
        assert len(links) == 2
        assert links[0]["title"] is None
        assert links[0]["text"] is None
        assert links[0]["url"] == "https://example.com/docs"
        assert links[1]["title"] == "API Reference"

    def test_strips_variants(self):
        for label in ("skip navigation", "JUMP TO CONTENT", "Skip to main"):
            assert server_module._clean_link_label(label) is None

    def test_keeps_meaningful_text(self):
        assert server_module._clean_link_label("Getting Started") == "Getting Started"
        assert server_module._clean_link_label("  ") is None
        assert server_module._clean_link_label(None) is None
