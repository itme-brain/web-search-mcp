"""File/content-type classification helpers."""

import mimetypes
from urllib.parse import urlparse
def _content_type_without_charset(content_type: str | None) -> str | None:
    if not content_type:
        return None
    return content_type.split(";", 1)[0].strip().lower() or None


def _guess_file_type(url: str, content_type: str | None) -> str:
    normalized_content_type = _content_type_without_charset(content_type)
    suffix = (urlparse(url).path.rsplit(".", 1)[-1].lower() if "." in urlparse(url).path else "")

    if normalized_content_type == "application/pdf" or suffix == "pdf":
        return "pdf"
    if (
        normalized_content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or suffix == "docx"
    ):
        return "docx"
    if (
        normalized_content_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        or suffix == "xlsx"
    ):
        return "xlsx"
    if (
        normalized_content_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        or suffix == "pptx"
    ):
        return "pptx"
    if normalized_content_type in {"text/html", "application/xhtml+xml"} or suffix in {"html", "htm", "xhtml"}:
        return "html"
    if normalized_content_type in {"text/markdown", "text/x-markdown"} or suffix == "md":
        return "markdown"
    if normalized_content_type == "application/json" or suffix == "json":
        return "json"
    if normalized_content_type in {"application/yaml", "application/x-yaml", "text/yaml", "text/x-yaml"} or suffix in {"yaml", "yml"}:
        return "yaml"
    if normalized_content_type in {"application/xml", "text/xml"} or suffix in {"xml", "rss", "atom"}:
        return "xml"
    if normalized_content_type in {"text/csv", "application/csv", "application/vnd.ms-excel"} or suffix == "csv":
        return "csv"
    if normalized_content_type and normalized_content_type.startswith("text/"):
        return "text"
    guessed_type, _ = mimetypes.guess_type(url)
    guessed_type = _content_type_without_charset(guessed_type)
    if guessed_type and guessed_type.startswith("text/"):
        return "text"
    return "unknown"

