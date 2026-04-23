"""Typed response models for MCP tool structured output."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChunkModel(StrictModel):
    text: str
    score: float


class ChunkSpecModel(StrictModel):
    """A chunk with a stable id — not rerank-scored.

    Returned on extract responses so callers can cherry-pick chunks by
    id on a follow-up call (`chunk_ids=[...]`) without re-scraping or
    re-reranking. IDs are indices into the chunk list derived from the
    first `_MAX_CONTENT_CHARS` of the document and are stable as long
    as the cached raw content is.
    """
    id: int
    text: str


class WarningModel(StrictModel):
    type: str
    source: str
    detail: str


class TimingModel(StrictModel):
    search: int | None = None
    scrape: int | None = None
    rerank: int | None = None
    total: int


class RerankerModel(StrictModel):
    name: str
    model: str


class HandoffModel(StrictModel):
    handler: str
    reason: str


class HeadingModel(StrictModel):
    level: int
    text: str


class OutgoingLinkModel(StrictModel):
    url: str
    text: str | None = None


class DocumentMetadataModel(StrictModel):
    author: str | None = None
    date: str | None = None
    site_name: str | None = None
    description: str | None = None
    word_count: int | None = None
    content_hash: str | None = None
    headings: list[HeadingModel] | None = None
    code_blocks: int | None = None
    outgoing_links: list[OutgoingLinkModel] | None = None


class SearchResultModel(StrictModel):
    rank: int
    search_rank: int
    title: str
    url: str
    normalized_url: str
    domain: str
    snippet: str
    content: str
    top_chunks: list[ChunkModel]
    score: float | None = None
    scraped: bool
    previously_seen: bool
    metadata: DocumentMetadataModel | None = None


class SearchMetaModel(StrictModel):
    num_results_requested: int
    num_results_returned: int
    scrape_top: int
    search_backend: str
    reranker: RerankerModel
    degraded: bool
    warnings: list[WarningModel]
    timings_ms: TimingModel


class SearchResponseModel(StrictModel):
    query: str
    time_range: str | None = None
    include_domains: list[str] | None = None
    exclude_domains: list[str] | None = None
    results: list[SearchResultModel]
    meta: SearchMetaModel


class ExtractResultModel(StrictModel):
    url: str
    normalized_url: str
    domain: str
    status: str
    content_type: str | None = None
    file_type: str | None = None
    title: str | None = None
    content: str
    chars_shown: int
    offset: int
    total_chars: int
    top_chunks: list[ChunkModel]
    chunks: list[ChunkSpecModel] = []
    cached: bool
    error: str | None = None
    handoff: HandoffModel | None = None
    metadata: DocumentMetadataModel | None = None


class ExtractMetaModel(StrictModel):
    urls_requested: int
    urls_succeeded: int
    urls_failed: int
    timings_ms: TimingModel


class ExtractResponseModel(StrictModel):
    query: str | None = None
    results: list[ExtractResultModel]
    meta: ExtractMetaModel


class MapResultModel(StrictModel):
    rank: int
    url: str
    normalized_url: str
    domain: str
    title: str | None = None
    link_text: str | None = None
    depth: int
    discovered_from: str | None = None
    link_type: str


class MapMetaModel(StrictModel):
    max_urls_requested: int
    urls_returned: int
    pages_visited: int
    warnings: list[WarningModel]
    timings_ms: TimingModel


class MapResponseModel(StrictModel):
    url: str
    results: list[MapResultModel]
    meta: MapMetaModel


class CrawlResultModel(StrictModel):
    rank: int
    url: str
    normalized_url: str
    domain: str
    title: str | None = None
    link_text: str | None = None
    depth: int
    discovered_from: str | None = None
    link_type: str
    status: str
    content_type: str | None = None
    content: str
    chars_shown: int
    offset: int
    total_chars: int
    cached: bool
    error: str | None = None
    metadata: DocumentMetadataModel | None = None


class CrawlMetaModel(StrictModel):
    max_urls_requested: int
    urls_discovered: int
    urls_returned: int
    urls_truncated_by_limit: int
    urls_deduplicated: int = 0
    urls_succeeded: int
    urls_failed: int
    warnings: list[WarningModel]
    timings_ms: TimingModel


class CrawlResponseModel(StrictModel):
    url: str
    results: list[CrawlResultModel]
    meta: CrawlMetaModel


def dump_response(model_cls: type[BaseModel], payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a tool payload and preserve omitted optional keys."""
    return model_cls.model_validate(payload).model_dump(mode="python", exclude_unset=True)
