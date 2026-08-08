"""Typed response models for MCP tool structured output."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChunkSpecModel(StrictModel):
    """A chunk with a stable id.

    Returned on extract responses as stable document chunk metadata.
    IDs are indices into the full cached document chunk list and are
    stable as long as the cached raw content is.
    """
    id: int
    text: str


class WarningModel(StrictModel):
    type: str
    source: str
    detail: str


class TimingModel(StrictModel):
    preprocessing: int | None = None
    search: int | None = None
    candidate_rerank: int | None = None
    scrape: int | None = None
    semantic: int | None = None
    rerank: int | None = None
    total: int


class RerankerModel(StrictModel):
    name: str
    model: str


class PreprocessingModel(StrictModel):
    enabled: bool = False
    configured: bool = False
    model: str
    artifact: dict[str, str]
    planning_used: bool = False
    digest_used: bool = False


class DocumentMetadataModel(StrictModel):
    """Citation metadata that accompanies a page response.

    Structural info (headings, code blocks, links) is not duplicated
    here — trafilatura's markdown output already carries it inline in
    the `content` field where the LLM can see it directly.
    """
    author: str | None = None
    date: str | None = None
    site_name: str | None = None
    description: str | None = None
    word_count: int | None = None
    language: str | None = None
    canonical_url: str | None = None
    final_url: str | None = None
    diagnostic: str | None = None


class SearchPassageModel(StrictModel):
    citation: str | None = None
    text: str
    score: float | None = None
    chunk_id: str | None = None
    resource_uri: str | None = None


class SearchResultModel(StrictModel):
    rank: int
    title: str
    url: str
    domain: str
    source_type: str | None = None
    retrieval_source: str | None = None
    best_score: float | None = None
    latest_date: str | None = None
    snippet: str | None = None
    content: str | None = None
    passages: list[SearchPassageModel] = []
    scraped: bool
    seen_recently: bool
    metadata: DocumentMetadataModel | None = None
    document_id: str | None = None
    resource_uri: str | None = None


class EvidenceReadResponseModel(StrictModel):
    reference: str
    kind: str
    document_id: str
    chunk_id: str | None = None
    url: str
    title: str | None = None
    content: str
    resource_uri: str


class SearchMetaModel(StrictModel):
    request_id: str | None = None
    profile: str = "search"
    intent: str = "general_web_research"
    candidate_pool_size: int = 0
    overview: list[str] = []
    gaps: list[str] = []
    next_actions: list[str] = []
    num_results_requested: int
    num_results_returned: int
    scrape_top: int
    max_passages: int | None = None
    max_chars_per_result: int | None = None
    search_queries: list[str] = []
    source_types: list[str] | None = None
    search_backend: str
    reranker: RerankerModel
    preprocessing: PreprocessingModel
    semantic_hits: int = 0
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
    domain: str
    status: str
    content_type: str | None = None
    file_type: str | None = None
    title: str | None = None
    content: str
    chars_shown: int
    total_chars: int
    truncated: bool = False
    total_chunks: int | None = None
    shown_chunk_ids: list[int] = []
    chunk_mode: str | None = None
    top_chunks: list[str] = []
    chunks: list[ChunkSpecModel] = []
    cached: bool
    error: str | None = None
    metadata: DocumentMetadataModel | None = None


class ExtractMetaModel(StrictModel):
    request_id: str | None = None
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
    domain: str
    title: str | None = None
    link_text: str | None = None
    depth: int
    discovered_from: str | None = None
    link_type: str


class MapMetaModel(StrictModel):
    request_id: str | None = None
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
    total_chars: int
    top_chunks: list[str] = []
    cached: bool
    error: str | None = None
    metadata: DocumentMetadataModel | None = None


class CrawlMetaModel(StrictModel):
    request_id: str | None = None
    max_urls_requested: int
    urls_discovered: int
    urls_returned: int
    urls_truncated_by_limit: int
    urls_deduplicated: int = 0
    sparse: bool = False
    sparsity_reason: str | None = None
    urls_succeeded: int
    urls_failed: int
    warnings: list[WarningModel]
    timings_ms: TimingModel


class CrawlResponseModel(StrictModel):
    url: str
    query: str | None = None
    results: list[CrawlResultModel]
    meta: CrawlMetaModel


def dump_response(model_cls: type[BaseModel], payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a tool payload and preserve omitted optional keys."""
    return model_cls.model_validate(payload).model_dump(mode="python", exclude_unset=True)
