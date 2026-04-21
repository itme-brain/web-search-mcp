"""Typed response models for MCP tool structured output."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChunkModel(StrictModel):
    text: str
    score: float


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
    cached: bool
    error: str | None = None
    handoff: HandoffModel | None = None


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
    max_depth: int
    urls_returned: int
    pages_visited: int
    same_domain_only: bool
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
    top_chunks: list[ChunkModel]
    cached: bool
    error: str | None = None
    score: float | None = None


class CrawlMetaModel(StrictModel):
    max_urls_requested: int
    max_depth: int
    urls_discovered: int
    urls_succeeded: int
    urls_failed: int
    same_domain_only: bool
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
