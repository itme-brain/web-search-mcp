"""Environment-backed settings and shared constants."""

import os
import re
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    searxng_url: str = "http://searxng:8080"
    crawl4ai_url: str = "http://crawl4ai:11235"
    crawl4ai_api_token: str = ""
    rerank_backend: str = "flashrank"
    rerank_model: str = "ms-marco-MiniLM-L-12-v2"
    rerank_device: str | None = None
    rerank_batch_size: int = 16
    rerank_max_length: int = 512
    request_timeout: int = 30
    max_results: int = 10
    max_scrape: int = 5
    max_candidates: int = 60
    max_pdf_bytes: int = 25 * 1024 * 1024
    enable_lfm_preprocessing: bool = False
    lfm_base_url: str = ""
    lfm_model: str = "LFM2.5-2.6B-Q8_0.gguf"
    lfm_hf_repo: str = "LiquidAI/LFM2.5-2.6B-GGUF"
    lfm_hf_file: str = "LFM2.5-2.6B-Q8_0.gguf"
    lfm_api_key: str = ""
    lfm_timeout: float = 45.0
    lfm_max_input_chars: int = 24000


settings = Settings()

SEARXNG_URL = settings.searxng_url
CRAWL4AI_URL = settings.crawl4ai_url
CRAWL4AI_API_TOKEN = settings.crawl4ai_api_token
RERANK_BACKEND = settings.rerank_backend
RERANK_MODEL = settings.rerank_model
RERANK_DEVICE = settings.rerank_device
RERANK_BATCH_SIZE = settings.rerank_batch_size
RERANK_MAX_LENGTH = settings.rerank_max_length
REQUEST_TIMEOUT = settings.request_timeout
MAX_RESULTS = settings.max_results
MAX_SCRAPE = settings.max_scrape
MAX_CANDIDATES = settings.max_candidates
MAX_PDF_BYTES = settings.max_pdf_bytes
ENABLE_LFM_PREPROCESSING = settings.enable_lfm_preprocessing
LFM_BASE_URL = settings.lfm_base_url.rstrip("/")
LFM_MODEL = settings.lfm_model
LFM_HF_REPO = settings.lfm_hf_repo
LFM_HF_FILE = settings.lfm_hf_file
LFM_API_KEY = settings.lfm_api_key
LFM_TIMEOUT = settings.lfm_timeout
LFM_MAX_INPUT_CHARS = settings.lfm_max_input_chars

_HTTP_TIMEOUT = max(REQUEST_TIMEOUT // 2, 10)
_MAX_CONTENT_CHARS = 20000
_MAX_EXTRACT_CONTENT_CHARS = 200000
_TITLE_DEDUP_THRESHOLD = 97.0
_TOP_CHUNKS = 3
_MAX_CHUNKS_PER_PAGE = 10
_CHUNK_GAP = "\n\n[…]\n\n"
_MAX_EXTRACT_URLS = 20
_MAX_MAP_URLS = 50
_SNIFF_MAX_BYTES = 8192
_DOWNLOAD_CHUNK_BYTES = 65536
_DISPLAY_CHUNK_COUNT = 3
_MIN_RELEVANCE_SCORE = 0.05

VALID_TIME_RANGES = frozenset({"day", "week", "month", "year"})

_WHITESPACE = re.compile(r"\s+")
_PILCROW_LINK = re.compile(r"\[¶\]\([^)]*\)")
_MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\([^)]+\)")
