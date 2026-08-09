import numpy as np
import pytest
from redis.exceptions import ResponseError

from web_search_mcp.storage import cache
from web_search_mcp.storage import semantic


class FakeSearchPipeline:
    def __init__(self, client):
        self.client = client
        self.commands = []

    def hset(self, key, mapping):
        self.commands.append(("hset", key, mapping))
        self.client.hashes[key] = mapping

    def expire(self, key, ttl):
        self.commands.append(("expire", key, ttl))
        self.client.expires[key] = ttl

    async def execute(self):
        return [1 for _ in self.commands]


class FakeSearchClient:
    def __init__(self):
        self.commands = []
        self.index_created = False
        self.hashes = {}
        self.expires = {}

    async def execute_command(self, *args):
        self.commands.append(args)
        command = args[0]
        if command == "FT.INFO":
            if not self.index_created:
                # Current valkey-bundle wording differs from older RediSearch.
                raise ResponseError("Index with name 'test-index' not found")
            return ["num_docs", len(self.hashes)]
        if command == "FT.CREATE":
            self.index_created = True
            return "OK"
        if command == "FT.SEARCH":
            return [
                1,
                "ws:semantic:chunk:abc",
                [
                    "id", "abc",
                    "url", "https://example.com/doc",
                    "domain", "example.com",
                    "title", "Doc",
                    "chunk_index", "2",
                    "text", "cached passage",
                    "metadata", '{"author": "Ada"}',
                    "updated_at", "123",
                    "distance", "0.2",
                ],
            ]
        raise AssertionError(f"unexpected command: {args}")

    def pipeline(self):
        return FakeSearchPipeline(self)


@pytest.fixture
def enabled_semantic(monkeypatch):
    monkeypatch.setattr(semantic, "ENABLED", True)
    monkeypatch.setattr(semantic, "_INDEX_READY", False)
    monkeypatch.setattr(semantic, "_INDEX_DIM", None)
    monkeypatch.setattr(semantic, "_LAST_ERROR", None)

    async def fake_embed(texts, *, is_query=False):
        return np.asarray([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    monkeypatch.setattr(semantic, "_embed", fake_embed)


@pytest.mark.asyncio
async def test_bge_embedding_instruction_is_query_only(monkeypatch):
    captured = []

    class FakeModel:
        def encode(self, texts, **_kwargs):
            captured.extend(texts)
            return np.asarray([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    monkeypatch.setattr(semantic, "_MODEL", FakeModel())
    monkeypatch.setattr(
        semantic,
        "QUERY_PREFIX",
        "Represent this sentence for searching relevant passages: ",
    )
    monkeypatch.setattr(semantic, "PASSAGE_PREFIX", "")

    await semantic._embed(["find this"], is_query=True)
    await semantic._embed(["passage text"], is_query=False)

    assert captured == [
        "Represent this sentence for searching relevant passages: find this",
        "passage text",
    ]


@pytest.mark.asyncio
async def test_index_page_creates_valkey_search_index_and_hashes_chunks(enabled_semantic):
    client = FakeSearchClient()
    cache.set_client(client)

    await semantic.index_page(
        "https://example.com/doc",
        "Doc",
        "first paragraph with enough words to become a chunk\n\nsecond paragraph also useful",
        {"author": "Ada"},
    )

    assert any(command[0] == "FT.CREATE" for command in client.commands)
    create_command = next(command for command in client.commands if command[0] == "FT.CREATE")
    assert create_command[-2:] == ("EF_CONSTRUCTION", "200")
    assert client.hashes
    key, mapping = next(iter(client.hashes.items()))
    assert key.startswith("ws:semantic:chunk:")
    assert mapping["url"] == "https://example.com/doc"
    assert mapping["domain"] == "example.com"
    assert mapping["metadata"] == '{"author": "Ada"}'
    assert isinstance(mapping["vector"], bytes)
    assert client.expires[key] == cache.SEMANTIC_INDEX_TTL_S


@pytest.mark.asyncio
async def test_search_uses_valkey_search_knn_and_decodes_records(enabled_semantic):
    client = FakeSearchClient()
    client.index_created = True
    cache.set_client(client)

    results = await semantic.search("query", top_k=1)

    search_command = next(command for command in client.commands if command[0] == "FT.SEARCH")
    assert any("KNN 1 @vector $query_vec AS distance" in str(part) for part in search_command)
    assert "SORTBY" not in search_command
    assert results == [{
        "id": "abc",
        "url": "https://example.com/doc",
        "domain": "example.com",
        "title": "Doc",
        "chunk_index": 2,
        "text": "cached passage",
        "metadata": {"author": "Ada"},
        "updated_at": 123,
        "distance": "0.2",
        "score": 0.8,
    }]
