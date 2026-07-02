"""Tests for the local (fastembed) embedding backend + prefetch helper.

fastembed is stubbed via importlib so no model is downloaded.
"""

from __future__ import annotations

import pytest

from core.embeddings import LocalEmbeddingClient, prefetch_local_model


class _FakeTextEmbedding:
    def __init__(self, model_name, cache_dir=None):
        self.model_name = model_name
        self.cache_dir = cache_dir

    def embed(self, texts):
        for t in texts:
            yield [float(len(t)), 1.0, 0.0]


class _FakeFastembed:
    TextEmbedding = _FakeTextEmbedding


@pytest.fixture
def fake_fastembed(monkeypatch):
    calls = {"imports": 0}

    def fake_import(name):
        calls["imports"] += 1
        if name == "fastembed":
            return _FakeFastembed
        raise ImportError(name)

    monkeypatch.setattr("core.embeddings.importlib.import_module", fake_import)
    return calls


class TestLocalEmbeddingClient:
    async def test_not_loaded_at_construction(self, fake_fastembed):
        LocalEmbeddingClient(model="m", cache_dir="c")
        assert fake_fastembed["imports"] == 0  # lazy — nothing imported/loaded yet

    async def test_embed_one(self, fake_fastembed):
        client = LocalEmbeddingClient(model="m", cache_dir="c")
        vec = await client.embed_one("hello")
        assert vec == [5.0, 1.0, 0.0]

    async def test_model_loaded_once(self, fake_fastembed):
        client = LocalEmbeddingClient(model="m", cache_dir="c")
        await client.embed_one("a")
        await client.embed_one("bb")
        await client.embed(["ccc", "dddd"])
        assert fake_fastembed["imports"] == 1  # loaded a single time, then cached

    async def test_embed_empty(self, fake_fastembed):
        client = LocalEmbeddingClient()
        assert await client.embed([]) == []

    async def test_defaults(self):
        client = LocalEmbeddingClient()
        assert client.model == "BAAI/bge-small-en-v1.5"
        assert client.cache_dir == "models"


class TestPrefetch:
    def test_prefetch_returns_dim(self, fake_fastembed):
        dim = prefetch_local_model("some-model", "some-cache")
        assert dim == 3  # _FakeTextEmbedding yields 3-dim vectors
