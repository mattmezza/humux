"""Admin API tests for the embedding status / prefetch / test endpoints."""

from __future__ import annotations

from typing import cast

from fastapi.testclient import TestClient

from api.admin import AgentState, create_admin_app
from core.config import Config
from core.config_store import ConfigStore

HEADERS = {"Authorization": "Bearer secret"}


class _StoreStub:
    """Minimal config store: auth + export_to_config(Config defaults)."""

    def __init__(self, overrides: dict | None = None):
        self._overrides = overrides or {}

    async def is_setup_complete(self) -> bool:
        return True

    async def get(self, key: str):
        if key == "admin.password_hash":
            return "hash"
        if key == "admin.password_salt":
            return "salt"
        return self._overrides.get(key)

    async def verify_admin_password(self, password: str) -> bool:
        return password == "secret"

    async def export_to_config(self) -> Config:
        cfg = Config()
        emb = cfg.memory.embedding
        for key, val in self._overrides.items():
            if key == "memory.embedding.provider":
                emb.provider = val
            elif key == "memory.embedding.model":
                emb.model = val
        return cfg


def _client(overrides: dict | None = None) -> TestClient:
    agent_state = AgentState(agent=None)
    app, _auth = create_admin_app(agent_state, cast(ConfigStore, _StoreStub(overrides)))
    return TestClient(app)


def test_embedding_status_local_default() -> None:
    resp = _client().get("/memory/embedding/status", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is True
    assert data["provider"] == "local"
    assert data["local"] is True
    assert "model_ready" in data  # bool (true/false) for local


def test_embedding_prefetch_invokes_helper(monkeypatch) -> None:
    seen = {}

    def fake_prefetch(model, cache_dir):
        seen["model"] = model
        seen["cache_dir"] = cache_dir
        return 384

    monkeypatch.setattr("core.embeddings.prefetch_local_model", fake_prefetch)

    resp = _client().post("/memory/embedding/prefetch", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["dimensions"] == 384
    assert seen["model"] == "BAAI/bge-small-en-v1.5"


def test_embedding_prefetch_rejects_remote(monkeypatch) -> None:
    resp = _client({"memory.embedding.provider": "openai"}).post(
        "/memory/embedding/prefetch", headers=HEADERS
    )
    assert resp.status_code == 400


def test_embedding_test_endpoint(monkeypatch) -> None:
    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def embed(self, texts):
            # First two related (close), third unrelated (orthogonal).
            return [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]][: len(texts)]

    monkeypatch.setattr("core.embeddings.LocalEmbeddingClient", _FakeClient)

    resp = _client().post("/memory/embedding/test", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["dimensions"] == 2
    assert data["similar_pair"] > data["unrelated_pair"]


def test_endpoints_require_auth() -> None:
    assert _client().get("/memory/embedding/status").status_code in (401, 403)


def test_memory_partial_renders() -> None:
    resp = _client().get("/partials/memory", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.text
    assert "Semantic memory (embeddings)" in body
    assert "Memory lifecycle" in body
    assert "Download model" in body
