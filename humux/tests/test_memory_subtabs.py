"""Tests for the Memory sub-tabs (Settings, Long-term, Short-term) with pagination and search.

Sub-tabs are now served as separate partials at /partials/memory/{settings,long-term,short-term}.
"""

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


def test_memory_wrapper_renders() -> None:
    """GET /partials/memory renders the wrapper with sub-tab navigation."""
    resp = _client().get("/partials/memory", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.text
    # Wrapper has the Alpine component and sub-tab nav
    assert 'x-data="memorySubtab()"' in body
    assert 'x-init="init()"' in body
    assert 'id="memory-sub"' in body
    assert "skeleton" in body
    # Sub-tab labels are present
    assert "Settings" in body
    assert "Long-term" in body
    assert "Short-term" in body


def test_memory_settings_subtab() -> None:
    """GET /partials/memory/settings renders the Settings sub-tab content."""
    resp = _client().get("/partials/memory/settings", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.text
    assert "Memory Settings" in body
    assert "Semantic memory (embeddings)" in body


def test_memory_long_term_subtab() -> None:
    """GET /partials/memory/long-term renders the long-term table."""
    resp = _client().get("/partials/memory/long-term", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.text
    # Long-term table headers are present
    assert "Category" in body or "Subject" in body or "Content" in body
    # Pagination controls present
    assert "Per page" in body or "Showing" in body


def test_memory_short_term_subtab() -> None:
    """GET /partials/memory/short-term renders the short-term table."""
    resp = _client().get("/partials/memory/short-term", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.text
    # Short-term table headers are present
    assert "Expires" in body or "Content" in body or "Scope" in body
    # Pagination controls present
    assert "Per page" in body or "Showing" in body


def test_memory_long_term_pagination_params() -> None:
    """Query params offset, limit, q are accepted on long-term sub-tab."""
    resp = _client().get(
        "/partials/memory/long-term?offset=10&limit=25&q=test",
        headers=HEADERS,
    )
    assert resp.status_code == 200


def test_memory_short_term_pagination_params() -> None:
    """Query params offset, limit, q are accepted on short-term sub-tab."""
    resp = _client().get(
        "/partials/memory/short-term?offset=0&limit=50&q=search",
        headers=HEADERS,
    )
    assert resp.status_code == 200


def test_sub_tab_nav_buttons_use_alpine() -> None:
    """Sub-tab buttons have Alpine @click handlers."""
    resp = _client().get("/partials/memory", headers=HEADERS)
    body = resp.text
    assert 'x-data="memorySubtab()"' in body
    assert 'x-init="init()"' in body
    assert "@click=\"select('settings')\"" in body or "@click='select(\"settings\")'" in body
    assert "@click=\"select('long-term')\"" in body or "@click='select(\"long-term\")'" in body
    assert "@click=\"select('short-term')\"" in body or "@click='select(\"short-term\")'" in body
    assert ":class" in body
    assert "'tab-link-active'" in body


def test_endpoints_require_auth() -> None:
    """Memory partial endpoints require authentication."""
    assert _client().get("/partials/memory").status_code in (401, 403)
    assert _client().get("/partials/memory/settings").status_code in (401, 403)
    assert _client().get("/partials/memory/long-term").status_code in (401, 403)
    assert _client().get("/partials/memory/short-term").status_code in (401, 403)
