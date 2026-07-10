"""Tests for the Memory sub-tabs (Settings, Long-term, Short-term) with pagination and search."""

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


def test_memory_partial_renders_settings_by_default() -> None:
    """GET /partials/memory renders the Settings view by default."""
    resp = _client().get("/partials/memory", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.text
    # Settings view includes the config panels
    assert "Memory Settings" in body
    assert "Semantic memory (embeddings)" in body
    assert "Memory lifecycle" in body
    # Sub-tab navigation is present
    assert "Settings" in body
    assert "Long-term" in body
    assert "Short-term" in body


def test_memory_partial_with_view_settings() -> None:
    """GET /partials/memory?view=settings renders the Settings sub-tab."""
    resp = _client().get("/partials/memory?view=settings", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.text
    assert "Memory Settings" in body
    assert "Semantic memory (embeddings)" in body
    assert "Settings" in body


def test_memory_partial_with_view_long_term_shows_table() -> None:
    """GET /partials/memory?view=long-term renders the long-term table."""
    resp = _client().get("/partials/memory?view=long-term", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.text
    # Long-term table headers are present
    assert "Category" in body or "Subject" in body or "Content" in body
    # Pagination controls present
    assert "page_size" in body or "Per page" in body or "Showing" in body
    assert "href=" in body or "hx-get" in body or "hx-target" in body


def test_memory_partial_with_view_short_term_shows_table() -> None:
    """GET /partials/memory?view=short-term renders the short-term table."""
    resp = _client().get("/partials/memory?view=short-term", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.text
    # Short-term table headers are present
    assert "Expires" in body or "Content" in body or "Scope" in body
    # Pagination controls present
    assert "page_size" in body or "Per page" in body or "Showing" in body


def test_memory_partial_invalid_view_falls_back_to_settings() -> None:
    """An unknown view value falls back to Settings."""
    resp = _client().get("/partials/memory?view=nonexistent", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.text
    assert "Memory Settings" in body


def test_memory_partial_long_term_pagination_params_accepted() -> None:
    """Query params offset, limit, q are accepted on long-term view."""
    resp = _client().get(
        "/partials/memory?view=long-term&offset=10&limit=25&q=test",
        headers=HEADERS,
    )
    assert resp.status_code == 200


def test_memory_partial_short_term_pagination_params_accepted() -> None:
    """Query params offset, limit, q are accepted on short-term view."""
    resp = _client().get(
        "/partials/memory?view=short-term&offset=0&limit=50&q=search",
        headers=HEADERS,
    )
    assert resp.status_code == 200


def test_memory_partial_with_legacy_tab_param() -> None:
    """Legacy URL with ?tab=memory still works (defaults to Settings view)."""
    # The admin dashboard passes ?tab=memory which loads /partials/memory
    resp = _client().get("/partials/memory", headers=HEADERS)
    assert resp.status_code == 200


def test_sub_tab_nav_buttons_link_correctly() -> None:
    """Sub-tab buttons have Alpine @click handlers (replaced hx-get)."""
    resp = _client().get("/partials/memory?view=settings", headers=HEADERS)
    body = resp.text
    # Nav is wrapped in an Alpine component
    assert 'x-data="memorySubtab()"' in body
    assert 'x-init="init()"' in body
    # Buttons use @click to invoke Alpine select() method
    assert '@click="select(\'settings\')"' in body or "@click='select(\"settings\")'" in body
    assert '@click="select(\'long-term\')"' in body or "@click='select(\"long-term\")'" in body
    assert '@click="select(\'short-term\')"' in body or "@click='select(\"short-term\")'" in body
    # Buttons use :class bindings for active state
    assert ":class" in body
    assert "'tab-link-active'" in body


def test_endpoints_require_auth() -> None:
    """Memory partial endpoint requires authentication."""
    assert _client().get("/partials/memory").status_code in (401, 403)
    assert _client().get("/partials/memory?view=settings").status_code in (401, 403)
    assert _client().get("/partials/memory?view=long-term").status_code in (401, 403)
    assert _client().get("/partials/memory?view=short-term").status_code in (401, 403)
