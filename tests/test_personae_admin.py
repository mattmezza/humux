"""Admin route tests for personae: CRUD, activate, and the tab partial."""

from __future__ import annotations

from typing import Any, cast

from fastapi.testclient import TestClient

from api.admin import AgentState, create_admin_app
from core.config import Config
from core.config_store import ConfigStore

AUTH = {"Authorization": "Bearer secret"}


class _Store:
    """Config-store stub backing get/set with a dict; personae live in tmp."""

    def __init__(self, tmp_path):
        self._data = {
            "agent.personae_db_path": str(tmp_path / "personae.db"),
            "agent.personae_dir": str(tmp_path / "seed"),  # missing = no gallery
        }

    async def is_setup_complete(self) -> bool:
        return True

    async def get(self, key: str):
        if key == "admin.password_hash":
            return "hash"
        if key == "admin.password_salt":
            return "salt"
        return self._data.get(key)

    async def set(self, key: str, value: str) -> None:
        self._data[key] = value

    async def verify_admin_password(self, password: str) -> bool:
        return password == "secret"

    async def get_all_redacted(self) -> dict:
        return {}


class _AgentStub:
    def __init__(self):
        self.config = Config()
        self.channels = {}


def _client(tmp_path):
    agent = _AgentStub()
    app, _ = create_admin_app(
        AgentState(agent=cast(Any, agent)), cast(ConfigStore, _Store(tmp_path))
    )
    return TestClient(app), agent


def test_persona_crud_and_activation(tmp_path) -> None:
    client, agent = _client(tmp_path)

    # Create via the guided fields.
    r = client.post(
        "/personae",
        json={
            "name": "coach",
            "role": "Fitness coach",
            "emoji": "🏋️",
            "voice": "en-US-GuyNeural",
            "personalia": "You are Forge.",
            "character": "Direct.",
            "skills": ["memory"],
            "tools": ["run_command"],
            "secrets": ["persona:coach:*"],
        },
        headers=AUTH,
    )
    assert r.status_code == 200
    assert "Fitness coach" in r.text  # partial lists the new card

    # Read back via JSON API.
    got = client.get("/personae/coach", headers=AUTH).json()
    assert got["voice"] == "en-US-GuyNeural"
    assert got["skills"] == ["memory"] and got["tools"] == ["run_command"]

    # Activate → persisted and hot-applied to the running agent.
    r = client.post("/personae/activate", json={"name": "coach"}, headers=AUTH)
    assert r.status_code == 200
    assert "✓ Active" in r.text
    assert agent.config.agent.active_persona == "coach"

    # Deleting the active persona reverts to the default identity.
    r = client.post("/personae/delete", json={"name": "coach"}, headers=AUTH)
    assert r.status_code == 200
    assert agent.config.agent.active_persona == ""
    assert client.get("/personae/coach", headers=AUTH).status_code == 404


def test_persona_raw_markdown_upsert(tmp_path) -> None:
    client, _ = _client(tmp_path)
    raw = "---\nrole: Writer\nskills: [memory]\ntools: []\npersonalia: |\n  Editor.\n---\n"
    r = client.post("/personae", json={"name": "writer", "raw": raw}, headers=AUTH)
    assert r.status_code == 200
    got = client.get("/personae/writer", headers=AUTH).json()
    assert got["role"] == "Writer" and got["skills"] == ["memory"]
    assert "Editor." in got["personalia"]


def test_activate_unknown_persona_404(tmp_path) -> None:
    client, _ = _client(tmp_path)
    assert (
        client.post("/personae/activate", json={"name": "ghost"}, headers=AUTH).status_code == 404
    )


def test_partial_personae_renders(tmp_path) -> None:
    client, _ = _client(tmp_path)
    r = client.get("/partials/personae", headers=AUTH)
    assert r.status_code == 200
    assert "Active persona" in r.text
