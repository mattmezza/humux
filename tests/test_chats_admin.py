"""Admin route tests for the Chats tab: list active contexts + per-chat bind."""

from __future__ import annotations

import asyncio
from typing import cast

from fastapi.testclient import TestClient

from api.admin import AgentState, create_admin_app
from core.config_store import ConfigStore
from core.history import ConversationHistory
from core.personae import PersonaStore

AUTH = {"Authorization": "Bearer secret"}


class _Store:
    """Config-store stub; personae + history live under tmp_path."""

    def __init__(self, tmp_path):
        self._data = {
            "agent.personae_db_path": str(tmp_path / "personae.db"),
            "agent.personae_dir": str(tmp_path / "seed"),
            "history.db_path": str(tmp_path / "history.db"),
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


def _client(tmp_path) -> TestClient:
    # agent=None so /chats/bind exercises the config-store fallback write path.
    app, _ = create_admin_app(AgentState(agent=None), cast(ConfigStore, _Store(tmp_path)))
    return TestClient(app)


async def _seed(tmp_path) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "coach.md").write_text("---\nrole: Fitness coach\n---\n")
    store = PersonaStore(db_path=str(tmp_path / "personae.db"), seed_dir=str(seed))
    await store.ensure_seeded()
    h = ConversationHistory(db_path=str(tmp_path / "history.db"))
    await h.add_turn("telegram", "u1", "user", "hi", "c1")


def _binding(tmp_path) -> str | None:
    h = ConversationHistory(db_path=str(tmp_path / "history.db"))
    return asyncio.run(h.get_chat_persona("telegram", "u1", "c1"))


def test_chats_partial_lists_active_contexts(tmp_path) -> None:
    asyncio.run(_seed(tmp_path))
    client = _client(tmp_path)
    r = client.get("/partials/chats", headers=AUTH)
    assert r.status_code == 200
    assert "Chats" in r.text
    assert "c1" in r.text  # the active chat shows up
    assert "Fitness coach" in r.text  # persona option available


def test_bind_and_unbind_persona(tmp_path) -> None:
    asyncio.run(_seed(tmp_path))
    client = _client(tmp_path)

    r = client.post(
        "/chats/bind",
        json={"channel": "telegram", "user_id": "u1", "chat_id": "c1", "persona": "coach"},
        headers=AUTH,
    )
    assert r.status_code == 200
    assert _binding(tmp_path) == "coach"

    r = client.post(
        "/chats/bind",
        json={"channel": "telegram", "user_id": "u1", "chat_id": "c1", "persona": ""},
        headers=AUTH,
    )
    assert r.status_code == 200
    assert _binding(tmp_path) is None


def test_bind_unknown_persona_404(tmp_path) -> None:
    asyncio.run(_seed(tmp_path))
    client = _client(tmp_path)
    r = client.post(
        "/chats/bind",
        json={"channel": "telegram", "user_id": "u1", "chat_id": "c1", "persona": "ghost"},
        headers=AUTH,
    )
    assert r.status_code == 404
    assert _binding(tmp_path) is None
