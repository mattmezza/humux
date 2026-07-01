"""Admin route tests for the Inspect tab: list active contexts, per-chat bind,
and the last-sent LLM payload view (#99)."""

from __future__ import annotations

import asyncio
from typing import cast

from fastapi.testclient import TestClient

from api.admin import AgentState, create_admin_app
from core import llm
from core.config_store import ConfigStore
from core.history import ConversationHistory

AUTH = {"Authorization": "Bearer secret"}


class _Store:
    """Config-store stub; agents + history live under tmp_path."""

    def __init__(self, tmp_path):
        self._data = {
            "agent.agents_db_path": str(tmp_path / "agents.db"),
            "agent.agents_dir": str(tmp_path / "seed"),
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

    async def set_many(self, values: dict) -> None:
        self._data.update(values)

    async def verify_admin_password(self, password: str) -> bool:
        return password == "secret"

    async def get_all_redacted(self) -> dict:
        return {}


def _client(tmp_path) -> TestClient:
    app, _ = create_admin_app(AgentState(agent=None), cast(ConfigStore, _Store(tmp_path)))
    return TestClient(app)


async def _seed(tmp_path) -> None:
    h = ConversationHistory(db_path=str(tmp_path / "history.db"))
    await h.add_turn("telegram", "u1", "user", "hi", "c1")


def test_inspect_partial_lists_active_contexts(tmp_path) -> None:
    asyncio.run(_seed(tmp_path))
    client = _client(tmp_path)
    r = client.get("/partials/inspect", headers=AUTH)
    assert r.status_code == 200
    assert "Inspect" in r.text
    assert "c1" in r.text  # the active chat shows up in the master list
    assert 'id="inspect-detail"' in r.text  # master/detail layout present
    # The agent dropdown was removed (one agent = one agent); no bind form.
    assert "/chats/bind" not in r.text


# ── Last-sent payload capture (#99) ─────────────────────────────────────────


def test_capture_records_per_context_and_suppresses_when_unset() -> None:
    llm.clear_captured()
    ctx = ("telegram", "u1", "c1")
    tok = llm.set_capture_context(ctx)
    try:
        llm.record_sent_payload(llm._capture_ctx.get(), {"model": "m", "messages": [], "tools": []})
    finally:
        llm.set_capture_context(None)  # mimic a subagent nulling the context
        llm.record_sent_payload(llm._capture_ctx.get(), {"model": "child"})
        llm.reset_capture_context(tok)
    got = llm.get_sent_payload(ctx)
    assert got is not None and got["model"] == "m"  # child write was a no-op
    llm.clear_captured()


def test_capture_lru_evicts_oldest() -> None:
    llm.clear_captured()
    cap = llm._CAPTURE_CAP
    for i in range(cap + 5):
        llm.record_sent_payload(("c", "u", str(i)), {"model": str(i)})
    assert len(llm._LAST_SENT) == cap
    assert llm.get_sent_payload(("c", "u", "0")) is None  # oldest evicted
    assert llm.get_sent_payload(("c", "u", str(cap + 4))) is not None
    llm.clear_captured()


def test_inspect_payload_renders_captured_request(tmp_path) -> None:
    asyncio.run(_seed(tmp_path))
    client = _client(tmp_path)
    llm.clear_captured()
    llm.record_sent_payload(
        ("telegram", "u1", "c1"),
        {
            "captured_at": 1_700_000_000.0,
            "provider": "anthropic",
            "model": "claude-test",
            "max_tokens": 8192,
            "system": "SYSTEM_PROMPT_MARKER",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "send_message"}],
        },
    )
    r = client.get(
        "/inspect/payload",
        params={"channel": "telegram", "user_id": "u1", "chat_id": "c1"},
        headers=AUTH,
    )
    assert r.status_code == 200
    assert "claude-test" in r.text
    assert "SYSTEM_PROMPT_MARKER" in r.text  # the exact system prompt is shown
    assert "send_message" in r.text  # tool defs included
    # Copy + download controls for the raw JSON, with a context-named file.
    assert "inspectCopyJSON(this)" in r.text
    assert 'data-filename="inspect-telegram-c1.json"' in r.text
    llm.clear_captured()


def test_inspect_payload_shows_context_usage_and_percent(tmp_path) -> None:
    # #116: the real prompt size (usage.context_tokens) + % of the model's window.
    asyncio.run(_seed(tmp_path))
    client = _client(tmp_path)
    llm.clear_captured()
    llm.record_sent_payload(
        ("telegram", "u1", "c1"),
        {
            "captured_at": 1_700_000_000.0,
            "provider": "anthropic",
            "model": "claude-opus-4-5",  # 200_000 window in CONTEXT_WINDOWS
            "max_tokens": 8192,
            "system": "S",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [],
            "usage": {"context_tokens": 4000, "input_tokens": 4000, "output_tokens": 10},
        },
    )
    r = client.get(
        "/inspect/payload",
        params={"channel": "telegram", "user_id": "u1", "chat_id": "c1"},
        headers=AUTH,
    )
    assert r.status_code == 200
    assert "4,000" in r.text  # token count, thousands-separated
    assert "200,000" in r.text  # window pulled from the shared CONTEXT_WINDOWS map
    assert "2.0% of context window" in r.text  # 4000 / 200000
    llm.clear_captured()


def test_inspect_payload_omits_usage_when_absent(tmp_path) -> None:
    # Payloads captured before a response (or a provider without usage) show no
    # meter rather than "0 tokens".
    asyncio.run(_seed(tmp_path))
    client = _client(tmp_path)
    llm.clear_captured()
    llm.record_sent_payload(
        ("telegram", "u1", "c1"),
        {
            "captured_at": 1_700_000_000.0,
            "provider": "x",
            "model": "m",
            "max_tokens": 100,
            "system": "S",
            "messages": [],
            "tools": [],
        },
    )
    r = client.get(
        "/inspect/payload",
        params={"channel": "telegram", "user_id": "u1", "chat_id": "c1"},
        headers=AUTH,
    )
    assert r.status_code == 200
    assert "of context window" not in r.text
    llm.clear_captured()


def test_inspect_payload_empty_for_uncaptured_context(tmp_path) -> None:
    asyncio.run(_seed(tmp_path))
    client = _client(tmp_path)
    llm.clear_captured()
    r = client.get(
        "/inspect/payload",
        params={"channel": "telegram", "user_id": "nobody", "chat_id": "none"},
        headers=AUTH,
    )
    assert r.status_code == 200
    assert "No payload captured yet" in r.text


def test_config_requires_restart_flags_startup_only_keys() -> None:
    from api.admin import _config_requires_restart

    # Read once at startup → restart required.
    assert _config_requires_restart({"voice.tts_enabled": "true"}) is True
    assert _config_requires_restart({"history.max_turns": "20"}) is True
    # Hot-applied by patch_config (agent.config swap + llm/memory rebuild) → no restart.
    assert _config_requires_restart({"memory.long_term_limit": "50"}) is False
    assert _config_requires_restart({"agent.model": "x", "agent.thinking_level": "high"}) is False
    assert _config_requires_restart({}) is False


def test_patch_config_restart_required_only_on_real_change(tmp_path) -> None:
    store = _Store(tmp_path)
    store._data["history.max_turns"] = "10"
    store._data["history.mode"] = "injection"
    app, _ = create_admin_app(AgentState(agent=None), cast(ConfigStore, store))
    client = TestClient(app)

    def patch(values):
        return client.patch("/config", json={"values": values}, headers=AUTH).json()

    # Mode flips (hot-applied) but max_turns unchanged → no restart, even though the
    # form re-sends max_turns.
    res = patch({"history.mode": "session", "history.max_turns": "10"})
    assert res["restart_required"] is False
    # max_turns actually changes → restart.
    assert patch({"history.max_turns": "20"})["restart_required"] is True
    # voice change → restart; hot-applied key → no restart.
    assert patch({"voice.tts_voice": "en-US-AvaNeural"})["restart_required"] is True
    assert patch({"memory.long_term_limit": "99"})["restart_required"] is False
