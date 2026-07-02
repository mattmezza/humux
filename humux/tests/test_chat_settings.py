"""Per-agent, per-chat Telegram trigger/DM permissions (#129)."""

from __future__ import annotations

import types
from typing import Any

import pytest

from core.agent import AgentCore
from core.agents import Agent, AgentStore, _as_chat_settings
from core.history import ConversationHistory

# ---- Agent.chat_permits: the enforcement predicate --------------------------


def test_chat_permits_defaults_to_everyone() -> None:
    a = Agent(name="coach")  # no settings at all
    assert a.chat_permits("-100", 7) is True
    a = Agent(name="coach", chat_settings={"-100": {"mode": "everyone", "users": []}})
    assert a.chat_permits("-100", 7) is True  # explicit everyone
    assert a.chat_permits("other", 7) is True  # unlisted chat


def test_chat_permits_nobody_blocks_all() -> None:
    a = Agent(name="coach", chat_settings={"c": {"mode": "nobody", "users": [7]}})
    assert a.chat_permits("c", 7) is False
    assert a.chat_permits("c", 8) is False


def test_chat_permits_users_allowlist() -> None:
    a = Agent(name="coach", chat_settings={"c": {"mode": "users", "users": [7, 9]}})
    assert a.chat_permits("c", 7) is True
    assert a.chat_permits("c", 9) is True
    assert a.chat_permits("c", 8) is False


# ---- _as_chat_settings: coercion + normalisation ----------------------------


def test_as_chat_settings_from_json_string_and_ids() -> None:
    got = _as_chat_settings('{"c": {"mode": "users", "users": ["5", "x", 6]}}')
    assert got == {"c": {"mode": "users", "users": [5, 6]}}  # non-numeric dropped


def test_as_chat_settings_drops_everyone_and_junk() -> None:
    assert _as_chat_settings({"c": {"mode": "everyone", "users": [1]}}) == {}
    assert _as_chat_settings({"c": {"mode": "bogus"}}) == {}  # bogus → everyone → dropped
    assert _as_chat_settings("not json") == {}
    assert _as_chat_settings(42) == {}
    assert _as_chat_settings({"c": "not-a-dict"}) == {}


# ---- Persistence round-trip -------------------------------------------------


@pytest.mark.asyncio
async def test_chat_settings_survive_upsert(tmp_path) -> None:
    store = AgentStore(db_path=str(tmp_path / "a.db"), seed_dir=None)
    await store.upsert(Agent(name="coach", chat_settings={"-100": {"mode": "users", "users": [7]}}))
    got = await store.get("coach")
    assert got is not None
    assert got.chat_settings == {"-100": {"mode": "users", "users": [7]}}


# ---- may_act_in_chat: resolve-then-gate integration -------------------------


def _fake_core(history: ConversationHistory, agents: AgentStore):
    fa: Any = types.SimpleNamespace(
        history=history,
        agents=agents,
        config=types.SimpleNamespace(agent=types.SimpleNamespace()),
    )
    for name in ("_load_agent", "_resolve_agent", "may_act_in_chat"):
        setattr(fa, name, types.MethodType(getattr(AgentCore, name), fa))
    return fa


@pytest.mark.asyncio
async def test_may_act_in_chat_gates_resolved_agent(tmp_path) -> None:
    store = AgentStore(db_path=str(tmp_path / "a.db"), seed_dir=None)
    await store.upsert(Agent(name="coach", chat_settings={"-100": {"mode": "users", "users": [7]}}))
    h = ConversationHistory(db_path=str(tmp_path / "h.db"))
    await h.set_chat_agent("telegram", "-100", "coach", "-100")  # bind group → coach
    core = _fake_core(h, store)

    assert await core.may_act_in_chat("telegram", "-100", "-100", 7) is True
    assert await core.may_act_in_chat("telegram", "-100", "-100", 8) is False
    # An unconfigured chat (resolves to no agent) is always allowed.
    assert await core.may_act_in_chat("telegram", "u1", "c1", 8) is True
