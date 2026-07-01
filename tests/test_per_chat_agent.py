"""Per-chat agent binding (#14): store layer, resolution ladder, topic folding."""

from __future__ import annotations

import types
from typing import Any

import pytest

from channels.telegram import TelegramChannel
from core.agent import AgentCore
from core.agents import AgentStore
from core.history import ConversationHistory

# ---- History store: bindings + list_chats ----------------------------------


@pytest.mark.asyncio
async def test_chat_agent_set_get_clear(tmp_path) -> None:
    h = ConversationHistory(db_path=str(tmp_path / "h.db"))
    assert await h.get_chat_agent("telegram", "u1", "c1") is None
    await h.set_chat_agent("telegram", "u1", "coach", "c1")
    assert await h.get_chat_agent("telegram", "u1", "c1") == "coach"
    # Upsert overwrites.
    await h.set_chat_agent("telegram", "u1", "writer", "c1")
    assert await h.get_chat_agent("telegram", "u1", "c1") == "writer"
    # Scoped to the triple — a different chat is unaffected.
    assert await h.get_chat_agent("telegram", "u1", "c2") is None
    await h.clear_chat_agent("telegram", "u1", "c1")
    assert await h.get_chat_agent("telegram", "u1", "c1") is None


@pytest.mark.asyncio
async def test_list_chats_unions_history_and_bindings(tmp_path) -> None:
    h = ConversationHistory(db_path=str(tmp_path / "h.db"))
    # A chat known only from history.
    await h.add_turn("telegram", "u1", "user", "hi", "c1")
    # A chat known only from a binding (a topic auto-bound before its first msg).
    await h.set_chat_agent("telegram", "u1", "coach", "c2")
    chats = await h.list_chats()
    by_chat = {c["chat_id"]: c for c in chats}
    assert by_chat["c1"]["agent"] == ""  # unbound
    assert by_chat["c2"]["agent"] == "coach"  # bound, no history yet
    assert all({"channel", "user_id", "chat_id", "agent"} <= c.keys() for c in chats)


@pytest.mark.asyncio
async def test_list_chats_orders_most_recent_first(tmp_path) -> None:
    import aiosqlite

    h = ConversationHistory(db_path=str(tmp_path / "h.db"))
    await h.add_turn("telegram", "u1", "user", "old", "old_chat")
    await h.add_turn("telegram", "u1", "user", "new", "new_chat")
    # Force distinct timestamps (datetime('now') is whole-second, so same-second
    # inserts would tie) — make new_chat unambiguously the most recent.
    sql = "UPDATE conversation_turns SET created_at=? WHERE chat_id=?"
    async with aiosqlite.connect(h.db_path) as db:
        await db.execute(sql, ("2020-01-01 00:00:00", "old_chat"))
        await db.execute(sql, ("2026-01-01 00:00:00", "new_chat"))
        await db.commit()
    chats = await h.list_chats()
    order = [c["chat_id"] for c in chats]
    assert order.index("new_chat") < order.index("old_chat")  # newest first
    by_chat = {c["chat_id"]: c for c in chats}
    assert by_chat["new_chat"]["last_active"] == "2026-01-01 00:00:00"


@pytest.mark.asyncio
async def test_clear_session_system_keeps_messages(tmp_path) -> None:
    h = ConversationHistory(db_path=str(tmp_path / "h.db"))
    await h.append_session_message("telegram", "u1", {"role": "user", "content": "hi"}, "c1")
    await h.set_session_system("telegram", "u1", "SYSTEM", "c1")
    await h.clear_session_system("telegram", "u1", "c1")
    assert await h.get_session_system("telegram", "u1", "c1") is None
    # Conversation preserved.
    assert await h.get_session("telegram", "u1", "c1") == [{"role": "user", "content": "hi"}]


# ---- Agent resolution ladder + auto-bind -----------------------------------


def _fake_agent(history: ConversationHistory, agents: AgentStore, active: str = ""):
    """A stand-in ``self`` carrying just what the agent methods touch, with the
    real ``AgentCore`` methods bound to it — avoids constructing the heavy core."""
    fa: Any = types.SimpleNamespace(
        history=history,
        agents=agents,
        config=types.SimpleNamespace(agent=types.SimpleNamespace(active_agent=active)),
    )
    for name in (
        "_load_agent",
        "_resolve_agent",
        "bind_chat_agent",
        "bind_chat_agent_by_label",
    ):
        setattr(fa, name, types.MethodType(getattr(AgentCore, name), fa))
    return fa


async def _seed_agents(tmp_path) -> AgentStore:
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "coach.md").write_text("---\nagent_name: Forge\nrole: Fitness coach\n---\n")
    (seed / "writer.md").write_text("---\nrole: Writer\n---\n")
    store = AgentStore(db_path=str(tmp_path / "p.db"), seed_dir=str(seed))
    await store.ensure_seeded()
    return store


@pytest.mark.asyncio
async def test_resolve_agent_ladder(tmp_path) -> None:
    h = ConversationHistory(db_path=str(tmp_path / "h.db"))
    store = await _seed_agents(tmp_path)

    # 3. Default identity when nothing is set.
    fa = _fake_agent(h, store, active="")
    assert await fa._resolve_agent("telegram", "u1", "c1") is None

    # 2. Global active agent.
    fa = _fake_agent(h, store, active="writer")
    p = await fa._resolve_agent("telegram", "u1", "c1")
    assert p is not None and p.name == "writer"

    # 1. Per-chat binding wins over global.
    await h.set_chat_agent("telegram", "u1", "coach", "c1")
    p = await fa._resolve_agent("telegram", "u1", "c1")
    assert p is not None and p.name == "coach"
    # A different chat still gets the global agent.
    p2 = await fa._resolve_agent("telegram", "u1", "c2")
    assert p2 is not None and p2.name == "writer"


@pytest.mark.asyncio
async def test_resolve_agent_bot_per_agent_channel(tmp_path) -> None:
    # Rung 0 (#29): a "telegram:<name>" channel binds straight to that agent,
    # outranking the per-chat binding and the global active agent.
    h = ConversationHistory(db_path=str(tmp_path / "h.db"))
    store = await _seed_agents(tmp_path)
    await h.set_chat_agent("telegram:coach", "u1", "writer", "c1")  # ignored by rung 0
    fa = _fake_agent(h, store, active="writer")
    p = await fa._resolve_agent("telegram:coach", "u1", "c1")
    assert p is not None and p.name == "coach"

    # Unknown agent in the channel name → fall through to the ordinary ladder.
    p2 = await fa._resolve_agent("telegram:ghost", "u1", "c2")
    assert p2 is not None and p2.name == "writer"  # global active agent


@pytest.mark.asyncio
async def test_resolve_agent_missing_binding_falls_through(tmp_path) -> None:
    h = ConversationHistory(db_path=str(tmp_path / "h.db"))
    store = await _seed_agents(tmp_path)
    await h.set_chat_agent("telegram", "u1", "ghost", "c1")  # not a real agent
    fa = _fake_agent(h, store, active="writer")
    p = await fa._resolve_agent("telegram", "u1", "c1")
    assert p is not None and p.name == "writer"  # falls through to global


@pytest.mark.asyncio
async def test_bind_by_label_matches_name_agentname_role(tmp_path) -> None:
    h = ConversationHistory(db_path=str(tmp_path / "h.db"))
    store = await _seed_agents(tmp_path)
    fa = _fake_agent(h, store)

    # Match by agent_name ("Forge"), case-insensitive.
    assert await fa.bind_chat_agent_by_label("telegram", "u1", "c1", "forge") == "coach"
    assert await h.get_chat_agent("telegram", "u1", "c1") == "coach"

    # Already bound → does not override a manual/earlier choice.
    assert await fa.bind_chat_agent_by_label("telegram", "u1", "c1", "Writer") is None
    assert await h.get_chat_agent("telegram", "u1", "c1") == "coach"

    # No match → None, no binding.
    assert await fa.bind_chat_agent_by_label("telegram", "u1", "c9", "nope") is None
    assert await h.get_chat_agent("telegram", "u1", "c9") is None

    # Match by role.
    assert await fa.bind_chat_agent_by_label("telegram", "u1", "c2", "writer") == "writer"


@pytest.mark.asyncio
async def test_bind_chat_agent_clears_session_system(tmp_path) -> None:
    h = ConversationHistory(db_path=str(tmp_path / "h.db"))
    store = await _seed_agents(tmp_path)
    fa = _fake_agent(h, store)
    await h.set_session_system("telegram", "u1", "OLD", "c1")
    await fa.bind_chat_agent("telegram", "u1", "c1", "coach")
    assert await h.get_session_system("telegram", "u1", "c1") is None
    # Empty name unbinds.
    await fa.bind_chat_agent("telegram", "u1", "c1", "")
    assert await h.get_chat_agent("telegram", "u1", "c1") is None


# ---- Telegram topic folding ------------------------------------------------


def _fold(topics_enabled: bool, chat, message):
    fake = types.SimpleNamespace(config=types.SimpleNamespace(topics_enabled=topics_enabled))
    return TelegramChannel._fold(fake, chat, message)


def test_fold_off_by_default() -> None:
    chat = types.SimpleNamespace(id=-100123)
    msg = types.SimpleNamespace(message_thread_id=45, is_topic_message=True)
    assert _fold(False, chat, msg) == -100123  # no folding when disabled


def test_fold_topic_message() -> None:
    chat = types.SimpleNamespace(id=-100123)
    msg = types.SimpleNamespace(message_thread_id=45, is_topic_message=True)
    assert _fold(True, chat, msg) == "-100123:45"


def test_fold_reply_chain_is_not_a_topic() -> None:
    # Non-forum reply-chains / discussion comments carry message_thread_id too,
    # but is_topic_message is False — they must NOT fold into a separate context.
    chat = types.SimpleNamespace(id=-100555)
    msg = types.SimpleNamespace(message_thread_id=789, is_topic_message=False)
    assert _fold(True, chat, msg) == -100555


def test_fold_general_topic_is_bare_chat() -> None:
    chat = types.SimpleNamespace(id=-100123)
    msg = types.SimpleNamespace(message_thread_id=None, is_topic_message=False)
    assert _fold(True, chat, msg) == -100123


def test_fold_no_chat_returns_none() -> None:
    msg = types.SimpleNamespace(message_thread_id=None, is_topic_message=False)
    assert _fold(True, None, msg) is None


def test_route_roundtrip() -> None:
    assert TelegramChannel._route("-100123:45") == (-100123, {"message_thread_id": 45})
    assert TelegramChannel._route(-100123) == (-100123, {})
    assert TelegramChannel._route("12345") == ("12345", {})  # bare numeric string, no thread
    assert TelegramChannel._route(12345) == (12345, {})
