"""Per-chat persona binding (#14): store layer, resolution ladder, topic folding."""

from __future__ import annotations

import types
from typing import Any

import pytest

from channels.telegram import TelegramChannel
from core.agent import AgentCore
from core.history import ConversationHistory
from core.personae import PersonaStore

# ---- History store: bindings + list_chats ----------------------------------


@pytest.mark.asyncio
async def test_chat_persona_set_get_clear(tmp_path) -> None:
    h = ConversationHistory(db_path=str(tmp_path / "h.db"))
    assert await h.get_chat_persona("telegram", "u1", "c1") is None
    await h.set_chat_persona("telegram", "u1", "coach", "c1")
    assert await h.get_chat_persona("telegram", "u1", "c1") == "coach"
    # Upsert overwrites.
    await h.set_chat_persona("telegram", "u1", "writer", "c1")
    assert await h.get_chat_persona("telegram", "u1", "c1") == "writer"
    # Scoped to the triple — a different chat is unaffected.
    assert await h.get_chat_persona("telegram", "u1", "c2") is None
    await h.clear_chat_persona("telegram", "u1", "c1")
    assert await h.get_chat_persona("telegram", "u1", "c1") is None


@pytest.mark.asyncio
async def test_list_chats_unions_history_and_bindings(tmp_path) -> None:
    h = ConversationHistory(db_path=str(tmp_path / "h.db"))
    # A chat known only from history.
    await h.add_turn("telegram", "u1", "user", "hi", "c1")
    # A chat known only from a binding (a topic auto-bound before its first msg).
    await h.set_chat_persona("telegram", "u1", "coach", "c2")
    chats = await h.list_chats()
    by_chat = {c["chat_id"]: c for c in chats}
    assert by_chat["c1"]["persona"] == ""  # unbound
    assert by_chat["c2"]["persona"] == "coach"  # bound, no history yet
    assert all({"channel", "user_id", "chat_id", "persona"} <= c.keys() for c in chats)


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


def _fake_agent(history: ConversationHistory, personae: PersonaStore, active: str = ""):
    """A stand-in ``self`` carrying just what the persona methods touch, with the
    real ``AgentCore`` methods bound to it — avoids constructing the heavy core."""
    fa: Any = types.SimpleNamespace(
        history=history,
        personae=personae,
        config=types.SimpleNamespace(agent=types.SimpleNamespace(active_persona=active)),
    )
    for name in (
        "_load_persona",
        "_resolve_persona",
        "bind_chat_persona",
        "bind_chat_persona_by_label",
    ):
        setattr(fa, name, types.MethodType(getattr(AgentCore, name), fa))
    return fa


async def _seed_personae(tmp_path) -> PersonaStore:
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "coach.md").write_text("---\nagent_name: Forge\nrole: Fitness coach\n---\n")
    (seed / "writer.md").write_text("---\nrole: Writer\n---\n")
    store = PersonaStore(db_path=str(tmp_path / "p.db"), seed_dir=str(seed))
    await store.ensure_seeded()
    return store


@pytest.mark.asyncio
async def test_resolve_persona_ladder(tmp_path) -> None:
    h = ConversationHistory(db_path=str(tmp_path / "h.db"))
    store = await _seed_personae(tmp_path)

    # 3. Default identity when nothing is set.
    fa = _fake_agent(h, store, active="")
    assert await fa._resolve_persona("telegram", "u1", "c1") is None

    # 2. Global active persona.
    fa = _fake_agent(h, store, active="writer")
    p = await fa._resolve_persona("telegram", "u1", "c1")
    assert p is not None and p.name == "writer"

    # 1. Per-chat binding wins over global.
    await h.set_chat_persona("telegram", "u1", "coach", "c1")
    p = await fa._resolve_persona("telegram", "u1", "c1")
    assert p is not None and p.name == "coach"
    # A different chat still gets the global persona.
    p2 = await fa._resolve_persona("telegram", "u1", "c2")
    assert p2 is not None and p2.name == "writer"


@pytest.mark.asyncio
async def test_resolve_persona_missing_binding_falls_through(tmp_path) -> None:
    h = ConversationHistory(db_path=str(tmp_path / "h.db"))
    store = await _seed_personae(tmp_path)
    await h.set_chat_persona("telegram", "u1", "ghost", "c1")  # not a real persona
    fa = _fake_agent(h, store, active="writer")
    p = await fa._resolve_persona("telegram", "u1", "c1")
    assert p is not None and p.name == "writer"  # falls through to global


@pytest.mark.asyncio
async def test_bind_by_label_matches_name_agentname_role(tmp_path) -> None:
    h = ConversationHistory(db_path=str(tmp_path / "h.db"))
    store = await _seed_personae(tmp_path)
    fa = _fake_agent(h, store)

    # Match by agent_name ("Forge"), case-insensitive.
    assert await fa.bind_chat_persona_by_label("telegram", "u1", "c1", "forge") == "coach"
    assert await h.get_chat_persona("telegram", "u1", "c1") == "coach"

    # Already bound → does not override a manual/earlier choice.
    assert await fa.bind_chat_persona_by_label("telegram", "u1", "c1", "Writer") is None
    assert await h.get_chat_persona("telegram", "u1", "c1") == "coach"

    # No match → None, no binding.
    assert await fa.bind_chat_persona_by_label("telegram", "u1", "c9", "nope") is None
    assert await h.get_chat_persona("telegram", "u1", "c9") is None

    # Match by role.
    assert await fa.bind_chat_persona_by_label("telegram", "u1", "c2", "writer") == "writer"


@pytest.mark.asyncio
async def test_bind_chat_persona_clears_session_system(tmp_path) -> None:
    h = ConversationHistory(db_path=str(tmp_path / "h.db"))
    store = await _seed_personae(tmp_path)
    fa = _fake_agent(h, store)
    await h.set_session_system("telegram", "u1", "OLD", "c1")
    await fa.bind_chat_persona("telegram", "u1", "c1", "coach")
    assert await h.get_session_system("telegram", "u1", "c1") is None
    # Empty name unbinds.
    await fa.bind_chat_persona("telegram", "u1", "c1", "")
    assert await h.get_chat_persona("telegram", "u1", "c1") is None


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
