"""Read-only live web transcript: share tokens, the public routes, and the
entry classification behind the page's progressive disclosures (/weburl)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from api.admin import AgentState, create_admin_app
from channels.telegram import _MENU_COMMANDS, TelegramChannel
from core.config_store import ConfigStore
from core.history import ConversationHistory
from core.transcript import strip_preamble, to_entries


class _Store:
    """Config-store stub; history lives under tmp_path."""

    def __init__(self, tmp_path):
        self._data = {
            "history.db_path": str(tmp_path / "history.db"),
            "agent.name": "Ada",
        }

    async def is_setup_complete(self) -> bool:
        return True

    async def get(self, key: str):
        if key == "admin.password_hash":
            return "hash"
        if key == "admin.password_salt":
            return "salt"
        return self._data.get(key)

    async def set_many(self, values: dict) -> None:
        self._data.update(values)

    async def verify_admin_password(self, password: str) -> bool:
        return password == "secret"

    async def get_all_redacted(self) -> dict:
        return {}


def _history(tmp_path) -> ConversationHistory:
    return ConversationHistory(db_path=str(tmp_path / "history.db"))


def _client(tmp_path) -> TestClient:
    app, _ = create_admin_app(AgentState(agent=None), cast(ConfigStore, _Store(tmp_path)))
    return TestClient(app)


# ── Token store ─────────────────────────────────────────────────────────────


def test_token_is_stable_per_context_and_unique_across_contexts(tmp_path) -> None:
    async def go():
        h = _history(tmp_path)
        first = await h.web_token("telegram", "u1", "c1")
        second = await h.web_token("telegram", "u1", "c1")  # a second /weburl
        other = await h.web_token("telegram", "u2", "c2")
        assert first == second  # the link stays shareable
        assert len(first) > 32 and other != first
        assert await h.resolve_web_token(first) == ("telegram", "u1", "c1")
        assert await h.resolve_web_token("nope") is None

    asyncio.run(go())


# ── Routes ──────────────────────────────────────────────────────────────────


def test_bad_token_is_404_on_both_routes(tmp_path) -> None:
    client = _client(tmp_path)
    assert client.get("/t/whatever").status_code == 404
    assert client.get("/t/whatever/messages").status_code == 404


def test_page_renders_with_agent_name(tmp_path) -> None:
    async def seed():
        return await _history(tmp_path).web_token("telegram", "u1", "c1")

    token = asyncio.run(seed())
    resp = _client(tmp_path).get(f"/t/{token}")
    assert resp.status_code == 200
    assert "Ada" in resp.text
    assert "Nothing here yet" in resp.text
    assert resp.headers["cache-control"] == "no-store"
    assert "noindex" in resp.headers["x-robots-tag"]


def test_page_offers_the_always_expand_thinking_toggle(tmp_path) -> None:
    """Thinking is collapsed behind a chip per turn; the toggle is what lets a
    reader who wants all of it stop clicking every one."""

    async def seed():
        return await _history(tmp_path).web_token("telegram", "u1", "c1")

    token = asyncio.run(seed())
    body = _client(tmp_path).get(f"/t/{token}").text
    assert 'id="think-toggle"' in body
    assert "expand-thinking" in body  # remembered per browser
    assert "think-sec" in body  # the class the toggle opens in bulk


def test_messages_are_scoped_to_the_context_and_respect_after(tmp_path) -> None:
    async def seed():
        h = _history(tmp_path)
        await h.add_turn("telegram", "u1", "user", "mine", "c1")
        await h.add_turn("telegram", "u1", "assistant", "yours", "c1")
        await h.add_turn("telegram", "u2", "user", "SECRET", "c2")  # another chat
        return await h.web_token("telegram", "u1", "c1")

    token = asyncio.run(seed())
    client = _client(tmp_path)
    resp = client.get(f"/t/{token}/messages")
    assert resp.status_code == 200
    data = resp.json()
    assert [e["text"] for e in data["entries"]] == ["mine", "yours"]
    assert [e["role"] for e in data["entries"]] == ["user", "assistant"]
    assert "SECRET" not in resp.text
    assert data["cursor"] > 0

    # The cursor pages forward: nothing new yet, then only the new turn.
    again = client.get(f"/t/{token}/messages", params={"after": data["cursor"]}).json()
    assert again["entries"] == [] and again["cursor"] == data["cursor"]

    async def more():
        await _history(tmp_path).add_turn("telegram", "u1", "user", "later", "c1")

    asyncio.run(more())
    tail = client.get(f"/t/{token}/messages", params={"after": data["cursor"]}).json()
    assert [e["text"] for e in tail["entries"]] == ["later"]


def test_session_mode_yields_typed_disclosure_entries(tmp_path) -> None:
    """A session-mode tool round classifies into reasoning / tool_call /
    tool_result / text; injection mode has none of them (plain text only)."""

    async def seed():
        h = _history(tmp_path)
        await h.append_session_message("telegram", "u1", {"role": "user", "content": "hi"}, "c1")
        await h.append_session_message(
            "telegram",
            "u1",
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "let me look"},
                    {"type": "text", "text": "one sec"},
                    {"type": "tool_use", "id": "t1", "name": "web_search", "input": {"q": "cats"}},
                ],
            },
            "c1",
        )
        await h.append_session_message(
            "telegram",
            "u1",
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "found"}],
            },
            "c1",
        )
        await h.append_session_message(
            "telegram", "u1", {"role": "assistant", "content": "done"}, "c1"
        )
        return await h.web_token("telegram", "u1", "c1")

    token = asyncio.run(seed())
    entries = _client(tmp_path).get(f"/t/{token}/messages").json()["entries"]
    kinds = [e["kind"] for e in entries]
    assert kinds == ["text", "reasoning", "tool_call", "text", "tool_result", "text"]
    call = entries[2]
    assert call["name"] == "web_search" and call["args"] == {"q": "cats"}
    assert entries[4]["name"] == "web_search" and entries[4]["text"] == "found"
    assert all(e["id"] and "at" in e for e in entries)


def test_injection_mode_has_text_entries_only(tmp_path) -> None:
    async def seed():
        h = _history(tmp_path)
        await h.add_turn("telegram", "u1", "user", "hello", "c1")
        await h.add_turn("telegram", "u1", "assistant", "hi there", "c1")
        return await h.web_token("telegram", "u1", "c1")

    token = asyncio.run(seed())
    entries = _client(tmp_path).get(f"/t/{token}/messages").json()["entries"]
    assert {e["kind"] for e in entries} == {"text"}


# ── Classification details ──────────────────────────────────────────────────


def test_openai_shapes_and_preamble_stripping() -> None:
    rows = [
        {
            "id": 1,
            "source": "session",
            "role": "",
            "created_at": "2026-08-20 10:00:00",
            "payload": {
                "role": "user",
                "content": (
                    "[Current date & time: Thursday, August 20, 2026 10:00 UTC]\n\n"
                    "<memories>\nlikes tea\n\nand cats\n</memories>\n\nwhat's up?"
                ),
            },
        },
        {
            "id": 2,
            "source": "session",
            "role": "",
            "created_at": "2026-08-20 10:00:01",
            "payload": {
                "role": "assistant",
                "content": None,
                "reasoning_content": "thinking hard",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}
                ],
            },
        },
        {
            "id": 3,
            "source": "session",
            "role": "",
            "created_at": "2026-08-20 10:00:02",
            "payload": {"role": "tool", "tool_call_id": "c1", "content": "file.txt"},
        },
        {
            "id": 4,
            "source": "session",
            "role": "",
            "created_at": "2026-08-20 10:00:03",
            "payload": {"role": "system", "content": "NEVER SHOW THIS"},
        },
    ]
    entries = to_entries(rows)
    assert [e["kind"] for e in entries] == ["text", "reasoning", "tool_call", "tool_result"]
    assert entries[0]["text"] == "what's up?"  # preamble + memories dropped
    assert entries[2]["args"] == {"cmd": "ls"}
    assert entries[3]["name"] == "bash"  # result labelled from its call
    assert not any("NEVER SHOW THIS" in str(e) for e in entries)


def test_strip_preamble_leaves_ordinary_messages_alone() -> None:
    assert strip_preamble("[note] to self") == "[note] to self"
    assert strip_preamble("plain text") == "plain text"


# ── Telegram command ────────────────────────────────────────────────────────


def test_weburl_command_replies_with_the_stable_link(tmp_path) -> None:
    async def go():
        history = _history(tmp_path)
        ch = object.__new__(TelegramChannel)
        ch.channel_name = "telegram"
        ch._fold = lambda chat, message=None: chat.id
        ch._convo_user_id = lambda chat, sender_id: str(sender_id)
        ch._may_act = AsyncMock(return_value=True)
        ch.agent = SimpleNamespace(history=history, _base_url=lambda: "https://agent.example")

        msg = SimpleNamespace(reply_text=AsyncMock(), message_id=1)
        update = SimpleNamespace(
            effective_user=SimpleNamespace(id=7),
            message=msg,
            effective_chat=SimpleNamespace(id=100, type="private"),
        )
        await ch._on_weburl_command(update, None)
        await ch._on_weburl_command(update, None)

        urls = [call.args[0] for call in msg.reply_text.await_args_list]
        assert urls[0] == urls[1]  # same token on a second call
        token = urls[0].rsplit("/t/", 1)[1]
        assert await history.resolve_web_token(token) == ("telegram", "7", "100")
        assert "localhost" not in urls[0]

    asyncio.run(go())


def test_weburl_command_respects_the_permission_gate(tmp_path) -> None:
    async def go():
        ch = object.__new__(TelegramChannel)
        ch.channel_name = "telegram"
        ch._fold = lambda chat, message=None: chat.id
        ch._convo_user_id = lambda chat, sender_id: str(sender_id)
        ch._may_act = AsyncMock(return_value=False)
        msg = SimpleNamespace(reply_text=AsyncMock(), message_id=1)
        update = SimpleNamespace(
            effective_user=SimpleNamespace(id=7),
            message=msg,
            effective_chat=SimpleNamespace(id=100, type="private"),
        )
        await ch._on_weburl_command(update, None)
        msg.reply_text.assert_not_awaited()

    asyncio.run(go())


def test_weburl_is_in_the_telegram_menu() -> None:
    assert any(cmd.command == "weburl" for cmd in _MENU_COMMANDS)
