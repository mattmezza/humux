import asyncio
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from channels.telegram import _MENU_COMMANDS, TELEGRAM_LIMIT, TelegramChannel, _chunk


def _dedup_channel() -> TelegramChannel:
    """Bare channel wired only for _on_text up to the create_task dispatch (#154)."""
    ch = object.__new__(TelegramChannel)
    ch.channel_name = "telegram"
    ch._bot_username = "coachbot"
    ch._last_inbound = {}
    ch._fold = lambda chat: None
    ch._remember_chat = lambda *a: None
    ch._reply_context = lambda m: ""
    ch._may_act = AsyncMock(return_value=True)
    ch._handle_text = AsyncMock()
    ch._turn_routing = lambda u, m, c: {
        "user_id": str(u.effective_user.id),
        "speaker_tag": "",
        "respond": True,
        "addressed": True,
    }
    return ch


def _text_update(text: str, user_id: int = 7, chat_id: int = 100) -> SimpleNamespace:
    msg = SimpleNamespace(text=text, message_id=1)
    user = SimpleNamespace(id=user_id, is_bot=False)
    chat = SimpleNamespace(id=chat_id, type="private")
    return SimpleNamespace(effective_user=user, message=msg, effective_chat=chat)


@pytest.mark.asyncio
async def test_duplicate_inbound_collapsed_to_one_turn() -> None:
    ch = _dedup_channel()
    await ch._on_text(_text_update("/yolo-on"), None)
    await ch._on_text(_text_update("/yolo-on"), None)  # redelivery
    await asyncio.sleep(0)
    assert ch._handle_text.await_count == 1  # second dropped


@pytest.mark.asyncio
async def test_same_text_from_different_senders_not_deduped() -> None:
    ch = _dedup_channel()
    await ch._on_text(_text_update("hi", user_id=7), None)
    await ch._on_text(_text_update("hi", user_id=8), None)  # a different person
    await asyncio.sleep(0)
    assert ch._handle_text.await_count == 2


def test_menu_command_names_are_telegram_valid() -> None:
    """Telegram rejects command names with anything but [a-z0-9_] — a hyphenated
    entry (e.g. yolo-on) would make set_my_commands fail at startup."""
    for cmd in _MENU_COMMANDS:
        assert re.fullmatch(r"[a-z0-9_]{1,32}", cmd.command), cmd.command
        assert cmd.description


@pytest.mark.asyncio
async def test_register_commands_full_replaces_menu() -> None:
    ch = object.__new__(TelegramChannel)
    ch.channel_name = "telegram"
    ch.app = SimpleNamespace(bot=AsyncMock())
    await ch.register_commands()
    ch.app.bot.set_my_commands.assert_awaited_once_with(_MENU_COMMANDS)


@pytest.mark.asyncio
async def test_register_commands_swallows_failure() -> None:
    """A menu-registration error must never stop the bot from polling."""
    ch = object.__new__(TelegramChannel)
    ch.channel_name = "telegram"
    ch.app = SimpleNamespace(bot=AsyncMock())
    ch.app.bot.set_my_commands.side_effect = RuntimeError("boom")
    await ch.register_commands()  # does not raise


@pytest.mark.asyncio
async def test_dedup_window_anchored_to_last_processed(monkeypatch) -> None:
    """A repeat within the window drops, but one after it is a fresh turn — the
    window is anchored to the last processed message, not slid on every drop."""
    ch = _dedup_channel()
    clock = {"t": 0.0}
    monkeypatch.setattr("channels.telegram.time.monotonic", lambda: clock["t"])
    await ch._on_text(_text_update("ping"), None)  # processed
    clock["t"] = 1.0
    await ch._on_text(_text_update("ping"), None)  # within 3s → dropped
    clock["t"] = 10.0
    await ch._on_text(_text_update("ping"), None)  # window passed → processed
    await asyncio.sleep(0)
    assert ch._handle_text.await_count == 2


def _channel_with_mock_bot(delay: float = 0.0) -> TelegramChannel:
    # Skip __init__ (it builds a real Application needing a bot token); _typing
    # only touches self.app.bot, the static _route helper and _PLACEHOLDER_DELAY.
    ch = object.__new__(TelegramChannel)
    ch.app = SimpleNamespace(bot=AsyncMock())
    ch.app.bot.send_message.return_value = SimpleNamespace(message_id=4242)
    ch._PLACEHOLDER_DELAY = delay
    return ch


async def _wait_for(predicate, ticks: int = 200) -> None:
    for _ in range(ticks):
        if predicate():
            return
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_slow_turn_posts_silent_placeholder_and_removes_it() -> None:
    # Web K ignores chat actions but renders messages, so a slow turn must leave a
    # real placeholder message behind and clean it up afterwards (#57).
    ch = _channel_with_mock_bot()

    async with ch._typing(123):
        await _wait_for(lambda: ch.app.bot.send_message.await_count > 0)

    args, kwargs = ch.app.bot.send_message.call_args
    assert args[0] == 123
    assert "Thinking" in args[1]
    # Silent: deleting a message does not retract its push, so it must not ping.
    assert kwargs["disable_notification"] is True
    ch.app.bot.delete_message.assert_awaited_once_with(123, 4242)


@pytest.mark.asyncio
async def test_fast_turn_posts_nothing() -> None:
    # Under the delay, a quick reply (native dots already cover it) must not flash
    # a throwaway bubble.
    ch = _channel_with_mock_bot(delay=60.0)

    async with ch._typing(123):
        pass

    ch.app.bot.send_message.assert_not_awaited()
    ch.app.bot.delete_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_placeholder_routes_to_forum_topic() -> None:
    # Folded "<chat>:<thread>" ids must carry message_thread_id so the placeholder
    # lands in the topic, not the main chat.
    ch = _channel_with_mock_bot()

    async with ch._typing("123:7"):
        await _wait_for(lambda: ch.app.bot.send_message.await_count > 0)

    _, kwargs = ch.app.bot.send_message.call_args
    assert kwargs.get("message_thread_id") == 7
    ch.app.bot.delete_message.assert_awaited_once_with(123, 4242)


@pytest.mark.asyncio
async def test_turn_ending_during_send_still_deletes_placeholder() -> None:
    # The leak the review found: if the turn ends while the placeholder send is
    # in flight, the bot must still delete it (not lose the id and orphan it).
    ch = _channel_with_mock_bot()
    gate = asyncio.Event()

    async def slow_send(*_a, **_k):
        await gate.wait()
        return SimpleNamespace(message_id=99)

    ch.app.bot.send_message.side_effect = slow_send

    async def run() -> None:
        async with ch._typing(123):
            await _wait_for(lambda: ch.app.bot.send_message.call_count > 0)

    task = asyncio.ensure_future(run())
    await _wait_for(lambda: ch.app.bot.send_message.call_count > 0)
    # The turn body has now exited and is blocked in _typing's finally, waiting on
    # the still-in-flight send. Releasing it must lead to a delete, not an orphan.
    gate.set()
    await task

    ch.app.bot.delete_message.assert_awaited_once_with(123, 99)


@pytest.mark.asyncio
async def test_placeholder_send_failure_is_non_fatal() -> None:
    # A failed placeholder send must not break the turn or trigger a delete.
    ch = _channel_with_mock_bot()
    ch.app.bot.send_message.side_effect = RuntimeError("blocked")

    async with ch._typing(123):
        await _wait_for(lambda: ch.app.bot.send_message.call_count > 0)

    ch.app.bot.delete_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_placeholder_reflects_approval_wait_then_thinking() -> None:
    # #147: while the agent is blocked on a permission prompt the placeholder must
    # say it's waiting, not still "Thinking…", and flip back once a decision lands.
    ch = _channel_with_mock_bot()

    async with ch._typing(123):
        await _wait_for(lambda: ch.app.bot.send_message.await_count > 0)
        # Agent hits an approval → the label switches to "Waiting…".
        ch._set_typing_waiting(123, True)
        await _wait_for(lambda: ch.app.bot.edit_message_text.await_count > 0)
        text, kwargs = ch.app.bot.edit_message_text.call_args
        assert "Waiting" in text[0]
        assert kwargs["message_id"] == 4242
        # User decides → back to "Thinking…".
        ch._set_typing_waiting(123, False)
        await _wait_for(lambda: ch.app.bot.edit_message_text.await_count > 1)
        assert "Thinking" in ch.app.bot.edit_message_text.call_args[0][0]

    ch.app.bot.delete_message.assert_awaited_once_with(123, 4242)


def test_set_typing_waiting_is_noop_without_active_placeholder() -> None:
    # A subagent / admin-API approval has no _typing wrapper for the chat; toggling
    # must not raise (registry miss), it just does nothing.
    ch = _channel_with_mock_bot()
    ch._set_typing_waiting(999, True)  # no entry for 999 → silent no-op


def test_chunk_keeps_pieces_under_limit_and_loses_nothing() -> None:
    # 50 lines of 200 chars = 10000 chars, well over the 4096 limit (#80).
    text = "\n".join(f"line{i} " + "x" * 200 for i in range(50))
    chunks = _chunk(text)
    assert len(chunks) > 1
    assert all(len(c) <= TELEGRAM_LIMIT for c in chunks)
    # Newline-joined chunks reconstruct the original — no data dropped, no dupes.
    assert "\n".join(chunks) == text


def test_chunk_hard_splits_a_single_oversized_line() -> None:
    # A heredoc with no newlines (the #80 incident) must still be split.
    text = "y" * (TELEGRAM_LIMIT * 2 + 17)
    chunks = _chunk(text)
    assert all(len(c) <= TELEGRAM_LIMIT for c in chunks)
    assert "".join(chunks) == text


def test_chunk_short_text_is_single_piece() -> None:
    assert _chunk("hello") == ["hello"]


@pytest.mark.asyncio
async def test_send_splits_long_reply_into_multiple_messages() -> None:
    # A >4096-char reply must be split, not crash the turn (#80).
    ch = _channel_with_mock_bot()
    await ch.send(123, "z" * (TELEGRAM_LIMIT + 500))
    assert ch.app.bot.send_message.await_count == 2
    for call in ch.app.bot.send_message.await_args_list:
        # call.args[1] is the rendered payload sent to Telegram.
        assert len(call.args[1]) <= TELEGRAM_LIMIT


@pytest.mark.asyncio
async def test_approval_request_chunks_and_keyboard_rides_last() -> None:
    # A long approval prompt must send across messages with the buttons only on
    # the final one, so the keyboard isn't lost (#80).
    ch = _channel_with_mock_bot()
    ch._last_chat_for_user = {}
    await ch.send_approval_request("123", "req1", "D" * (TELEGRAM_LIMIT + 500))
    calls = ch.app.bot.send_message.await_args_list
    assert len(calls) >= 2
    assert all(len(c.args[1]) <= TELEGRAM_LIMIT for c in calls)
    assert all(c.kwargs.get("reply_markup") is None for c in calls[:-1])
    assert calls[-1].kwargs.get("reply_markup") is not None


# --- /zz_skill_* commands (#178) ---


def _skills_agent(entries, content="BODY"):
    """Stub AgentCore with a skills engine returning ``entries``."""
    agent = SimpleNamespace()
    agent.skills = SimpleNamespace(
        index_entries=AsyncMock(return_value=entries),
        get_skill_content=AsyncMock(return_value=content),
    )
    agent.process = AsyncMock()
    return agent


@pytest.mark.asyncio
async def test_register_commands_appends_skill_entries() -> None:
    ch = object.__new__(TelegramChannel)
    ch.channel_name = "telegram"
    ch.app = SimpleNamespace(bot=AsyncMock())
    ch.agent = _skills_agent([{"name": "artifact-hosting", "summary": "Publish pages"}])
    await ch.register_commands()
    (commands,) = ch.app.bot.set_my_commands.await_args.args
    by_name = {c.command: c.description for c in commands}
    assert by_name["zz_skill_artifact_hosting"] == "Publish pages"
    for c in commands:  # every generated name must be Telegram-valid
        assert re.fullmatch(r"[a-z0-9_]{1,32}", c.command), c.command


@pytest.mark.asyncio
async def test_zz_skill_command_loads_into_history_and_chat() -> None:
    ch = _dedup_channel()
    ch.agent = _skills_agent([{"name": "artifact-hosting", "summary": "s"}], content="SKILL BODY")
    ch.send = AsyncMock()
    await ch._on_text(_text_update("/zz_skill_artifact_hosting"), None)
    await asyncio.sleep(0)
    ch._handle_text.assert_not_awaited()  # handled as a command, not a turn
    ch.agent.process.assert_awaited_once()
    kwargs = ch.agent.process.await_args.kwargs
    assert kwargs["respond"] is False  # record-only: rides into history
    assert "SKILL BODY" in kwargs["message"]
    ch.send.assert_awaited_once()
    assert "SKILL BODY" in ch.send.await_args.args[1]


@pytest.mark.asyncio
async def test_zz_skill_unknown_name_reports_error() -> None:
    ch = _dedup_channel()
    ch.agent = _skills_agent([])
    ch.send = AsyncMock()
    await ch._on_text(_text_update("/zz_skill_nope"), None)
    await asyncio.sleep(0)
    ch.agent.process.assert_not_awaited()
    assert "Unknown skill" in ch.send.await_args.args[1]


@pytest.mark.asyncio
async def test_zz_skill_aimed_at_other_bot_is_ignored() -> None:
    ch = _dedup_channel()  # _bot_username = "coachbot"
    ch.agent = _skills_agent([{"name": "weather", "summary": "s"}])
    ch.send = AsyncMock()
    await ch._on_text(_text_update("/zz_skill_weather@otherbot"), None)
    await asyncio.sleep(0)
    ch.send.assert_not_awaited()
    ch.agent.process.assert_not_awaited()
