import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from channels.telegram import TELEGRAM_LIMIT, TelegramChannel, _chunk


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
