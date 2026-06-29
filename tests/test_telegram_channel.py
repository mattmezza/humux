from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from channels.telegram import TelegramChannel


def _channel_with_mock_bot() -> TelegramChannel:
    # Skip __init__ (it builds a real Application needing a bot token); _typing
    # only touches self.app.bot and the static _route helper.
    ch = object.__new__(TelegramChannel)
    ch.app = SimpleNamespace(bot=AsyncMock())
    ch.app.bot.send_message.return_value = SimpleNamespace(message_id=4242)
    return ch


@pytest.mark.asyncio
async def test_typing_posts_and_removes_placeholder_message() -> None:
    # Web K ignores chat actions but renders messages, so the turn must leave a
    # real placeholder message behind and clean it up afterwards (#57).
    ch = _channel_with_mock_bot()

    async with ch._typing(123):
        pass

    ch.app.bot.send_message.assert_awaited_once()
    args, _ = ch.app.bot.send_message.call_args
    assert args[0] == 123
    assert "Thinking" in args[1]
    ch.app.bot.delete_message.assert_awaited_once_with(123, 4242)


@pytest.mark.asyncio
async def test_typing_routes_placeholder_to_forum_topic() -> None:
    # Folded "<chat>:<thread>" ids must carry message_thread_id so the
    # placeholder lands in the topic, not the main chat.
    ch = _channel_with_mock_bot()

    async with ch._typing("123:7"):
        pass

    _, kwargs = ch.app.bot.send_message.call_args
    assert kwargs.get("message_thread_id") == 7
    ch.app.bot.delete_message.assert_awaited_once_with(123, 4242)


@pytest.mark.asyncio
async def test_typing_survives_placeholder_send_failure() -> None:
    # A failed placeholder send must not break the turn or trigger a delete.
    ch = _channel_with_mock_bot()
    ch.app.bot.send_message.side_effect = RuntimeError("blocked")

    async with ch._typing(123):
        pass

    ch.app.bot.delete_message.assert_not_awaited()
