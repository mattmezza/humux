"""Tests for scheduler helpers and job registration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.scheduler import (
    _parse_cron,
    run_agent_task,
    run_memory_consolidation,
    run_system_command,
    set_agent_context,
)


def test_parse_cron_valid_expression() -> None:
    parsed = _parse_cron("0 7 * * *")
    assert parsed == {"minute": "0", "hour": "7"}


def test_parse_cron_invalid_expression() -> None:
    with pytest.raises(ValueError):
        _parse_cron("* * *")


def test_parse_cron_all_wildcards() -> None:
    parsed = _parse_cron("* * * * *")
    assert parsed == {}


def test_parse_cron_full_fields() -> None:
    parsed = _parse_cron("30 7 1 6 1-5")
    assert parsed == {"minute": "30", "hour": "7", "day": "1", "month": "6", "day_of_week": "1-5"}


@pytest.mark.asyncio
async def test_run_system_command_logs_nonzero_exit(monkeypatch) -> None:
    agent = SimpleNamespace(executor=SimpleNamespace(run_command_trusted=AsyncMock()))
    agent.executor.run_command_trusted.return_value = {
        "stdout": "",
        "stderr": "boom",
        "exit_code": 1,
    }
    set_agent_context(agent)

    await run_system_command("echo fail")

    agent.executor.run_command_trusted.assert_awaited_once_with("echo fail")


@pytest.mark.asyncio
async def test_run_memory_consolidation_calls_store(monkeypatch) -> None:
    memory = SimpleNamespace(consolidate_and_cleanup=AsyncMock(return_value={}))
    llm_sentinel = object()
    agent = SimpleNamespace(
        memory=memory,
        llm=llm_sentinel,
        config=SimpleNamespace(
            memory=SimpleNamespace(
                consolidation_model="model",
                consolidation_provider="anthropic",
                consolidation_thinking_level="",
            ),
        ),
        _memory_llm=lambda self_provider, thinking_level="": llm_sentinel,
    )
    set_agent_context(agent)

    await run_memory_consolidation()

    memory.consolidate_and_cleanup.assert_awaited_once_with(llm=llm_sentinel, model="model")


@pytest.mark.asyncio
async def test_run_agent_task_sends_to_owner() -> None:
    channel = AsyncMock()
    agent = SimpleNamespace(
        channels={"telegram": channel},
        process=AsyncMock(return_value=SimpleNamespace(text="done")),
        config=SimpleNamespace(
            channels=SimpleNamespace(telegram=SimpleNamespace(allowed_user_ids=[123]))
        ),
        job_store=None,
    )
    set_agent_context(agent)

    await run_agent_task("do thing", channel="telegram")

    agent.process.assert_awaited_once()
    channel.send.assert_awaited_once_with(123, "done")


@pytest.mark.asyncio
async def test_run_agent_task_persona_job_generates_as_persona() -> None:
    # A "telegram:<persona>" job (#29) is generated AS that persona (persona_name
    # forced) while keeping the "system" execution mode, and delivered via that
    # bot to its own owner (the bot's allowlist, not the global one).
    channel = AsyncMock()
    channel.config = SimpleNamespace(allowed_user_ids=[99])
    agent = SimpleNamespace(
        channels={"telegram:coach": channel},
        process=AsyncMock(return_value=SimpleNamespace(text="done")),
        config=SimpleNamespace(
            channels=SimpleNamespace(telegram=SimpleNamespace(allowed_user_ids=[1]))
        ),
        job_store=None,
    )
    set_agent_context(agent)

    await run_agent_task("ping", channel="telegram:coach")

    _, kwargs = agent.process.call_args
    assert kwargs["persona_name"] == "coach"
    assert kwargs["channel"] == "system"  # execution mode unchanged
    channel.send.assert_awaited_once_with(99, "done")  # coach bot → coach's owner


@pytest.mark.asyncio
async def test_run_agent_task_runs_as_origin_persona_and_chat() -> None:
    # Issue #71: a job carrying an origin persona + chat runs AS that persona and
    # is delivered back to that chat — not the default identity in the owner DM.
    channel = AsyncMock()
    agent = SimpleNamespace(
        channels={"telegram": channel},
        process=AsyncMock(return_value=SimpleNamespace(text="done")),
        config=SimpleNamespace(
            channels=SimpleNamespace(telegram=SimpleNamespace(allowed_user_ids=[123]))
        ),
        job_store=None,
    )
    set_agent_context(agent)

    await run_agent_task(
        "do thing", channel="telegram", persona="coach", origin_chat_id="-100200:7"
    )

    _, kwargs = agent.process.call_args
    assert kwargs["persona_name"] == "coach"  # not the default identity
    channel.send.assert_awaited_once_with("-100200:7", "done")  # origin chat, not owner 123


@pytest.mark.asyncio
async def test_run_agent_task_marks_oneshot_done() -> None:
    """One-shot jobs should be marked 'done' after execution."""
    channel = AsyncMock()
    job_store = AsyncMock()
    job_store.get_job = AsyncMock(
        return_value={"id": "test-once", "schedule": "once", "status": "active"}
    )
    job_store.update_status = AsyncMock()

    agent = SimpleNamespace(
        channels={"telegram": channel},
        process=AsyncMock(return_value=SimpleNamespace(text="result")),
        config=SimpleNamespace(
            channels=SimpleNamespace(telegram=SimpleNamespace(allowed_user_ids=[42]))
        ),
        job_store=job_store,
    )
    set_agent_context(agent)

    await run_agent_task("do once", channel="telegram", job_id="test-once")

    job_store.update_status.assert_awaited_once_with("test-once", "done")


@pytest.mark.asyncio
async def test_run_agent_task_silent_no_updates() -> None:
    """Silent agent tasks with no meaningful response should not send a message."""
    channel = AsyncMock()
    agent = SimpleNamespace(
        channels={"telegram": channel},
        process=AsyncMock(return_value=SimpleNamespace(text="[NO_UPDATES]")),
        config=SimpleNamespace(
            channels=SimpleNamespace(telegram=SimpleNamespace(allowed_user_ids=[42]))
        ),
        job_store=None,
    )
    set_agent_context(agent)

    await run_agent_task("check email", channel="telegram", silent=True)

    agent.process.assert_awaited_once()
    channel.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_agent_task_no_channel() -> None:
    """If the target channel is not registered, the response is dropped."""
    agent = SimpleNamespace(
        channels={},
        process=AsyncMock(return_value=SimpleNamespace(text="done")),
        config=SimpleNamespace(
            channels=SimpleNamespace(telegram=SimpleNamespace(allowed_user_ids=[42]))
        ),
        job_store=None,
    )
    set_agent_context(agent)

    # Should not raise
    await run_agent_task("do thing", channel="telegram")
    agent.process.assert_awaited_once()
