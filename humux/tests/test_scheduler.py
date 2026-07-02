"""Tests for scheduler helpers and job registration."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from core.scheduler import (
    AgentScheduler,
    _parse_cron,
    run_agent_task,
    run_memory_consolidation,
    run_system_command,
    set_agent_context,
)


def _make_scheduler(job_store) -> AgentScheduler:
    """Build an AgentScheduler with a stub agent (UTC tz) and the given job store."""
    agent = SimpleNamespace(config=SimpleNamespace(agent=SimpleNamespace(timezone="UTC")))
    return AgentScheduler(agent, job_store)


def _iso(offset_minutes: int) -> str:
    return (datetime.now(ZoneInfo("UTC")) + timedelta(minutes=offset_minutes)).isoformat()


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
    # The default agent's bot is the bare "telegram" channel and carries its own
    # allowlist (#133); the owner is its first allowed user.
    channel = AsyncMock()
    channel.config = SimpleNamespace(allowed_user_ids=[123])
    agent = SimpleNamespace(
        channels={"telegram": channel},
        process=AsyncMock(return_value=SimpleNamespace(text="done")),
        config=SimpleNamespace(),
        job_store=None,
    )
    set_agent_context(agent)

    await run_agent_task("do thing", channel="telegram")

    agent.process.assert_awaited_once()
    channel.send.assert_awaited_once_with(123, "done")


@pytest.mark.asyncio
async def test_run_agent_task_agent_job_generates_as_agent() -> None:
    # A "telegram:<agent>" job (#29) is generated AS that agent (agent_name
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
    assert kwargs["agent_name"] == "coach"
    assert kwargs["channel"] == "system"  # execution mode unchanged
    channel.send.assert_awaited_once_with(99, "done")  # coach bot → coach's owner


@pytest.mark.asyncio
async def test_run_agent_task_runs_as_origin_agent_and_chat() -> None:
    # Issue #71: a job carrying an origin agent + chat runs AS that agent and
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
        "do thing", channel="telegram", agent_name="coach", origin_chat_id="-100200:7"
    )

    _, kwargs = agent.process.call_args
    assert kwargs["agent_name"] == "coach"  # not the default identity
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


@pytest.mark.asyncio
async def test_retire_if_past_marks_done() -> None:
    """An active one-shot whose run_at has elapsed is retired to 'done'."""
    job_store = AsyncMock()
    sched = _make_scheduler(job_store)
    retired = await sched._retire_if_past(
        {"id": "old", "schedule": "once", "status": "active", "run_at": _iso(-60)}
    )
    assert retired is True
    job_store.update_status.assert_awaited_once_with("old", "done")


@pytest.mark.asyncio
async def test_retire_if_past_keeps_future_oneshot() -> None:
    """A future one-shot is left alone."""
    job_store = AsyncMock()
    sched = _make_scheduler(job_store)
    retired = await sched._retire_if_past(
        {"id": "soon", "schedule": "once", "status": "active", "run_at": _iso(60)}
    )
    assert retired is False
    job_store.update_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_retire_if_past_ignores_cron() -> None:
    """Cron jobs are never retired regardless of timing."""
    job_store = AsyncMock()
    sched = _make_scheduler(job_store)
    retired = await sched._retire_if_past(
        {"id": "daily", "schedule": "cron", "status": "active", "cron": "0 7 * * *"}
    )
    assert retired is False
    job_store.update_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_load_jobs_retires_past_oneshot_and_skips_registration() -> None:
    """load_jobs marks a past one-shot done and does not register it in APScheduler."""
    job_store = AsyncMock()
    job_store.list_jobs = AsyncMock(
        return_value=[
            {
                "id": "stale",
                "type": "agent",
                "schedule": "once",
                "status": "active",
                "run_at": _iso(-120),
                "task": "x",
                "channel": "telegram",
            }
        ]
    )
    sched = _make_scheduler(job_store)
    await sched.load_jobs()
    job_store.update_status.assert_awaited_once_with("stale", "done")
    assert sched.scheduler.get_job("stale") is None


@pytest.mark.asyncio
async def test_load_jobs_registers_future_oneshot() -> None:
    """A future one-shot is registered, not retired."""
    job_store = AsyncMock()
    job_store.list_jobs = AsyncMock(
        return_value=[
            {
                "id": "future",
                "type": "agent",
                "schedule": "once",
                "status": "active",
                "run_at": _iso(120),
                "task": "x",
                "channel": "telegram",
            }
        ]
    )
    sched = _make_scheduler(job_store)
    await sched.load_jobs()
    job_store.update_status.assert_not_awaited()
    assert sched.scheduler.get_job("future") is not None
