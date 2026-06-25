"""Tests for optional tools, per-turn datetime injection, and prompt caching."""

from __future__ import annotations

import pytest

from core.config import Config
from core.history import ConversationHistory
from core.prompt_builder import build_prompt_sections
from core.tools import active_tool_prompts, tool_env

# ---------------------------------------------------------------------------
# Tools registry
# ---------------------------------------------------------------------------


def test_gh_tool_inactive_by_default() -> None:
    cfg = Config()
    assert active_tool_prompts(cfg) == []
    assert tool_env(cfg) == {}


def test_gh_tool_env_and_advert_when_enabled() -> None:
    cfg = Config()
    cfg.tools.gh.enabled = True
    cfg.tools.gh.token = "ghp_secret"
    assert tool_env(cfg) == {"GH_TOKEN": "ghp_secret"}
    blocks = active_tool_prompts(cfg)
    assert len(blocks) == 1
    assert "gh" in blocks[0]


def test_gh_enabled_without_token_has_no_env() -> None:
    cfg = Config()
    cfg.tools.gh.enabled = True  # no token
    assert tool_env(cfg) == {}
    # Still advertised so the agent knows the capability exists.
    assert active_tool_prompts(cfg)


# ---------------------------------------------------------------------------
# Static system prompt: no datetime, tool advert gated on activation
# ---------------------------------------------------------------------------


def _sections(cfg: Config):
    return build_prompt_sections(
        config=cfg,
        history_mode="session",
        skills_index="",
        memories="",
        reflections="",
        decomposed_goal=None,
    )


def test_static_prompt_has_no_datetime() -> None:
    cfg = Config()
    sections = _sections(cfg)
    # The static prompt must not bake in a concrete date/time (it is injected
    # per turn instead), so the prefix stays stable and cacheable.
    assert "Today is" not in sections.full_prompt
    assert "Current time:" not in sections.full_prompt
    assert cfg.agent.timezone in sections.intro


def test_tools_section_only_when_enabled() -> None:
    cfg = Config()
    assert _sections(cfg).tools == ""
    cfg.tools.gh.enabled = True
    cfg.tools.gh.token = "ghp_x"
    sections = _sections(cfg)
    assert "<tools>" in sections.tools
    assert sections.tools in sections.full_prompt


# ---------------------------------------------------------------------------
# Session system snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_system_snapshot_roundtrip(tmp_path) -> None:
    history = ConversationHistory(db_path=str(tmp_path / "h.db"))
    assert await history.get_session_system("telegram", "u1") is None
    await history.set_session_system("telegram", "u1", "SYSTEM-A")
    assert await history.get_session_system("telegram", "u1") == "SYSTEM-A"


@pytest.mark.asyncio
async def test_session_system_survives_new_instance(tmp_path) -> None:
    db = str(tmp_path / "h.db")
    h1 = ConversationHistory(db_path=db)
    await h1.set_session_system("telegram", "u1", "SYSTEM-A")
    # Fresh instance (cold cache) must load the snapshot from disk.
    h2 = ConversationHistory(db_path=db)
    assert await h2.get_session_system("telegram", "u1") == "SYSTEM-A"


@pytest.mark.asyncio
async def test_clear_session_drops_system_snapshot(tmp_path) -> None:
    history = ConversationHistory(db_path=str(tmp_path / "h.db"))
    await history.set_session_system("telegram", "u1", "SYSTEM-A")
    await history.clear_session("telegram", "u1")
    assert await history.get_session_system("telegram", "u1") is None


@pytest.mark.asyncio
async def test_clear_drops_system_snapshot(tmp_path) -> None:
    history = ConversationHistory(db_path=str(tmp_path / "h.db"))
    await history.set_session_system("telegram", "u1", "SYSTEM-A")
    await history.clear("telegram", "u1")
    assert await history.get_session_system("telegram", "u1") is None


# ---------------------------------------------------------------------------
# Agent: per-turn preamble + user-message injection + session caching
# ---------------------------------------------------------------------------


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from core.agent import AgentCore

    return AgentCore(Config())


def test_turn_preamble_carries_datetime(agent) -> None:
    preamble = agent._turn_preamble(None)
    assert "Current date & time" in preamble
    # No execution plan when the goal was not decomposed.
    assert "execution_plan" not in preamble


def test_build_user_message_prepends_preamble(agent) -> None:
    preamble = agent._turn_preamble(None)
    msg = agent._build_user_message("hello", None, preamble)
    assert msg["role"] == "user"
    assert msg["content"].startswith(preamble)
    assert msg["content"].endswith("hello")


def test_build_user_message_no_preamble_is_plain(agent) -> None:
    msg = agent._build_user_message("hello", None, "")
    assert msg["content"] == "hello"


@pytest.mark.asyncio
async def test_session_system_built_once_and_reused(agent, monkeypatch) -> None:
    calls = {"n": 0}

    async def fake_build(*args, **kwargs) -> str:
        calls["n"] += 1
        return f"SYSTEM-{calls['n']}"

    monkeypatch.setattr(agent, "_build_system_prompt", fake_build)

    first = await agent._session_system_prompt("telegram", "u1", "")
    second = await agent._session_system_prompt("telegram", "u1", "")
    assert first == second == "SYSTEM-1"
    assert calls["n"] == 1  # built only once for the session

    # After /new (clear), it rebuilds.
    await agent.history.clear_session("telegram", "u1")
    third = await agent._session_system_prompt("telegram", "u1", "")
    assert third == "SYSTEM-2"
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# Per-action write state — one write's outcome must not block a different one
# ---------------------------------------------------------------------------


def _job_call(call_id: str, **params):
    from core.llm import LLMToolCall

    return LLMToolCall(id=call_id, name="manage_jobs", arguments={"action": "create", **params})


async def _approve(name, params, channel, user_id):
    return "approved"


async def _ok_manage_jobs(params):
    return {"ok": True, "job_id": "job_" + params.get("task", ""), "task": params.get("task")}


@pytest.mark.asyncio
async def test_write_signature_distinguishes_distinct_actions(agent) -> None:
    a = agent._write_signature("manage_jobs", {"action": "create", "task": "A"})
    b = agent._write_signature("manage_jobs", {"action": "create", "task": "B"})
    a_again = agent._write_signature("manage_jobs", {"task": "A", "action": "create"})
    assert a != b  # different params → different signature
    assert a == a_again  # key order does not matter


@pytest.mark.asyncio
async def test_distinct_writes_are_independent_after_success(agent, monkeypatch) -> None:
    """A completed write must not block a *different* subsequent write."""
    monkeypatch.setattr(agent, "_request_approval", _approve)
    monkeypatch.setattr(agent, "_tool_manage_jobs", _ok_manage_jobs)
    agent.channels = {"telegram": object()}  # presence so approval path runs

    state = agent._new_request_state()
    first = await agent._execute_tool(
        _job_call("1", task="ping mum", run_at="2026-07-01T09:00:00"),
        "telegram",
        "u1",
        state,
    )
    second = await agent._execute_tool(
        _job_call("2", task="ping dad", run_at="2026-07-02T09:00:00"),
        "telegram",
        "u1",
        state,
    )
    assert first.get("ok") is True
    assert second.get("ok") is True  # not blocked by "already fulfilled"


@pytest.mark.asyncio
async def test_identical_write_is_deduplicated(agent, monkeypatch) -> None:
    """An identical repeated write within a turn is still suppressed.

    Uses send_email — a plain write — because manage_jobs is deliberately
    exempt from the generic guard (it has its own id-based guard, see #11).
    """
    from core.llm import LLMToolCall

    async def _ok_send_email(params):
        return {"ok": True}

    monkeypatch.setattr(agent, "_request_approval", _approve)
    monkeypatch.setattr(agent, "_tool_send_email", _ok_send_email)
    agent.channels = {"telegram": object()}

    args = {"account": "a", "to": "x@y.z", "subject": "s", "body": "b"}
    state = agent._new_request_state()
    first = await agent._execute_tool(
        LLMToolCall(id="1", name="send_email", arguments=args), "telegram", "u1", state
    )
    repeat = await agent._execute_tool(
        LLMToolCall(id="2", name="send_email", arguments=dict(args)), "telegram", "u1", state
    )
    assert first.get("ok") is True
    assert "already completed" in repeat.get("error", "")


# ---------------------------------------------------------------------------
# Job creation (#11): block only a live duplicate id, never a prior write
# ---------------------------------------------------------------------------


async def _no_sync(job_id):  # scheduler.sync_job stub — no APScheduler in tests
    return None


@pytest.mark.asyncio
async def test_brand_new_job_id_never_blocked(agent, monkeypatch) -> None:
    """A brand-new job id creates even after another job was made this turn."""
    monkeypatch.setattr(agent, "_request_approval", _approve)
    monkeypatch.setattr(agent.scheduler, "sync_job", _no_sync)
    agent.channels = {"telegram": object()}

    state = agent._new_request_state()
    await agent._execute_tool(
        _job_call("1", job_id="setup", task="t", run_at="2026-07-01T09:00:00"),
        "telegram",
        "u1",
        state,
    )
    new = await agent._execute_tool(
        _job_call(
            "2",
            job_id="flight-monitor-lx1272",
            task="watch flight",
            run_at="2026-07-02T09:00:00",
        ),
        "telegram",
        "u1",
        state,
    )
    assert new.get("ok") is True
    assert new.get("job_id") == "flight-monitor-lx1272"


@pytest.mark.asyncio
async def test_recreate_active_job_id_blocked_by_id_not_generic_guard(agent, monkeypatch) -> None:
    """Recreating a live id is blocked with an id-based message, not 'already completed'."""
    monkeypatch.setattr(agent, "_request_approval", _approve)
    monkeypatch.setattr(agent.scheduler, "sync_job", _no_sync)
    agent.channels = {"telegram": object()}

    state = agent._new_request_state()
    first = await agent._execute_tool(
        _job_call("1", job_id="flight-x", task="t", run_at="2026-07-01T09:00:00"),
        "telegram",
        "u1",
        state,
    )
    second = await agent._execute_tool(
        _job_call("2", job_id="flight-x", task="t", run_at="2026-07-01T09:00:00"),
        "telegram",
        "u1",
        state,
    )
    assert first.get("ok") is True
    assert "already exists and is active" in second.get("error", "")
    assert "already completed" not in second.get("error", "")


@pytest.mark.asyncio
async def test_cancelled_job_id_can_be_recreated(agent, monkeypatch) -> None:
    """A done/cancelled id is free to recreate (only live ids block)."""
    monkeypatch.setattr(agent, "_request_approval", _approve)
    monkeypatch.setattr(agent.scheduler, "sync_job", _no_sync)
    agent.channels = {"telegram": object()}

    state = agent._new_request_state()
    await agent._execute_tool(
        _job_call("1", job_id="flight-z", task="t", run_at="2026-07-01T09:00:00"),
        "telegram",
        "u1",
        state,
    )
    await agent._tool_manage_jobs({"action": "cancel", "job_id": "flight-z"})
    again = await agent._execute_tool(
        _job_call("2", job_id="flight-z", task="t2", run_at="2026-07-03T09:00:00"),
        "telegram",
        "u1",
        state,
    )
    assert again.get("ok") is True


@pytest.mark.asyncio
async def test_skipping_one_write_does_not_block_a_different_one(agent, monkeypatch) -> None:
    """Skipping a write blocks only that exact action, not other writes."""
    decisions = {"ping mum": "skipped", "ping dad": "approved"}

    async def fake_approval(name, params, channel, user_id):
        return decisions.get(params.get("task"), "approved")

    monkeypatch.setattr(agent, "_request_approval", fake_approval)
    monkeypatch.setattr(agent, "_tool_manage_jobs", _ok_manage_jobs)
    agent.channels = {"telegram": object()}

    state = agent._new_request_state()
    skipped = await agent._execute_tool(
        _job_call("1", task="ping mum", run_at="2026-07-01T09:00:00"),
        "telegram",
        "u1",
        state,
    )
    other = await agent._execute_tool(
        _job_call("2", task="ping dad", run_at="2026-07-02T09:00:00"),
        "telegram",
        "u1",
        state,
    )
    assert "skipped" in skipped.get("error", "")
    assert other.get("ok") is True  # the skip did not leak onto a different write


# ---------------------------------------------------------------------------
# Batch approval — multiple writes in one turn share a single prompt (#12)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_approval_asks_once_for_multiple_writes(agent, monkeypatch) -> None:
    """Several writes in one turn must trigger exactly one approval prompt."""
    prompts = {"n": 0}

    async def fake_await(description, channel, user_id, tool_name=None, params=None):
        prompts["n"] += 1
        return "approved"

    monkeypatch.setattr(agent, "_await_approval", fake_await)
    monkeypatch.setattr(agent, "_tool_manage_jobs", _ok_manage_jobs)
    agent.channels = {"telegram": object()}

    state = agent._new_request_state()
    c1 = _job_call("1", task="ping mum", run_at="2026-07-01T09:00:00")
    c2 = _job_call("2", task="ping dad", run_at="2026-07-02T09:00:00")

    await agent._batch_approve_writes([c1, c2], "telegram", "u1", state)
    assert prompts["n"] == 1  # one prompt covered both writes

    r1 = await agent._execute_tool(c1, "telegram", "u1", state)
    r2 = await agent._execute_tool(c2, "telegram", "u1", state)
    assert prompts["n"] == 1  # execution reused the batch decision, no re-prompt
    assert r1.get("ok") is True and r2.get("ok") is True


@pytest.mark.asyncio
async def test_batch_approval_denied_blocks_every_write(agent, monkeypatch) -> None:
    """Denying the batch blocks all of its writes, not just one."""

    async def deny(description, channel, user_id, tool_name=None, params=None):
        return "denied"

    monkeypatch.setattr(agent, "_await_approval", deny)
    monkeypatch.setattr(agent, "_tool_manage_jobs", _ok_manage_jobs)
    agent.channels = {"telegram": object()}

    state = agent._new_request_state()
    c1 = _job_call("1", task="ping mum", run_at="2026-07-01T09:00:00")
    c2 = _job_call("2", task="ping dad", run_at="2026-07-02T09:00:00")

    await agent._batch_approve_writes([c1, c2], "telegram", "u1", state)
    r1 = await agent._execute_tool(c1, "telegram", "u1", state)
    r2 = await agent._execute_tool(c2, "telegram", "u1", state)
    assert "denied" in r1.get("error", "")
    assert "denied" in r2.get("error", "")


@pytest.mark.asyncio
async def test_single_write_is_not_batched(agent, monkeypatch) -> None:
    """A lone write is left to the per-call path, not the batch prompt."""
    prompts = {"n": 0}

    async def fake_await(description, channel, user_id, tool_name=None, params=None):
        prompts["n"] += 1
        return "approved"

    monkeypatch.setattr(agent, "_await_approval", fake_await)
    agent.channels = {"telegram": object()}

    state = agent._new_request_state()
    await agent._batch_approve_writes(
        [_job_call("1", task="ping mum", run_at="2026-07-01T09:00:00")],
        "telegram",
        "u1",
        state,
    )
    assert prompts["n"] == 0  # nothing to batch for a single write
    assert state["write_decisions"] == {}
