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
