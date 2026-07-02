"""Tests for conversation compaction, token-usage capture, and the /clear alias."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.compaction import (
    compact_messages,
    compaction_threshold_tokens,
    effective_window,
    should_compact,
)
from core.config import CompactionConfig, Config
from core.history import ConversationHistory
from core.llm import _anthropic_usage, _openai_usage


class FakeLLM:
    """Minimal stand-in exposing the async generate_text used by compaction."""

    def __init__(self, summary: str = "SUMMARY") -> None:
        self.summary = summary
        self.calls = 0

    async def generate_text(self, *, model: str, prompt: str, max_tokens: int = 2048) -> str:
        self.calls += 1
        return self.summary


# ---------------------------------------------------------------------------
# Threshold logic
# ---------------------------------------------------------------------------


def test_effective_window_known_and_fallback() -> None:
    cfg = CompactionConfig(context_window=12345)
    assert effective_window(cfg, "claude-haiku-4-5") == 200_000
    assert effective_window(cfg, "claude-opus-4-8") == 1_000_000
    assert effective_window(cfg, "some-unknown-model") == 12345


def test_threshold_percent_vs_tokens() -> None:
    pct = CompactionConfig(threshold_type="percent", threshold_percent=80)
    assert compaction_threshold_tokens(pct, "claude-haiku-4-5") == 160_000
    tok = CompactionConfig(threshold_type="tokens", threshold_tokens=150_000)
    assert compaction_threshold_tokens(tok, "claude-haiku-4-5") == 150_000


def test_should_compact_gates() -> None:
    cfg = CompactionConfig(enabled=True, threshold_type="tokens", threshold_tokens=1000)
    assert should_compact(cfg, 1000, "m") is True
    assert should_compact(cfg, 999, "m") is False
    assert should_compact(cfg, 0, "m") is False  # no usage info
    disabled = CompactionConfig(enabled=False, threshold_type="tokens", threshold_tokens=1000)
    assert should_compact(disabled, 5000, "m") is False


# ---------------------------------------------------------------------------
# Usage capture
# ---------------------------------------------------------------------------


def test_anthropic_usage_sums_cache_into_context() -> None:
    resp = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=300,
            cache_creation_input_tokens=50,
        )
    )
    u = _anthropic_usage(resp)
    assert u["context_tokens"] == 450  # 100 + 300 + 50
    assert u["output_tokens"] == 20


def test_openai_usage_uses_prompt_tokens() -> None:
    resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=777, completion_tokens=33))
    u = _openai_usage(resp)
    assert u["context_tokens"] == 777
    assert u["output_tokens"] == 33


def test_usage_none_when_absent() -> None:
    assert _anthropic_usage(SimpleNamespace(usage=None)) is None
    assert _openai_usage(SimpleNamespace(usage=None)) is None


# ---------------------------------------------------------------------------
# compact_messages
# ---------------------------------------------------------------------------


def _session_with_tool_pair() -> list[dict]:
    return [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t", "name": "x", "input": {}}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "r"}]},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]


@pytest.mark.asyncio
async def test_compact_keeps_recent_and_summarizes_rest() -> None:
    llm = FakeLLM("THE SUMMARY")
    msgs = _session_with_tool_pair()
    result = await compact_messages(llm, "m", msgs, keep_recent_turns=1)
    assert result is not None
    new, summary = result
    assert summary == "THE SUMMARY"
    assert llm.calls == 1
    # Rebuilt as: user(summary), assistant(ack), then the last real user turn verbatim.
    assert new[0]["role"] == "user" and "THE SUMMARY" in new[0]["content"]
    assert new[1]["role"] == "assistant"
    assert new[2:] == [{"role": "user", "content": "u3"}, {"role": "assistant", "content": "a3"}]
    # Valid alternation: never two same-role in a row.
    roles = [m["role"] for m in new]
    assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))


@pytest.mark.asyncio
async def test_compact_does_not_split_tool_pair() -> None:
    # Keeping 2 turns cuts before u2 — the tool_use/tool_result pair stays together
    # in the summarized prefix, never straddling the boundary.
    llm = FakeLLM()
    msgs = _session_with_tool_pair()
    new, _ = await compact_messages(llm, "m", msgs, keep_recent_turns=2)
    tail = new[2:]
    # The kept tail must start with a real user message (not a tool_result carrier).
    first = tail[0]
    assert first["role"] == "user" and isinstance(first["content"], str)


@pytest.mark.asyncio
async def test_compact_noop_when_too_few_turns() -> None:
    llm = FakeLLM()
    msgs = _session_with_tool_pair()  # 3 real user turns
    assert await compact_messages(llm, "m", msgs, keep_recent_turns=3) is None
    assert llm.calls == 0


# ---------------------------------------------------------------------------
# History.replace_session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replace_session_rewrites_messages(tmp_path) -> None:
    h = ConversationHistory(db_path=str(tmp_path / "h.db"))
    await h.append_session_message("telegram", "u", {"role": "user", "content": "old"})
    await h.replace_session("telegram", "u", [{"role": "user", "content": "new"}])
    assert await h.get_session("telegram", "u") == [{"role": "user", "content": "new"}]
    # Cold instance reads the rewritten rows.
    h2 = ConversationHistory(db_path=str(tmp_path / "h.db"))
    assert await h2.get_session("telegram", "u") == [{"role": "user", "content": "new"}]


# ---------------------------------------------------------------------------
# Agent integration: _maybe_compact + /clear alias
# ---------------------------------------------------------------------------


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from core.agent import AgentCore

    a = AgentCore(Config())
    a.history_mode = "session"
    return a


@pytest.mark.asyncio
async def test_maybe_compact_replaces_session_and_notifies(agent, monkeypatch) -> None:
    agent.config.compaction.enabled = True
    agent.config.compaction.threshold_type = "tokens"
    agent.config.compaction.threshold_tokens = 100
    agent.config.compaction.keep_recent_turns = 1

    for m in _session_with_tool_pair():
        await agent.history.append_session_message("telegram", "u", m, "")

    monkeypatch.setattr(agent, "_background_llm", lambda provider, thinking_level="": FakeLLM("S"))
    response = SimpleNamespace(usage={"context_tokens": 999})

    notice = await agent._maybe_compact("telegram", "u", "", response)
    assert notice is not None and "summarized" in notice.lower()
    session = await agent.history.get_session("telegram", "u")
    assert "S" in session[0]["content"]
    assert session[-2:] == [
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]


@pytest.mark.asyncio
async def test_maybe_compact_below_threshold_noop(agent, monkeypatch) -> None:
    agent.config.compaction.enabled = True
    agent.config.compaction.threshold_type = "tokens"
    agent.config.compaction.threshold_tokens = 100000
    for m in _session_with_tool_pair():
        await agent.history.append_session_message("telegram", "u", m, "")
    monkeypatch.setattr(agent, "_background_llm", lambda provider, thinking_level="": FakeLLM("S"))
    response = SimpleNamespace(usage={"context_tokens": 50})
    assert await agent._maybe_compact("telegram", "u", "", response) is None


@pytest.mark.asyncio
async def test_clear_alias_clears_conversation(agent) -> None:
    resp = await agent.process("/clear", channel="telegram", user_id="u", chat_id="")
    assert resp.text == "Conversation cleared."


@pytest.mark.asyncio
async def test_new_still_clears_conversation(agent) -> None:
    resp = await agent.process("/new", channel="telegram", user_id="u", chat_id="")
    assert resp.text == "Conversation cleared."
