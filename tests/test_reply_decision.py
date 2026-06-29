"""Reply decision (#36): the gate that keeps the agent quiet in group chats.

Covers the one-shot classifier (should_reply), the group-vs-DM heuristic, the
rate-limit backstop, and the end-to-end suppression path through process().
"""

from __future__ import annotations

import pytest

from core.config import Config
from core.reply_decision import should_reply


class _LLMStub:
    """Minimal LLM stub returning a canned generate_text response."""

    def __init__(self, response: str):
        self._response = response
        self.calls = 0

    async def generate_text(self, *, model: str, prompt: str, max_tokens: int = 1024) -> str:
        self.calls += 1
        return self._response


class _BoomLLM:
    async def generate_text(self, *, model: str, prompt: str, max_tokens: int = 1024) -> str:
        raise RuntimeError("classifier exploded")


# -- should_reply unit tests --------------------------------------------------


@pytest.mark.asyncio
async def test_reply_when_model_says_reply() -> None:
    llm = _LLMStub("REPLY")
    assert await should_reply(llm, "m", "can you check my calendar?", "Coach") is True
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_skip_when_model_says_skip() -> None:
    llm = _LLMStub("SKIP — addressed to another bot")
    assert await should_reply(llm, "m", "@OtherBot what time is it?", "Coach") is False


@pytest.mark.asyncio
async def test_fail_open_on_error() -> None:
    # A classifier failure must never drop a real message.
    assert await should_reply(_BoomLLM(), "m", "hello there", "Coach") is True


@pytest.mark.asyncio
async def test_fail_open_on_blank_and_junk() -> None:
    assert await should_reply(_LLMStub(""), "m", "hi", "Coach") is True  # empty model output
    assert await should_reply(_LLMStub("maybe?"), "m", "hi", "Coach") is True  # unparseable
    # Empty message short-circuits without an LLM call.
    quiet = _LLMStub("SKIP")
    assert await should_reply(quiet, "m", "   ", "Coach") is True
    assert quiet.calls == 0


# -- AgentCore gate: helpers + integration ------------------------------------


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from core.agent import AgentCore

    cfg = Config()
    cfg.reply_decision.enabled = True
    cfg.goal_decomposition.enabled = False  # keep the gate the only background call
    return AgentCore(cfg)


def test_is_group_chat_heuristic(agent) -> None:
    assert agent._is_group_chat("111", "-999") is True  # telegram group id
    assert agent._is_group_chat("111", "111") is False  # telegram/whatsapp DM
    assert agent._is_group_chat("111", "12345@g.us") is True  # whatsapp group jid
    assert agent._is_group_chat("111", "-999:42") is True  # forum topic
    assert agent._is_group_chat("111", "") is False  # no chat id → not a group


def test_reserve_reply_caps_window(agent) -> None:
    cfg = agent.config.reply_decision
    cfg.max_replies_per_window = 3
    cfg.window_seconds = 120
    for _ in range(3):
        assert agent._reserve_reply("telegram", "-999", cfg) is not None
    # Fourth reservation within the window is refused.
    assert agent._reserve_reply("telegram", "-999", cfg) is None
    # A different chat keeps its own budget.
    assert agent._reserve_reply("telegram", "-888", cfg) is not None


def test_release_frees_a_slot(agent) -> None:
    cfg = agent.config.reply_decision
    cfg.max_replies_per_window = 1
    ts = agent._reserve_reply("telegram", "-999", cfg)
    assert ts is not None
    assert agent._reserve_reply("telegram", "-999", cfg) is None  # full
    agent._release_reply("telegram", "-999", ts)  # SKIP gives the slot back
    assert agent._reserve_reply("telegram", "-999", cfg) is not None  # room again


def test_reserve_prunes_expired_window(agent) -> None:
    cfg = agent.config.reply_decision
    cfg.max_replies_per_window = 1
    cfg.window_seconds = 120
    # A timestamp older than the window must be pruned, not counted.
    agent._reply_times[("telegram", "-999")] = [1.0]
    assert agent._reserve_reply("telegram", "-999", cfg) is not None


# Stub the heavy dispatch so process() exercises the gate without real inference.
def _stub_dispatch(agent, monkeypatch, text="ok"):
    from core.models import AgentResponse

    async def _fake(*a, **k):
        return AgentResponse(text=text)

    monkeypatch.setattr(agent, "_process_injection", _fake)
    monkeypatch.setattr(agent, "_process_session", _fake)


@pytest.mark.asyncio
async def test_process_suppresses_on_skip(agent, monkeypatch) -> None:
    monkeypatch.setattr(agent, "_background_llm", lambda *a, **k: _LLMStub("SKIP"))
    _stub_dispatch(agent, monkeypatch)

    resp = await agent.process(
        message="lol same", channel="telegram", user_id="111", chat_id="-999"
    )
    assert resp.text == ""
    # The reserved slot is released on SKIP, so a quiet chat keeps its budget.
    assert agent._reply_times.get(("telegram", "-999"), []) == []


@pytest.mark.asyncio
async def test_process_replies_in_group_and_records_slot(agent, monkeypatch) -> None:
    monkeypatch.setattr(agent, "_background_llm", lambda *a, **k: _LLMStub("REPLY"))
    _stub_dispatch(agent, monkeypatch, text="here you go")

    resp = await agent.process(
        message="can you help me?", channel="telegram", user_id="111", chat_id="-999"
    )
    assert resp.text == "here you go"
    # A real reply keeps its reserved slot — the backstop counts it.
    assert len(agent._reply_times[("telegram", "-999")]) == 1


@pytest.mark.asyncio
async def test_process_skips_gate_in_dm(agent, monkeypatch) -> None:
    # In a DM (chat_id == user_id) the gate must be bypassed entirely, so
    # should_reply is never consulted.
    import core.agent as agent_mod

    calls: list[int] = []

    async def _spy(*a, **k):
        calls.append(1)
        return True

    monkeypatch.setattr(agent_mod, "should_reply", _spy)
    _stub_dispatch(agent, monkeypatch, text="hi")

    resp = await agent.process(
        message="hey there friend", channel="telegram", user_id="111", chat_id="111"
    )
    assert resp.text == "hi"
    assert calls == []  # gate skipped in a 1:1 chat


@pytest.mark.asyncio
async def test_concurrent_burst_is_bounded_by_cap(agent, monkeypatch) -> None:
    # The whole point of the backstop: a bursty bot-to-bot loop (many messages
    # arriving as concurrent tasks for one chat) must not sail past the cap.
    # With reserve-before-await, exactly `max_replies_per_window` get through.
    import asyncio

    cfg = agent.config.reply_decision
    cfg.max_replies_per_window = 6
    cfg.window_seconds = 120
    monkeypatch.setattr(agent, "_background_llm", lambda *a, **k: _LLMStub("REPLY"))
    _stub_dispatch(agent, monkeypatch, text="r")

    resps = await asyncio.gather(
        *(
            agent.process(message=f"loop {i}", channel="telegram", user_id="111", chat_id="-999")
            for i in range(50)
        )
    )
    replied = sum(1 for r in resps if r.text)
    assert replied == 6  # not 50 — the cap held under concurrency


@pytest.mark.asyncio
async def test_process_suppresses_when_rate_capped(agent, monkeypatch) -> None:
    import time

    cfg = agent.config.reply_decision
    cfg.max_replies_per_window = 2
    agent._reply_times[("telegram", "-999")] = [time.time()] * 2
    # Rate cap is checked before the classifier — it must not even be called.
    monkeypatch.setattr(agent, "_background_llm", lambda *a, **k: _BoomLLM())
    _stub_dispatch(agent, monkeypatch)
    resp = await agent.process(
        message="still going", channel="telegram", user_id="111", chat_id="-999"
    )
    assert resp.text == ""
