"""Reply decision (#36): the gate that keeps the agent quiet in group chats.

Covers group eligibility, the reply rate cap, and model routing through process().
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from core.config import Config, ReplyDecisionConfig
from core.decisions import DecisionClient


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from core.agent import AgentCore

    cfg = Config()
    cfg.reply_decision.enabled = True
    cfg.decisions.provider = "typesafe"
    cfg.goal_decomposition.enabled = False  # keep the gate the only background call
    cfg.memory.embedding.enabled = False  # no sidecar in tests → 12s timeout per embed call
    agent = AgentCore(cfg)
    monkeypatch.setattr(
        agent, "_background_llm", Mock(side_effect=AssertionError("unexpected LLM"))
    )
    return agent


@pytest.fixture
def evaluate(monkeypatch):
    evaluate = AsyncMock(return_value={"reply": {"type": "noul", "noul": 0.95}})
    monkeypatch.setattr(DecisionClient, "_evaluate", evaluate)
    return evaluate


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
async def test_process_suppresses_on_skip(agent, monkeypatch, evaluate) -> None:
    evaluate.return_value = {"reply": {"type": "noul", "noul": 0.05}}
    _stub_dispatch(agent, monkeypatch)

    resp = await agent.process(
        message="lol same",
        channel="telegram",
        user_id="111",
        chat_id="-999",
        addressed=False,
    )
    assert resp.text == ""
    evaluate.assert_awaited_once()
    # The reserved slot is released on SKIP, so a quiet chat keeps its budget.
    assert agent._reply_times.get(("telegram", "-999"), []) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("answers", [{"reply": {"type": "noul", "noul": 0.95}}, {}])
async def test_process_replies_in_group_and_records_slot(
    agent, monkeypatch, evaluate, answers
) -> None:
    evaluate.return_value = answers
    _stub_dispatch(agent, monkeypatch, text="here you go")

    resp = await agent.process(
        message="can you help me?",
        channel="telegram",
        user_id="111",
        chat_id="-999",
        addressed=False,
    )
    assert resp.text == "here you go"
    evaluate.assert_awaited_once()
    # A real reply keeps its reserved slot — the backstop counts it.
    assert len(agent._reply_times[("telegram", "-999")]) == 1


@pytest.mark.asyncio
async def test_process_skips_gate_in_dm(agent, monkeypatch, evaluate) -> None:
    _stub_dispatch(agent, monkeypatch, text="hi")

    resp = await agent.process(
        message="hey there friend",
        channel="telegram",
        user_id="111",
        chat_id="111",
        addressed=False,
    )
    assert resp.text == "hi"
    evaluate.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_config_without_decision_provider_only_applies_cap(
    agent, monkeypatch, evaluate
) -> None:
    agent.config.decisions.provider = ""
    agent.config.reply_decision = ReplyDecisionConfig.model_validate(
        {
            "enabled": True,
            "provider": "openrouter",
            "model": "old-model",
            "thinking_level": "low",
            "max_replies_per_window": 1,
        }
    )
    _stub_dispatch(agent, monkeypatch)

    args = {"channel": "telegram", "user_id": "111", "chat_id": "-999", "addressed": False}
    assert (await agent.process("first message", **args)).text == "ok"
    assert (await agent.process("second message", **args)).text == ""
    evaluate.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_burst_is_bounded_by_cap(agent, monkeypatch, evaluate) -> None:
    # The whole point of the backstop: a bursty bot-to-bot loop (many messages
    # arriving as concurrent tasks for one chat) must not sail past the cap.
    # With reserve-before-await, exactly `max_replies_per_window` get through.
    import asyncio

    cfg = agent.config.reply_decision
    cfg.max_replies_per_window = 6
    cfg.window_seconds = 120
    _stub_dispatch(agent, monkeypatch, text="r")

    resps = await asyncio.gather(
        *(
            agent.process(
                message=f"loop {i}",
                channel="telegram",
                user_id="111",
                chat_id="-999",
                addressed=False,
            )
            for i in range(50)
        )
    )
    replied = sum(1 for r in resps if r.text)
    assert replied == 6  # not 50 — the cap held under concurrency
    assert evaluate.await_count == 6


@pytest.mark.asyncio
async def test_process_suppresses_when_rate_capped(agent, monkeypatch, evaluate) -> None:
    import time

    cfg = agent.config.reply_decision
    cfg.max_replies_per_window = 2
    agent._reply_times[("telegram", "-999")] = [time.time()] * 2
    # Rate cap is checked before the classifier — it must not even be called.
    _stub_dispatch(agent, monkeypatch)
    resp = await agent.process(
        message="still going",
        channel="telegram",
        user_id="111",
        chat_id="-999",
        addressed=False,
    )
    assert resp.text == ""
    evaluate.assert_not_awaited()
