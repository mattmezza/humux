"""/stop command + Stop-button turn abort (#146).

A running turn holds the per-chat lock, so the abort signal reaches it out of
band via ``_active_turns`` (set by ``request_stop``); the tool loop checks the
flag between rounds and bails with ``_STOPPED_MESSAGE`` instead of running the
rest of the turn.
"""

from __future__ import annotations

import asyncio

import pytest

from core.agent import _STOPPED_MESSAGE
from core.config import Config
from core.llm import LLMResponse, LLMToolCall


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from core.agent import AgentCore

    cfg = Config()
    cfg.agent.llm_provider = "deepseek"
    cfg.agent.model = "deepseek-v4-flash"
    cfg.memory.embedding.enabled = False
    cfg.task_reflection.enabled = False
    cfg.goal_decomposition.enabled = False
    return AgentCore(cfg)


class _LoopingLLM:
    """Always wants another tool call — a turn that never ends on its own."""

    provider = "deepseek"

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, **_kw) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text="",
            tool_calls=[LLMToolCall(id=f"c{self.calls}", name="web_search", arguments={"q": "x"})],
        )

    def assistant_message(self, response: LLMResponse) -> dict:
        return {"role": "assistant", "content": response.text}

    def tool_result_messages(self, results: list[dict]) -> list[dict]:
        return [{"role": "user", "content": results}]


def test_request_stop_idle_vs_active(agent) -> None:
    key = ("telegram", "u", "55")
    assert agent.request_stop(*key) is False  # nothing running
    agent._active_turns_map()[key] = asyncio.Event()
    assert agent.request_stop(*key) is True  # active turn → flagged
    assert agent._active_turns_map()[key].is_set()


@pytest.mark.asyncio
async def test_stop_command_idle_is_noop(agent) -> None:
    resp = await agent.process("/stop", "telegram", "u", chat_id="55")
    assert "Nothing to stop" in resp.text


@pytest.mark.asyncio
async def test_stop_command_aborts_running_turn(agent) -> None:
    agent.llm = _LoopingLLM()

    async def fake_tool(call, channel, user_id, request_state):
        await asyncio.sleep(0.002)  # pace rounds so /stop lands well before the cap
        return {"ok": True}

    agent._execute_tool = fake_tool

    async def run_turn():
        return await agent.process("do a lot", "telegram", "u", chat_id="55", message_id=1)

    turn = asyncio.create_task(run_turn())
    # Wait for the turn to register its abort Event (it does so before the tool loop).
    for _ in range(200):
        await asyncio.sleep(0.001)
        if ("telegram", "u", "55") in agent._active_turns_map():
            break
    else:
        turn.cancel()
        pytest.fail("turn never registered an abort Event")

    stop = await agent.process("/stop", "telegram", "u", chat_id="55")
    assert stop.text == ""  # the aborting turn delivers the notice, not /stop

    resp = await asyncio.wait_for(turn, timeout=1)
    assert resp.text == _STOPPED_MESSAGE
    # The loop broke instead of spinning to the round cap.
    assert agent.llm.calls < 50
    # The session records the stop so the next turn knows it was interrupted.
    # (Session is the only history mode now; the old injection-mode conversation_turns
    # table that get_messages reads is no longer written on this path.)
    session = await agent.history.get_session("telegram", "u", "55")
    assert session[-1]["role"] == "assistant" and session[-1]["content"] == _STOPPED_MESSAGE
