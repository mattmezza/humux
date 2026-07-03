"""Mid-turn steering (#145).

An addressed follow-up sent while a turn is running for its chat is buffered and
injected into that turn between tool rounds — the user redirects without waiting
for the whole tool loop. The turn acks pickup with a 👀 reaction. The buffer is
per conversation key, so a steer never crosses into another chat.
"""

from __future__ import annotations

import asyncio

import pytest

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


class _RecordingLLM:
    """Loops a fixed number of tool calls, then answers. Records every ``messages``
    array it was asked to generate on, so a test can assert what the model saw."""

    provider = "deepseek"

    def __init__(self, rounds: int = 3) -> None:
        self.rounds = rounds
        self.calls = 0
        self.seen: list[list[dict]] = []

    async def generate(self, *, messages, **_kw) -> LLMResponse:
        # Deep-enough copy: capture content refs before later rounds mutate the list.
        self.seen.append(list(messages))
        self.calls += 1
        if self.calls <= self.rounds:
            return LLMResponse(
                text="",
                tool_calls=[
                    LLMToolCall(id=f"c{self.calls}", name="web_search", arguments={"q": "x"})
                ],
            )
        return LLMResponse(text="done", tool_calls=[])

    def assistant_message(self, response: LLMResponse) -> dict:
        return {"role": "assistant", "content": response.text}

    def tool_result_messages(self, results: list[dict]) -> list[dict]:
        return [{"role": "user", "content": results}]


class _FakeChannel:
    def __init__(self) -> None:
        self.reactions: list[tuple] = []

    async def react(self, chat_id, message_id, emoji) -> None:
        self.reactions.append((chat_id, message_id, emoji))


# --- unit: buffer mechanics ------------------------------------------------


def test_drain_is_scoped_per_chat(agent) -> None:
    a = ("telegram", "u", "1")
    b = ("telegram", "u", "2")
    agent._steer_map().setdefault(a, []).append({"text": "hi", "consumed": False})
    assert agent._drain_steer(*b) == []  # other chat sees nothing
    got = agent._drain_steer(*a)
    assert [e["text"] for e in got] == ["hi"]
    assert got[0]["consumed"] is True  # drained entries are marked for the depositor
    assert agent._drain_steer(*a) == []  # popped, not re-served


@pytest.mark.asyncio
async def test_pop_formats_and_acks(agent) -> None:
    agent.channels = {"telegram": _FakeChannel()}
    key = ("telegram", "u", "1")
    agent._steer_map().setdefault(key, []).append(
        {"text": "use the calendar API instead", "message_id": 7, "consumed": False}
    )
    msg = await agent._pop_steer_message(*key)
    assert msg["role"] == "user"
    assert "<steering_message>" in msg["content"]
    assert "use the calendar API instead" in msg["content"]
    assert agent.channels["telegram"].reactions == [("1", 7, "👀")]
    assert await agent._pop_steer_message(*key) is None  # drained


# --- integration: injection into a running turn ----------------------------


async def _wait_active(agent, key, task):
    for _ in range(400):
        await asyncio.sleep(0.005)
        if key in agent._active_turns_map():
            return
    task.cancel()
    pytest.fail("turn never registered an abort Event")


@pytest.mark.asyncio
async def test_steer_injected_into_running_turn(agent) -> None:
    agent.llm = _RecordingLLM(rounds=3)
    agent.channels = {"telegram": _FakeChannel()}

    async def fake_tool(call, channel, user_id, request_state):
        await asyncio.sleep(0.01)  # pace rounds so the steer lands mid-loop
        return {"ok": True}

    agent._execute_tool = fake_tool
    key = ("telegram", "u", "55")

    turn = asyncio.create_task(
        agent.process("start work", "telegram", "u", chat_id="55", message_id=1)
    )
    await _wait_active(agent, key, turn)

    steer = await agent.process(
        "actually, use the calendar API", "telegram", "u", chat_id="55", message_id=2
    )
    assert steer.text == ""  # consumed by the running turn — no duplicate reply

    resp = await asyncio.wait_for(turn, timeout=3)
    assert resp.text == "done"
    # The steering text reached at least one LLM call, wrapped for the model.
    flat = "".join(str(m.get("content")) for seen in agent.llm.seen for m in seen)
    assert "<steering_message>" in flat
    assert "use the calendar API" in flat
    # And the follow-up got a 👀 ack.
    assert (("55", 2, "👀")) in agent.channels["telegram"].reactions


@pytest.mark.asyncio
async def test_idle_message_is_not_a_steer(agent) -> None:
    # No turn running → not buffered as a steer; falls through to normal processing.
    key = ("telegram", "u", "9")
    assert agent._active_turns_map().get(key) is None
    called = {}

    async def fake_impl(message, *a, **k):
        called["msg"] = message
        from core.agent import AgentResponse

        return AgentResponse(text="normal")

    agent._process_impl = fake_impl
    resp = await agent.process("do a thing", "telegram", "u", chat_id="9")
    assert resp.text == "normal" and called["msg"] == "do a thing"
    assert not agent._steer_map().get(key)  # nothing buffered
