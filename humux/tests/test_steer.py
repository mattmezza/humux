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


def test_drain_is_scoped_and_defers_consume(agent) -> None:
    a = ("telegram", "u", "1")
    b = ("telegram", "u", "2")
    agent._steer_map().setdefault(a, []).append({"text": "hi", "consumed": False})
    assert agent._drain_steer(*b) == []  # other chat sees nothing
    got = agent._drain_steer(*a)
    assert [e["text"] for e in got] == ["hi"]
    # Draining does NOT consume — that waits until the steer has reached the model,
    # so a failed generate() leaves it to run as its own turn (#145).
    assert got[0]["consumed"] is False
    assert agent._drain_steer(*a) == []  # popped, not re-served


def test_steer_message_formats(agent) -> None:
    msg = agent._steer_message([{"text": "use the calendar API instead", "message_id": 7}])
    assert msg["role"] == "user"
    assert "<steering_message>" in msg["content"]
    assert "use the calendar API instead" in msg["content"]


@pytest.mark.asyncio
async def test_commit_marks_and_acks(agent) -> None:
    agent.channels = {"telegram": _FakeChannel()}
    entries = [{"text": "x", "message_id": 7, "consumed": False}]
    await agent._commit_steer("telegram", "1", entries)
    assert entries[0]["consumed"] is True  # committed only after the model saw it
    assert agent.channels["telegram"].reactions == [("1", 7, "👀")]


# --- integration: injection into a running turn ----------------------------


async def _wait_active(agent, key, task):
    for _ in range(400):
        await asyncio.sleep(0.001)
        if key in agent._active_turns_map():
            return
    task.cancel()
    pytest.fail("turn never registered an abort Event")


@pytest.mark.asyncio
async def test_steer_injected_into_running_turn(agent) -> None:
    agent.llm = _RecordingLLM(rounds=3)
    agent.channels = {"telegram": _FakeChannel()}

    async def fake_tool(call, channel, user_id, request_state):
        await asyncio.sleep(0.002)  # pace rounds so the steer lands mid-loop
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

    resp = await asyncio.wait_for(turn, timeout=1)
    assert resp.text == "done"
    # The steering text reached at least one LLM call, wrapped for the model.
    flat = "".join(str(m.get("content")) for seen in agent.llm.seen for m in seen)
    assert "<steering_message>" in flat
    assert "use the calendar API" in flat
    # And the follow-up got a 👀 ack.
    assert (("55", 2, "👀")) in agent.channels["telegram"].reactions


class _FailOnSteeredRoundLLM:
    """call 1 → a tool call; call 2 (the steered round) → raises; later calls answer.

    Reproduces the failure the review found: the generate() carrying the steer errors."""

    provider = "deepseek"

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, *, messages, **_kw) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                text="",
                tool_calls=[LLMToolCall(id="c1", name="web_search", arguments={"q": "x"})],
            )
        if self.calls == 2:
            raise RuntimeError("provider 5xx on the steered round")
        return LLMResponse(text="recovered", tool_calls=[])

    def assistant_message(self, response: LLMResponse) -> dict:
        return {"role": "assistant", "content": response.text}

    def tool_result_messages(self, results: list[dict]) -> list[dict]:
        return [{"role": "user", "content": results}]


@pytest.mark.asyncio
async def test_steer_not_lost_when_carrying_generate_fails(agent) -> None:
    # Injection mode: no sticky-session safety net, so the unconsumed-on-failure
    # fallback is the only thing keeping the steer alive.
    agent.history_mode = "injection"
    agent.llm = _FailOnSteeredRoundLLM()
    agent.channels = {"telegram": _FakeChannel()}

    async def fake_tool(call, channel, user_id, request_state):
        await asyncio.sleep(0.003)  # window for the steer to land before the failing round
        return {"ok": True}

    agent._execute_tool = fake_tool
    key = ("telegram", "u", "77")

    turn = asyncio.create_task(
        agent.process("start work", "telegram", "u", chat_id="77", message_id=1)
    )
    await _wait_active(agent, key, turn)

    steer_task = asyncio.create_task(
        agent.process("actually, do it differently", "telegram", "u", chat_id="77", message_id=2)
    )

    with pytest.raises(RuntimeError):
        await asyncio.wait_for(turn, timeout=1)  # the failing round propagates

    # The steer was never confirmed to the model, so it runs as its own turn
    # instead of being silently dropped.
    steer_resp = await asyncio.wait_for(steer_task, timeout=1)
    assert steer_resp.text == "recovered"
    # No false ack: 👀 fires only on a committed steer, and this one never committed.
    assert agent.channels["telegram"].reactions == []


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


@pytest.mark.asyncio
async def test_webhook_event_steers_running_turn_when_steerable(agent) -> None:
    # #266: a GitHub event landing on a thread whose turn is still running is
    # injected into THAT turn (steerable=True), not queued as a duplicate.
    agent.llm = _RecordingLLM(rounds=3)
    agent.channels = {}

    async def fake_tool(call, channel, user_id, request_state):
        await asyncio.sleep(0.002)
        return {"ok": True}

    agent._execute_tool = fake_tool
    chat = "github:dev:acme/widgets#7"
    key = ("system", "github", chat)
    turn = asyncio.create_task(
        agent.process("[GitHub] issue opened: build it", "system", "github", chat_id=chat)
    )
    await _wait_active(agent, key, turn)

    steer = await agent.process(
        "[GitHub] new comment: also cover the empty case",
        "system",
        "github",
        chat_id=chat,
        steerable=True,
    )
    assert steer.text == ""  # consumed by the running turn

    resp = await asyncio.wait_for(turn, timeout=1)
    assert resp.text == "done"
    flat = "".join(str(m.get("content")) for seen in agent.llm.seen for m in seen)
    assert "<steering_message>" in flat
    assert "also cover the empty case" in flat


@pytest.mark.asyncio
async def test_system_without_steerable_queues_not_steers(agent) -> None:
    # The scheduler and other system callers keep the old behaviour: a second
    # message for a busy chat waits for the lock and runs as its own turn.
    agent.llm = _RecordingLLM(rounds=2)
    agent.channels = {}

    async def fake_tool(call, channel, user_id, request_state):
        await asyncio.sleep(0.002)
        return {"ok": True}

    agent._execute_tool = fake_tool
    key = ("system", "scheduler", "job:1")
    turn = asyncio.create_task(agent.process("run job", "system", "scheduler", chat_id="job:1"))
    await _wait_active(agent, key, turn)

    follow = asyncio.create_task(
        agent.process("second event", "system", "scheduler", chat_id="job:1")
    )
    assert (await asyncio.wait_for(turn, timeout=1)).text == "done"
    assert (await asyncio.wait_for(follow, timeout=1)).text == "done"  # own turn
    flat = "".join(str(m.get("content")) for seen in agent.llm.seen for m in seen)
    assert "<steering_message>" not in flat
