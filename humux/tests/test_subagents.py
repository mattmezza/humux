"""Tests for subagents — scope narrowing, registry, the run primitive, and
scheduled-job wiring (issue #15)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.agents import Agent
from core.config import Config
from core.job_store import VALID_TYPES, JobStore
from core.llm import LLMResponse, LLMToolCall
from core.subagents import (
    FILE_HANDOFF_INSTRUCTION,
    SubagentRegistry,
    SubagentRun,
    narrow_scope,
    normalize_effort,
    resolve_cap,
    short_summary,
)

# ---------------------------------------------------------------------------
# narrow_scope — inherit, never widen ([] / None == "all")
# ---------------------------------------------------------------------------


def test_narrow_scope_parent_unrestricted_takes_child() -> None:
    assert narrow_scope([], ["a"]) == ["a"]
    assert narrow_scope(None, ["a", "b"]) == ["a", "b"]


def test_narrow_scope_child_unspecified_inherits_parent() -> None:
    assert narrow_scope(["a", "b"], []) == ["a", "b"]
    assert narrow_scope(["a"], None) == ["a"]


def test_narrow_scope_both_restricted_is_intersection() -> None:
    # The child can never gain a name the parent lacks.
    assert narrow_scope(["a", "b"], ["b", "c"]) == ["b"]
    assert narrow_scope(["a"], ["b"]) == []


def test_narrow_scope_both_empty_stays_all() -> None:
    assert narrow_scope([], []) == []


# ---------------------------------------------------------------------------
# SubagentRegistry
# ---------------------------------------------------------------------------


def _run(run_id: str, status: str = "running") -> SubagentRun:
    return SubagentRun(run_id=run_id, agent="", task="t", status=status)


def test_registry_register_list_and_active_count() -> None:
    reg = SubagentRegistry()
    reg.register(_run("a"))
    reg.register(_run("b"))
    assert reg.active_count() == 2
    assert {r.run_id for r in reg.list_runs()} == {"a", "b"}
    reg.finish("a", "done", result="ok")
    assert reg.active_count() == 1
    assert {r.run_id for r in reg.list_runs(active_only=True)} == {"b"}
    assert reg.get("a").result == "ok"


def test_registry_cancel_only_running() -> None:
    reg = SubagentRegistry()
    reg.register(_run("a"))
    assert reg.cancel("a") is True
    assert reg.get("a").status == "cancelled"
    # already finished / unknown → False
    assert reg.cancel("a") is False
    assert reg.cancel("missing") is False


def test_registry_trims_finished_runs() -> None:
    reg = SubagentRegistry()
    for i in range(60):
        reg.register(_run(f"r{i}"))
        reg.finish(f"r{i}", "done")
    # Only the most recent finished runs are kept (cap 50).
    assert len(reg.list_runs()) == 50


def test_short_summary_first_nonempty_line_capped() -> None:
    assert short_summary("\n\nhello world\nmore") == "hello world"
    assert short_summary("x" * 400).endswith("…")
    assert len(short_summary("x" * 400)) == 281  # 280 + ellipsis


def test_running_for_filters_by_chat_and_drops_finished() -> None:
    reg = SubagentRegistry()
    here = SubagentRun(run_id="a", agent="", task="t", origin_channel="tg", origin_chat_id="1")
    other = SubagentRun(run_id="b", agent="", task="t", origin_channel="tg", origin_chat_id="2")
    reg.register(here)
    reg.register(other)

    # Running runs appear every turn; the other chat's run is never included.
    assert [r.run_id for r in reg.running_for("tg", "1")] == ["a"]
    assert [r.run_id for r in reg.running_for("tg", "1")] == ["a"]

    # Once finished it drops out of the running list (its result goes to history).
    reg.finish("a", "done", result="answer")
    assert reg.running_for("tg", "1") == []


# ---------------------------------------------------------------------------
# AgentCore.run_subagent — built with a scripted fake LLM (no network)
# ---------------------------------------------------------------------------


class _ScriptedLLM:
    """Returns a fixed sequence of LLMResponses; trivial message builders."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.provider = "deepseek"

    async def generate(self, **_kw) -> LLMResponse:
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(text="(done)", tool_calls=[])

    def assistant_message(self, response: LLMResponse) -> dict:
        return {"role": "assistant", "content": response.text}

    def tool_result_messages(self, results: list[dict]) -> list[dict]:
        return [{"role": "user", "content": results}]


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from core.agent import AgentCore

    cfg = Config()
    cfg.agent.llm_provider = "deepseek"
    cfg.agent.model = "deepseek-v4-flash"
    cfg.memory.embedding.enabled = False  # keep retrieval lexical (no model load)
    return AgentCore(cfg)


@pytest.mark.asyncio
async def test_run_subagent_sync_returns_result(agent) -> None:
    agent.llm = _ScriptedLLM([LLMResponse(text="the answer", tool_calls=[])])
    result = await agent.run_subagent(task="do a thing")
    assert result["ok"] is True
    assert result["result"] == "the answer"
    assert result["summary"] == "the answer"
    run = agent.subagents.get(result["run_id"])
    assert run.status == "done"


@pytest.mark.asyncio
async def test_run_subagent_disabled(agent) -> None:
    agent.config.subagents.enabled = False
    result = await agent.run_subagent(task="x")
    assert "disabled" in result["error"].lower()


@pytest.mark.asyncio
async def test_run_subagent_depth_cap(agent) -> None:
    agent.config.subagents.recursion_depth = 2
    # A caller already at the ceiling cannot spawn.
    result = await agent.run_subagent(task="x", parent_state={"depth": 2})
    assert "recursion depth" in result["error"].lower()


@pytest.mark.asyncio
async def test_run_subagent_unknown_agent(agent) -> None:
    result = await agent.run_subagent(task="x", agent_name="does-not-exist")
    assert "not found" in result["error"].lower()


@pytest.mark.asyncio
async def test_run_subagent_step_budget_stops_loop(agent) -> None:
    agent.config.subagents.max_steps = 1
    # Always asks for a tool → would loop forever without the cap.
    call = LLMToolCall(id="1", name="web_search", arguments={"query": "q"})
    agent.llm = _ScriptedLLM([LLMResponse(text="", tool_calls=[call]) for _ in range(5)])
    result = await agent.run_subagent(task="loop")
    assert result["ok"] is True
    assert "budget" in result["result"].lower()


@pytest.mark.asyncio
async def test_run_subagent_token_budget_stops_loop(agent) -> None:
    agent.config.subagents.token_budget = 100
    agent.config.subagents.max_steps = 100  # ensure the token budget is the limiter
    call = LLMToolCall(id="1", name="web_search", arguments={"query": "q"})
    # Each round reports 80 tokens; cumulative exceeds 100 after the second call.
    agent.llm = _ScriptedLLM(
        [
            LLMResponse(text="", tool_calls=[call], usage={"input_tokens": 80, "output_tokens": 0})
            for _ in range(10)
        ]
    )
    result = await agent.run_subagent(task="loop")
    assert result["ok"] is True
    assert "budget" in result["result"].lower()


@pytest.mark.asyncio
async def test_background_subagent_notifies_user_digests_context(agent, monkeypatch) -> None:
    channel = AsyncMock()
    agent.channels["telegram"] = channel
    agent.llm = _ScriptedLLM([LLMResponse(text="raw verbose findings: CHF 599 ...", tool_calls=[])])

    # Stub the summary inference: (chat notification, context digest).
    async def fake_summary(batch):
        return "Cheapest is CHF 599.", "iPhone 17e 256GB at CHF 599; entry model."

    monkeypatch.setattr(agent, "_summarize_subagent_batch", fake_summary)

    # A trailing assistant turn for the digest to merge into (keeps alternation).
    await agent.history.add_turn("telegram", "u1", "user", "price?", "555")
    await agent.history.add_turn("telegram", "u1", "assistant", "On it.", "555")

    result = await agent.run_subagent(
        task="price check",
        origin_channel="telegram",
        origin_user_id="u1",
        origin_chat_id="555",
        background=True,
    )
    run = agent.subagents.get(result["run_id"])
    await run._task

    assert run.status == "done"
    # Chat: the one-line NOTIFICATION only — never the raw output.
    channel.send.assert_awaited_once_with("555", "Cheapest is CHF 599.")
    # Context: the concise digest is kept (merged), the raw output never is.
    turns = await agent.history.get_messages("telegram", "u1", "555")
    blob = str(turns[-1]["content"])
    assert [t["role"] for t in turns] == ["user", "assistant"]  # still alternating
    assert "iPhone 17e 256GB at CHF 599" in blob
    assert "raw verbose findings" not in blob


@pytest.mark.asyncio
async def test_background_batch_delivers_once_when_all_done(agent, monkeypatch) -> None:
    calls: list[list[str]] = []

    async def fake_deliver(channel, user_id, chat_id, batch):
        calls.append([r.run_id for r in batch])

    monkeypatch.setattr(agent, "_summarize_and_deliver", fake_deliver)

    common = dict(background=True, origin_channel="telegram", origin_chat_id="c")
    r1 = SubagentRun(run_id="s1", agent="", task="a", origin_user_id="u", **common)
    r2 = SubagentRun(run_id="s2", agent="", task="b", origin_user_id="u", **common)
    agent.subagents.register(r1)
    agent.subagents.register(r2)

    # First finishes → the other is still running → barrier holds, no delivery.
    agent.subagents.finish("s1", "done", result="x")
    await agent._maybe_deliver_subagent_batch(r1)
    assert calls == []

    # Last finishes → barrier releases → ONE delivery over the whole batch.
    agent.subagents.finish("s2", "done", result="y")
    await agent._maybe_deliver_subagent_batch(r2)
    assert len(calls) == 1
    assert sorted(calls[0]) == ["s1", "s2"]
    assert r1.synthesized and r2.synthesized


@pytest.mark.asyncio
async def test_cancelling_a_sibling_releases_a_deferred_reply(agent, monkeypatch) -> None:
    """Regression: a done run that deferred to a still-running sibling must not be
    orphaned when the user cancels that sibling (the lost-reply blocker)."""
    import asyncio

    calls: list[list[str]] = []

    async def fake_deliver(channel, user_id, chat_id, batch):
        calls.append(sorted(r.run_id for r in batch))

    monkeypatch.setattr(agent, "_summarize_and_deliver", fake_deliver)

    gate = asyncio.Event()

    async def fake_loop(task, agent, state, run):
        if task == "B-task":
            await gate.wait()  # block so this run is "still running" when A finishes
            return "B done"
        return "A done"

    monkeypatch.setattr(agent, "_run_subagent_loop", fake_loop)

    origin = dict(origin_channel="telegram", origin_user_id="u", origin_chat_id="c")
    b_res = await agent.run_subagent(task="B-task", background=True, **origin)
    a_res = await agent.run_subagent(task="A-task", background=True, **origin)

    # A completes but B is still running → A defers (not delivered, not lost).
    await agent.subagents.get(a_res["run_id"])._task
    assert calls == []
    assert agent.subagents.get(a_res["run_id"]).synthesized is False

    # User cancels B → its cancel path must release A's deferred reply.
    agent.subagents.cancel(b_res["run_id"])
    with pytest.raises(asyncio.CancelledError):
        await agent.subagents.get(b_res["run_id"])._task

    # A's reply was delivered (not lost), and only A — B was cancelled.
    assert calls == [[a_res["run_id"]]]
    assert agent.subagents.get(a_res["run_id"]).synthesized is True


def test_summary_parsing_and_fallback() -> None:
    from core.subagents import _parse_summary, fallback_summary

    n, d = _parse_summary("NOTIFICATION: Cheapest is CHF 599.\nDIGEST: iPhone 17e 256GB CHF 599.")
    assert n == "Cheapest is CHF 599."
    assert "iPhone 17e" in d
    # No markers → first non-empty line becomes the notification.
    assert _parse_summary("Just one line")[0] == "Just one line"
    assert _parse_summary("") == ("", "")
    # Truncation fallback from raw items (no LLM).
    items = [("task a", "line1\nline2", "", "done"), ("task b", "r b", "", "done")]
    notif, digest = fallback_summary(items)
    assert "line1" in notif and "r b" in notif
    assert "- task a:" in digest


@pytest.mark.asyncio
async def test_summarize_batch_calls_llm_and_parses() -> None:
    from core.subagents import summarize_batch

    class FakeLLM:
        async def generate_text(self, *, model, prompt, max_tokens=600):
            return "NOTIFICATION: Done — 3 results.\nDIGEST: A, B and C found."

    notif, digest = await summarize_batch(FakeLLM(), "m", [("t", "r", "", "done")])
    assert notif == "Done — 3 results."
    assert digest == "A, B and C found."


@pytest.mark.asyncio
async def test_run_subagent_background_respects_concurrency(agent) -> None:
    agent.config.subagents.max_concurrent = 1
    # Pre-fill one running slot.
    agent.subagents.register(SubagentRun(run_id="busy", agent="", task="t"))
    result = await agent.run_subagent(task="x", background=True)
    assert "concurrent" in result["error"].lower() or "max" in result["error"].lower()


@pytest.mark.asyncio
async def test_spawn_subagent_not_deduplicated_in_turn(agent) -> None:
    """Two identical spawns in one turn must both run (each is a distinct run)."""
    from core.llm import LLMToolCall

    agent.llm = _ScriptedLLM(
        [LLMResponse(text="a", tool_calls=[]), LLMResponse(text="b", tool_calls=[])]
    )
    state = agent._new_request_state(
        None, origin={"channel": "system", "user_id": "u", "chat_id": ""}
    )
    call = LLMToolCall(id="x", name="spawn_subagent", arguments={"task": "same task"})
    r1 = await agent._execute_tool(call, "system", "u", state)
    r2 = await agent._execute_tool(call, "system", "u", state)
    assert r1.get("ok") is True
    assert r2.get("ok") is True
    assert r1["run_id"] != r2["run_id"]


def test_finish_does_not_overwrite_terminal_state() -> None:
    """A late normal completion cannot un-cancel a run."""
    reg = SubagentRegistry()
    reg.register(_run("a"))
    assert reg.cancel("a") is True
    assert reg.finish("a", "done", result="late") is False  # no-op
    assert reg.get("a").status == "cancelled"
    assert reg.get("a").result == ""


def test_narrow_agent_intersects_scopes(agent) -> None:
    parent = Agent(name="p", skills=["s1", "s2"], tools=["a", "b"], secrets=["x"])
    requested = Agent(name="child", skills=[], tools=["b", "c"], secrets=["y"])
    child = agent._narrow_agent(requested, {"agent_obj": parent})
    assert child.name == "child"
    assert child.skills == ["s1", "s2"]  # child unspecified → inherits parent
    assert child.tools == ["b"]  # intersection, never 'c'
    assert child.secrets == []  # 'y' not in parent's ['x']


def test_subagent_status_note_lists_only_running_runs(agent) -> None:
    agent.subagents.register(
        SubagentRun(
            run_id="r1",
            agent="coding-helper",
            task="t",
            origin_channel="repl",
            origin_chat_id="repl",
            progress="step 2",
        )
    )
    note = agent._subagent_status_note("repl", "repl")
    assert "r1" in note and "running" in note and "step 2" in note
    # Scoped to the chat: a different chat sees nothing.
    assert agent._subagent_status_note("repl", "other-chat") == ""

    # Once finished it leaves the preamble (the agent synthesises a reply instead).
    agent.subagents.finish("r1", "done", result="the iPhone 17e is CHF 599")
    assert agent._subagent_status_note("repl", "repl") == ""


# ---------------------------------------------------------------------------
# Scheduled subagent jobs
# ---------------------------------------------------------------------------


def test_disabled_drops_spawn_subagent_from_llm_tools() -> None:
    from core.agent import TOOLS, apply_feature_gates

    def names(ts):
        return {t["name"] for t in ts}

    base = dict(secrets_available=True)
    assert "spawn_subagent" in names(apply_feature_gates(TOOLS, **base, subagents_enabled=True))
    assert "spawn_subagent" not in names(
        apply_feature_gates(TOOLS, **base, subagents_enabled=False)
    )


def test_disabled_hides_spawn_subagent_from_agent_scope() -> None:
    from api.admin import GATEABLE_TOOLS, gateable_tools_for

    assert "spawn_subagent" in gateable_tools_for()
    assert set(gateable_tools_for()) == set(GATEABLE_TOOLS)
    assert "spawn_subagent" not in gateable_tools_for(subagents_enabled=False)


def test_subagent_is_valid_job_type() -> None:
    assert "subagent" in VALID_TYPES


@pytest.mark.asyncio
async def test_job_store_persists_agent(tmp_path) -> None:
    store = JobStore(db_path=str(tmp_path / "jobs.db"))
    job = await store.upsert_job(
        "j1", type="subagent", schedule="cron", cron="0 9 * * *", task="brief", agent="analyst"
    )
    assert job["type"] == "subagent"
    assert job["agent"] == "analyst"
    fetched = await store.get_job("j1")
    assert fetched["agent"] == "analyst"


@pytest.mark.asyncio
async def test_job_store_migrates_agent_column(tmp_path) -> None:
    """A DB created before the agent column gains it on next open."""
    import sqlite3

    db_path = str(tmp_path / "old.db")
    with sqlite3.connect(db_path) as db:
        db.execute(
            "CREATE TABLE jobs (id TEXT PRIMARY KEY, type TEXT, schedule TEXT, cron TEXT, "
            "run_at TEXT, task TEXT, channel TEXT, status TEXT, created_by TEXT, "
            "description TEXT, created_at TEXT, updated_at TEXT)"
        )
        db.execute("INSERT INTO jobs (id, type) VALUES ('legacy', 'agent')")
        db.commit()

    store = JobStore(db_path=db_path)
    job = await store.upsert_job("new", type="subagent", agent="coach")
    assert job["agent"] == "coach"
    legacy = await store.get_job("legacy")
    assert legacy["agent"] == ""  # backfilled default


@pytest.mark.asyncio
async def test_run_subagent_task_delivers_to_owner() -> None:
    from core.scheduler import run_subagent_task, set_agent_context

    channel = AsyncMock()
    channel.config = SimpleNamespace(allowed_user_ids=[7])
    agent = SimpleNamespace(
        channels={"telegram": channel},
        run_subagent=AsyncMock(return_value={"ok": True, "result": "scheduled out"}),
        config=SimpleNamespace(),
        job_store=None,
    )
    set_agent_context(agent)

    await run_subagent_task(agent_name="analyst", task="weekly review", channel="telegram")

    agent.run_subagent.assert_awaited_once()
    channel.send.assert_awaited_once_with(7, "scheduled out")


# ---------------------------------------------------------------------------
# Caller-sized runs: max_steps / token_budget / thinking_effort + file handoff
# ---------------------------------------------------------------------------


def test_resolve_cap_defaults_clamps_and_coerces() -> None:
    assert resolve_cap(None, 12) == 12  # caller didn't choose → configured ceiling
    assert resolve_cap(3, 12) == 3  # honoured below the ceiling
    assert resolve_cap(999, 12) == 12  # config is a ceiling, never exceeded
    assert resolve_cap(0, 12, floor=1) == 1  # floored
    assert resolve_cap("nope", 12) == 12  # garbage degrades to the ceiling
    assert resolve_cap("4", 12) == 4  # numeric string coerces
    assert resolve_cap(float("inf"), 12) == 12  # int(inf) overflows → ceiling, no crash


def test_normalize_effort_maps_and_inherits() -> None:
    assert normalize_effort(None) is None  # inherit the caller's level
    assert normalize_effort("") is None
    assert normalize_effort("off") == ""  # reasoning off
    assert normalize_effort("HIGH") == "high"  # case-insensitive
    assert normalize_effort("medium") == "medium"
    assert normalize_effort("bogus") is None  # unknown → safe inherit default


class _RecordingLLM(_ScriptedLLM):
    """A scripted LLM that also records the system prompt of each call."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(responses)
        self.systems: list[str] = []

    async def generate(self, *, system: str = "", **_kw) -> LLMResponse:
        self.systems.append(system)
        return await super().generate()


@pytest.mark.asyncio
async def test_subagent_system_prompt_carries_file_handoff(agent) -> None:
    rec = _RecordingLLM([LLMResponse(text="done", tool_calls=[])])
    agent.llm = rec
    await agent.run_subagent(task="x")
    assert any(FILE_HANDOFF_INSTRUCTION in s for s in rec.systems)


@pytest.mark.asyncio
async def test_run_subagent_caller_max_steps_is_the_limiter(agent) -> None:
    agent.config.subagents.max_steps = 100  # high config ceiling …
    call = LLMToolCall(id="1", name="web_search", arguments={"query": "q"})
    agent.llm = _ScriptedLLM([LLMResponse(text="", tool_calls=[call]) for _ in range(10)])
    result = await agent.run_subagent(task="loop", max_steps=1)  # … caller wants just 1
    assert result["ok"] is True
    assert "budget" in result["result"].lower()
    assert agent.subagents.get(result["run_id"]).max_steps == 1


@pytest.mark.asyncio
async def test_run_subagent_clamps_caps_to_config_ceiling(agent) -> None:
    agent.config.subagents.max_steps = 5
    agent.config.subagents.token_budget = 9000
    agent.llm = _ScriptedLLM([LLMResponse(text="done", tool_calls=[])])
    result = await agent.run_subagent(task="x", max_steps=999, token_budget=10**9)
    run = agent.subagents.get(result["run_id"])
    assert run.max_steps == 5
    assert run.token_budget == 9000


@pytest.mark.asyncio
async def test_run_subagent_effort_inherits_by_default(agent, monkeypatch) -> None:
    agent.llm = _ScriptedLLM([LLMResponse(text="ok", tool_calls=[])])
    monkeypatch.setattr(
        agent, "_background_llm", lambda *a, **k: pytest.fail("should not clone when inheriting")
    )
    result = await agent.run_subagent(task="x")  # no thinking_effort
    assert result["ok"] is True
    assert agent.subagents.get(result["run_id"]).effort is None


@pytest.mark.asyncio
async def test_run_subagent_effort_uses_scoped_client(agent, monkeypatch) -> None:
    agent.llm = _ScriptedLLM([LLMResponse(text="ok", tool_calls=[])])
    captured: dict = {}

    def spy(provider, level=""):
        captured["level"] = level
        return _ScriptedLLM([LLMResponse(text="ok", tool_calls=[])])

    monkeypatch.setattr(agent, "_background_llm", spy)
    result = await agent.run_subagent(task="x", thinking_effort="high")
    assert result["ok"] is True
    assert captured["level"] == "high"
    assert agent.subagents.get(result["run_id"]).effort == "high"


# ---------------------------------------------------------------------------
# Agent roster — let the agent pick a specialist, selection stays user-led
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agents_roster_lists_name_and_role(agent, monkeypatch) -> None:
    agents = [
        Agent(name="coding-helper", role="Writes and reviews code"),
        Agent(name="writing-editor", role="Edits prose\nsecond line ignored"),
    ]
    monkeypatch.setattr(agent.agents, "list_agents", AsyncMock(return_value=agents))
    block = await agent._agents_roster_block(None)
    assert "<agents>" in block
    assert "- coding-helper — Writes and reviews code" in block
    assert "- writing-editor — Edits prose" in block  # only the first role line
    assert "second line ignored" not in block


@pytest.mark.asyncio
async def test_agents_roster_marks_current_and_gates(agent, monkeypatch) -> None:
    agents = [Agent(name="me", role="r1"), Agent(name="other", role="r2")]
    monkeypatch.setattr(agent.agents, "list_agents", AsyncMock(return_value=agents))
    block = await agent._agents_roster_block(Agent(name="me", role="r1"))
    assert "- me (you) — r1" in block
    # an agent whose tool scope excludes spawn_subagent gets no roster
    scoped = Agent(name="me", tools=["web_search"])
    assert await agent._agents_roster_block(scoped) == ""
    # nor when subagents are disabled
    agent.config.subagents.enabled = False
    assert await agent._agents_roster_block(None) == ""


@pytest.mark.asyncio
async def test_agents_roster_only_offered_on_main_turn(agent, monkeypatch) -> None:
    monkeypatch.setattr(
        agent.agents,
        "list_agents",
        AsyncMock(return_value=[Agent(name="coding-helper", role="code")]),
    )
    # subagent preamble (offer_agents defaults False) → no roster leaks in
    assert "<agents>" not in await agent._turn_preamble(None, query="x")
    # main turn opts in
    assert "<agents>" in await agent._turn_preamble(None, query="x", offer_agents=True)


@pytest.mark.asyncio
async def test_run_subagent_unknown_agent_lists_available(agent, monkeypatch) -> None:
    monkeypatch.setattr(
        agent.agents,
        "list_agents",
        AsyncMock(return_value=[Agent(name="coding-helper"), Agent(name="analyst")]),
    )
    result = await agent.run_subagent(task="x", agent_name="nope")
    assert "not found" in result["error"].lower()
    assert "coding-helper" in result["error"] and "analyst" in result["error"]
