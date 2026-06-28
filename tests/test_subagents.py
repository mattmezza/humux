"""Tests for subagents — scope narrowing, registry, the run primitive, and
scheduled-job wiring (issue #15)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.config import Config
from core.job_store import VALID_TYPES, JobStore
from core.llm import LLMResponse, LLMToolCall
from core.personae import Persona
from core.subagents import SubagentRegistry, SubagentRun, narrow_scope, short_summary

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
    return SubagentRun(run_id=run_id, persona="", task="t", status=status)


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
    here = SubagentRun(run_id="a", persona="", task="t", origin_channel="tg", origin_chat_id="1")
    other = SubagentRun(run_id="b", persona="", task="t", origin_channel="tg", origin_chat_id="2")
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
async def test_run_subagent_unknown_persona(agent) -> None:
    result = await agent.run_subagent(task="x", persona_name="does-not-exist")
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
async def test_run_subagent_background_reports_to_origin(agent) -> None:
    channel = AsyncMock()
    agent.channels["telegram"] = channel
    agent.llm = _ScriptedLLM([LLMResponse(text="bg result", tool_calls=[])])

    # The spawning turn already left a user+assistant pair in history.
    await agent.history.add_turn("telegram", "u1", "user", "do it async", "555")
    await agent.history.add_turn("telegram", "u1", "assistant", "started in the background", "555")

    result = await agent.run_subagent(
        task="async work",
        origin_channel="telegram",
        origin_user_id="u1",
        origin_chat_id="555",
        background=True,
    )
    assert result["background"] is True
    assert result["status"] == "running"

    run = agent.subagents.get(result["run_id"])
    await run._task  # let the background loop finish

    assert run.status == "done"
    # Delivered to the chat...
    channel.send.assert_awaited_once()
    sent_chat, sent_text = channel.send.await_args.args
    assert sent_chat == "555"
    assert "bg result" in sent_text
    # ...and folded into the trailing assistant turn (history stays alternating).
    turns = await agent.history.get_messages("telegram", "u1", "555")
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert "started in the background" in str(turns[-1]["content"])
    assert "bg result" in str(turns[-1]["content"])


@pytest.mark.asyncio
async def test_run_subagent_background_respects_concurrency(agent) -> None:
    agent.config.subagents.max_concurrent = 1
    # Pre-fill one running slot.
    agent.subagents.register(SubagentRun(run_id="busy", persona="", task="t"))
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


def test_narrow_persona_intersects_scopes(agent) -> None:
    parent = Persona(name="p", skills=["s1", "s2"], tools=["a", "b"], secrets=["x"])
    requested = Persona(name="child", skills=[], tools=["b", "c"], secrets=["y"])
    child = agent._narrow_persona(requested, {"persona_obj": parent})
    assert child.name == "child"
    assert child.skills == ["s1", "s2"]  # child unspecified → inherits parent
    assert child.tools == ["b"]  # intersection, never 'c'
    assert child.secrets == []  # 'y' not in parent's ['x']


def test_subagent_status_note_lists_only_running_runs(agent) -> None:
    agent.subagents.register(
        SubagentRun(
            run_id="r1",
            persona="coding-helper",
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

    # Once finished it leaves the preamble (its result is in history instead).
    agent.subagents.finish("r1", "done", result="the iPhone 17e is CHF 599")
    assert agent._subagent_status_note("repl", "repl") == ""


@pytest.mark.asyncio
async def test_record_subagent_in_history_merges_into_assistant_turn(agent) -> None:
    run = SubagentRun(
        run_id="r9",
        persona="coding-helper",
        task="t",
        origin_channel="repl",
        origin_user_id="repl",
        origin_chat_id="repl",
    )
    # No prior assistant turn → a fresh assistant turn is added.
    await agent.history.add_turn("repl", "repl", "user", "hello", "repl")
    await agent.history.add_turn("repl", "repl", "assistant", "on it", "repl")
    await agent._record_subagent_in_history(run, "🤖 done: CHF 599")

    turns = await agent.history.get_messages("repl", "repl", "repl")
    assert [t["role"] for t in turns] == ["user", "assistant"]  # merged, not appended
    assert "CHF 599" in str(turns[-1]["content"])


# ---------------------------------------------------------------------------
# Scheduled subagent jobs
# ---------------------------------------------------------------------------


def test_disabled_drops_spawn_subagent_from_llm_tools() -> None:
    from core.agent import TOOLS, apply_feature_gates

    def names(ts):
        return {t["name"] for t in ts}

    base = dict(secrets_available=True, artifacts_enabled=True)
    assert "spawn_subagent" in names(apply_feature_gates(TOOLS, **base, subagents_enabled=True))
    assert "spawn_subagent" not in names(
        apply_feature_gates(TOOLS, **base, subagents_enabled=False)
    )


def test_disabled_hides_spawn_subagent_from_persona_scope() -> None:
    from api.admin import GATEABLE_TOOLS, gateable_tools_for

    assert "spawn_subagent" in gateable_tools_for(True)
    assert set(gateable_tools_for(True)) == set(GATEABLE_TOOLS)
    assert "spawn_subagent" not in gateable_tools_for(True, subagents_enabled=False)


def test_subagent_is_valid_job_type() -> None:
    assert "subagent" in VALID_TYPES


@pytest.mark.asyncio
async def test_job_store_persists_persona(tmp_path) -> None:
    store = JobStore(db_path=str(tmp_path / "jobs.db"))
    job = await store.upsert_job(
        "j1", type="subagent", schedule="cron", cron="0 9 * * *", task="brief", persona="analyst"
    )
    assert job["type"] == "subagent"
    assert job["persona"] == "analyst"
    fetched = await store.get_job("j1")
    assert fetched["persona"] == "analyst"


@pytest.mark.asyncio
async def test_job_store_migrates_persona_column(tmp_path) -> None:
    """A DB created before the persona column gains it on next open."""
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
    job = await store.upsert_job("new", type="subagent", persona="coach")
    assert job["persona"] == "coach"
    legacy = await store.get_job("legacy")
    assert legacy["persona"] == ""  # backfilled default


@pytest.mark.asyncio
async def test_run_subagent_task_delivers_to_owner() -> None:
    from core.scheduler import run_subagent_task, set_agent_context

    channel = AsyncMock()
    agent = SimpleNamespace(
        channels={"telegram": channel},
        run_subagent=AsyncMock(return_value={"ok": True, "result": "scheduled out"}),
        config=SimpleNamespace(
            channels=SimpleNamespace(telegram=SimpleNamespace(allowed_user_ids=[7]))
        ),
        job_store=None,
    )
    set_agent_context(agent)

    await run_subagent_task(persona="analyst", task="weekly review", channel="telegram")

    agent.run_subagent.assert_awaited_once()
    channel.send.assert_awaited_once_with(7, "scheduled out")
