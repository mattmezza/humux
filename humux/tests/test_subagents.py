"""Tests for subagents — scope narrowing, registry, the run primitive, and
scheduled-job wiring (issue #15)."""

from __future__ import annotations

import asyncio
import time
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
    derive_label,
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
        self.thinking_level = ""

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
    cfg.history.mode = "injection"  # exercises the windowed add_turn/get_messages store
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
    """Background refuses (never queues) when the *chat* is at its slot cap —
    the gate counts runs actually executing, not merely registered."""
    agent.config.subagents.max_concurrent = 1
    chat_key = agent._subagent_chat_key("telegram", "c1")
    async with agent.subagents.slot(chat_key, lambda: 1):
        assert agent.subagents.slots_in_use(chat_key) == 1
        result = await agent.run_subagent(
            task="x",
            background=True,
            origin_channel="telegram",
            origin_chat_id="c1",
        )
    assert "error" in result
    assert "max 1" in result["error"].lower()
    # Another chat is unaffected by this chat's cap.
    assert agent.subagents.slots_in_use(agent._subagent_chat_key("telegram", "other")) == 0


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
            origin_channel="cli",
            origin_chat_id="cli:s1",
            progress="step 2",
        )
    )
    note = agent._subagent_status_note("cli", "cli:s1")
    assert "r1" in note and "running" in note and "step 2" in note
    # Scoped to the chat: a different chat sees nothing.
    assert agent._subagent_status_note("cli", "other-chat") == ""

    # Once finished it leaves the preamble (the agent synthesises a reply instead).
    agent.subagents.finish("r1", "done", result="the iPhone 17e is CHF 599")
    assert agent._subagent_status_note("cli", "cli:s1") == ""


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
    # The effort-scoped client is a clone: the run generates with the overridden
    # thinking level while the main client stays untouched.
    captured: dict = {}

    class _Recording(_ScriptedLLM):
        thinking_level = ""

        async def generate(self, **kw) -> LLMResponse:
            captured["level"] = self.thinking_level
            return await super().generate(**kw)

    agent.llm = _Recording([LLMResponse(text="ok", tool_calls=[])])
    result = await agent.run_subagent(task="x", thinking_effort="high")
    assert result["ok"] is True
    assert captured["level"] == "high"
    assert agent.llm.thinking_level == ""  # main client untouched
    assert agent.subagents.get(result["run_id"]).effort == "high"


# ---------------------------------------------------------------------------
# Per-spawn model/provider override, gated by subagents.allowed_models (#299)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_subagent_model_override(agent, monkeypatch) -> None:
    agent.config.subagents.allowed_models = ["anthropic:claude-sonnet-4-5"]
    agent.llm = _ScriptedLLM([LLMResponse(text="ok", tool_calls=[])])
    built: dict = {}

    def _fake_background_llm(provider, thinking_level=""):
        built["provider"] = provider
        return agent.llm

    monkeypatch.setattr(agent, "_background_llm", _fake_background_llm)

    # Allowed: the model alone resolves its provider off the allowlist.
    result = await agent.run_subagent(task="x", model="claude-sonnet-4-5")
    assert result["ok"] is True
    assert built["provider"] == "anthropic"
    run = agent.subagents.get(result["run_id"])
    assert (run.provider, run.model) == ("anthropic", "claude-sonnet-4-5")

    # Not on the list: refused with a readable error, no run started.
    built.clear()
    refused = await agent.run_subagent(task="x", model="gpt-9")
    assert "not allowed" in refused["error"]
    assert "anthropic:claude-sonnet-4-5" in refused["error"]
    assert built == {}

    # Omitted: inherits the caller's client exactly as before (#299 back-compat).
    result = await agent.run_subagent(task="x")
    assert result["ok"] is True
    assert built == {}
    run = agent.subagents.get(result["run_id"])
    assert (run.provider, run.model) == ("", "")


@pytest.mark.asyncio
async def test_run_subagent_model_override_needs_an_allowlist(agent) -> None:
    agent.llm = _ScriptedLLM([LLMResponse(text="ok", tool_calls=[])])
    refused = await agent.run_subagent(task="x", model="anything")  # allowed_models empty
    assert "not enabled" in refused["error"]


def test_spawn_subagent_description_names_allowed_models(agent) -> None:
    agent.config.subagents.allowed_models = ["deepseek:deepseek-v4-flash"]
    spawn = next(t for t in agent._tools_for_turn(None) if t["name"] == "spawn_subagent")
    assert "deepseek:deepseek-v4-flash" in spawn["description"]
    # The shared module-level schema is untouched by the per-turn copy.
    agent.config.subagents.allowed_models = []
    spawn = next(t for t in agent._tools_for_turn(None) if t["name"] == "spawn_subagent")
    assert "deepseek:deepseek-v4-flash" not in spawn["description"]


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


# ---------------------------------------------------------------------------
# Fan-out — spawn_subagents: concurrency, per-chat cap, budgets, abort
# ---------------------------------------------------------------------------


def _fanout_state(agent, channel: str = "cli", user: str = "u", chat: str = "c1", depth: int = 0):
    """A request_state shaped like a live turn's, so the fan-out finds its origin."""
    return agent._new_request_state(
        None,
        depth=depth,
        origin={"channel": channel, "user_id": user, "chat_id": chat},
    )


def _install_overlap_loop(agent, monkeypatch, delay: float = 0.02, per_task: dict | None = None):
    """Replace the subagent's inner loop with a sleeper that records peak overlap.

    Patched *inside* the per-chat slot gate (``_run_subagent_loop`` still runs),
    so the peak concurrent count is exactly what the gate allowed.
    """
    tracker = {"active": 0, "peak": 0, "finished": []}

    async def fake_inner(task, child_agent, child_state, run):
        tracker["active"] += 1
        tracker["peak"] = max(tracker["peak"], tracker["active"])
        await asyncio.sleep((per_task or {}).get(task, delay))
        tracker["active"] -= 1
        tracker["finished"].append(run.label)
        run.stopped_reason = "complete"
        return f"result:{task}"

    monkeypatch.setattr(agent, "_run_subagent_loop_inner", fake_inner)
    return tracker


@pytest.mark.asyncio
async def test_fanout_runs_children_concurrently(agent, monkeypatch) -> None:
    """Wall clock ≈ one child, not the sum — and all four overlap."""
    agent.config.subagents.max_concurrent = 4
    tracker = _install_overlap_loop(agent, monkeypatch, delay=0.05)
    tasks = [{"task": f"t{i}", "label": f"l{i}"} for i in range(4)]

    started = time.monotonic()
    out = await agent._tool_spawn_subagents({"tasks": tasks}, "cli", "u", _fanout_state(agent))
    elapsed = time.monotonic() - started

    assert out["ok"] is True
    assert out["succeeded"] == 4 and out["failed"] == 0
    assert tracker["peak"] == 4  # genuinely parallel, not a serial loop
    assert elapsed < 0.05 * 3  # ≈ one child; the serial cost would be 4×


@pytest.mark.asyncio
async def test_fanout_queues_at_the_per_chat_cap(agent, monkeypatch) -> None:
    """Over the cap the excess queues (never refuses): peak == max_concurrent,
    every subtask still completes."""
    agent.config.subagents.max_concurrent = 2
    tracker = _install_overlap_loop(agent, monkeypatch, delay=0.02)
    tasks = [{"task": f"t{i}", "label": f"l{i}"} for i in range(4)]

    out = await agent._tool_spawn_subagents({"tasks": tasks}, "cli", "u", _fanout_state(agent))

    assert tracker["peak"] == 2
    assert out["succeeded"] == 4
    assert {r["status"] for r in out["results"]} == {"done"}


@pytest.mark.asyncio
async def test_slots_are_per_chat_not_global(agent, monkeypatch) -> None:
    """A chat sitting at its cap blocks only itself — another chat runs at once."""
    agent.config.subagents.max_concurrent = 1
    _install_overlap_loop(agent, monkeypatch, delay=0.0)
    key_a = agent._subagent_chat_key("cli", "A")

    async with agent.subagents.slot(key_a, lambda: 1):
        # Chat B is unaffected by chat A's full pool.
        res = await asyncio.wait_for(
            agent.run_subagent(task="x", origin_channel="cli", origin_chat_id="B"), timeout=2
        )
        assert res["ok"] is True
        # Chat A itself queues behind the held slot instead of running.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                agent.run_subagent(task="y", origin_channel="cli", origin_chat_id="A"),
                timeout=0.05,
            )


@pytest.mark.asyncio
async def test_fanout_results_are_ordered_by_index(agent, monkeypatch) -> None:
    """Children finishing out of order still come back in request order."""
    agent.config.subagents.max_concurrent = 4
    per_task = {"t0": 0.04, "t1": 0.02, "t2": 0.0}
    tracker = _install_overlap_loop(agent, monkeypatch, per_task=per_task)
    tasks = [{"task": f"t{i}", "label": f"l{i}"} for i in range(3)]

    out = await agent._tool_spawn_subagents({"tasks": tasks}, "cli", "u", _fanout_state(agent))

    assert tracker["finished"] == ["l2", "l1", "l0"]  # completion order is reversed
    assert [r["index"] for r in out["results"]] == [0, 1, 2]
    assert [r["label"] for r in out["results"]] == ["l0", "l1", "l2"]
    assert [r["result"] for r in out["results"]] == ["result:t0", "result:t1", "result:t2"]


@pytest.mark.asyncio
async def test_fanout_labels_derive_from_the_task_when_unset(agent, monkeypatch) -> None:
    _install_overlap_loop(agent, monkeypatch, delay=0.0)
    tasks = [
        {"task": "check the swiss iPhone price today"},
        {"task": "summarise the release notes", "label": "notes"},
    ]

    out = await agent._tool_spawn_subagents({"tasks": tasks}, "cli", "u", _fanout_state(agent))

    assert [r["label"] for r in out["results"]] == ["check the swiss iPhone", "notes"]


@pytest.mark.asyncio
async def test_fanout_reports_partial_failure_per_index(agent, monkeypatch) -> None:
    """A bad subtask fails its own index; the siblings still run and the
    aggregate carries a note so the model can't silently drop it."""
    _install_overlap_loop(agent, monkeypatch, delay=0.0)
    tasks = [
        {"task": "good one", "label": "a"},
        {"task": "bad one", "label": "b", "agent": "does-not-exist"},
        {"task": "good two", "label": "c"},
    ]

    out = await agent._tool_spawn_subagents({"tasks": tasks}, "cli", "u", _fanout_state(agent))

    assert out["ok"] is False
    assert out["succeeded"] == 2 and out["failed"] == 1
    statuses = [r["status"] for r in out["results"]]
    assert statuses == ["done", "error", "done"]
    assert "Agent not found" in out["results"][1]["error"]
    assert "note" in out and "did not complete" in out["note"]


@pytest.mark.asyncio
async def test_fanout_refuses_beyond_max_fanout(agent) -> None:
    agent.config.subagents.max_fanout = 2
    out = await agent._tool_spawn_subagents(
        {"tasks": [{"task": f"t{i}"} for i in range(3)]}, "cli", "u", _fanout_state(agent)
    )
    assert "max_fanout" in out["error"]
    assert "2" in out["error"]


@pytest.mark.asyncio
async def test_nested_fanout_is_refused(agent) -> None:
    """A subagent (depth ≥ 1) may delegate singly but never fan out again."""
    out = await agent._tool_spawn_subagents(
        {"tasks": [{"task": "a"}, {"task": "b"}]},
        "system",
        "u",
        _fanout_state(agent, depth=1),
    )
    assert "Nested fan-out" in out["error"]


@pytest.mark.asyncio
async def test_subagent_loop_is_not_offered_the_fanout_tool(agent) -> None:
    """Belt to the handler's braces: the plural tool is stripped from the schemas
    a subagent's loop sends, while the singular stays below the depth ceiling."""

    class _ToolCapturingLLM(_ScriptedLLM):
        def __init__(self, responses):
            super().__init__(responses)
            self.offered: list[set[str]] = []

        async def generate(self, *, tools=(), **kw) -> LLMResponse:
            self.offered.append({t["name"] for t in tools})
            return await super().generate()

    agent.llm = _ToolCapturingLLM([LLMResponse(text="ok", tool_calls=[])])
    await agent.run_subagent(task="x")

    offered = agent.llm.offered[0]
    assert "spawn_subagents" not in offered
    assert "spawn_subagent" in offered  # depth 1 < recursion_depth, so still allowed
    assert "generate_image" not in offered


# ---------------------------------------------------------------------------
# FanoutBudget — one reserve-then-refund pool per turn, shared by the whole tree
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fanout_token_pool_starts_only_what_fits(agent, monkeypatch) -> None:
    agent.config.subagents.turn_token_budget = 2500
    _install_overlap_loop(agent, monkeypatch, delay=0.0)
    tasks = [{"task": f"t{i}", "label": f"l{i}", "token_budget": 1000} for i in range(3)]

    out = await agent._tool_spawn_subagents({"tasks": tasks}, "cli", "u", _fanout_state(agent))

    assert out["succeeded"] == 2 and out["failed"] == 1
    refused = out["results"][2]
    assert refused["status"] == "error"
    assert "tokens" in refused["error"] and "left this turn" in refused["error"]


@pytest.mark.asyncio
async def test_fanout_spawn_pool_is_shared_across_spawns_in_one_turn(agent, monkeypatch) -> None:
    """One pool per turn, carried on request_state — a later spawn draws from
    what the earlier one already took."""
    agent.config.subagents.max_spawns_per_turn = 2
    _install_overlap_loop(agent, monkeypatch, delay=0.0)
    state = _fanout_state(agent)

    first = await agent._tool_spawn_subagents(
        {"tasks": [{"task": "a"}, {"task": "b"}]}, "cli", "u", state
    )
    assert first["succeeded"] == 2
    assert state["fanout_budget"].spawns_left == 0

    second = await agent._tool_spawn_subagents(
        {"tasks": [{"task": "c"}, {"task": "d"}]}, "cli", "u", state
    )
    assert second["ok"] is False
    assert second["failed"] == 2
    assert all("Spawn budget" in r["error"] for r in second["results"])
    assert "No subtask could be started" in second["note"]


@pytest.mark.asyncio
async def test_completed_run_refunds_its_unspent_reservation(agent) -> None:
    """Reserve the full budget up front, hand back what the run didn't spend."""
    agent.config.subagents.turn_token_budget = 50_000
    agent.llm = _ScriptedLLM(
        [LLMResponse(text="done", tool_calls=[], usage={"input_tokens": 30, "output_tokens": 20})]
    )
    state = _fanout_state(agent)

    res = await agent.run_subagent(task="x", token_budget=5000, parent_state=state)

    assert res["ok"] is True
    budget = state["fanout_budget"]
    assert budget.tokens_left == 50_000 - 50  # 5000 reserved, 4950 refunded
    assert budget.spawns_left == agent.config.subagents.max_spawns_per_turn - 1


# ---------------------------------------------------------------------------
# Run telemetry: steps / tokens_used / stopped_reason
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_reports_steps_tokens_and_stopped_reason(agent) -> None:
    agent.config.subagents.max_steps = 2
    call = LLMToolCall(id="1", name="web_search", arguments={"query": "q"})
    agent.llm = _ScriptedLLM(
        [
            LLMResponse(text="", tool_calls=[call], usage={"input_tokens": 40, "output_tokens": 10})
            for _ in range(6)
        ]
    )

    res = await agent.run_subagent(task="loop")

    run = agent.subagents.get(res["run_id"])
    assert run.stopped_reason == "max_steps" == res["stopped_reason"]
    assert res["steps"] == run.steps == 2
    assert res["tokens_used"] == run.tokens_used == 150  # 3 rounds × 50


@pytest.mark.asyncio
async def test_sync_result_carries_telemetry_on_a_clean_run(agent) -> None:
    agent.llm = _ScriptedLLM(
        [
            LLMResponse(
                text="the answer", tool_calls=[], usage={"input_tokens": 7, "output_tokens": 3}
            )
        ]
    )
    res = await agent.run_subagent(task="x")
    assert res["stopped_reason"] == "complete"
    assert res["steps"] == 0
    assert res["tokens_used"] == 10


# ---------------------------------------------------------------------------
# Cancellation — /stop mid-fan-out, and cascade to a run's children
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_abort_mid_fanout_cancels_every_child(agent, monkeypatch) -> None:
    abort = asyncio.Event()
    agent._active_turns_map()[("cli", "u", "c1")] = abort

    async def fake_inner(task, child_agent, child_state, run):
        abort.set()  # the user hits /stop once the fan-out is under way
        await asyncio.sleep(5)
        raise AssertionError("child should have been cancelled")

    monkeypatch.setattr(agent, "_run_subagent_loop_inner", fake_inner)

    out = await agent._tool_spawn_subagents(
        {"tasks": [{"task": "a", "label": "a"}, {"task": "b", "label": "b"}]},
        "cli",
        "u",
        _fanout_state(agent),
    )

    assert [r["status"] for r in out["results"]] == ["cancelled", "cancelled"]
    assert out["ok"] is False and out["succeeded"] == 0


def test_cancel_cascades_to_child_runs() -> None:
    """A create_task child outlives its parent's cancellation, so cancel walks
    the tree instead of orphaning the fan-out."""
    reg = SubagentRegistry()
    reg.register(SubagentRun(run_id="p", agent="", task="t", children=["c1", "c2"]))
    reg.register(_run("c1"))
    reg.register(_run("c2"))

    assert reg.cancel("p") is True
    assert [reg.get(rid).status for rid in ("p", "c1", "c2")] == ["cancelled"] * 3


# ---------------------------------------------------------------------------
# Tool exposure + labels
# ---------------------------------------------------------------------------


def test_legacy_spawn_subagent_allowlist_also_gets_the_plural(agent) -> None:
    """Agents scoped before the plural tool existed may still fan out."""
    scoped = Agent(name="legacy", tools=["spawn_subagent"])
    names = {t["name"] for t in agent._tools_for_turn(scoped)}
    assert {"spawn_subagent", "spawn_subagents"} <= names
    # An agent scoped away from spawning gets neither.
    assert "spawn_subagents" not in {
        t["name"] for t in agent._tools_for_turn(Agent(name="x", tools=["web_search"]))
    }


def test_derive_label_prefers_the_model_label_then_the_task() -> None:
    assert derive_label("pricing", "look up the price") == "pricing"
    assert derive_label("  spaced  out ", "task") == "spaced out"
    assert derive_label("", "look up the swiss price of an iPhone") == "look up the swiss"
    assert derive_label("", "", 2) == "task2"
    assert len(derive_label("z" * 99, "")) == 32
