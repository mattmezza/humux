"""Tests for task reflection module."""

from __future__ import annotations

import json

import aiosqlite
import pytest

from core.task_reflection import ReflectionStore, _extract_json_object


@pytest.fixture
async def store(tmp_path):
    db_path = str(tmp_path / "reflections.db")
    reflection = ReflectionStore(db_path=db_path, max_reflections=50)
    await reflection._ensure_schema()
    return reflection


class _LLMStub:
    """Minimal LLM stub that returns a canned response."""

    def __init__(self, response: str):
        self._response = response
        self.call_count = 0

    async def generate_text(self, *, model: str, prompt: str, max_tokens: int = 1024) -> str:
        self.call_count += 1
        return self._response


async def _count_rows(db_path: str) -> int:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM reflections")
        row = await cursor.fetchone()
        return row[0] if row else 0


# -- reflect_on_task tests --


@pytest.mark.asyncio
async def test_reflect_stores_lesson(store) -> None:
    response = json.dumps(
        {
            "outcome": "partial",
            "task_summary": "Tried to check weather",
            "lesson": "Weather API timed out; should retry with a different provider",
            "tool_issues": [
                {
                    "tool": "run_command",
                    "issue": "API timeout",
                    "suggestion": "Use backup weather API",
                }
            ],
            "category": "api",
        }
    )
    llm = _LLMStub(response)
    tool_log = [
        {
            "name": "run_command",
            "args": {"command": "curl weather.api"},
            "result": {"error": "timeout"},
        },
    ]

    stored = await store.reflect_on_task(
        llm=llm,
        model="test-model",
        user_msg="What's the weather?",
        agent_msg="Sorry, the weather API timed out.",
        tool_log=tool_log,
    )

    assert stored is True
    assert await _count_rows(store.db_path) == 1


@pytest.mark.asyncio
async def test_reflect_skips_empty_lesson(store) -> None:
    response = json.dumps(
        {
            "outcome": "success",
            "task_summary": "Sent an email",
            "lesson": "",
            "tool_issues": [],
            "category": "general",
        }
    )
    llm = _LLMStub(response)
    tool_log = [
        {"name": "send_email", "args": {}, "result": {"ok": True}},
    ]

    stored = await store.reflect_on_task(
        llm=llm,
        model="test-model",
        user_msg="Send email to Marco",
        agent_msg="Email sent.",
        tool_log=tool_log,
    )

    assert stored is False
    assert await _count_rows(store.db_path) == 0


@pytest.mark.asyncio
async def test_reflect_skips_when_no_tools_used(store) -> None:
    llm = _LLMStub("should not be called")
    stored = await store.reflect_on_task(
        llm=llm,
        model="test-model",
        user_msg="Hello",
        agent_msg="Hi!",
        tool_log=[],
    )
    assert stored is False
    assert llm.call_count == 0


@pytest.mark.asyncio
async def test_reflect_handles_llm_error(store) -> None:
    class _ErrorLLM:
        async def generate_text(self, **kwargs) -> str:
            raise RuntimeError("API error")

    tool_log = [{"name": "web_search", "args": {}, "result": {"results": []}}]
    stored = await store.reflect_on_task(
        llm=_ErrorLLM(),
        model="test-model",
        user_msg="Search something",
        agent_msg="Here are the results",
        tool_log=tool_log,
    )
    assert stored is False


@pytest.mark.asyncio
async def test_reflect_handles_invalid_json(store) -> None:
    llm = _LLMStub("This is not valid JSON")
    tool_log = [{"name": "run_command", "args": {}, "result": {"ok": True}}]
    stored = await store.reflect_on_task(
        llm=llm,
        model="test-model",
        user_msg="Do something",
        agent_msg="Done",
        tool_log=tool_log,
    )
    assert stored is False


@pytest.mark.asyncio
async def test_reflect_deduplicates_lessons(store) -> None:
    response = json.dumps(
        {
            "outcome": "failure",
            "task_summary": "Weather check failed",
            "lesson": "Weather API is unreliable",
            "tool_issues": [],
            "category": "api",
        }
    )
    llm = _LLMStub(response)
    tool_log = [{"name": "run_command", "args": {}, "result": {"error": "timeout"}}]

    stored1 = await store.reflect_on_task(
        llm=llm, model="m", user_msg="Weather?", agent_msg="Failed", tool_log=tool_log
    )
    assert stored1 is True

    stored2 = await store.reflect_on_task(
        llm=llm, model="m", user_msg="Weather again?", agent_msg="Failed again", tool_log=tool_log
    )
    assert stored2 is False  # Duplicate lesson
    assert await _count_rows(store.db_path) == 1


@pytest.mark.asyncio
async def test_reflect_validates_outcome(store) -> None:
    """Invalid outcome values should be normalized to 'success'."""
    response = json.dumps(
        {
            "outcome": "invalid_value",
            "task_summary": "Something happened",
            "lesson": "Important lesson here",
            "tool_issues": [],
            "category": "general",
        }
    )
    llm = _LLMStub(response)
    tool_log = [{"name": "run_command", "args": {}, "result": {"ok": True}}]

    stored = await store.reflect_on_task(
        llm=llm, model="m", user_msg="msg", agent_msg="resp", tool_log=tool_log
    )
    assert stored is True

    # Verify the stored outcome was normalized
    async with aiosqlite.connect(store.db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT outcome FROM reflections")
        row = await cursor.fetchone()
        assert row["outcome"] == "success"


@pytest.mark.asyncio
async def test_reflect_validates_category(store) -> None:
    """Invalid category values should be normalized to 'general'."""
    response = json.dumps(
        {
            "outcome": "success",
            "task_summary": "Something happened",
            "lesson": "Another lesson",
            "tool_issues": [],
            "category": "invalid_category",
        }
    )
    llm = _LLMStub(response)
    tool_log = [{"name": "run_command", "args": {}, "result": {"ok": True}}]

    stored = await store.reflect_on_task(
        llm=llm, model="m", user_msg="msg", agent_msg="resp", tool_log=tool_log
    )
    assert stored is True

    async with aiosqlite.connect(store.db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT category FROM reflections")
        row = await cursor.fetchone()
        assert row["category"] == "general"


# -- format_for_prompt tests --


@pytest.mark.asyncio
async def test_format_for_prompt_empty(store) -> None:
    result = await store.format_for_prompt()
    assert result == ""


@pytest.mark.asyncio
async def test_format_for_prompt_with_reflections(store) -> None:
    # Insert some reflections directly
    async with aiosqlite.connect(store.db_path) as db:
        await db.execute(
            "INSERT INTO reflections (task_summary, outcome, lesson, tool_issues, category) "
            "VALUES (?, ?, ?, ?, ?)",
            ("Weather check", "partial", "Use backup weather API", None, "api"),
        )
        await db.execute(
            "INSERT INTO reflections (task_summary, outcome, lesson, tool_issues, category) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                "Email send",
                "success",
                "himalaya folder flag is case-sensitive",
                json.dumps(
                    [
                        {
                            "tool": "run_command",
                            "issue": "wrong case",
                            "suggestion": "Use INBOX not inbox",
                        }
                    ]
                ),
                "tool",
            ),
        )
        await db.commit()

    result = await store.format_for_prompt()
    assert "## Lessons from past tasks" in result
    assert "Use backup weather API" in result
    assert "[partial]" in result
    assert "himalaya folder flag is case-sensitive" in result
    assert "Use INBOX not inbox" in result


@pytest.mark.asyncio
async def test_format_for_prompt_excludes_empty_lessons(store) -> None:
    """Reflections with empty lessons should not appear in prompt."""
    async with aiosqlite.connect(store.db_path) as db:
        await db.execute(
            "INSERT INTO reflections (task_summary, outcome, lesson, category) VALUES (?, ?, ?, ?)",
            ("Boring task", "success", "", "general"),
        )
        await db.execute(
            "INSERT INTO reflections (task_summary, outcome, lesson, category) VALUES (?, ?, ?, ?)",
            ("Important task", "failure", "Always validate input", "planning"),
        )
        await db.commit()

    result = await store.format_for_prompt()
    assert "Always validate input" in result
    assert "Boring task" not in result


# -- _extract_json_object tests --


class TestExtractJsonObject:
    def test_plain_json(self):
        assert _extract_json_object('{"a": 1}') == {"a": 1}

    def test_empty_string(self):
        assert _extract_json_object("") is None

    def test_no_json(self):
        assert _extract_json_object("No JSON here") is None

    def test_code_fenced(self):
        raw = '```json\n{"a": 1}\n```'
        assert _extract_json_object(raw) == {"a": 1}

    def test_with_preamble(self):
        raw = 'Here is the result:\n{"a": 1}\nDone.'
        assert _extract_json_object(raw) == {"a": 1}

    def test_nested_braces(self):
        raw = '{"outer": {"inner": 1}}'
        assert _extract_json_object(raw) == {"outer": {"inner": 1}}


# -- _format_tool_log tests --


class TestFormatToolLog:
    def test_empty_log(self):
        result = ReflectionStore._format_tool_log([])
        assert result == "(no tools used)"

    def test_successful_tool(self):
        log = [{"name": "web_search", "result": {"query": "test", "results": []}}]
        result = ReflectionStore._format_tool_log(log)
        assert "web_search: OK" in result

    def test_failed_tool(self):
        log = [{"name": "run_command", "result": {"error": "command not found"}}]
        result = ReflectionStore._format_tool_log(log)
        assert "run_command: ERROR" in result
        assert "command not found" in result

    def test_truncates_long_results(self):
        log = [{"name": "run_command", "result": {"stdout": "x" * 1000}}]
        result = ReflectionStore._format_tool_log(log)
        assert "run_command: OK" in result
