"""Tests for goal decomposition module."""

from __future__ import annotations

import json

import pytest

from core.goal_decomposition import (
    DecomposedGoal,
    SubGoal,
    _extract_json_object,
    classify_complexity,
    decompose_goal,
)


class _LLMStub:
    """Minimal LLM stub that returns a canned response."""

    def __init__(self, response: str):
        self._response = response
        self.call_count = 0

    async def generate_text(self, *, model: str, prompt: str, max_tokens: int = 1024) -> str:
        self.call_count += 1
        return self._response


# -- classify_complexity tests --


@pytest.mark.asyncio
async def test_classify_complex_message() -> None:
    llm = _LLMStub("COMPLEX")
    result = await classify_complexity(llm, "test-model", "Plan my trip to Tokyo next month")
    assert result is True
    assert llm.call_count == 1


@pytest.mark.asyncio
async def test_classify_simple_message() -> None:
    llm = _LLMStub("SIMPLE")
    result = await classify_complexity(llm, "test-model", "What's the weather like right now?")
    assert result is False


@pytest.mark.asyncio
async def test_classify_short_message_skips_llm() -> None:
    """Messages shorter than 20 chars are auto-classified as simple (no LLM call)."""
    llm = _LLMStub("COMPLEX")
    result = await classify_complexity(llm, "test-model", "Hi")
    assert result is False
    assert llm.call_count == 0  # No LLM call made


@pytest.mark.asyncio
async def test_classify_returns_false_on_error() -> None:
    """LLM errors should gracefully return False (simple)."""

    class _ErrorLLM:
        async def generate_text(self, **kwargs) -> str:
            raise RuntimeError("API error")

    result = await classify_complexity(
        _ErrorLLM(), "test-model", "Plan my trip to Tokyo next month"
    )
    assert result is False


@pytest.mark.asyncio
async def test_classify_handles_lowercase_response() -> None:
    llm = _LLMStub("complex")
    result = await classify_complexity(
        llm, "test-model", "Help me prepare for the interview next week"
    )
    assert result is True


@pytest.mark.asyncio
async def test_classify_handles_response_with_extra_text() -> None:
    llm = _LLMStub("COMPLEX\n\nThis is a multi-step request.")
    result = await classify_complexity(
        llm, "test-model", "Help me prepare for the interview next week"
    )
    assert result is True


# -- decompose_goal tests --


@pytest.mark.asyncio
async def test_decompose_goal_returns_plan() -> None:
    response = json.dumps(
        {
            "goal": "Plan trip to Tokyo",
            "steps": [
                {
                    "id": 1,
                    "title": "Check passport",
                    "description": "Verify passport is valid",
                    "depends_on": [],
                },
                {
                    "id": 2,
                    "title": "Search flights",
                    "description": "Find direct flights ZRH-NRT",
                    "depends_on": [1],
                },
                {
                    "id": 3,
                    "title": "Book hotel",
                    "description": "Find hotel near Shinjuku",
                    "depends_on": [1],
                },
            ],
        }
    )
    llm = _LLMStub(response)
    result = await decompose_goal(llm, "test-model", "Plan my trip to Tokyo")

    assert result is not None
    assert result.goal == "Plan trip to Tokyo"
    assert len(result.steps) == 3
    assert result.steps[0].title == "Check passport"
    assert result.steps[1].depends_on == [1]


@pytest.mark.asyncio
async def test_decompose_goal_handles_code_fence() -> None:
    response = (
        "```json\n"
        '{"goal": "Setup", "steps": [{"id": 1, "title": "Install",'
        ' "description": "Install deps", "depends_on": []}]}\n'
        "```"
    )
    llm = _LLMStub(response)
    result = await decompose_goal(llm, "test-model", "Set up my new laptop")

    assert result is not None
    assert result.goal == "Setup"
    assert len(result.steps) == 1


@pytest.mark.asyncio
async def test_decompose_goal_handles_preamble_text() -> None:
    response = (
        "Here is the decomposition:\n"
        '{"goal": "Prepare", "steps": [{"id": 1, "title": "Research",'
        ' "description": "Look up info", "depends_on": []}]}\n'
        "Hope that helps!"
    )
    llm = _LLMStub(response)
    result = await decompose_goal(llm, "test-model", "Prepare for interview")

    assert result is not None
    assert result.goal == "Prepare"


@pytest.mark.asyncio
async def test_decompose_goal_returns_none_on_invalid_json() -> None:
    llm = _LLMStub("This is not JSON at all")
    result = await decompose_goal(llm, "test-model", "Plan my trip")
    assert result is None


@pytest.mark.asyncio
async def test_decompose_goal_returns_none_on_empty_steps() -> None:
    response = json.dumps({"goal": "Something", "steps": []})
    llm = _LLMStub(response)
    result = await decompose_goal(llm, "test-model", "Do something")
    assert result is None


@pytest.mark.asyncio
async def test_decompose_goal_returns_none_on_error() -> None:
    class _ErrorLLM:
        async def generate_text(self, **kwargs) -> str:
            raise RuntimeError("API error")

    result = await decompose_goal(_ErrorLLM(), "test-model", "Plan my trip")
    assert result is None


@pytest.mark.asyncio
async def test_decompose_goal_caps_at_6_steps() -> None:
    steps = [
        {"id": i, "title": f"Step {i}", "description": f"Do thing {i}", "depends_on": []}
        for i in range(1, 10)  # 9 steps
    ]
    response = json.dumps({"goal": "Big plan", "steps": steps})
    llm = _LLMStub(response)
    result = await decompose_goal(llm, "test-model", "Big complex request")

    assert result is not None
    assert len(result.steps) == 6  # Capped at 6


# -- _extract_json_object tests --


class TestExtractJsonObject:
    def test_plain_json(self):
        result = _extract_json_object('{"a": 1}')
        assert result == {"a": 1}

    def test_empty_string(self):
        assert _extract_json_object("") is None

    def test_whitespace_only(self):
        assert _extract_json_object("   ") is None

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
        result = _extract_json_object(raw)
        assert result == {"outer": {"inner": 1}}

    def test_braces_in_strings(self):
        raw = '{"content": "a {b} c"}'
        result = _extract_json_object(raw)
        assert result == {"content": "a {b} c"}

    def test_array_returns_none(self):
        """Arrays should not be returned (we want objects only)."""
        assert _extract_json_object("[1, 2, 3]") is None


# -- DecomposedGoal.format_for_prompt tests --


class TestFormatForPrompt:
    def test_basic_format(self):
        goal = DecomposedGoal(
            goal="Plan trip",
            steps=[
                SubGoal(id=1, title="Check passport", description="Verify validity", depends_on=[]),
                SubGoal(id=2, title="Book flight", description="Search flights", depends_on=[1]),
            ],
        )
        result = goal.format_for_prompt()
        assert "Overall goal: Plan trip" in result
        assert "1. Check passport" in result
        assert "2. Book flight (after step 1)" in result
        assert "Verify validity" in result

    def test_multiple_dependencies(self):
        goal = DecomposedGoal(
            goal="Setup",
            steps=[
                SubGoal(id=1, title="A", description="Do A", depends_on=[]),
                SubGoal(id=2, title="B", description="Do B", depends_on=[]),
                SubGoal(id=3, title="C", description="Do C", depends_on=[1, 2]),
            ],
        )
        result = goal.format_for_prompt()
        assert "3. C (after steps 1, 2)" in result
