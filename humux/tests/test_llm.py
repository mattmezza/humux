"""Tests for thinking-level plumbing in LLMClient."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from core import llm
from core.config import AgentConfig
from core.llm import LLMClient


def test_thinking_level_normalized_and_defaulted() -> None:
    assert LLMClient("anthropic", "x", thinking_level=" HIGH ").thinking_level == "high"
    assert LLMClient("anthropic", "x").thinking_level == ""


def test_from_agent_config_carries_thinking_level() -> None:
    cfg = AgentConfig(llm_provider="openai", openai_api_key="x", thinking_level="low")
    assert LLMClient.from_agent_config(cfg).thinking_level == "low"


def test_from_agent_config_openrouter_defaults_base_url() -> None:
    """OpenRouter (#128) routes through the OpenAI-compatible client with the
    gateway base URL when none is configured."""
    cfg = AgentConfig(llm_provider="openrouter", openrouter_api_key="k")
    client = LLMClient.from_agent_config(cfg)
    assert client.provider == "openrouter"
    assert "openrouter.ai/api/v1" in str(client._client.base_url)


def test_from_agent_config_openrouter_base_url_override() -> None:
    cfg = AgentConfig(
        llm_provider="openrouter",
        openrouter_api_key="k",
        openrouter_base_url="https://example.test/v1",
    )
    client = LLMClient.from_agent_config(cfg)
    assert "example.test" in str(client._client.base_url)


@pytest.mark.asyncio
async def test_anthropic_generate_sends_effort_when_set() -> None:
    client = LLMClient("anthropic", "x", thinking_level="medium")
    create = AsyncMock(return_value=type("R", (), {"content": [], "usage": None})())
    client._client = type("C", (), {"messages": type("M", (), {"create": create})()})()

    await client.generate(model="claude-4-6-opus", system="s", messages=[], tools=[])

    kwargs = create.await_args.kwargs
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "medium"}


@pytest.mark.asyncio
async def test_anthropic_generate_omits_effort_when_off() -> None:
    client = LLMClient("anthropic", "x")
    create = AsyncMock(return_value=type("R", (), {"content": [], "usage": None})())
    client._client = type("C", (), {"messages": type("M", (), {"create": create})()})()

    await client.generate(model="claude-4-6-opus", system="s", messages=[], tools=[])

    kwargs = create.await_args.kwargs
    assert "thinking" not in kwargs
    assert "output_config" not in kwargs


@pytest.mark.asyncio
async def test_anthropic_generate_text_sends_effort_when_set() -> None:
    """Background tasks (memory/reflection/etc.) honor the client's level too."""
    client = LLMClient("anthropic", "x", thinking_level="low")
    create = AsyncMock(return_value=type("R", (), {"content": []})())
    client._client = type("C", (), {"messages": type("M", (), {"create": create})()})()

    await client.generate_text(model="claude-4-6-opus", prompt="hi")

    kwargs = create.await_args.kwargs
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "low"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [("max_tokens", True), ("end_turn", False)],
)
async def test_anthropic_generate_flags_truncation(stop_reason: str, expected: bool) -> None:
    """stop_reason == 'max_tokens' means the response (incl. tool args) was cut off (#77)."""
    client = LLMClient("anthropic", "x")
    resp = type("R", (), {"content": [], "usage": None, "stop_reason": stop_reason})()
    create = AsyncMock(return_value=resp)
    client._client = type("C", (), {"messages": type("M", (), {"create": create})()})()

    out = await client.generate(model="claude-4-6-opus", system="s", messages=[], tools=[])
    assert out.truncated is expected


@pytest.mark.asyncio
async def test_openai_generate_flags_truncation() -> None:
    """OpenAI/DeepSeek signal the same cut-off via finish_reason == 'length' (#77)."""
    client = LLMClient("openai", "x")
    msg = type(
        "Msg",
        (),
        {
            "tool_calls": None,
            "content": "hi",
            "reasoning_content": None,
            "reasoning": None,
            "model_dump": lambda self, exclude_none=True: {"role": "assistant", "content": "hi"},
        },
    )()
    choice = type("Choice", (), {"message": msg, "finish_reason": "length"})()
    resp = type("R", (), {"choices": [choice], "usage": None})()
    create = AsyncMock(return_value=resp)
    completions = type("Co", (), {"create": create})()
    client._client = type("C", (), {"chat": type("Ch", (), {"completions": completions})()})()

    out = await client.generate(model="deepseek-v4-flash", system="s", messages=[], tools=[])
    assert out.truncated is True


@pytest.mark.asyncio
async def test_generate_backfills_usage_into_captured_payload() -> None:
    """The Inspect tab needs the real context size (#116); generate() writes the
    response's usage back onto the payload it captured before the call."""
    client = LLMClient("anthropic", "x")
    usage = type(
        "U",
        (),
        {
            "input_tokens": 1200,
            "output_tokens": 30,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 0,
        },
    )()
    resp = type("R", (), {"content": [], "usage": usage, "stop_reason": "end_turn"})()
    create = AsyncMock(return_value=resp)
    client._client = type("C", (), {"messages": type("M", (), {"create": create})()})()

    llm.clear_captured()
    ctx = ("telegram", "u1", "c1")
    tok = llm.set_capture_context(ctx)
    try:
        await client.generate(model="claude-opus-4-8", system="s", messages=[], tools=[])
    finally:
        llm.reset_capture_context(tok)
    captured = llm.get_sent_payload(ctx)
    assert captured is not None
    assert captured["usage"]["context_tokens"] == 2000  # 1200 input + 800 cache read
    llm.clear_captured()


def test_truncation_tool_results_carries_notice() -> None:
    """A truncated round feeds back the notice per pending call instead of executing it (#77)."""
    import json

    from core.agent import _TRUNCATION_NOTICE, _truncation_tool_results

    call = type("Call", (), {"id": "tu_1"})()
    response = type("Resp", (), {"tool_calls": [call]})()
    results = _truncation_tool_results(response)
    assert results == [
        {
            "type": "tool_result",
            "tool_use_id": "tu_1",
            "content": json.dumps({"error": _TRUNCATION_NOTICE}),
        }
    ]


def test_reasoning_kwargs_per_provider() -> None:
    assert LLMClient("openai", "x", thinking_level="high")._reasoning_kwargs() == {
        "reasoning_effort": "high"
    }
    assert LLMClient("anthropic", "x")._reasoning_kwargs() == {}
    # unknown level value is ignored (off)
    assert LLMClient("anthropic", "x", thinking_level="bogus")._reasoning_kwargs() == {}
