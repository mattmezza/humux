"""Tests for thinking-level plumbing in LLMClient."""

from __future__ import annotations

import json
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import BadRequestError

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
async def test_openrouter_records_reported_spend_for_chat_and_background_text(monkeypatch):
    record = AsyncMock()
    monkeypatch.setattr("core.metrics.record_usage", record)
    client = LLMClient("openai", "test-key", "https://openrouter.ai/api/v1")
    response = {
        "id": "gen-test",
        "object": "chat.completion",
        "model": "router/resolved",
        "created": 0,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": 200,
            "completion_tokens": 10,
            "total_tokens": 210,
            "cost": 0,
            "prompt_tokens_details": {"cached_tokens": 80, "cache_write_tokens": 50},
        },
    }
    transport = httpx.MockTransport(lambda _req: httpx.Response(200, json=response))
    http_client = httpx.AsyncClient(transport=transport)
    async with client._client as sdk, sdk.with_options(http_client=http_client) as sdk_with_mock:
        client._client = sdk_with_mock
        assert (
            await client.generate(model="router/test", system="", messages=[], tools=[])
        ).text == "hi"
        assert await client.generate_text(model="router/test", prompt="hi") == "hi"

    assert record.call_count == 2
    for call in record.call_args_list:
        assert call.kwargs == {
            "provider": "openrouter",
            "model": "router/test",
            "pricing_model": "router/resolved",
            "cost_usd": 0.0,
            "input_tokens": 70,
            "output_tokens": 10,
            "cache_read_input_tokens": 80,
            "cache_creation_input_tokens": 50,
            "agent": "",
        }


@pytest.mark.asyncio
async def test_anthropic_generate_sends_effort_when_set() -> None:
    client = LLMClient("anthropic", "x", thinking_level="medium")
    create = AsyncMock(return_value=type("R", (), {"content": [], "usage": None})())
    client._client = type("C", (), {"messages": type("M", (), {"create": create})()})()

    await client.generate(model="claude-4-6-opus", system="s", messages=[], tools=[])

    kwargs = create.await_args.kwargs
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "medium"}
    assert kwargs["max_tokens"] == 4096
    assert "max_completion_tokens" not in kwargs


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
    """Background tasks (memory/compaction/etc.) honor the client's level too."""
    client = LLMClient("anthropic", "x", thinking_level="low")
    create = AsyncMock(return_value=type("R", (), {"content": []})())
    client._client = type("C", (), {"messages": type("M", (), {"create": create})()})()

    await client.generate_text(model="claude-4-6-opus", prompt="hi")

    kwargs = create.await_args.kwargs
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "low"}
    assert kwargs["max_tokens"] == 1024
    assert "max_completion_tokens" not in kwargs


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


# --- #317: an empty response keeps its raw shape; the turn decides what to say --


def _openai_msg(content: str, reasoning: str | None = None) -> Any:
    return type(
        "Msg",
        (),
        {
            "tool_calls": None,
            "content": content,
            "reasoning_content": reasoning,
            "reasoning": None,
            "model_dump": lambda self, exclude_none=True: {
                "role": "assistant",
                "content": content,
            },
        },
    )()


def _openai_resp(content: str, finish_reason: str, reasoning: str | None = None) -> Any:
    choice = type(
        "Choice", (), {"message": _openai_msg(content, reasoning), "finish_reason": finish_reason}
    )()
    return type("R", (), {"choices": [choice], "usage": None})()


def _openai_client(
    create: AsyncMock,
    *,
    provider: str = "openai",
    base_url: str | None = None,
    thinking_level: str = "",
) -> LLMClient:
    client = LLMClient(provider, "x", base_url, thinking_level)
    completions = type("Co", (), {"create": create})()
    client._client = type("C", (), {"chat": type("Ch", (), {"completions": completions})()})()
    return client


@pytest.mark.asyncio
async def test_openai_generate_keeps_reasoning_out_of_text() -> None:
    """A reasoning model can answer only in reasoning_content. generate() reports
    both fields as they came — surfacing the CoT as the reply is the agent loop's
    call (only on a first, tool-free round), never the client's (#317)."""
    create = AsyncMock(return_value=_openai_resp("", "stop", reasoning="the answer is 42"))
    client = _openai_client(create)

    out = await client.generate(model="stealth/ox-alpha", system="s", messages=[], tools=[])

    assert create.await_count == 1
    assert out.text == ""
    assert out.reasoning == "the answer is 42"
    assert out.truncated is False


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
    ctx = ("telegram", "u1", "c1", "")
    tok = llm.set_capture_context(ctx)
    try:
        await client.generate(model="claude-opus-4-8", system="s", messages=[], tools=[])
    finally:
        llm.reset_capture_context(tok)
    captured = llm.get_sent_payload(ctx)
    assert captured is not None
    assert captured["usage"]["context_tokens"] == 2000  # 1200 input + 800 cache read
    llm.clear_captured()


def test_list_captured_is_newest_first() -> None:
    llm.clear_captured()
    for i in range(3):
        llm.record_sent_payload(("cli", "u", "c", f"sub_{i}"), {"model": str(i)})
    assert llm.list_captured() == [
        ("cli", "u", "c", "sub_2"),
        ("cli", "u", "c", "sub_1"),
        ("cli", "u", "c", "sub_0"),
    ]
    llm.clear_captured()


def test_capture_run_id_is_a_separate_slot_from_the_main_turn() -> None:
    """A subagent's payload is keyed by its run id, so it can't clobber the
    parent turn's last-sent payload (#99 + fan-out)."""
    llm.clear_captured()
    main = ("telegram", "u1", "c1", "")
    child = ("telegram", "u1", "c1", "sub_abc")
    llm.record_sent_payload(main, {"model": "parent"})
    llm.record_sent_payload(child, {"model": "child"})
    assert llm.get_sent_payload(main)["model"] == "parent"
    assert llm.get_sent_payload(child)["model"] == "child"
    assert llm.list_captured()[0] == child  # newest first
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


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-5.4", "gpt-5.2", "gpt-5-mini"])
async def test_openai_sends_completion_limit_for_chat_and_text(model: str) -> None:
    async def create(**kwargs):
        if "max_tokens" in kwargs:
            raise _unsupported_parameter("max_tokens")
        assert kwargs["max_completion_tokens"] == 256
        return _openai_resp("hi", "stop")

    request = AsyncMock(side_effect=create)
    client = _openai_client(request)
    response = await client.generate(model=model, system="s", messages=[], tools=[], max_tokens=256)
    assert response.text == "hi"
    assert await client.generate_text(model=model, prompt="hi", max_tokens=256) == "hi"
    assert request.await_count == 2


def _unsupported_parameter(parameter: str) -> BadRequestError:
    body = {
        "code": "unsupported_parameter",
        "param": parameter,
        "message": f"Unsupported parameter: {parameter}",
    }
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    return BadRequestError(
        body["message"], response=httpx.Response(400, request=request), body=body
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("first_call", ["chat", "text"])
async def test_openai_falls_back_and_caches_limit_by_endpoint_and_model(
    first_call: str, monkeypatch
) -> None:
    monkeypatch.setattr(llm, "_OPENAI_TOKEN_PARAMETERS", OrderedDict())

    async def create(**kwargs):
        if "max_completion_tokens" in kwargs:
            raise _unsupported_parameter("max_completion_tokens")
        assert kwargs["max_tokens"] == 42
        return _openai_resp("hi", "stop")

    request = AsyncMock(side_effect=create)
    primary = _openai_client(request, base_url="https://legacy.test/v1")
    second = _openai_client(request, base_url="https://legacy.test/v1")

    async def call(client: LLMClient, mode: str) -> str:
        if mode == "text":
            return await client.generate_text(model="custom", prompt="hi", max_tokens=42)
        response = await client.generate(
            model="custom", system="s", messages=[], tools=[], max_tokens=42
        )
        return response.text

    assert await call(primary, first_call) == "hi"
    assert await call(second, "chat" if first_call == "text" else "text") == "hi"
    assert request.await_count == 3
    assert request.await_args_list[0].kwargs["max_completion_tokens"] == 42

    modern_request = AsyncMock(return_value=_openai_resp("hi", "stop"))
    modern = _openai_client(modern_request, base_url="https://modern.test/v1")
    assert await call(modern, "text") == "hi"
    assert modern_request.await_args.kwargs["max_completion_tokens"] == 42
    assert await second.generate_text(model="other", prompt="hi", max_tokens=42) == "hi"
    assert request.await_args_list[3].kwargs["max_completion_tokens"] == 42


@pytest.mark.asyncio
async def test_openai_does_not_retry_unrelated_unsupported_parameter() -> None:
    error = _unsupported_parameter("reasoning_effort")
    create = AsyncMock(side_effect=error)
    client = _openai_client(create)
    with pytest.raises(BadRequestError) as exc:
        await client.generate_text(model="custom", prompt="hi")
    assert exc.value is error
    assert create.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["google", "grok", "deepseek", "openrouter"])
async def test_other_providers_use_legacy_limit(provider: str) -> None:
    create = AsyncMock(return_value=_openai_resp("hi", "stop"))
    client = _openai_client(create, provider=provider)
    assert await client.generate_text(model="custom", prompt="hi", max_tokens=30) == "hi"
    assert create.await_args.kwargs["max_tokens"] == 30
    assert "max_completion_tokens" not in create.await_args.kwargs


@pytest.mark.asyncio
async def test_openai_recovers_when_endpoint_stops_accepting_legacy_limit(monkeypatch) -> None:
    cache = OrderedDict({("https://upgraded.test/v1/", "gpt-6.2-sol"): "max_tokens"})
    monkeypatch.setattr(llm, "_OPENAI_TOKEN_PARAMETERS", cache)
    create = AsyncMock(
        side_effect=[
            _unsupported_parameter("max_tokens"),
            _openai_resp("hi", "stop"),
            _openai_resp("hi", "stop"),
        ]
    )
    client = _openai_client(create, base_url="https://upgraded.test/v1")

    response = await client.generate(
        model="gpt-6.2-sol", system="s", messages=[], tools=[], max_tokens=42
    )
    assert response.text == "hi"
    assert await client.generate_text(model="gpt-6.2-sol", prompt="hi", max_tokens=42) == "hi"
    assert create.await_args_list[0].kwargs["max_tokens"] == 42
    assert create.await_args_list[1].kwargs["max_completion_tokens"] == 42
    assert create.await_args_list[2].kwargs["max_completion_tokens"] == 42
    assert create.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "base_url"),
    [
        ("gpt-6.2-sol", "https://api.openai.com/v1"),
        ("gpt-6.2-astra-2026-11-01", "https://eu.api.openai.com/v1"),
    ],
)
async def test_gpt_62_responses_tool_round_trip(model: str, base_url: str, monkeypatch) -> None:
    monkeypatch.setattr("core.metrics.record_usage", AsyncMock())
    client = LLMClient("openai", "x", base_url, thinking_level="high")
    tool_call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "weather",
        "arguments": '{"city":"Zurich"}',
        "status": "completed",
    }
    first_output = [
        {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "encrypted"},
        tool_call,
    ]
    answer = {
        "type": "message",
        "role": "assistant",
        "id": "msg_1",
        "status": "completed",
        "content": [{"type": "output_text", "text": "It is sunny.", "annotations": []}],
    }
    messages = [
        {"role": "user", "content": "Use my current city."},
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_old",
                    "function": {"name": "location", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_old", "content": "Zurich"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is the weather like here?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            ],
        },
    ]
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        requests.append(json.loads(request.content))
        first = len(requests) == 1
        return httpx.Response(
            200,
            json={
                "id": f"resp_{len(requests)}",
                "object": "response",
                "model": model,
                "status": "completed",
                "output": first_output if first else [answer],
                "usage": {
                    "input_tokens": 1500,
                    "output_tokens": 100,
                    "total_tokens": 1600,
                    "input_tokens_details": {"cached_tokens": 1024},
                    "output_tokens_details": {"reasoning_tokens": 65},
                },
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    tools = [
        {
            "name": "weather",
            "description": "Look up the weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
    ]
    async with client._client as sdk, sdk.with_options(http_client=http_client) as responses:
        client._client = responses
        first = await client.generate(
            model=model, system="Be concise", messages=messages, tools=tools
        )
        assert first.tool_calls == [llm.LLMToolCall("call_1", "weather", {"city": "Zurich"})]
        assert first.usage == {
            "input_tokens": 1500,
            "output_tokens": 100,
            "context_tokens": 1500,
            "cache_read_input_tokens": 1024,
            "cache_creation_input_tokens": 0,
        }

        messages.append(client.assistant_message(first))
        messages.extend(
            client.tool_result_messages([{"tool_use_id": "call_1", "content": "Sunny"}])
        )
        final = await client.generate(model=model, system="Be concise", messages=messages, tools=[])

    first_request, followup = requests
    assert final.text == "It is sunny."
    assert final.tool_calls == []
    assert first_request["model"] == model
    assert first_request["max_output_tokens"] == 4096
    assert first_request["reasoning"] == {"effort": "high"}
    assert first_request["store"] is False
    assert first_request["include"] == ["reasoning.encrypted_content"]
    assert first_request["tools"] == [
        {
            "type": "function",
            "name": "weather",
            "description": "Look up the weather",
            "parameters": tools[0]["input_schema"],
            "strict": False,
        }
    ]
    assert first_request["input"][0] == {"role": "developer", "content": "Be concise"}
    assert first_request["input"][3:5] == [
        {"type": "function_call", "call_id": "call_old", "name": "location", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_old", "output": "Zurich"},
    ]
    assert first_request["input"][-1]["content"][-1] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,AA==",
        "detail": "auto",
    }
    assert followup["input"] == [
        *first_request["input"],
        *first_output,
        {"type": "function_call_output", "call_id": "call_1", "output": "Sunny"},
    ]
    assert "tools" not in followup


@pytest.mark.asyncio
async def test_gpt_62_response_flags_incomplete_tool_call() -> None:
    response = SimpleNamespace(
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output_text="",
        output=[SimpleNamespace(type="function_call", status="incomplete", arguments='{"city":')],
        usage=None,
    )
    create = AsyncMock(return_value=response)
    client = LLMClient("openai", "x", "https://api.openai.com/v1", thinking_level="none")
    client._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    result = await client.generate(model="gpt-6.2-sol", system="s", messages=[], tools=[])

    assert result.truncated
    assert result.tool_calls == []
    assert create.await_args.kwargs["reasoning"] == {"effort": "low"}


def test_responses_replays_anthropic_tool_history_with_user_steering() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Look up the current city first"},
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_1", "name": "location", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Oslo"},
                {"type": "text", "text": "Please use Celsius."},
            ],
        },
    ]
    assert llm._responses_input("", messages) == [
        {"role": "assistant", "content": "Let me check."},
        {"type": "function_call", "call_id": "toolu_1", "name": "location", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "toolu_1", "output": "Oslo"},
        {"role": "user", "content": [{"type": "input_text", "text": "Please use Celsius."}]},
    ]


@pytest.mark.asyncio
async def test_gpt_62_text_generation_uses_responses_with_default_reasoning() -> None:
    create = AsyncMock(
        return_value=SimpleNamespace(
            output_text="ok",
            output=[],
            incomplete_details=None,
            usage=None,
        )
    )
    client = LLMClient("openai", "x", "https://api.openai.com/v1")
    client._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    assert await client.generate_text(model="gpt-6.2-sol", prompt="hi") == "ok"
    assert create.await_args.kwargs["input"] == [{"role": "user", "content": "hi"}]
    assert "reasoning" not in create.await_args.kwargs


@pytest.mark.asyncio
async def test_openai_compatible_gateway_keeps_chat_completions_for_gpt_62() -> None:
    create = AsyncMock(return_value=_openai_resp("hi", "stop"))
    client = _openai_client(create, base_url="https://gateway.test/v1", thinking_level="high")
    previous_turn = {
        "role": "assistant",
        "content": "Previous reply",
        llm._RESPONSES_ITEMS: ["native state"],
    }
    result = await client.generate(
        model="gpt-6.2-sol", system="s", messages=[previous_turn], tools=[{"name": "weather"}]
    )

    assert result.text == "hi"
    args = create.await_args.kwargs
    assert args["reasoning_effort"] == "high"
    assert args["tools"][0]["function"]["name"] == "weather"
    assert args["messages"][-1] == {"role": "assistant", "content": "Previous reply"}
    assert llm._RESPONSES_ITEMS in previous_turn
