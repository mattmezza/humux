"""Reasoning metadata and provider request shapes."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI

from core import reasoning
from core.agents import _as_llm_config
from core.config import Config
from core.llm import LLMClient
from core.subagents import normalize_effort


@pytest.mark.parametrize(
    ("provider", "base_url", "model", "level", "expected"),
    [
        ("openai", None, "gpt-5.4", "xhigh", {"reasoning_effort": "xhigh"}),
        ("deepseek", None, "deepseek-flash", "none", {"reasoning_effort": "none"}),
        (
            "openrouter",
            None,
            "deepseek/deepseek-v4.1-flash",
            "max",
            {"extra_body": {"reasoning": {"effort": "max"}}},
        ),
        (
            "google",
            None,
            "gemini-2.5-flash",
            "budget:0",
            {"extra_body": {"extra_body": {"google": {"thinking_config": {"thinking_budget": 0}}}}},
        ),
        (
            "anthropic",
            None,
            "claude-haiku-4-5",
            "budget:2048",
            {"thinking": {"type": "enabled", "budget_tokens": 2048}},
        ),
        (
            "anthropic",
            None,
            "claude-opus-4-5",
            "high",
            {"output_config": {"effort": "high"}},
        ),
        (
            "openai",
            "http://localhost:30000/v1",
            "local",
            "0",
            {"extra_body": {"reasoning_effort": 0}},
        ),
        (
            "openai",
            "http://localhost:30000/v1",
            "local",
            "effort:ultra",
            {"extra_body": {"reasoning_effort": "ultra"}},
        ),
        ("deepseek", None, "deepseek-flash", "sglang:75", {}),
        ("deepseek", None, "deepseek-flash", "", {}),
    ],
)
def test_reasoning_settings_encode_for_correct_endpoint(provider, base_url, model, level, expected):
    assert reasoning.reasoning_kwargs(provider, base_url, model, level) == expected


async def test_sglang_numeric_effort_reaches_both_completion_paths():
    endpoint = "http://localhost:30000/v1"
    model = "deepseek-ai/DeepSeek-V4.1-Flash"
    client = LLMClient("openai", "unused", endpoint, thinking_level="sglang:75")
    await client._client.close()
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "test",
                        },
                    }
                ],
            },
        )

    async with AsyncOpenAI(
        api_key="unused",
        base_url=endpoint,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    ) as sdk:
        client._client = sdk
        await client.generate(model=model, system="test", messages=[], tools=[])
        await client.generate_text(model=model, prompt="test")

    assert len(requests) == 2
    for body in requests:
        assert body["chat_template_kwargs"] == {"thinking": True, "reasoning_effort": 75}
        assert "reasoning_effort" not in body


@pytest.mark.parametrize(
    ("provider", "model", "extra", "levels", "budget_min"),
    [
        (
            "deepseek",
            "deepseek-flash",
            {"effort": {"supported_levels": ["low", "high", "max"]}},
            ["none", "low", "high", "max"],
            None,
        ),
        (
            "openrouter",
            "test/reasoner",
            {
                "reasoning": {
                    "supported_efforts": ["high", "none"],
                    "mandatory": True,
                    "supports_max_tokens": True,
                }
            },
            ["high"],
            0,
        ),
    ],
)
async def test_discovers_live_openai_compatible_metadata(
    monkeypatch,
    provider,
    model,
    extra,
    levels,
    budget_min,
):
    def handle(request):
        assert request.url.path.endswith("/models")
        items = [
            {"id": "other", "object": "model", "owned_by": "test", "created": 0},
            {"id": model, "object": "model", "owned_by": "test", "created": 0, **extra},
        ]
        return httpx.Response(200, json={"object": "list", "data": items})

    def sdk_client(**kwargs):
        return AsyncOpenAI(
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
            **kwargs,
        )

    monkeypatch.setattr("openai.AsyncOpenAI", sdk_client)
    discovered = await reasoning.discover_options(provider, "test", None, model)
    assert discovered["source"] == "provider"
    assert discovered["levels"] == levels
    assert discovered.get("budget", {}).get("min") == budget_min


async def test_anthropic_discovers_xhigh_and_manual_budget(monkeypatch):
    sdk = AsyncMock()
    sdk.__aenter__.return_value = sdk
    sdk.models.retrieve.return_value.capabilities = {
        "effort": {"xhigh": {"supported": True}, "max": {"supported": False}},
        "thinking": {"supported": True, "types": {"enabled": {"supported": True}}},
    }
    monkeypatch.setattr(reasoning, "AsyncAnthropic", lambda **kwargs: sdk)

    result = await reasoning.discover_options("anthropic", "test", None, "claude")
    assert result["levels"] == ["xhigh"]
    assert result["budget"]["min"] == 1024
    assert reasoning.reasoning_kwargs("anthropic", None, "claude", "xhigh")["output_config"] == {
        "effort": "xhigh",
    }


async def test_docs_options_leave_unknown_models_unclassified():
    known = await reasoning.discover_options("openai", "", None, "gpt-5.4-2026-03-05")
    unknown = await reasoning.discover_options("openai", "", None, "future-model")
    assert known["levels"] == ["none", "low", "medium", "high", "xhigh"]
    assert known["source"] == "docs"
    assert unknown["levels"] == [] and unknown["supported"] is None


def test_per_agent_and_subagent_settings_preserve_numeric_values_and_default():
    assert _as_llm_config({"thinking_level": 0}) == {"thinking_level": "0"}
    assert _as_llm_config({"thinking_level": ""}) == {"thinking_level": ""}
    assert normalize_effort("sglang:75") == "sglang:75"
    assert normalize_effort("sglang:0") is None


def test_yaml_numeric_efforts_use_the_same_settings_as_the_ui():
    config = Config.model_validate(
        {
            "agent": {"thinking_level": 75},
            "memory": {"extraction_thinking_level": 0},
        }
    )
    assert config.agent.thinking_level == "75"
    assert config.memory.extraction_thinking_level == "0"
