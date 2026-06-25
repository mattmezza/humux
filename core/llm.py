"""LLM client abstraction for multiple providers."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from typing import Any, cast

from anthropic import AsyncAnthropic

_DEFAULT_BASE_URLS = {
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
    "grok": "https://api.x.ai/v1",
    "deepseek": "https://api.deepseek.com",
}

_ANTHROPIC_MODEL_ALIASES = {
    "claude-4-5-haiku": "claude-haiku-4-5",
}


@dataclass
class LLMToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[LLMToolCall]
    raw: object | None = None
    # Token usage for the request, when the provider reports it. Keys:
    # input_tokens, output_tokens, cache_read_input_tokens,
    # cache_creation_input_tokens, context_tokens (= full prompt size).
    usage: dict[str, int] | None = None


def _anthropic_usage(response: Any) -> dict[str, int] | None:
    u = getattr(response, "usage", None)
    if u is None:
        return None
    inp = getattr(u, "input_tokens", 0) or 0
    out = getattr(u, "output_tokens", 0) or 0
    cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
    cache_creation = getattr(u, "cache_creation_input_tokens", 0) or 0
    # The true context size is the uncached input plus everything served
    # from / written to the cache this request.
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_creation,
        "context_tokens": inp + cache_read + cache_creation,
    }


def _openai_usage(response: Any) -> dict[str, int] | None:
    u = getattr(response, "usage", None)
    if u is None:
        return None
    prompt = getattr(u, "prompt_tokens", 0) or 0
    completion = getattr(u, "completion_tokens", 0) or 0
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "context_tokens": prompt,
    }


def _openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for tool in tools:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
        )
    return converted


def _normalize_provider(provider: str) -> str:
    value = (provider or "").strip().lower()
    return value or "anthropic"


def _normalize_model(provider: str, model: str) -> str:
    value = (model or "").strip()
    if provider == "anthropic":
        return _ANTHROPIC_MODEL_ALIASES.get(value, value)
    return value


class LLMClient:
    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str | None = None,
        thinking_level: str = "",
    ):
        self.provider = _normalize_provider(provider)
        # "" (off) | "low" | "medium" | "high" — applied only to the main generate() call
        self.thinking_level = (thinking_level or "").strip().lower()
        self._client: Any
        if self.provider == "anthropic":
            self._client = AsyncAnthropic(api_key=api_key)
        else:
            resolved_base = base_url or _DEFAULT_BASE_URLS.get(self.provider)
            try:
                module = importlib.import_module("openai")
                client_class = cast(Any, getattr(module, "AsyncOpenAI"))
            except Exception as exc:
                raise RuntimeError("openai package is required for this provider") from exc
            client_kwargs: dict[str, Any] = {
                "api_key": api_key,
                "base_url": resolved_base or None,
            }
            self._client = cast(Any, client_class)(**client_kwargs)  # type: ignore[call-arg]

    def _reasoning_kwargs(self) -> dict[str, Any]:
        """Provider-specific request kwargs for the configured thinking level.

        Empty when no level is set, so non-reasoning calls are untouched.
        """
        level = self.thinking_level
        if level not in ("low", "medium", "high"):
            return {}
        if self.provider == "anthropic":
            return {"thinking": {"type": "adaptive"}, "output_config": {"effort": level}}
        return {"reasoning_effort": level}

    @classmethod
    def from_agent_config(cls, config) -> LLMClient:
        provider = _normalize_provider(getattr(config, "llm_provider", "anthropic"))
        thinking = getattr(config, "thinking_level", "")
        if provider == "anthropic":
            return cls(provider, getattr(config, "anthropic_api_key", ""), thinking_level=thinking)
        if provider == "openai":
            return cls(
                provider,
                getattr(config, "openai_api_key", ""),
                getattr(config, "openai_base_url", ""),
                thinking_level=thinking,
            )
        if provider == "google":
            return cls(
                provider,
                getattr(config, "google_api_key", ""),
                getattr(config, "google_base_url", ""),
                thinking_level=thinking,
            )
        if provider == "grok":
            return cls(
                provider,
                getattr(config, "grok_api_key", ""),
                getattr(config, "grok_base_url", ""),
                thinking_level=thinking,
            )
        if provider == "deepseek":
            return cls(
                provider,
                getattr(config, "deepseek_api_key", ""),
                getattr(config, "deepseek_base_url", ""),
                thinking_level=thinking,
            )
        return cls("anthropic", getattr(config, "anthropic_api_key", ""), thinking_level=thinking)

    async def generate(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> LLMResponse:
        resolved_model = _normalize_model(self.provider, model)
        if self.provider == "anthropic":
            client_any = cast(Any, self._client)
            messages_client = cast(Any, getattr(client_any, "messages"))  # type: ignore[attr-defined]
            # Mark the (static) system prompt as a cache breakpoint so the
            # tools + system prefix is cached and not reprocessed every turn.
            system_param: Any = system
            if system:
                system_param = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            response = await messages_client.create(
                model=resolved_model,
                max_tokens=max_tokens,
                system=cast(Any, system_param),
                messages=cast(Any, messages),
                tools=cast(Any, tools),
                **self._reasoning_kwargs(),
            )
            tool_calls = []
            text_parts = []
            for block in response.content:
                block_any = cast(Any, block)
                if getattr(block_any, "type", None) == "tool_use":
                    tool_calls.append(
                        LLMToolCall(
                            id=getattr(block_any, "id", ""),
                            name=getattr(block_any, "name", ""),
                            arguments=getattr(block_any, "input", {}),
                        )
                    )
                if getattr(block_any, "type", None) == "text":
                    text_parts.append(getattr(block_any, "text", ""))
            return LLMResponse(
                text="\n".join(text_parts).strip(),
                tool_calls=tool_calls,
                raw=response.content,
                usage=_anthropic_usage(response),
            )

        openai_tools = _openai_tools(tools)
        client_any = cast(Any, self._client)
        full_messages = [{"role": "system", "content": system}, *messages]
        response = await client_any.chat.completions.create(
            model=resolved_model,
            max_tokens=max_tokens,
            messages=cast(Any, full_messages),
            tools=cast(Any, openai_tools),
            **self._reasoning_kwargs(),
        )
        message = response.choices[0].message
        tool_calls = []
        for call in message.tool_calls or []:
            args = {}
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(LLMToolCall(id=call.id, name=call.function.name, arguments=args))
        return LLMResponse(
            text=(message.content or "").strip(),
            tool_calls=tool_calls,
            raw=message.model_dump(exclude_none=True),
            usage=_openai_usage(response),
        )

    def assistant_message(self, response: LLMResponse) -> dict[str, Any]:
        if self.provider == "anthropic":
            return {"role": "assistant", "content": response.raw or response.text}
        if isinstance(response.raw, dict):
            return response.raw
        return {"role": "assistant", "content": response.text}

    def tool_result_messages(self, tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.provider == "anthropic":
            return [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": r.get("tool_use_id", ""),
                            "content": r.get("content", ""),
                        }
                        for r in tool_results
                    ],
                }
            ]
        return [
            {
                "role": "tool",
                "tool_call_id": r.get("tool_use_id", ""),
                "content": r.get("content", ""),
            }
            for r in tool_results
        ]

    async def generate_text(self, *, model: str, prompt: str, max_tokens: int = 1024) -> str:
        resolved_model = _normalize_model(self.provider, model)
        if self.provider == "anthropic":
            client_any = cast(Any, self._client)
            messages_client = cast(Any, getattr(client_any, "messages"))  # type: ignore[attr-defined]
            response = await messages_client.create(
                model=resolved_model,
                max_tokens=max_tokens,
                messages=cast(Any, [{"role": "user", "content": prompt}]),
                **self._reasoning_kwargs(),
            )
            for block in response.content:
                block_any = cast(Any, block)
                if getattr(block_any, "type", None) == "text":
                    return str(getattr(block_any, "text", "")).strip()
            return ""

        client_any = cast(Any, self._client)
        response = await client_any.chat.completions.create(
            model=resolved_model,
            max_tokens=max_tokens,
            messages=cast(Any, [{"role": "user", "content": prompt}]),
            **self._reasoning_kwargs(),
        )
        return (response.choices[0].message.content or "").strip()
