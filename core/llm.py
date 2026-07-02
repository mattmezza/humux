"""LLM client abstraction for multiple providers."""

from __future__ import annotations

import contextvars
import importlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, cast

from anthropic import AsyncAnthropic

# Dedicated logger for model chain-of-thought. Silent by default (WARNING);
# the REPL bumps it to INFO to stream reasoning live without spamming server logs.
reasoning_log = logging.getLogger("core.llm.reasoning")
reasoning_log.setLevel(logging.WARNING)

# ── Inference-payload capture (admin Inspect tab, #99) ─────────────────────
# Last full request sent to the model, per conversation context — the exact
# system prompt + history window + tool defs that ran. In-memory only, bounded;
# the agent sets the context for a turn, generate() records the payload, the
# admin Inspect tab reads it back. ponytail: process-global dict, fine for one
# agent process; move onto the agent instance if multi-tenant ever lands.
# Context key = (channel, user_id, chat_id) — same triple as ConversationHistory.
_capture_ctx: contextvars.ContextVar = contextvars.ContextVar("humux_llm_capture_ctx", default=None)
_LAST_SENT: OrderedDict[tuple[str, str, str], dict[str, Any]] = OrderedDict()
_CAPTURE_CAP = 100  # ponytail: LRU cap; bump if you watch >100 live chats


def set_capture_context(ctx: tuple[str, str, str] | None) -> Any:
    """Bind the conversation context that generate() should record under.

    Pass ``None`` to suppress capture (e.g. subagents, which run inside the
    spawner's context but must not overwrite its captured payload). Returns a
    token for :func:`reset_capture_context`."""
    return _capture_ctx.set(ctx)


def reset_capture_context(token: Any) -> None:
    _capture_ctx.reset(token)


def record_sent_payload(ctx: tuple[str, str, str] | None, payload: dict[str, Any]) -> None:
    """Store ``payload`` as the last-sent request for ``ctx`` (no-op if None)."""
    if ctx is None:
        return
    _LAST_SENT[ctx] = payload
    _LAST_SENT.move_to_end(ctx)
    while len(_LAST_SENT) > _CAPTURE_CAP:
        _LAST_SENT.popitem(last=False)


def get_sent_payload(ctx: tuple[str, str, str]) -> dict[str, Any] | None:
    return _LAST_SENT.get(ctx)


def clear_captured() -> None:
    """Drop all captured payloads (used by tests)."""
    _LAST_SENT.clear()


_DEFAULT_BASE_URLS = {
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
    "grok": "https://api.x.ai/v1",
    "deepseek": "https://api.deepseek.com",
    "openrouter": "https://openrouter.ai/api/v1",  # #128: OpenAI-compatible gateway
}

_ANTHROPIC_MODEL_ALIASES = {
    "claude-4-5-haiku": "claude-haiku-4-5",
}

# Vision-capable model heuristic — single source of truth for "can this model
# read images natively". Mirrored client-side by modelSupportsVision() in the
# admin UI (api/templates/base.html + wizard/llm.html); keep the two in sync.
# ponytail: substring heuristic, extend as model ids change. The fallback only
# engages when this returns False, so a wrong "True" just means no captioning.
_VISION_TEXT_ONLY = ("deepseek",)  # families with no image input
_VISION_CAPABLE_PATTERNS = (
    "claude",
    "gpt-4o",
    "gpt-4.1",
    "gpt-5",
    "o1",
    "o3",
    "o4",
    "gemini",
    "grok-4",
    "grok-2-vision",
    "vision",
    "llava",
    "pixtral",
)


def model_supports_vision(provider: str, model: str) -> bool:
    """True when (provider, model) accepts image input natively."""
    mid = (model or "").lower()
    if not mid:
        return False
    if any(p in mid for p in _VISION_TEXT_ONLY):
        return False
    # Anthropic and Google line-ups have no text-only chat models, so trust the
    # provider even for an unrecognized id; others fall back to name patterns.
    if _normalize_provider(provider) in ("anthropic", "google"):
        return True
    return any(p in mid for p in _VISION_CAPABLE_PATTERNS)


@dataclass
class LLMToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[LLMToolCall]
    reasoning: str = ""  # model chain-of-thought, when the provider exposes it
    raw: object | None = None
    # Token usage for the request, when the provider reports it. Keys:
    # input_tokens, output_tokens, cache_read_input_tokens,
    # cache_creation_input_tokens, context_tokens (= full prompt size).
    usage: dict[str, int] | None = None
    # True when the provider stopped at the output-token limit (Anthropic
    # stop_reason == "max_tokens" / OpenAI finish_reason == "length"), i.e. the
    # response — including any tool-call arguments — was cut off mid-stream. The
    # agent loop surfaces this instead of running a half-built tool call.
    truncated: bool = False


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


def _as_content_blocks(content: Any) -> list[dict[str, Any]]:
    """Normalise a message ``content`` to a list of content-part blocks."""
    if isinstance(content, list):
        return content
    return [{"type": "text", "text": content if isinstance(content, str) else str(content)}]


def _coalesce_user_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive ``user`` messages into one, so a run of user turns is
    sent as a single turn.

    Group multi-agent rooms (#30) record every message they see — including ones
    the bot stays silent on — so the replayed history can hold several user turns
    in a row before the bot's reply. Anthropic requires strict user/assistant
    alternation, so this normalises the array right before the API call instead of
    making every caller track it. String contents join with a blank line; anything
    multimodal falls back to concatenated content-part blocks (valid for both
    Anthropic and the OpenAI-compatible providers). Assistant/tool messages are
    left untouched — only plain user turns are ever produced back-to-back.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        if out and out[-1].get("role") == "user" and msg.get("role") == "user":
            prev, cur = out[-1].get("content"), msg.get("content")
            if isinstance(prev, str) and isinstance(cur, str):
                merged: Any = f"{prev}\n\n{cur}"
            else:
                merged = _as_content_blocks(prev) + _as_content_blocks(cur)
            out[-1] = {**out[-1], "content": merged}
        else:
            out.append(dict(msg))
    return out


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
        # Sampling temperature (#12). None = use the provider default. Set by the
        # caller on the main agent client; applied via _sampling_kwargs().
        self.temperature: float | None = None
        self._client: Any
        if self.provider == "anthropic":
            self._client = AsyncAnthropic(api_key=api_key, timeout=60)
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
                "timeout": 60,
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

    def _sampling_kwargs(self) -> dict[str, Any]:
        """Reasoning kwargs plus an explicit temperature when configured (#12).

        Temperature is skipped on reasoning calls — several providers reject an
        explicit temperature alongside reasoning_effort/thinking.
        """
        kwargs = self._reasoning_kwargs()
        if self.temperature is not None and not kwargs:
            kwargs["temperature"] = self.temperature
        return kwargs

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
        if provider == "openrouter":
            # OpenAI-compatible; empty base_url falls back to _DEFAULT_BASE_URLS.
            return cls(
                provider,
                getattr(config, "openrouter_api_key", ""),
                getattr(config, "openrouter_base_url", ""),
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
        # Collapse any run of consecutive user turns (group rooms record silent
        # turns between replies, #30) so the array honours strict alternation.
        messages = _coalesce_user_messages(messages)
        # Snapshot the exact request for the Inspect tab (#99). Shallow-copy the
        # lists so the caller's later mutations (tool-result ping-pong) don't edit
        # what we stored; each generate() overwrites, so the slot holds last-sent.
        # We keep a reference so the real token usage from the response can be
        # backfilled below (#116) — the Inspect tab shows context size + % window.
        payload = {
            "captured_at": time.time(),
            "provider": self.provider,
            "model": resolved_model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": list(messages),
            "tools": list(tools),
        }
        record_sent_payload(_capture_ctx.get(), payload)
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
                **self._sampling_kwargs(),
            )
            tool_calls = []
            text_parts = []
            reasoning_parts = []
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
                if getattr(block_any, "type", None) == "thinking":
                    reasoning_parts.append(getattr(block_any, "thinking", ""))
            reasoning = "\n".join(p for p in reasoning_parts if p).strip()
            if reasoning:
                reasoning_log.info("%s", reasoning)
            usage = _anthropic_usage(response)
            payload["usage"] = usage  # backfill real context size for Inspect (#116)
            return LLMResponse(
                text="\n".join(text_parts).strip(),
                tool_calls=tool_calls,
                reasoning=reasoning,
                raw=response.content,
                usage=usage,
                truncated=getattr(response, "stop_reason", None) == "max_tokens",
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
        # DeepSeek/others expose CoT as message.reasoning_content (or .reasoning).
        reasoning = (
            getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or ""
        ).strip()
        if reasoning:
            reasoning_log.info("%s", reasoning)
        usage = _openai_usage(response)
        payload["usage"] = usage  # backfill real context size for Inspect (#116)
        return LLMResponse(
            text=(message.content or "").strip(),
            tool_calls=tool_calls,
            reasoning=reasoning,
            raw=message.model_dump(exclude_none=True),
            usage=usage,
            truncated=getattr(response.choices[0], "finish_reason", None) == "length",
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
                **self._sampling_kwargs(),
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
