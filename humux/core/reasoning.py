"""Discovery and request encoding for provider-specific reasoning settings."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

from anthropic import AsyncAnthropic

DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
    "grok": "https://api.x.ai/v1",
    "deepseek": "https://api.deepseek.com",
    "openrouter": "https://openrouter.ai/api/v1",
}
PROVIDERS = {"anthropic", *DEFAULT_BASE_URLS}
CLAUDE_EFFORT = ("low", "medium", "high", "xhigh", "max")
GATEWAY_EFFORT = ("none", "minimal", *CLAUDE_EFFORT)
_CLAUDE_MODES = ("adaptive", "disabled", "between_tools")
_INTEGER = re.compile(r"-?[0-9]{1,9}")
_EFFORT = re.compile(r"[a-z][a-z0-9_-]{0,39}")


def normalize_setting(value: object) -> str | None:
    """Normalize a stored string or integer while preserving zero and explicit defaults."""
    if value is None:
        return ""
    if type(value) not in (str, int):
        return None
    level = str(value).strip().lower()
    if level in ("", "off", *GATEWAY_EFFORT, *_CLAUDE_MODES) or _INTEGER.fullmatch(level):
        return level
    mode, _, argument = level.partition(":")
    if mode == "effort" and _EFFORT.fullmatch(argument):
        return level
    if _INTEGER.fullmatch(argument):
        if mode == "budget" and int(argument) >= -1:
            return level
        if mode == "sglang" and 1 <= int(argument) <= 100:
            return level
    return None


def _native_endpoint(provider: str, base_url: str | None) -> bool:
    if provider == "anthropic":
        return True
    default = DEFAULT_BASE_URLS.get(provider, "")
    return not base_url or urlsplit(base_url).netloc == urlsplit(default).netloc


def reasoning_kwargs(
    provider: str, base_url: str | None, model: str, level: object
) -> dict[str, Any]:
    setting = normalize_setting(level)
    if setting in (None, "", "off"):
        return {}
    kind, _, argument = setting.partition(":")
    if provider == "anthropic":
        if setting in CLAUDE_EFFORT:
            kwargs = {"output_config": {"effort": setting}}
            # Opus 4.5 supports effort and manual extended thinking.
            if not model.startswith(("claude-opus-4-5", "claude-4-5-opus")):
                kwargs["thinking"] = {"type": "adaptive"}
            return kwargs
        if setting in _CLAUDE_MODES:
            return {"thinking": {"type": setting}}
        if kind == "budget" and int(argument) >= 1024:
            return {"thinking": {"type": "enabled", "budget_tokens": int(argument)}}
        return {}

    native = _native_endpoint(provider, base_url)
    if kind == "sglang" and not native:
        return {
            "extra_body": {
                "chat_template_kwargs": {
                    "thinking": True,
                    "reasoning_effort": int(argument),
                }
            }
        }
    if kind == "budget" and provider == "google" and native:
        return {
            "extra_body": {
                "extra_body": {
                    "google": {
                        "thinking_config": {"thinking_budget": int(argument)},
                    }
                }
            }
        }
    if provider == "openrouter" and native:
        if kind == "budget" and int(argument) >= 0:
            return {"extra_body": {"reasoning": {"max_tokens": int(argument)}}}
        if setting in GATEWAY_EFFORT:
            return {"extra_body": {"reasoning": {"effort": setting}}}
    if not native and _INTEGER.fullmatch(setting):
        return {"extra_body": {"reasoning_effort": int(setting)}}
    if kind == "effort":
        return {"extra_body": {"reasoning_effort": argument}}
    if setting in GATEWAY_EFFORT:
        return {"reasoning_effort": setting}
    return {}


def _effort_options(values: list) -> list[str]:
    levels = []
    for value in values:
        level = normalize_setting(value)
        if level in GATEWAY_EFFORT or (level and _INTEGER.fullmatch(level)):
            levels.append(level)
        elif isinstance(value, str) and _EFFORT.fullmatch(value):
            levels.append(f"effort:{value}")
    return list(dict.fromkeys(levels))


def _advertised_options(provider: str, model: dict, *, native: bool) -> dict | None:
    """Read the public DeepSeek and OpenRouter model metadata shapes."""
    effort = model.get("effort") or {}
    if isinstance(effort, dict) and isinstance(effort.get("supported_levels"), list):
        levels = _effort_options(effort["supported_levels"])
        if provider == "deepseek" and native and levels and "none" not in levels:
            levels.insert(0, "none")
        return {"levels": levels, "supported": bool(levels)}

    reasoning = model.get("reasoning")
    if not isinstance(reasoning, dict):
        parameters = model.get("supported_parameters")
        if provider == "openrouter" and native and isinstance(parameters, list):
            if "reasoning" in parameters or "reasoning_effort" in parameters:
                return {"levels": [], "supported": True}
            if model.get("id") not in {"openrouter/auto", "openrouter/free"}:
                return {"levels": [], "supported": False}
        return None
    advertised = reasoning.get("supported_efforts", [])
    if advertised is None:
        levels = list(GATEWAY_EFFORT)
    else:
        levels = _effort_options(advertised) if isinstance(advertised, list) else []
    if reasoning.get("mandatory"):
        levels = [level for level in levels if level != "none"]
    result: dict = {"levels": levels, "supported": True}
    if reasoning.get("supports_max_tokens") and provider == "openrouter" and native:
        result["budget"] = {"prefix": "budget:", "label": "Reasoning token budget", "min": 0}
    return result


def _documented_options(provider: str, model: str) -> dict | None:
    """Known Chat Completions options when the official Models API omits them."""
    # https://developers.openai.com/api/docs/guides/reasoning
    # https://ai.google.dev/gemini-api/docs/openai#thinking
    # https://docs.x.ai/developers/model-capabilities/text/reasoning
    # Only match exact model families and dated snapshots. Other aliases can move
    # to different model families independently of this catalog.
    model = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)
    standard = ["low", "medium", "high"]
    budget = None
    if provider == "openai":
        if model in {
            "gpt-5.2",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-5.5",
        }:
            levels = ["none", *standard, "xhigh"]
        elif model == "gpt-5.1":
            levels = ["none", *standard]
        elif model in {"gpt-5", "gpt-5-mini", "gpt-5-nano"}:
            levels = ["minimal", *standard]
        elif model in {"o1", "o3", "o3-mini", "o4-mini"}:
            levels = standard
        elif model in {
            "gpt-6.2-astra",
            "gpt-6.2-sol",
            "gpt-6.1-sol",
            "gpt-6-astra",
            "gpt-6-sol",
            "gpt-6-luna",
        }:
            levels = [*standard, "xhigh", "max"]
        else:
            return None
    elif provider == "grok":
        if model in {"grok-4.6", "grok-4.7"}:
            levels = [*standard, "xhigh"]
        elif model == "grok-4.5":
            levels = standard
        elif model == "grok-4.3":
            levels = ["none", *standard]
        else:
            return None
    elif provider == "google":
        if model == "gemini-3-pro-preview":
            levels = ["low", "high"]
        elif model in {
            "gemini-4-carbon",
            "gemini-4-argon",
            "gemini-3.8-flash",
            "gemini-3.1-pro-preview",
        }:
            levels = standard
        elif model in {
            "gemini-3.6-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-3.1-flash-lite-preview",
            "gemini-3-flash-preview",
        }:
            levels = ["minimal", *standard]
        elif model == "gemini-2.5-pro":
            levels = standard
            budget = {
                "prefix": "budget:",
                "label": "Reasoning token budget",
                "min": 128,
                "max": 32768,
                "dynamic": True,
            }
        elif model in {"gemini-2.5-flash", "gemini-2.5-flash-lite"}:
            levels = ["none", *standard]
            budget = {
                "prefix": "budget:",
                "label": "Reasoning token budget",
                "min": 0 if model.endswith("-flash") else 512,
                "max": 24576,
                "dynamic": True,
            }
        elif model in {"gemini-4-neon", "gemini-4-hydrogen"}:
            levels = [*standard, "xhigh"]
        else:
            return None
    else:
        return None
    result: dict = {"levels": levels, "supported": True}
    if budget:
        result["budget"] = budget
    return result


async def discover_options(provider: str, api_key: str, base_url: str | None, model: str) -> dict:
    """Discover a model's options without making a generation request."""
    if provider == "anthropic":
        if not api_key:
            return {"ok": False, "error": "Save the Anthropic API key first"}
        if model == "claude-4-5-haiku":
            model = "claude-haiku-4-5"
        async with AsyncAnthropic(api_key=api_key, timeout=10, max_retries=0) as client:
            info = await client.models.retrieve(model)
        caps = getattr(info, "capabilities", None)
        if caps is None:
            return {"ok": True, "supported": None, "levels": [], "source": "unknown"}
        if not isinstance(caps, dict):
            caps = caps.model_dump()
        effort = caps.get("effort") or {}
        thinking = caps.get("thinking") or {}
        modes = thinking.get("types") or {}
        levels = [level for level in CLAUDE_EFFORT if (effort.get(level) or {}).get("supported")]
        levels.extend(mode for mode in _CLAUDE_MODES if (modes.get(mode) or {}).get("supported"))
        result = {
            "ok": True,
            "levels": levels,
            "source": "provider",
            "supported": bool(thinking.get("supported") or levels),
        }
        if (modes.get("enabled") or {}).get("supported"):
            result["budget"] = {
                "prefix": "budget:",
                "label": "Reasoning token budget",
                "min": 1024,
            }
        return result

    native = _native_endpoint(provider, base_url)
    model = model.removeprefix("models/") if provider == "google" else model
    if native and provider in {"openai", "google", "grok"}:
        options = _documented_options(provider, model)
        return (
            {"ok": True, "source": "docs", **options}
            if options
            else {
                "ok": True,
                "supported": None,
                "levels": [],
                "source": "unknown",
            }
        )
    if not api_key and native and provider != "openrouter":
        return {"ok": False, "error": "Save the provider API key first"}

    result = {"ok": True, "supported": None, "levels": [], "source": "unknown"}
    if not native and re.search(r"deepseek[-_]v?4[._-]1[-_]flash", model, re.IGNORECASE):
        # A served model ID doesn't identify the inference engine.
        result["budget"] = {
            "prefix": "sglang:",
            "label": "SGLang numeric effort",
            "min": 1,
            "max": 100,
            "conditional": "For SGLang endpoints",
        }

    from openai import AsyncOpenAI

    async with AsyncOpenAI(
        api_key=api_key or "unused",
        base_url=base_url or DEFAULT_BASE_URLS[provider],
        timeout=10,
        max_retries=0,
    ) as client:
        response = await client.models.list()
        for entry in response.data:
            if entry.id == model:
                options = _advertised_options(provider, entry.model_dump(), native=native)
                if options is not None:
                    return {**result, "source": "provider", **options}
                if (
                    provider == "deepseek"
                    and native
                    and model
                    in {
                        "deepseek-flash",
                        "deepseek-v4-pro",
                    }
                ):
                    return {
                        "ok": True,
                        "source": "docs",
                        "supported": True,
                        "levels": ["none", "low", "high", "max"],
                    }
                break
    return result
