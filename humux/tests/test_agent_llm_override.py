"""Per-agent LLM inference override: an agent's ``llm`` dict wins key-by-key
over the global LLM config (senior agent → big model, junior → cheap one);
no override = the shared main client, byte-for-byte unchanged behaviour."""

from __future__ import annotations

import pytest

from core.agents import Agent, AgentStore, _as_llm_config
from core.config import Config
from core.llm import LLMResponse


@pytest.fixture
def core(tmp_path, monkeypatch):
    from core.agent import AgentCore

    monkeypatch.chdir(tmp_path)
    cfg = Config()
    cfg.agent.llm_provider = "deepseek"
    cfg.agent.model = "deepseek-v4-flash"
    cfg.agent.max_tokens = 8192
    cfg.agent.temperature = 0.5
    cfg.memory.embedding.enabled = False
    return AgentCore(cfg)


def test_no_override_returns_main_client(core):
    for agent in (None, Agent(name="plain")):
        llm, model, max_tokens = core._agent_llm(agent)
        assert llm is core.llm
        assert model == "deepseek-v4-flash" and max_tokens == 8192


def test_full_override(core):
    a = Agent(
        name="senior",
        llm={
            "provider": "deepseek",  # same provider → cloned client
            "model": "deepseek-reasoner",
            "thinking_level": "high",
            "max_tokens": 32000,
            "temperature": 0.2,
        },
    )
    llm, model, max_tokens = core._agent_llm(a)
    assert llm is not core.llm  # clone, main client untouched
    assert core.llm.thinking_level == "" and core.llm.temperature == 0.5
    assert llm.provider == "deepseek"
    assert llm.thinking_level == "high" and llm.temperature == 0.2
    assert model == "deepseek-reasoner" and max_tokens == 32000


def test_partial_override_inherits_the_rest(core):
    a = Agent(name="junior", llm={"model": "deepseek-chat"})
    llm, model, max_tokens = core._agent_llm(a)
    assert model == "deepseek-chat"
    assert max_tokens == 8192  # inherited
    assert llm.provider == "deepseek" and llm.temperature == 0.5  # inherited


def test_cross_provider_override_uses_global_credentials(core):
    core.config.agent.anthropic_api_key = "sk-test"
    a = Agent(name="senior", llm={"provider": "anthropic", "model": "claude-4-6-opus"})
    llm, model, _ = core._agent_llm(a)
    assert llm.provider == "anthropic" and model == "claude-4-6-opus"


async def test_llm_override_persists_through_store(tmp_path):
    store = AgentStore(db_path=str(tmp_path / "a.db"), seed_dir=None)
    await store.upsert(Agent(name="senior", llm={"model": "claude-4-6-opus", "max_tokens": 64000}))
    loaded = await store.get("senior")
    assert loaded.llm == {"model": "claude-4-6-opus", "max_tokens": 64000}
    # And an agent saved without one stays inherit-everything.
    await store.upsert(Agent(name="junior"))
    assert (await store.get("junior")).llm == {}


def test_coercer_drops_junk():
    assert _as_llm_config({"provider": " Anthropic ", "max_tokens": "9000"}) == {
        "provider": "anthropic",
        "max_tokens": 9000,
    }
    assert _as_llm_config('{"model": "m", "temperature": 0.1}') == {
        "model": "m",
        "temperature": 0.1,
    }
    assert _as_llm_config({"temperature": 99, "max_tokens": -1, "thinking_level": "ultra"}) == {}
    assert _as_llm_config("broken json") == {}


def test_openrouter_override_resolves_global_key_and_base_url(core):
    """A cross-provider override to OpenRouter picks up the globally configured
    key/base URL (from the LLM tab), never the agent's own (it has none)."""
    core.config.agent.openrouter_api_key = "sk-or-test"
    core.config.agent.openrouter_base_url = "https://or.example.test/v1"
    a = Agent(name="routed", llm={"provider": "openrouter", "model": "stealth/ox-alpha"})

    llm, model, max_tokens = core._agent_llm(a)

    assert llm.provider == "openrouter" and model == "stealth/ox-alpha"
    assert llm._client.api_key == "sk-or-test"
    assert "or.example.test" in str(llm._client.base_url)


def test_openrouter_override_falls_back_to_default_base_url(core):
    """No configured base URL → the OpenRouter gateway default is used."""
    core.config.agent.openrouter_api_key = "sk-or-test"
    a = Agent(name="routed", llm={"provider": "openrouter", "model": "deepseek/deepseek-v4-flash"})

    llm, _, _ = core._agent_llm(a)

    assert llm.provider == "openrouter"
    assert "openrouter.ai/api/v1" in str(llm._client.base_url)


def test_agent_llm_override_never_mutates_main_client(core):
    """The override builds a fresh/cloned client — the main client's temperature
    and thinking level must survive an override-bearing turn (#317 regression)."""
    core.config.agent.openrouter_api_key = "sk-or-test"
    a = Agent(name="senior", llm={"provider": "openrouter", "model": "stealth/ox-alpha"})

    core._agent_llm(a)

    assert core.llm.provider == "deepseek"
    assert core.llm.thinking_level == "" and core.llm.temperature == 0.5


class _SilentFirstResponseLLM:
    """Returns an empty, non-truncated, tool-free response from the very first
    call — a provider glitch, not a deliberate react-only turn."""

    provider = "deepseek"

    def __init__(self, reasoning: str = "") -> None:
        self._reasoning = reasoning

    async def generate(self, **_kw) -> LLMResponse:
        return LLMResponse(text="", tool_calls=[], reasoning=self._reasoning)

    def assistant_message(self, response: LLMResponse) -> dict:
        return {"role": "assistant", "content": response.text}

    def tool_result_messages(self, results: list[dict]) -> list[dict]:
        return [{"role": "user", "content": results}]


@pytest.mark.asyncio
async def test_empty_first_response_is_not_a_silent_turn(core):
    """#317: a first response with no text and no tool calls surfaces a clear
    notice instead of an empty reply — while react-only turns (which always run
    at least one tool round) keep their deliberate silence (test_reactions)."""
    from core.agent import _EMPTY_RESPONSE_MESSAGE

    core.llm = _SilentFirstResponseLLM()
    resp = await core.process("hi", "telegram", "u", chat_id="55")

    assert resp.text == _EMPTY_RESPONSE_MESSAGE


@pytest.mark.asyncio
async def test_reasoning_only_first_response_is_surfaced(core):
    """#317: a reasoning model (stealth/ox-alpha & co) can leave content empty and
    put the whole answer in chain-of-thought — on a first, tool-free round that
    reasoning is the reply, rather than nothing."""
    core.llm = _SilentFirstResponseLLM(reasoning="the answer is 42")
    resp = await core.process("hi", "telegram", "u", chat_id="55")

    assert resp.text == "the answer is 42"


class _BoomLLM:
    """Raises the given exception on every generate — a hard provider failure."""

    provider = "deepseek"

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def generate(self, **_kw):
        raise self._exc

    def assistant_message(self, response: LLMResponse) -> dict:
        return {"role": "assistant", "content": response.text}

    def tool_result_messages(self, results: list[dict]) -> list[dict]:
        return [{"role": "user", "content": results}]


@pytest.mark.asyncio
async def test_provider_api_error_surfaces_to_the_user(core):
    """#317: a bad model id on OpenRouter must surface the exact API error instead
    of dying silently in the channel layer."""
    from httpx import Request, Response
    from openai import NotFoundError

    req = Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    resp = Response(404, request=req)
    core.llm = _BoomLLM(
        NotFoundError(
            "Model not found: deepseek/deepseek-v4-flash-0731",
            response=resp,
            body=None,
        )
    )

    out = await core.process("hi", "telegram", "u", chat_id="55")

    assert "model call failed" in out.text
    assert "404" in out.text
    assert "deepseek-v4-flash-0731" in out.text


@pytest.mark.asyncio
async def test_provider_timeout_error_surfaces_without_status(core):
    """A connection-level APIError (no HTTP status) still reaches the user."""
    from openai import APIConnectionError

    core.llm = _BoomLLM(APIConnectionError(request=None))

    out = await core.process("hi", "telegram", "u", chat_id="55")

    assert "model call failed" in out.text
    assert "(HTTP" not in out.text
