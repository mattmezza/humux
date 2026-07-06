"""Per-agent LLM inference override: an agent's ``llm`` dict wins key-by-key
over the global LLM config (senior agent → big model, junior → cheap one);
no override = the shared main client, byte-for-byte unchanged behaviour."""

from __future__ import annotations

import pytest

from core.agents import Agent, AgentStore, _as_llm_config
from core.config import Config


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
