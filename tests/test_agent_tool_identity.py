"""Per-agent tool identity (#93): the active agent's own credentials are
injected into ``run_command``, and never the owner's when it opts out.

These exercise the real dispatch path in ``AgentCore._execute_tool`` with a
stubbed executor that captures the per-call ``tool_env`` override.
"""

from __future__ import annotations

import pytest

from core.agent import AgentCore
from core.agents import Agent
from core.config import Config
from core.llm import LLMToolCall
from core.permissions import PermissionLevel
from core.secret_store import SecretStore
from core.tools import gh_token_secret_name
from core.vault import InfraVault


@pytest.fixture
async def agent(tmp_path) -> AgentCore:
    # Infra vault keyed so per-agent tokens are storable/resolvable headlessly.
    store = SecretStore(db_path=str(tmp_path / "config.db"), infra_vault=InfraVault("machine-key"))
    await store.set_infra_secret(gh_token_secret_name("hopper"), "hopper-token")
    await store.load_infra_cache()
    config = Config()
    config.tools.gh.enabled = True
    config.tools.gh.token = "owner-token"  # the system-wide default
    config.tools.browser.enabled = True
    a = AgentCore(config, secret_store=store)
    # Refresh the executor's shared default to match the config above.
    from core.tools import tool_env

    a.executor.tool_env = tool_env(config)
    return a


async def _run(agent: AgentCore, ag: Agent | None, monkeypatch) -> dict:
    captured: dict = {}

    async def fake_exec(command, timeout, cwd=None, tool_env=None):
        captured["tool_env"] = tool_env
        return {"stdout": "", "stderr": "", "exit_code": 0}

    monkeypatch.setattr(agent.executor, "_exec", fake_exec)
    monkeypatch.setattr(agent.permissions, "check", lambda *a, **k: PermissionLevel.ALWAYS)
    rs = agent._new_request_state(ag)
    call = LLMToolCall(
        id="1", name="run_command", arguments={"command": "gh issue list", "purpose": "p"}
    )
    await agent._execute_tool(call, "system", "u", rs)
    return captured


async def test_no_agent_uses_shared_default(agent: AgentCore, monkeypatch) -> None:
    # No agent → no per-call override; the executor's shared default applies.
    cap = await _run(agent, None, monkeypatch)
    assert cap["tool_env"] is None


async def test_agent_with_own_token(agent: AgentCore, monkeypatch) -> None:
    hopper = Agent(name="hopper", tool_config={"gh": {"enabled": True}})
    cap = await _run(agent, hopper, monkeypatch)
    assert cap["tool_env"]["GH_TOKEN"] == "hopper-token"  # its own, not the owner's


async def test_agent_gh_enabled_but_no_token_does_not_borrow_owner(
    agent: AgentCore, monkeypatch
) -> None:
    # gh switched on but no token stored for this agent → no GH_TOKEN at all,
    # rather than silently inheriting the owner's.
    atlas = Agent(name="atlas", tool_config={"gh": {"enabled": True}})
    cap = await _run(agent, atlas, monkeypatch)
    assert "GH_TOKEN" not in cap["tool_env"]


async def test_agent_gh_disabled_strips_owner_token(agent: AgentCore, monkeypatch) -> None:
    lingua = Agent(name="lingua", tool_config={"gh": {"enabled": False}})
    cap = await _run(agent, lingua, monkeypatch)
    assert "GH_TOKEN" not in cap["tool_env"]


async def test_agent_browser_profile_injected(agent: AgentCore, monkeypatch) -> None:
    hopper = Agent(name="hopper", tool_config={"browser": {"enabled": True, "profile": "hop"}})
    cap = await _run(agent, hopper, monkeypatch)
    assert cap["tool_env"]["BROWSER_PROFILE"] == "hop"


async def test_agent_without_tool_config_inherits(agent: AgentCore, monkeypatch) -> None:
    # An agent that never configured tools still gets the owner token (migration:
    # unchanged behaviour until you opt an agent in).
    plain = Agent(name="plain")
    cap = await _run(agent, plain, monkeypatch)
    assert cap["tool_env"]["GH_TOKEN"] == "owner-token"


async def test_agent_gh_token_secret_reference(agent: AgentCore, monkeypatch) -> None:
    # token_secret reuses an existing infra-vault secret instead of a per-agent
    # copy — e.g. a shared PAT the owner already stored.
    await agent.secret_store.set_infra_secret("SHARED_PAT", "shared-token")
    await agent.secret_store.load_infra_cache()
    atlas = Agent(name="atlas", tool_config={"gh": {"enabled": True, "token_secret": "SHARED_PAT"}})
    cap = await _run(agent, atlas, monkeypatch)
    assert cap["tool_env"]["GH_TOKEN"] == "shared-token"


async def test_subagent_keeps_agent_tool_identity(agent: AgentCore) -> None:
    # A subagent spawned AS an agent must keep that agent's tool identity — else
    # it falls back to the owner's token (the bleed #93 prevents). _narrow_agent
    # narrows skills/tools/secrets but tool_config is identity, copied verbatim.
    parent_state = agent._new_request_state(None)
    enabled = Agent(name="hopper", tool_config={"gh": {"enabled": True}})
    assert agent._narrow_agent(enabled, parent_state).tool_setting("gh") == {"enabled": True}
    # An agent explicitly DENIED gh stays denied as a subagent.
    denied = Agent(name="lingua", tool_config={"gh": {"enabled": False}})
    assert agent._narrow_agent(denied, parent_state).tool_setting("gh") == {"enabled": False}
