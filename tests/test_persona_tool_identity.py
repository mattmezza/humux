"""Per-persona tool identity (#93): the active persona's own credentials are
injected into ``run_command``, and never the owner's when it opts out.

These exercise the real dispatch path in ``AgentCore._execute_tool`` with a
stubbed executor that captures the per-call ``tool_env`` override.
"""

from __future__ import annotations

import pytest

from core.agent import AgentCore
from core.config import Config
from core.llm import LLMToolCall
from core.permissions import PermissionLevel
from core.personae import Persona
from core.secret_store import SecretStore
from core.tools import gh_token_secret_name
from core.vault import InfraVault


@pytest.fixture
async def agent(tmp_path) -> AgentCore:
    # Infra vault keyed so per-persona tokens are storable/resolvable headlessly.
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


async def _run(agent: AgentCore, persona: Persona | None, monkeypatch) -> dict:
    captured: dict = {}

    async def fake_exec(command, timeout, cwd=None, tool_env=None):
        captured["tool_env"] = tool_env
        return {"stdout": "", "stderr": "", "exit_code": 0}

    monkeypatch.setattr(agent.executor, "_exec", fake_exec)
    monkeypatch.setattr(agent.permissions, "check", lambda *a, **k: PermissionLevel.ALWAYS)
    rs = agent._new_request_state(persona)
    call = LLMToolCall(
        id="1", name="run_command", arguments={"command": "gh issue list", "purpose": "p"}
    )
    await agent._execute_tool(call, "system", "u", rs)
    return captured


async def test_no_persona_uses_shared_default(agent: AgentCore, monkeypatch) -> None:
    # No persona → no per-call override; the executor's shared default applies.
    cap = await _run(agent, None, monkeypatch)
    assert cap["tool_env"] is None


async def test_persona_with_own_token(agent: AgentCore, monkeypatch) -> None:
    hopper = Persona(name="hopper", tool_config={"gh": {"enabled": True}})
    cap = await _run(agent, hopper, monkeypatch)
    assert cap["tool_env"]["GH_TOKEN"] == "hopper-token"  # its own, not the owner's


async def test_persona_gh_enabled_but_no_token_does_not_borrow_owner(
    agent: AgentCore, monkeypatch
) -> None:
    # gh switched on but no token stored for this persona → no GH_TOKEN at all,
    # rather than silently inheriting the owner's.
    atlas = Persona(name="atlas", tool_config={"gh": {"enabled": True}})
    cap = await _run(agent, atlas, monkeypatch)
    assert "GH_TOKEN" not in cap["tool_env"]


async def test_persona_gh_disabled_strips_owner_token(agent: AgentCore, monkeypatch) -> None:
    lingua = Persona(name="lingua", tool_config={"gh": {"enabled": False}})
    cap = await _run(agent, lingua, monkeypatch)
    assert "GH_TOKEN" not in cap["tool_env"]


async def test_persona_browser_profile_injected(agent: AgentCore, monkeypatch) -> None:
    hopper = Persona(name="hopper", tool_config={"browser": {"enabled": True, "profile": "hop"}})
    cap = await _run(agent, hopper, monkeypatch)
    assert cap["tool_env"]["BROWSER_PROFILE"] == "hop"


async def test_persona_without_tool_config_inherits(agent: AgentCore, monkeypatch) -> None:
    # A persona that never configured tools still gets the owner token (migration:
    # unchanged behaviour until you opt a persona in).
    plain = Persona(name="plain")
    cap = await _run(agent, plain, monkeypatch)
    assert cap["tool_env"]["GH_TOKEN"] == "owner-token"


async def test_persona_gh_token_secret_reference(agent: AgentCore, monkeypatch) -> None:
    # token_secret reuses an existing infra-vault secret instead of a per-persona
    # copy — e.g. a shared PAT the owner already stored.
    await agent.secret_store.set_infra_secret("SHARED_PAT", "shared-token")
    await agent.secret_store.load_infra_cache()
    atlas = Persona(
        name="atlas", tool_config={"gh": {"enabled": True, "token_secret": "SHARED_PAT"}}
    )
    cap = await _run(agent, atlas, monkeypatch)
    assert cap["tool_env"]["GH_TOKEN"] == "shared-token"


async def test_subagent_keeps_persona_tool_identity(agent: AgentCore) -> None:
    # A subagent spawned AS a persona must keep that persona's tool identity — else
    # it falls back to the owner's token (the bleed #93 prevents). _narrow_persona
    # narrows skills/tools/secrets but tool_config is identity, copied verbatim.
    parent_state = agent._new_request_state(None)
    enabled = Persona(name="hopper", tool_config={"gh": {"enabled": True}})
    assert agent._narrow_persona(enabled, parent_state).tool_setting("gh") == {"enabled": True}
    # A persona explicitly DENIED gh stays denied as a subagent.
    denied = Persona(name="lingua", tool_config={"gh": {"enabled": False}})
    assert agent._narrow_persona(denied, parent_state).tool_setting("gh") == {"enabled": False}
