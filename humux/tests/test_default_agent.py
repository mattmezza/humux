"""The fallback "default" agent is a normal agent row flagged with ``is_default``
(the #115-follow-up: identity by property, not a reserved slug). It renames,
reassigns and deletes like any agent, and runs with its OWN scope + tools — being
default is purely which agent answers when none is selected."""

from __future__ import annotations

import pytest

from core.agent import AgentCore, _agent_scope
from core.agents import (
    SEED_DEFAULT_SLUG,
    Agent,
    AgentStore,
    default_agent_from_values,
)
from core.config import Config


def test_default_agent_has_normal_scope():
    # A normal agent (incl. the default) scopes memory/permissions to its slug;
    # only a bare None (no agent at all) is the shared "" scope.
    assert _agent_scope(None) == ""
    assert _agent_scope(Agent(name="assistant", is_default=True)) == "assistant"
    assert _agent_scope(Agent(name="coach")) == "coach"


async def test_seed_flags_one_default(tmp_path):
    store = AgentStore(
        db_path=str(tmp_path / "a.db"),
        seed_dir=None,
        default_identity=default_agent_from_values(character="Base", agent_name="Aria"),
    )
    d = await store.get_default()
    assert d is not None and d.is_default and d.name == SEED_DEFAULT_SLUG
    assert d.character == "Base" and d.agent_name == "Aria"


async def test_seed_idempotent_no_overwrite(tmp_path):
    db = str(tmp_path / "a.db")
    ident = lambda: default_agent_from_values(character="Base")  # noqa: E731
    store = AgentStore(db_path=db, seed_dir=None, default_identity=ident())
    d = await store.get_default()
    d.character = "Edited"
    await store.upsert(d)  # upsert must not clear the is_default flag
    store2 = AgentStore(db_path=db, seed_dir=None, default_identity=ident())
    again = await store2.get_default()
    assert again.character == "Edited" and again.is_default


async def test_seed_skips_if_slug_taken(tmp_path):
    # A custom agent already using the seed slug is never clobbered.
    store = AgentStore(
        db_path=str(tmp_path / "a.db"),
        seed_dir=None,
        default_identity=default_agent_from_values(character="Base"),
    )
    await store.upsert(Agent(name=SEED_DEFAULT_SLUG, character="Mine"))
    await store.set_default("")  # ensure nothing is flagged
    # ensure_seeded runs on list; it must not overwrite the existing slug.
    names = {a.name for a in await store.list_agents()}
    assert names == {SEED_DEFAULT_SLUG}
    assert (await store.get(SEED_DEFAULT_SLUG)).character == "Mine"


async def test_seed_respects_tombstone(tmp_path):
    db = str(tmp_path / "a.db")
    ident = lambda: default_agent_from_values(character="Base")  # noqa: E731
    store = AgentStore(db_path=db, seed_dir=None, default_identity=ident())
    assert await store.get_default() is not None
    await store.delete(SEED_DEFAULT_SLUG)  # deliberately remove the default
    store2 = AgentStore(db_path=db, seed_dir=None, default_identity=ident())
    assert await store2.get_default() is None  # not resurrected (#102 tombstone)


async def test_set_default_moves_flag(tmp_path):
    store = AgentStore(
        db_path=str(tmp_path / "a.db"),
        seed_dir=None,
        default_identity=default_agent_from_values(),
    )
    await store.upsert(Agent(name="coach"))
    assert await store.set_default("coach") is True
    flagged = [a.name for a in await store.list_agents() if a.is_default]
    assert flagged == ["coach"]  # exactly one, reassigned
    assert (await store.get_default()).name == "coach"


async def test_set_default_rejects_empty_and_unknown(tmp_path):
    store = AgentStore(
        db_path=str(tmp_path / "a.db"),
        seed_dir=None,
        default_identity=default_agent_from_values(),
    )
    assert await store.set_default("") is False  # there is always exactly one default
    assert await store.set_default("ghost") is False  # no such agent


@pytest.fixture
def core(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = Config()
    cfg.agent.character = "You are Base."
    cfg.agent.name = "Aria"
    cfg.agent.llm_provider = "deepseek"
    cfg.memory.embedding.enabled = False
    return AgentCore(cfg)


async def test_resolve_falls_back_to_flagged_default(core):
    resolved = await core._resolve_agent("telegram", "u", "c")
    assert resolved is not None and resolved.is_default and resolved.name == SEED_DEFAULT_SLUG
    assert resolved.character == "You are Base." and resolved.agent_name == "Aria"
    # Normal agent → its scope is its slug, not the shared "".
    rs = core._new_request_state(resolved)
    assert rs["agent_name"] == SEED_DEFAULT_SLUG


async def test_default_is_renamable_and_stays_default(core):
    await core.agents.get_default()  # seed the default first
    await core.agents.rename(SEED_DEFAULT_SLUG, "aria")
    d = await core.agents.get_default()
    assert d is not None and d.name == "aria" and d.is_default  # flag travels with the row


async def test_default_skipped_by_110_account_grant(tmp_path):
    # The #110 one-time grant preserves pre-existing agents' any-account access;
    # the newly-seeded default is NOT pre-existing, so it stays #110-gated (its
    # bindings come only from config / its editor).
    from core.agents import bind_existing_accounts

    store = AgentStore(
        db_path=str(tmp_path / "a.db"),
        seed_dir=None,
        default_identity=default_agent_from_values(),
    )
    await store.upsert(Agent(name="coach"))
    n = await bind_existing_accounts(store, ["work", "personal"], ["gcal"])
    assert n == 1  # only the unbound custom agent was granted
    assert (await store.get_default()).email_accounts == []  # default untouched
    assert (await store.get("coach")).email_access("work") == "read_write"


# --- kill-switch (#115 flw): a disabled agent processes nothing ---------------


async def test_set_enabled_toggles_and_upsert_preserves(tmp_path):
    store = AgentStore(
        db_path=str(tmp_path / "a.db"), seed_dir=None, default_identity=default_agent_from_values()
    )
    d = await store.get_default()
    assert d.enabled is True  # agents start enabled
    assert await store.set_enabled(d.name, False) is True
    assert (await store.get(d.name)).enabled is False
    # Editing the agent (upsert) must NOT flip the on/off state back.
    edited = await store.get(d.name)
    edited.character = "changed"
    await store.upsert(edited)
    assert (await store.get(d.name)).enabled is False
    assert await store.set_enabled("ghost", True) is False


async def test_disabled_agent_processes_nothing(core):
    d = await core.agents.get_default()
    await core.agents.set_enabled(d.name, False)

    class _Boom:  # if the gate fails, the turn would reach the LLM
        provider = "deepseek"

        async def generate(self, **_kw):
            raise AssertionError("a disabled agent must not call the LLM")

    core.llm = _Boom()
    resp = await core.process(message="hi", channel="telegram", user_id="u", chat_id="c")
    assert resp.text == ""  # dropped, silent

    # Even the command shortcuts are dropped when disabled — no /new clear, no
    # /yolo flip (the gate sits BEFORE those branches).
    flipped = []
    core.permissions.set_yolo = lambda *a, **k: flipped.append(a)  # type: ignore[assignment]

    async def _cmd(msg):
        return await core.process(message=msg, channel="telegram", user_id="u", chat_id="c")

    assert (await _cmd("/new")).text == ""  # not "Conversation cleared."
    assert (await _cmd("/yolo-on")).text == ""
    assert flipped == []  # YOLO never flipped for a disabled agent

    # Re-enabling lets it process again (now the scripted LLM would run).
    await core.agents.set_enabled(d.name, True)
    assert (await core.agents.get_default()).enabled is True


async def test_disabled_agent_cannot_be_spawned(core):
    await core.agents.upsert(Agent(name="coach"))
    await core.agents.set_enabled("coach", False)
    core.config.subagents.enabled = True
    result = await core.run_subagent(task="x", agent_name="coach")
    assert "disabled" in result["error"].lower()
