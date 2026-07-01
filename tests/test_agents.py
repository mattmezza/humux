"""Tests for the agent engine: store, scoping, and prompt selection."""

from __future__ import annotations

import pytest

from core.agent import scoped_tools
from core.agents import Agent, AgentStore, parse_markdown, to_markdown
from core.config import Config
from core.prompt_builder import build_prompt_sections


def test_parse_frontmatter_only() -> None:
    md = """---
role: Fitness coach
emoji: "🏋️"
voice: en-US-GuyNeural
skills: [scheduling, memory]
tools: [run_command]
secrets: [agent:fitness:key]
personalia: |
  You are Forge.
character: |
  Direct.
---
"""
    p = parse_markdown(md, name="fitness")
    assert p.role == "Fitness coach"
    assert p.voice == "en-US-GuyNeural"
    assert p.skills == ["scheduling", "memory"]
    assert p.tools == ["run_command"]
    assert p.secrets == ["agent:fitness:key"]
    # #98: legacy personalia folds into character (prepended), so both land there.
    assert "Forge" in p.character and "Direct" in p.character
    assert p.character.index("Forge") < p.character.index("Direct")


def test_parse_body_appended_to_character() -> None:
    md = "---\nrole: X\ncharacter: Base.\n---\nExtra body prose."
    p = parse_markdown(md, name="x")
    assert "Base." in p.character and "Extra body prose." in p.character


def test_markdown_roundtrip() -> None:
    p = Agent(
        name="t",
        role="R",
        emoji="🤖",
        voice="en-GB-SoniaNeural",
        character="How.",
        skills=["memory"],
        tools=["send_message"],
        secrets=[],
    )
    p2 = parse_markdown(to_markdown(p), name="t")
    assert (p2.role, p2.voice, p2.skills, p2.tools) == (p.role, p.voice, p.skills, p.tools)
    assert p2.character.strip() == "How."


def test_allow_semantics() -> None:
    blank = Agent(name="d")  # empty allowlists = everything
    assert blank.allows_skill("anything") and blank.allows_tool("anything")
    scoped = Agent(name="s", skills=["memory"], tools=["run_command"])
    assert scoped.allows_skill("memory") and not scoped.allows_skill("email")
    assert scoped.allows_tool("run_command") and not scoped.allows_tool("send_email")


def test_scoped_tools_filters_but_keeps_load_skill() -> None:
    from core.agent import TOOLS

    assert scoped_tools(None) is TOOLS  # no agent = all tools
    p = Agent(name="s", tools=["run_command"])
    names = {t["name"] for t in scoped_tools(p)}
    assert "run_command" in names
    assert "send_email" not in names
    assert "load_skill" in names  # always retained — core mechanic


def test_gateable_tools_in_sync_with_tools() -> None:
    # The admin UI lists GATEABLE_TOOLS for the scope checkboxes; it must stay
    # in sync with the real tool set (every tool except the always-on ones:
    # load_skill, the vault discovery/request tools — issue #19 — and
    # recall_memory / remember, which mirror always-on scoped memory access — #47/#13).
    from api.admin import GATEABLE_TOOLS
    from core.agent import TOOLS

    always_on = {
        "load_skill",
        "search_skills",
        "list_skills",
        "recall_memory",
        "remember",
        "list_secrets",
        "request_secret",
    }
    assert set(GATEABLE_TOOLS) | always_on == {t["name"] for t in TOOLS}


def test_prompt_uses_agent_identity() -> None:
    cfg = Config()
    cfg.agent.name = "Clio"
    cfg.agent.character = "DEFAULT-CHARACTER"
    agent = Agent(
        name="coach",
        agent_name="Forge",
        role="Fitness coach",
        character="AGENT-CH",
    )
    sections = build_prompt_sections(
        config=cfg,
        history_mode="injection",
        skills_index="",
        memories="",
        reflections="",
        decomposed_goal=None,
        agent=agent,
    )
    full = sections.full_prompt
    assert "AGENT-CH" in full
    assert "DEFAULT-CHARACTER" not in full
    assert "Fitness coach" in full  # active-role line
    assert "You are Forge" in full  # agent agent_name overrides global name
    assert "You are Clio" not in full

    # No agent → configured identity, unchanged behaviour.
    default = build_prompt_sections(
        config=cfg,
        history_mode="injection",
        skills_index="",
        memories="",
        reflections="",
        decomposed_goal=None,
    )
    assert "DEFAULT-CHARACTER" in default.full_prompt


@pytest.mark.asyncio
async def test_store_seed_lists_files(tmp_path) -> None:
    (tmp_path / "coach.md").write_text("---\nrole: Coach\nskills: [memory]\n---\n")
    store = AgentStore(db_path=str(tmp_path / "p.db"), seed_dir=tmp_path)
    listed = await store.list_agents()
    assert [p.name for p in listed] == ["coach"]
    assert (await store.get("coach")).role == "Coach"


@pytest.mark.asyncio
async def test_store_crud(tmp_path) -> None:
    # No seed dir, so delete is not undone by re-seeding.
    store = AgentStore(db_path=str(tmp_path / "p.db"), seed_dir=tmp_path / "missing")
    await store.upsert(Agent(name="coach", role="Coach", skills=["memory"]))
    assert (await store.get("coach")).role == "Coach"

    await store.upsert(Agent(name="coach", role="Updated", skills=["memory", "weather"]))
    got = await store.get("coach")
    assert got.role == "Updated" and got.skills == ["memory", "weather"]

    assert await store.delete("coach") is True
    assert await store.get("coach") is None


@pytest.mark.asyncio
async def test_store_rename(tmp_path) -> None:
    store = AgentStore(db_path=str(tmp_path / "p.db"), seed_dir=tmp_path / "missing")
    await store.upsert(Agent(name="coach", role="Coach", skills=["memory"]))

    # Happy path: the row moves to the new slug, fields intact.
    assert await store.rename("coach", "trainer") is True
    assert await store.get("coach") is None
    moved = await store.get("trainer")
    assert moved is not None and moved.role == "Coach" and moved.skills == ["memory"]

    # Renaming a missing slug is a no-op (False), not an error.
    assert await store.rename("ghost", "whoever") is False

    # Collision with an existing slug is rejected.
    await store.upsert(Agent(name="writer", role="Writer"))
    with pytest.raises(ValueError):
        await store.rename("trainer", "writer")
    # Both survive the rejected rename.
    assert (await store.get("trainer")).role == "Coach"
    assert (await store.get("writer")).role == "Writer"


@pytest.mark.asyncio
async def test_rename_seeded_agent_does_not_reseed_old_slug(tmp_path) -> None:
    """Renaming a *seeded* agent must not leave the old slug to be re-seeded
    from its gallery file: that resurrected it as a duplicate copy (#102). A
    tombstone on the old slug suppresses the re-seed."""
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "fitness-coach.md").write_text("---\nrole: Fitness coach\n---\n")
    store = AgentStore(db_path=str(tmp_path / "p.db"), seed_dir=seed)
    assert [p.name for p in await store.list_agents()] == ["fitness-coach"]

    await store.rename("fitness-coach", "my-coach")
    # Only the renamed row survives — the old stem is not re-seeded.
    assert {p.name for p in await store.list_agents()} == {"my-coach"}


@pytest.mark.asyncio
async def test_delete_seeded_agent_does_not_reseed(tmp_path) -> None:
    """Deleting a *seeded* agent must actually remove it, not have it re-seeded
    from its gallery file on the next list (#102)."""
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "fitness-coach.md").write_text("---\nrole: Fitness coach\n---\n")
    store = AgentStore(db_path=str(tmp_path / "p.db"), seed_dir=seed)
    assert [p.name for p in await store.list_agents()] == ["fitness-coach"]

    assert await store.delete("fitness-coach") is True
    assert [p.name for p in await store.list_agents()] == []

    # Re-creating the slug deliberately clears the tombstone, so a later edit sticks.
    await store.upsert(Agent(name="fitness-coach", role="Back"))
    assert (await store.get("fitness-coach")).role == "Back"
    assert [p.name for p in await store.list_agents()] == ["fitness-coach"]
