"""Tests for the persona engine: store, scoping, and prompt selection."""

from __future__ import annotations

import pytest

from core.agent import scoped_tools
from core.config import Config
from core.personae import Persona, PersonaStore, parse_markdown, to_markdown
from core.prompt_builder import build_prompt_sections


def test_parse_frontmatter_only() -> None:
    md = """---
role: Fitness coach
emoji: "🏋️"
voice: en-US-GuyNeural
skills: [scheduling, memory]
tools: [run_command]
secrets: [persona:fitness:key]
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
    assert p.secrets == ["persona:fitness:key"]
    assert "Forge" in p.personalia and "Direct" in p.character


def test_parse_body_appended_to_character() -> None:
    md = "---\nrole: X\ncharacter: Base.\n---\nExtra body prose."
    p = parse_markdown(md, name="x")
    assert "Base." in p.character and "Extra body prose." in p.character


def test_markdown_roundtrip() -> None:
    p = Persona(
        name="t",
        role="R",
        emoji="🤖",
        voice="en-GB-SoniaNeural",
        personalia="Who.",
        character="How.",
        skills=["memory"],
        tools=["send_message"],
        secrets=[],
    )
    p2 = parse_markdown(to_markdown(p), name="t")
    assert (p2.role, p2.voice, p2.skills, p2.tools) == (p.role, p.voice, p.skills, p.tools)
    assert p2.personalia.strip() == "Who." and p2.character.strip() == "How."


def test_allow_semantics() -> None:
    blank = Persona(name="d")  # empty allowlists = everything
    assert blank.allows_skill("anything") and blank.allows_tool("anything")
    scoped = Persona(name="s", skills=["memory"], tools=["run_command"])
    assert scoped.allows_skill("memory") and not scoped.allows_skill("email")
    assert scoped.allows_tool("run_command") and not scoped.allows_tool("send_email")


def test_scoped_tools_filters_but_keeps_load_skill() -> None:
    from core.agent import TOOLS

    assert scoped_tools(None) is TOOLS  # no persona = all tools
    p = Persona(name="s", tools=["run_command"])
    names = {t["name"] for t in scoped_tools(p)}
    assert "run_command" in names
    assert "send_email" not in names
    assert "load_skill" in names  # always retained — core mechanic


def test_gateable_tools_in_sync_with_tools() -> None:
    # The admin UI lists GATEABLE_TOOLS for the scope checkboxes; it must stay
    # in sync with the real tool set (every tool except the always-on ones:
    # load_skill, the vault discovery/request tools — issue #19 — and
    # recall_memory, which mirrors always-on scoped memory injection — #47).
    from api.admin import GATEABLE_TOOLS
    from core.agent import TOOLS

    always_on = {
        "load_skill",
        "search_skills",
        "list_skills",
        "recall_memory",
        "list_secrets",
        "request_secret",
    }
    assert set(GATEABLE_TOOLS) | always_on == {t["name"] for t in TOOLS}


def test_prompt_uses_persona_identity() -> None:
    cfg = Config()
    cfg.agent.name = "Clio"
    cfg.agent.personalia = "DEFAULT-PERSONALIA"
    cfg.agent.character = "DEFAULT-CHARACTER"
    persona = Persona(
        name="coach",
        agent_name="Forge",
        role="Fitness coach",
        personalia="PERSONA-ID",
        character="PERSONA-CH",
    )
    sections = build_prompt_sections(
        config=cfg,
        history_mode="injection",
        skills_index="",
        memories="",
        reflections="",
        decomposed_goal=None,
        persona=persona,
    )
    full = sections.full_prompt
    assert "PERSONA-ID" in full and "PERSONA-CH" in full
    assert "DEFAULT-PERSONALIA" not in full and "DEFAULT-CHARACTER" not in full
    assert "Fitness coach" in full  # active-role line
    assert "You are Forge" in full  # persona agent_name overrides global name
    assert "You are Clio" not in full

    # No persona → configured identity, unchanged behaviour.
    default = build_prompt_sections(
        config=cfg,
        history_mode="injection",
        skills_index="",
        memories="",
        reflections="",
        decomposed_goal=None,
    )
    assert "DEFAULT-PERSONALIA" in default.full_prompt


@pytest.mark.asyncio
async def test_store_seed_lists_files(tmp_path) -> None:
    (tmp_path / "coach.md").write_text("---\nrole: Coach\nskills: [memory]\n---\n")
    store = PersonaStore(db_path=str(tmp_path / "p.db"), seed_dir=tmp_path)
    listed = await store.list_personae()
    assert [p.name for p in listed] == ["coach"]
    assert (await store.get("coach")).role == "Coach"


@pytest.mark.asyncio
async def test_store_crud(tmp_path) -> None:
    # No seed dir, so delete is not undone by re-seeding.
    store = PersonaStore(db_path=str(tmp_path / "p.db"), seed_dir=tmp_path / "missing")
    await store.upsert(Persona(name="coach", role="Coach", skills=["memory"]))
    assert (await store.get("coach")).role == "Coach"

    await store.upsert(Persona(name="coach", role="Updated", skills=["memory", "weather"]))
    got = await store.get("coach")
    assert got.role == "Updated" and got.skills == ["memory", "weather"]

    assert await store.delete("coach") is True
    assert await store.get("coach") is None


@pytest.mark.asyncio
async def test_store_rename(tmp_path) -> None:
    store = PersonaStore(db_path=str(tmp_path / "p.db"), seed_dir=tmp_path / "missing")
    await store.upsert(Persona(name="coach", role="Coach", skills=["memory"]))

    # Happy path: the row moves to the new slug, fields intact.
    assert await store.rename("coach", "trainer") is True
    assert await store.get("coach") is None
    moved = await store.get("trainer")
    assert moved is not None and moved.role == "Coach" and moved.skills == ["memory"]

    # Renaming a missing slug is a no-op (False), not an error.
    assert await store.rename("ghost", "whoever") is False

    # Collision with an existing slug is rejected.
    await store.upsert(Persona(name="writer", role="Writer"))
    with pytest.raises(ValueError):
        await store.rename("trainer", "writer")
    # Both survive the rejected rename.
    assert (await store.get("trainer")).role == "Coach"
    assert (await store.get("writer")).role == "Writer"


@pytest.mark.asyncio
async def test_rename_seeded_persona_reseeds_old_slug(tmp_path) -> None:
    """Characterise (not endorse) the gallery seam: seeding keys off the file
    stem, so renaming a *seeded* persona leaves the old slug to be re-seeded on
    the next list. User-created personae have no seed file, so they rename clean.
    """
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "fitness-coach.md").write_text("---\nrole: Fitness coach\n---\n")
    store = PersonaStore(db_path=str(tmp_path / "p.db"), seed_dir=seed)
    assert [p.name for p in await store.list_personae()] == ["fitness-coach"]

    await store.rename("fitness-coach", "my-coach")
    # list_personae re-seeds the now-missing gallery stem alongside the renamed row.
    assert {p.name for p in await store.list_personae()} == {"fitness-coach", "my-coach"}
