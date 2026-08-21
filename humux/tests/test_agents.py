"""Tests for the agent engine: store, scoping, and prompt selection."""

from __future__ import annotations

import pytest

from core.agent import scoped_tools
from core.agents import Agent, AgentStore, parse_markdown, to_markdown
from core.config import Config
from core.executor import ToolExecutor
from core.prompt_builder import build_prompt_sections, display_prefixes


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
    assert p.tools == ["bash"]
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
        voice="en-GB-SoniaNeural",
        character="How.",
        skills=["memory"],
        tools=["send_message"],
        secrets=[],
    )
    p2 = parse_markdown(to_markdown(p), name="t")
    assert (p2.role, p2.voice, p2.skills, p2.tools) == (p.role, p.voice, p.skills, p.tools)
    assert p2.character.strip() == "How."


def test_spawnable_agents_roundtrip_and_coercion() -> None:
    # Frontmatter list, plus the form-input shapes _as_list must coerce.
    p = parse_markdown("---\nspawnable_agents: [qa, legal]\n---\n", name="boss")
    assert p.spawnable_agents == ["qa", "legal"]
    assert parse_markdown("---\nspawnable_agents: qa, legal\n---\n", name="b").spawnable_agents == [
        "qa",
        "legal",
    ]
    assert parse_markdown(
        "---\nspawnable_agents: |\n  qa\n  legal\n---\n", name="b"
    ).spawnable_agents == ["qa", "legal"]
    assert parse_markdown("---\nrole: X\n---\n", name="b").spawnable_agents == []  # absent = []
    # Survives the markdown round-trip.
    back = parse_markdown(to_markdown(p), name="boss")
    assert back.spawnable_agents == ["qa", "legal"]


@pytest.mark.asyncio
async def test_store_roundtrips_spawnable_agents(tmp_path) -> None:
    store = AgentStore(db_path=str(tmp_path / "p.db"), seed_dir=tmp_path / "missing")
    await store.upsert(Agent(name="boss", spawnable_agents=["qa", "legal"]))
    assert (await store.get("boss")).spawnable_agents == ["qa", "legal"]
    # Clearing the team sticks (upsert overwrites, not merges).
    await store.upsert(Agent(name="boss", spawnable_agents=[]))
    assert (await store.get("boss")).spawnable_agents == []


def test_allow_semantics() -> None:
    blank = Agent(name="d")  # empty allowlists = everything
    assert blank.allows_skill("anything") and blank.allows_tool("anything")
    scoped = Agent(name="s", skills=["memory"], tools=["bash"])
    assert scoped.allows_skill("memory") and not scoped.allows_skill("email")
    assert scoped.allows_tool("bash") and not scoped.allows_tool("send_email")


def test_scoped_tools_filters_but_keeps_memory_and_vault() -> None:
    from core.agent import TOOLS

    assert scoped_tools(None) is TOOLS  # no agent = all tools
    p = Agent(name="s", tools=["bash"])
    names = {t["name"] for t in scoped_tools(p)}
    assert "bash" in names
    assert "send_email" not in names
    assert "recall_memory" in names  # always retained — core mechanic


def test_legacy_tool_names_normalized() -> None:
    # Agent docs saved before the #178 rename keep working: execution tools map
    # 1:1; the read-only list_dir/grep and the removed skill tools DROP OUT —
    # never widened to the execution-capable bash.
    p = Agent(
        name="old",
        tools=[
            "run_command",
            "read_file",
            "write_file",
            "edit_file",
            "list_dir",
            "grep",
            "load_skill",
            "send_email",
        ],
    )
    assert p.tools == ["bash", "read", "write", "edit", "send_email"]
    assert p.allows_tool("bash") and p.allows_tool("read") and p.allows_tool("send_email")
    assert not p.allows_tool("spawn_subagent")


def test_legacy_scope_of_only_removed_tools_stays_restrictive() -> None:
    # An agent scoped to ONLY removed tools must not widen to "all" (tools == []).
    p = Agent(name="locked", tools=["load_skill", "search_skills", "list_dir", "grep"])
    assert p.tools == ["remember"]
    assert not p.allows_tool("bash")
    assert not p.allows_tool("send_email")


def test_gateable_tools_in_sync_with_tools() -> None:
    # The admin UI lists GATEABLE_TOOLS for the scope checkboxes; it must stay
    # in sync with the real tool set (every tool except the always-on ones:
    # the vault discovery/request tools — issue #19 — and recall_memory /
    # remember, which mirror always-on scoped memory access — #47/#13).
    from api.admin import GATEABLE_TOOLS
    from core.agent import TOOLS

    always_on = {
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
        decomposed_goal=None,
    )
    assert "DEFAULT-CHARACTER" in default.full_prompt


def test_workspace_section_namespaces_by_slug() -> None:
    # #149/#151: harness on → a <workspace> block appears, exposes the root path,
    # and namespaces under the agent slug; off → no block.
    cfg = Config()
    cfg.workspace.enabled = True
    cfg.workspace.directory = "/data/ws"

    def build(agent):
        return build_prompt_sections(
            config=cfg,
            history_mode="injection",
            skills_index="",
            memories="",
            decomposed_goal=None,
            agent=agent,
        ).workspace

    scoped = build(Agent(name="coach"))
    assert "/data/ws" in scoped  # CWD affordance exposed
    assert "coach/" in scoped  # namespaced under the slug

    assert "default/" in build(None)  # no agent → "default" fallback

    cfg.workspace.enabled = False
    assert build(Agent(name="coach")) == ""  # harness off → no block


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


def _prompt(cfg, agent=None) -> str:
    return build_prompt_sections(
        config=cfg,
        history_mode="injection",
        skills_index="",
        memories="",
        decomposed_goal=None,
        agent=agent,
    ).full_prompt


def test_delegation_block_present_and_gated() -> None:
    """The tool schemas say HOW to spawn; the orchestration skill says WHEN — but
    the model only reads that skill if it already suspects fan-out. This block is
    the trigger, so it has to be in the prompt whenever spawning is possible."""
    cfg = Config()
    cfg.subagents.enabled = True
    assert "<delegation>" in _prompt(cfg)

    cfg.subagents.enabled = False
    assert "<delegation>" not in _prompt(cfg)

    cfg.subagents.enabled = True
    walled_off = Agent(name="solo", tools=["bash"])  # allowlist without spawn_subagent
    assert "<delegation>" not in _prompt(cfg, walled_off)


def test_allowlist_note_dedup_never_widens_the_advertised_set() -> None:
    """The note is display-only de-dup: every prefix it advertises must be one the
    executor actually permits, and no genuinely distinct prefix may be hidden."""
    allowed = ToolExecutor.ALLOWED_PREFIXES
    shown = display_prefixes()

    # Nothing advertised that the guard would reject (expand the `a|b X` folds).
    for entry in shown:
        head, _, rest = entry.partition(" ")
        for variant in [f"{name} {rest}".strip() for name in head.split("|")]:
            assert variant in allowed, variant

    # Nothing genuinely distinct hidden: every real prefix is still covered by a
    # displayed one (either itself or a shorter prefix that subsumes it).
    flat = [
        f"{name} {e.partition(' ')[2]}".strip()
        for e in shown
        for name in e.partition(" ")[0].split("|")
    ]
    for prefix in allowed:
        assert any(prefix.startswith(v) for v in flat), prefix

    # And it actually de-duplicated: the subsumed tools/ scripts are gone.
    assert "python3 /app/tools/skills.py" not in shown
    assert "python3|python /app/skills/" in shown


def test_voice_block_gated_on_channel() -> None:
    """TTS on is not enough: a channel that can't play audio (cli) must not be told
    it can speak. Unknown/default ("") keeps the historical behaviour."""
    cfg = Config()
    cfg.voice.tts_enabled = True

    def voice_for(channel: str) -> str:
        return build_prompt_sections(
            config=cfg,
            history_mode="injection",
            channel=channel,
            skills_index="",
            memories="",
            decomposed_goal=None,
        ).voice

    assert "<voice>" in voice_for("")  # admin preview / tests
    assert "<voice>" in voice_for("telegram")
    assert "<voice>" in voice_for("telegram:coach")
    assert voice_for("cli") == ""
    assert voice_for("system") == ""

    cfg.voice.tts_enabled = False
    assert voice_for("telegram") == ""
