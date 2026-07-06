"""Tests for SkillsEngine and the skill file validator."""

from __future__ import annotations

import pytest

from core.skills import (
    SkillsEngine,
    SkillsStore,
    ValidationError,
    validate_skill_dir,
    validate_skill_file,
)


@pytest.mark.asyncio
async def test_get_index_block_empty_db(tmp_path) -> None:
    db_path = str(tmp_path / "skills.db")
    engine = SkillsEngine(db_path=db_path, seed_dir=tmp_path)
    assert await engine.get_index_block() == ""


@pytest.mark.asyncio
async def test_get_index_block_lists_seeded_skills(tmp_path) -> None:
    (tmp_path / "alpha.md").write_text("Alpha skill")
    (tmp_path / "beta.md").write_text("Beta skill")

    db_path = str(tmp_path / "skills.db")
    engine = SkillsEngine(db_path=db_path, seed_dir=tmp_path)
    index = await engine.get_index_block()

    assert '<skill name="alpha">Alpha skill</skill>' in index
    assert '<skill name="beta">Beta skill</skill>' in index
    assert "skills.py show" in index  # loading instructions ride with the index


@pytest.mark.asyncio
async def test_get_skill_content_reads_seeded_skill(tmp_path) -> None:
    (tmp_path / "memory.md").write_text("# Memory\n\nUse sqlite3.")
    db_path = str(tmp_path / "skills.db")
    engine = SkillsEngine(db_path=db_path, seed_dir=tmp_path)

    content = await engine.get_skill_content("memory")

    assert "Use sqlite3." in content


# --- index_entries (backs the index block, admin view, Telegram commands) ---


def _engine_with(tmp_path, **skills) -> SkillsEngine:
    for name, summary in skills.items():
        (tmp_path / f"{name}.md").write_text(summary)
    return SkillsEngine(db_path=str(tmp_path / "skills.db"), seed_dir=tmp_path)


@pytest.mark.asyncio
async def test_index_entries_returns_name_summary(tmp_path) -> None:
    engine = _engine_with(tmp_path, email="send and read email", weather="fetch the forecast")
    entries = await engine.index_entries()
    by_name = {e["name"]: e["summary"] for e in entries}
    assert by_name == {"email": "send and read email", "weather": "fetch the forecast"}


@pytest.mark.asyncio
async def test_index_entries_scoped_to_allowlist(tmp_path) -> None:
    engine = _engine_with(tmp_path, email="send email", weather="forecast", news="headlines")
    entries = await engine.index_entries(allow=["email", "news"])
    assert {e["name"] for e in entries} == {"email", "news"}


# --- Validator tests ---


def test_valid_h1_title(tmp_path) -> None:
    path = tmp_path / "good.md"
    path.write_text("# My Skill\n\nSome content.\n")
    errors = validate_skill_file(path)
    assert errors == []


def test_missing_h1_title(tmp_path) -> None:
    path = tmp_path / "bad.md"
    path.write_text("Just some text without a title.\n")
    errors = validate_skill_file(path)
    assert any("Missing H1 title" in str(e) for e in errors)


def test_closed_code_blocks(tmp_path) -> None:
    path = tmp_path / "good.md"
    path.write_text("# Skill\n\n```bash\necho hello\n```\n")
    errors = validate_skill_file(path)
    assert errors == []


def test_unclosed_code_block(tmp_path) -> None:
    path = tmp_path / "bad.md"
    path.write_text("# Skill\n\n```bash\necho hello\n")
    errors = validate_skill_file(path)
    assert any("Unclosed" in str(e) for e in errors)


def test_allowed_command_prefix(tmp_path) -> None:
    path = tmp_path / "good.md"
    path.write_text("# Skill\n\n```bash\ncurl -s wttr.in/London\n```\n")
    errors = validate_skill_file(path)
    assert errors == []


def test_disallowed_command_prefix(tmp_path) -> None:
    path = tmp_path / "bad.md"
    path.write_text("# Skill\n\n```bash\nevil_tool --do-bad-things\n```\n")
    errors = validate_skill_file(path)
    assert any("not in the allowed prefix list" in str(e) for e in errors)


def test_piped_allowed_commands(tmp_path) -> None:
    path = tmp_path / "good.md"
    path.write_text("# Skill\n\n```bash\nhimalaya envelope list -o json | jq '.[].subject'\n```\n")
    errors = validate_skill_file(path)
    assert errors == []


def test_known_tool_name_is_valid(tmp_path) -> None:
    path = tmp_path / "good.md"
    path.write_text('# Skill\n\n```bash\nwrite_file(path="test.txt", content="hello")\n```\n')
    errors = validate_skill_file(path)
    assert errors == []


def test_unlabeled_block_not_checked(tmp_path) -> None:
    """Unlabeled code blocks (tables, tool call examples) are not validated."""
    path = tmp_path / "good.md"
    path.write_text(
        "# Skill\n\n```\n| Col1 | Col2 |\n|------|------|\n| A    | B    |\n```\n"
        '\n```\nwrite_file(path="test.txt", content="hello")\n```\n'
    )
    errors = validate_skill_file(path)
    assert errors == []


def test_python3_tool_invocation(tmp_path) -> None:
    path = tmp_path / "good.md"
    path.write_text(
        "# Skill\n\n```bash\npython3 ./tools/browser.py read --url https://example.com\n```\n"
    )
    errors = validate_skill_file(path)
    assert errors == []


def test_python3_c_script(tmp_path) -> None:
    path = tmp_path / "good.md"
    path.write_text('# Skill\n\n```bash\npython3 -c "import sys; print(sys.version)"\n```\n')
    errors = validate_skill_file(path)
    assert errors == []


def test_backslash_continuation(tmp_path) -> None:
    """Continuation lines after \\ should be treated as part of the same command."""
    path = tmp_path / "good.md"
    path.write_text(
        "# Skill\n\n```bash\n"
        "python3 /app/tools/browser.py act --url https://site/login \\"
        '\n  --steps \'[{"click":"#btn"}]\'\n```\n'
    )
    errors = validate_skill_file(path)
    assert errors == []


def test_long_python_invocation(tmp_path) -> None:
    """Multi-line python3 command with continuation."""
    path = tmp_path / "good.md"
    path.write_text(
        "# Skill\n\n```bash\n"
        "python3 /app/tools/skills.py upsert --name weather --stdin --write-seed \\"
        "\n  < skills/weather.md\n```\n"
    )
    errors = validate_skill_file(path)
    assert errors == []


def test_pipe_continuation(tmp_path) -> None:
    """Pipe at end of line is fine."""
    path = tmp_path / "good.md"
    path.write_text(
        "# Skill\n\n```bash\n"
        "himalaya envelope list -a personal -s 10 -o json \\\n  | jq '.[].subject'\n```\n"
    )
    errors = validate_skill_file(path)
    assert errors == []


def test_strict_mode_warns_missing_description(tmp_path) -> None:
    path = tmp_path / "strict.md"
    path.write_text("# Minimal\n\n```bash\necho hi\n```\n")
    errors = validate_skill_file(path, strict=True)
    assert any("Missing description" in str(e) for e in errors)


def test_strict_mode_no_warning_with_description(tmp_path) -> None:
    path = tmp_path / "good.md"
    path.write_text("# Full\n\nDescription paragraph.\n\n```bash\necho hi\n```\n")
    errors = validate_skill_file(path, strict=True)
    assert not any("Missing description" in str(e) for e in errors)


def test_validate_skill_dir(tmp_path) -> None:
    (tmp_path / "a.md").write_text("# A\n\n```bash\necho ok\n```\n")
    (tmp_path / "b.md").write_text("# B\n\n```bash\ncurl example.com\n```\n")
    errors = validate_skill_dir(tmp_path)
    assert errors == []


def test_validate_skill_dir_finds_errors(tmp_path) -> None:
    (tmp_path / "good.md").write_text("# Good\n\n```bash\necho ok\n```\n")
    (tmp_path / "bad.md").write_text("No H1 title\n\n```bash\nunknown_cmd\n```\n")
    errors = validate_skill_dir(tmp_path)
    assert len(errors) >= 2


def test_validation_error_str(tmp_path) -> None:
    err = ValidationError(file="test.md", line=5, message="something wrong")
    s = str(err)
    assert s.startswith("E:")
    assert "test.md" in s
    assert "5" in s

    warn = ValidationError(file="test.md", line=3, message="style issue", severity="warning")
    assert str(warn).startswith("W:")


def test_validate_nonexistent_file(tmp_path) -> None:
    path = tmp_path / "nonexistent.md"
    errors = validate_skill_file(path)
    assert any("not found" in str(e) for e in errors)


def test_validate_nonexistent_dir(tmp_path) -> None:
    errors = validate_skill_dir(tmp_path / "no_such_dir")
    assert any("not found" in str(e) for e in errors)


# --- Agent Skills format: SKILL.md dirs, frontmatter, installs (#65) ---

SPEC_SKILL = """---
name: docx
description: Create and edit Word documents via bundled scripts.
---

# docx

Run `python3 scripts/convert.py`.
"""


def _spec_skill_dir(root, name="docx", content=SPEC_SKILL) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(content)
    (d / "scripts").mkdir()
    (d / "scripts" / "convert.py").write_text('print("hi")\n')


@pytest.mark.asyncio
async def test_seed_spec_dir_skill(tmp_path) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()
    _spec_skill_dir(seed)
    store = SkillsStore(
        db_path=str(tmp_path / "s.db"), seed_dir=seed, installed_dir=tmp_path / "inst"
    )
    skills = await store.list_skills()
    assert [s["name"] for s in skills] == ["docx"]
    # Frontmatter description becomes the summary.
    assert skills[0]["summary"] == "Create and edit Word documents via bundled scripts."
    # Content carries the bundled-files pointer + the raw SKILL.md.
    skill = await store.get_skill("docx")
    assert "Bundled files for this skill" in skill["content"]
    assert str(seed / "docx") in skill["content"]
    assert skill["origin"] is None
    assert skill["stale_seed"] is False


@pytest.mark.asyncio
async def test_flat_and_spec_skills_coexist(tmp_path) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "weather.md").write_text("# Weather\n\ncurl wttr.in")
    _spec_skill_dir(seed)
    store = SkillsStore(
        db_path=str(tmp_path / "s.db"), seed_dir=seed, installed_dir=tmp_path / "inst"
    )
    skills = await store.list_skills()
    assert [s["name"] for s in skills] == ["docx", "weather"]


@pytest.mark.asyncio
async def test_installed_dir_seeded_with_origin(tmp_path) -> None:
    inst = tmp_path / "inst"
    _spec_skill_dir(inst)
    (inst / "docx" / ".origin").write_text("https://example.com/repo#skills/docx\n")
    store = SkillsStore(
        db_path=str(tmp_path / "s.db"), seed_dir=tmp_path / "seed", installed_dir=inst
    )
    skill = await store.get_skill("docx")
    assert skill["origin"] == "https://example.com/repo#skills/docx"


@pytest.mark.asyncio
async def test_installed_skill_is_read_only(tmp_path) -> None:
    inst = tmp_path / "inst"
    _spec_skill_dir(inst)
    (inst / "docx" / ".origin").write_text("https://example.com/repo#skills/docx\n")
    store = SkillsStore(
        db_path=str(tmp_path / "s.db"), seed_dir=tmp_path / "seed", installed_dir=inst
    )
    await store.ensure_seeded()
    with pytest.raises(ValueError, match="read-only"):
        await store.upsert_skill("docx", "# hacked")


@pytest.mark.asyncio
async def test_delete_installed_skill_removes_dir(tmp_path) -> None:
    inst = tmp_path / "inst"
    _spec_skill_dir(inst)
    store = SkillsStore(
        db_path=str(tmp_path / "s.db"), seed_dir=tmp_path / "seed", installed_dir=inst
    )
    await store.ensure_seeded()
    assert await store.delete_skill("docx") is True
    assert not (inst / "docx").exists()
    # ensure_seeded can't resurrect it.
    await store.ensure_seeded()
    assert await store.get_skill("docx") is None


@pytest.mark.asyncio
async def test_install_from_git_and_update(tmp_path) -> None:
    import subprocess

    repo = tmp_path / "repo"
    _spec_skill_dir(repo / "document-skills")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"],
        check=True,
    )
    store = SkillsStore(
        db_path=str(tmp_path / "s.db"),
        seed_dir=tmp_path / "seed",
        installed_dir=tmp_path / "inst",
    )
    result = await store.install_from_git(str(repo), "document-skills/docx")
    assert result["name"] == "docx"
    assert result["origin"] == f"{repo}#document-skills/docx"
    skill = await store.get_skill("docx")
    assert skill["origin"] == result["origin"]
    assert (tmp_path / "inst" / "docx" / "scripts" / "convert.py").exists()
    # Update re-fetches from the recorded origin.
    updated = await store.update_installed_skill("docx")
    assert updated["name"] == "docx"


@pytest.mark.asyncio
async def test_install_rejects_missing_skill_md(tmp_path) -> None:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("nope")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"],
        check=True,
    )
    store = SkillsStore(
        db_path=str(tmp_path / "s.db"),
        seed_dir=tmp_path / "seed",
        installed_dir=tmp_path / "inst",
    )
    with pytest.raises(ValueError, match="No SKILL.md"):
        await store.install_from_git(str(repo), "")


def test_validate_spec_skill_frontmatter(tmp_path) -> None:
    _spec_skill_dir(tmp_path)
    errors = validate_skill_file(tmp_path / "docx" / "SKILL.md")
    assert errors == []


def test_validate_spec_skill_missing_frontmatter(tmp_path) -> None:
    d = tmp_path / "docx"
    d.mkdir()
    (d / "SKILL.md").write_text("# docx\n\nNo frontmatter here.")
    errors = validate_skill_file(d / "SKILL.md")
    assert any("frontmatter" in str(e).lower() for e in errors)


def test_validate_spec_skill_name_dir_mismatch(tmp_path) -> None:
    d = tmp_path / "other"
    d.mkdir()
    (d / "SKILL.md").write_text(SPEC_SKILL)
    errors = validate_skill_file(d / "SKILL.md")
    assert any("does not match directory" in str(e) for e in errors)


def test_spec_skill_commands_downgrade_to_warnings(tmp_path) -> None:
    d = tmp_path / "docx"
    d.mkdir()
    (d / "SKILL.md").write_text(SPEC_SKILL + "\n```bash\nnpm install\n```\n")
    errors = validate_skill_file(d / "SKILL.md")
    assert errors  # the disallowed command is flagged...
    assert all(e.severity == "warning" for e in errors)  # ...but only as a warning


def test_spec_skill_python_and_pdftoppm_no_warnings(tmp_path) -> None:
    d = tmp_path / "docx"
    d.mkdir()
    (d / "SKILL.md").write_text(
        SPEC_SKILL + "\n```bash\npython scripts/x.py in.docx\npdftoppm -png out.pdf page\n```\n"
    )
    assert validate_skill_file(d / "SKILL.md") == []
