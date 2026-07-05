"""Tests for SkillsEngine and the skill file validator."""

from __future__ import annotations

import pytest

from core.skills import SkillsEngine, validate_skill_file, validate_skill_dir, ValidationError


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
    path.write_text("# Skill\n\n```bash\nwrite_file(path=\"test.txt\", content=\"hello\")\n```\n")
    errors = validate_skill_file(path)
    assert errors == []


def test_unlabeled_block_not_checked(tmp_path) -> None:
    """Unlabeled code blocks (tables, tool call examples) are not validated."""
    path = tmp_path / "good.md"
    path.write_text("# Skill\n\n```\n| Col1 | Col2 |\n|------|------|\n| A    | B    |\n```\n\n```\nwrite_file(path=\"test.txt\", content=\"hello\")\n```\n")
    errors = validate_skill_file(path)
    assert errors == []


def test_python3_tool_invocation(tmp_path) -> None:
    path = tmp_path / "good.md"
    path.write_text("# Skill\n\n```bash\npython3 ./tools/browser.py read --url https://example.com\n```\n")
    errors = validate_skill_file(path)
    assert errors == []


def test_python3_c_script(tmp_path) -> None:
    path = tmp_path / "good.md"
    path.write_text("# Skill\n\n```bash\npython3 -c \"import sys; print(sys.version)\"\n```\n")
    errors = validate_skill_file(path)
    assert errors == []


def test_backslash_continuation(tmp_path) -> None:
    """Continuation lines after \\ should be treated as part of the same command."""
    path = tmp_path / "good.md"
    path.write_text(
        "# Skill\n\n```bash\npython3 /app/tools/browser.py act --url https://site/login \\\n  --steps '[{\"click\":\"#btn\"}]'\n```\n"
    )
    errors = validate_skill_file(path)
    assert errors == []


def test_long_python_invocation(tmp_path) -> None:
    """Multi-line python3 command with continuation."""
    path = tmp_path / "good.md"
    path.write_text(
        "# Skill\n\n```bash\npython3 /app/tools/skills.py upsert --name weather --stdin --write-seed \\\n  < skills/weather.md\n```\n"
    )
    errors = validate_skill_file(path)
    assert errors == []


def test_pipe_continuation(tmp_path) -> None:
    """Pipe at end of line is fine."""
    path = tmp_path / "good.md"
    path.write_text(
        "# Skill\n\n```bash\nhimalaya envelope list -a personal -s 10 -o json \\\n  | jq '.[].subject'\n```\n"
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

