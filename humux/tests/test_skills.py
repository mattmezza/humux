"""Tests for SkillsEngine."""

from __future__ import annotations

import pytest

from core.skills import SkillsEngine


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

    assert "- alpha: Alpha skill" in index
    assert "- beta: Beta skill" in index


@pytest.mark.asyncio
async def test_get_skill_content_reads_seeded_skill(tmp_path) -> None:
    (tmp_path / "memory.md").write_text("# Memory\n\nUse sqlite3.")
    db_path = str(tmp_path / "skills.db")
    engine = SkillsEngine(db_path=db_path, seed_dir=tmp_path)

    content = await engine.get_skill_content("memory")

    assert "Use sqlite3." in content


# --- on-demand discovery: index_entries + search_index (#50) ---


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


@pytest.mark.asyncio
async def test_search_index_ranks_name_hit_first(tmp_path) -> None:
    # "weather" matches the name of one skill and the summary of another; the
    # name hit must outrank the summary-only hit.
    engine = _engine_with(
        tmp_path,
        weather="forecast lookups",
        travel="plan trips, mentions weather in passing",
    )
    hits = await engine.search_index("weather")
    assert [h["name"] for h in hits] == ["weather", "travel"]


@pytest.mark.asyncio
async def test_search_index_no_match_is_empty(tmp_path) -> None:
    engine = _engine_with(tmp_path, email="send email")
    assert await engine.search_index("quantumchromodynamics") == []


@pytest.mark.asyncio
async def test_search_index_empty_query_browses(tmp_path) -> None:
    engine = _engine_with(tmp_path, a="x", b="y", c="z")
    hits = await engine.search_index("", limit=2)
    assert len(hits) == 2  # empty query = cheap browse, capped by limit


@pytest.mark.asyncio
async def test_search_index_respects_allowlist(tmp_path) -> None:
    engine = _engine_with(tmp_path, email="send email", webmail="email in the browser")
    hits = await engine.search_index("email", allow=["email"])
    assert {h["name"] for h in hits} == {"email"}
