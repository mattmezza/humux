"""Two-tier scoped memory: shared pool + per-agent private memory (#42)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import aiosqlite
import pytest

from core.memory import (
    MemoryStore,
    _extraction_scope_block,
    _resolve_extracted_scope,
    _scope_filter,
)


@pytest.fixture
async def store(tmp_path):
    s = MemoryStore(db_path=str(tmp_path / "memory.db"), long_term_limit=50)
    await s._ensure_schema()
    return s


def _contents(rows: list[dict]) -> set[str]:
    return {r["content"] for r in rows}


# --- the isolation invariant ------------------------------------------------


async def test_private_memory_invisible_to_other_agents(store):
    await store._insert_long_term("fact", "matteo", "shared owner fact", scope="")
    await store._insert_long_term("health", "matteo", "coach-only fact", scope="coach")
    await store._insert_long_term("work", "matteo", "finance-only fact", scope="finance")

    # The coach sees shared + its own, never finance's private memory.
    coach = _contents(await store.get_long_term("coach"))
    assert coach == {"shared owner fact", "coach-only fact"}

    finance = _contents(await store.get_long_term("finance"))
    assert finance == {"shared owner fact", "finance-only fact"}

    # The default identity (no agent) sees shared only.
    default = _contents(await store.get_long_term(""))
    assert default == {"shared owner fact"}

    # scope=None is the admin/owner view — everything.
    all_rows = _contents(await store.get_long_term(None))
    assert all_rows == {"shared owner fact", "coach-only fact", "finance-only fact"}


async def test_short_term_scoped(store):
    await store._store_short_term({"content": "shared now", "ttl_hours": 8}, scope="")
    await store._store_short_term({"content": "coach now", "ttl_hours": 8}, scope="coach")

    assert _contents(await store.get_short_term("")) == {"shared now"}
    assert _contents(await store.get_short_term("coach")) == {"shared now", "coach now"}
    assert _contents(await store.get_short_term("finance")) == {"shared now"}


async def test_format_for_prompt_scoped(store):
    await store._insert_long_term("fact", "matteo", "lives in zurich", scope="")
    await store._insert_long_term("health", "matteo", "training for a marathon", scope="coach")

    coach_block = await store.format_for_prompt(scope="coach")
    assert "training for a marathon" in coach_block
    assert "lives in zurich" in coach_block

    finance_block = await store.format_for_prompt(scope="finance")
    assert "training for a marathon" not in finance_block
    assert "lives in zurich" in finance_block


async def test_dedup_candidates_bounded_to_scope(store):
    # An identical fact stored privately under another agent must not be a
    # dedup/UPDATE/DELETE candidate for this agent.
    await store._insert_long_term("fact", "x", "secret number is 42", scope="finance")
    similar = await store._retrieve_similar_long_term("x", "secret number is 42", scope="coach")
    assert similar == []  # finance's private row is invisible to coach
    # ...but visible within finance's own scope.
    own = await store._retrieve_similar_long_term("x", "secret number is 42", scope="finance")
    assert any(r["content"] == "secret number is 42" for r in own)


async def test_hygiene_never_merges_across_scopes(store):
    # Same content in two different private scopes: each scope-partition has a
    # single member, so no cluster forms and the merge LLM is never invoked.
    await store._insert_long_term("fact", "x", "duplicate fact text", scope="coach")
    await store._insert_long_term("fact", "x", "duplicate fact text", scope="finance")

    llm = AsyncMock()
    llm.generate_text = AsyncMock(return_value='{"updates": [], "deletes": []}')
    removed = await store._hygiene_pass(llm, "model")

    assert removed == 0
    llm.generate_text.assert_not_called()  # no cross-scope cluster to resolve
    rows = await store.get_long_term(None)
    assert len(rows) == 2  # both survive


# --- migration: legacy rows default to shared -------------------------------


async def test_migration_defaults_existing_rows_to_shared(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "CREATE TABLE long_term (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "category TEXT NOT NULL, subject TEXT NOT NULL, content TEXT NOT NULL, "
            "updated_at DATETIME DEFAULT (datetime('now')))"
        )
        await db.execute(
            "CREATE TABLE short_term (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "content TEXT NOT NULL, expires_at DATETIME NOT NULL)"
        )
        await db.execute(
            "INSERT INTO long_term (category, subject, content) VALUES ('f', 'm', 'old')"
        )
        await db.commit()

    store = MemoryStore(db_path=db_path)
    await store._ensure_schema()  # must add scope without dropping the legacy row

    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("PRAGMA table_info(long_term)")
        assert "scope" in {r[1] for r in await cur.fetchall()}
        cur = await db.execute("PRAGMA table_info(short_term)")
        assert "scope" in {r[1] for r in await cur.fetchall()}
        cur = await db.execute("SELECT scope FROM long_term WHERE content = 'old'")
        assert (await cur.fetchone())[0] == ""  # legacy row is shared


# --- scope-resolution helpers (pure) ----------------------------------------


def test_scope_filter():
    assert _scope_filter(None) == ("", ())
    assert _scope_filter("") == (" AND scope = ''", ())
    assert _scope_filter("coach") == (" AND scope IN ('', ?)", ("coach",))


def test_resolve_extracted_scope():
    # Private only when an agent is active AND the model tagged it private.
    assert _resolve_extracted_scope({"scope": "private"}, "coach") == "coach"
    assert _resolve_extracted_scope({"scope": "PRIVATE"}, "coach") == "coach"
    assert _resolve_extracted_scope({"scope": "shared"}, "coach") == ""
    assert _resolve_extracted_scope({}, "coach") == ""
    # No active agent → always shared, even if tagged private.
    assert _resolve_extracted_scope({"scope": "private"}, "") == ""


def test_extraction_scope_block():
    assert _extraction_scope_block("") == ""
    block = _extraction_scope_block("coach")
    assert "coach" in block and "private" in block


# --- inline editing from the admin UI (#158) --------------------------------


async def _one_id(store, table: str) -> int:
    async with aiosqlite.connect(store.db_path) as db:
        cursor = await db.execute(f"SELECT id FROM {table} LIMIT 1")  # noqa: S608
        return (await cursor.fetchone())[0]


async def test_update_long_term_changes_fields_and_scope(store):
    await store._insert_long_term("fact", "matteo", "old content", scope="")
    n = await store.update_long_term(
        await _one_id(store, "long_term"),
        category="work",
        subject="matteo",
        content="new content",
        scope="coach",
    )
    assert n == 1
    # Moved to the coach scope with the new category.
    coach = {r["content"]: r for r in await store.get_long_term("coach")}
    assert coach["new content"]["category"] == "work"


async def test_update_long_term_reembeds_only_on_text_change(store, monkeypatch):
    calls = []

    async def fake_blob(text):
        calls.append(text)
        return b"vec"

    monkeypatch.setattr(store, "_embed_blob", fake_blob)
    await store._insert_long_term("fact", "s", "content", scope="")
    mid = await _one_id(store, "long_term")
    calls.clear()

    # Scope-only edit → no re-embed.
    await store.update_long_term(mid, category="fact", subject="s", content="content", scope="x")
    assert calls == []
    # Content edit → re-embed.
    await store.update_long_term(mid, category="fact", subject="s", content="changed", scope="x")
    assert calls == ["s: changed"]


async def test_update_long_term_missing_returns_zero(store):
    assert (
        await store.update_long_term(999, category="fact", subject="s", content="c", scope="") == 0
    )


async def test_update_short_term(store):
    await store._store_short_term(
        {"content": "temp fact", "context": "", "ttl_hours": 24}, scope=""
    )
    n = await store.update_short_term(
        await _one_id(store, "short_term"),
        content="edited",
        scope="coach",
        expires_at="2099-01-01 00:00:00",
    )
    assert n == 1
    coach = {r["content"] for r in await store.get_short_term("coach")}
    assert "edited" in coach
