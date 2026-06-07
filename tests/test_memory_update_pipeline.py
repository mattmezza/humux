"""Tests for the unified ADD/UPDATE/DELETE/NOOP long-term update pipeline.

Covers MemoryStore.update_memory and the lexical candidate retrieval that
feeds it (issue #5, Tier 1).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from core.memory import (
    MemoryStore,
    _normalize_subject,
    _similarity,
    _tokens,
)


@pytest.fixture
async def store(tmp_path):
    db_path = str(tmp_path / "memory.db")
    s = MemoryStore(db_path=db_path, long_term_limit=50)
    await s._ensure_schema()
    return s


async def _insert_long_term(store: MemoryStore, category: str, subject: str, content: str) -> int:
    async with aiosqlite.connect(store.db_path) as db:
        cursor = await db.execute(
            "INSERT INTO long_term (category, subject, content, source, confidence) "
            "VALUES (?, ?, ?, 'test', 'stated')",
            (category, subject, content),
        )
        await db.commit()
        return cursor.lastrowid


async def _rows(store: MemoryStore) -> list[dict]:
    async with aiosqlite.connect(store.db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, category, subject, content FROM long_term ORDER BY id"
        )
        return [dict(r) for r in await cursor.fetchall()]


class _DecisionLLM:
    """LLM stub that returns a fixed update decision (dict) and records calls."""

    def __init__(self, decision: dict | str):
        self._decision = decision if isinstance(decision, str) else json.dumps(decision)
        self.calls = 0
        self.last_prompt: str | None = None

    async def generate_text(self, *, model: str, prompt: str, max_tokens: int = 1024) -> str:
        self.calls += 1
        self.last_prompt = prompt
        return self._decision


# -- update_memory --


class TestUpdateMemory:
    async def test_new_fact_into_empty_db_adds_without_llm(self, store):
        """No existing memories → ADD directly, no LLM call."""
        llm = _DecisionLLM({"operation": "NOOP"})  # should never be consulted

        op = await store.update_memory(
            llm, "m", {"category": "fact", "subject": "matteo", "content": "Lives in Zurich"}
        )

        assert op == "ADD"
        assert llm.calls == 0
        rows = await _rows(store)
        assert len(rows) == 1
        assert rows[0]["content"] == "Lives in Zurich"

    async def test_unrelated_existing_still_adds_without_llm(self, store):
        """Existing memory shares no tokens → no candidate → ADD without LLM."""
        await _insert_long_term(store, "food", "simge", "Allergic to peanuts")
        llm = _DecisionLLM({"operation": "NOOP"})

        op = await store.update_memory(
            llm, "m", {"category": "work", "subject": "matteo", "content": "Software engineer"}
        )

        assert op == "ADD"
        assert llm.calls == 0
        assert len(await _rows(store)) == 2

    async def test_semantic_duplicate_noop(self, store):
        """A near-duplicate of an existing memory → LLM rules NOOP, nothing added."""
        await _insert_long_term(store, "health", "matteo", "Allergic to shellfish")
        llm = _DecisionLLM({"operation": "NOOP"})

        op = await store.update_memory(
            llm, "m", {"category": "health", "subject": "matteo", "content": "Cannot eat shellfish"}
        )

        assert op == "NOOP"
        assert llm.calls == 1
        assert len(await _rows(store)) == 1

    async def test_refinement_updates_in_place(self, store):
        """LLM returns UPDATE with a merged content → existing row is rewritten."""
        rid = await _insert_long_term(store, "work", "matteo", "Uses a desk at work")
        llm = _DecisionLLM(
            {
                "operation": "UPDATE",
                "id": rid,
                "category": "work",
                "subject": "matteo",
                "content": "Uses a standing desk at work",
            }
        )

        op = await store.update_memory(
            llm, "m", {"category": "work", "subject": "matteo", "content": "Standing desk now"}
        )

        assert op == "UPDATE"
        rows = await _rows(store)
        assert len(rows) == 1
        assert rows[0]["content"] == "Uses a standing desk at work"

    async def test_contradiction_deletes(self, store):
        """LLM returns DELETE → the contradicted memory is removed, none added."""
        rid = await _insert_long_term(store, "work", "matteo", "Uses a standing desk at work")
        llm = _DecisionLLM({"operation": "DELETE", "id": rid})

        op = await store.update_memory(
            llm,
            "m",
            {"category": "work", "subject": "matteo", "content": "Switched back to sitting desk"},
        )

        assert op == "DELETE"
        assert len(await _rows(store)) == 0

    async def test_add_alongside_existing(self, store):
        """LLM returns ADD even though a lexically similar memory exists."""
        await _insert_long_term(store, "routine", "matteo", "Runs on Mondays")
        llm = _DecisionLLM({"operation": "ADD"})

        op = await store.update_memory(
            llm,
            "m",
            {"category": "routine", "subject": "matteo", "content": "Runs on Thursdays too"},
        )

        assert op == "ADD"
        assert len(await _rows(store)) == 2

    async def test_malformed_output_is_safe_noop(self, store):
        """Non-JSON LLM output → no-op, no mutation."""
        await _insert_long_term(store, "health", "matteo", "Allergic to shellfish")
        llm = _DecisionLLM("totally not json")

        op = await store.update_memory(
            llm, "m", {"category": "health", "subject": "matteo", "content": "Cannot eat shellfish"}
        )

        assert op == "NOOP"
        assert len(await _rows(store)) == 1

    async def test_llm_exception_is_safe_noop(self, store):
        """An LLM error mid-decision must not crash or mutate state."""
        await _insert_long_term(store, "health", "matteo", "Allergic to shellfish")
        llm = AsyncMock()
        llm.generate_text.side_effect = RuntimeError("API down")

        op = await store.update_memory(
            llm, "m", {"category": "health", "subject": "matteo", "content": "Cannot eat shellfish"}
        )

        assert op == "NOOP"
        assert len(await _rows(store)) == 1

    async def test_update_with_invalid_id_is_noop(self, store):
        """UPDATE targeting an id not in the candidate set is rejected."""
        await _insert_long_term(store, "health", "matteo", "Allergic to shellfish")
        llm = _DecisionLLM({"operation": "UPDATE", "id": 999, "content": "x"})

        op = await store.update_memory(
            llm, "m", {"category": "health", "subject": "matteo", "content": "Cannot eat shellfish"}
        )

        assert op == "NOOP"
        rows = await _rows(store)
        assert rows[0]["content"] == "Allergic to shellfish"

    async def test_delete_with_invalid_id_is_noop(self, store):
        await _insert_long_term(store, "health", "matteo", "Allergic to shellfish")
        llm = _DecisionLLM({"operation": "DELETE", "id": 999})

        op = await store.update_memory(
            llm, "m", {"category": "health", "subject": "matteo", "content": "Cannot eat shellfish"}
        )

        assert op == "NOOP"
        assert len(await _rows(store)) == 1

    async def test_empty_content_is_noop_without_llm(self, store):
        llm = _DecisionLLM({"operation": "ADD"})
        op = await store.update_memory(
            llm, "m", {"category": "fact", "subject": "matteo", "content": "  "}
        )
        assert op == "NOOP"
        assert llm.calls == 0
        assert len(await _rows(store)) == 0

    async def test_subject_normalised_on_add(self, store):
        """Subjects are lowercased in code, not just via prompt instruction."""
        llm = _DecisionLLM({"operation": "NOOP"})
        await store.update_memory(
            llm, "m", {"category": "fact", "subject": "Matteo", "content": "Lives in Zurich"}
        )
        rows = await _rows(store)
        assert rows[0]["subject"] == "matteo"

    async def test_update_prompt_includes_timestamps(self, store):
        """The decision prompt carries created/updated timestamps (issue #8)."""
        await _insert_long_term(store, "health", "matteo", "Allergic to shellfish")
        llm = _DecisionLLM({"operation": "NOOP"})

        await store.update_memory(
            llm, "m", {"category": "health", "subject": "matteo", "content": "Cannot eat shellfish"}
        )

        assert "created" in (llm.last_prompt or "")
        assert "updated" in (llm.last_prompt or "")


# -- lexical retrieval --


class TestRetrieveSimilar:
    async def test_subject_boost_ranks_same_subject_first(self, store):
        await _insert_long_term(store, "fact", "simge", "enjoys shellfish dishes")
        await _insert_long_term(store, "fact", "matteo", "dislikes loud music")

        similar = await store._retrieve_similar_long_term("matteo", "allergic to shellfish")

        # Both share a token with the candidate; the same-subject row wins.
        assert similar[0]["subject"] == "matteo"

    async def test_caps_at_top_k(self, store):
        for i in range(20):
            await _insert_long_term(store, "fact", "matteo", f"likes hiking trip {i}")

        similar = await store._retrieve_similar_long_term("matteo", "likes hiking")

        assert len(similar) <= store._UPDATE_TOP_K

    async def test_no_overlap_returns_empty(self, store):
        await _insert_long_term(store, "fact", "simge", "speaks turkish")
        similar = await store._retrieve_similar_long_term("matteo", "owns a bicycle")
        assert similar == []


# -- pure helpers --


class TestHelpers:
    def test_normalize_subject(self):
        assert _normalize_subject("  Matteo ") == "matteo"
        assert _normalize_subject("") == ""
        assert _normalize_subject(None) == ""

    def test_tokens_drops_stopwords_and_single_chars(self):
        assert _tokens("The user is a developer") == {"user", "developer"}

    def test_similarity_jaccard(self):
        assert _similarity({"a", "b"}, {"a", "b"}) == 1.0
        assert _similarity({"a", "b"}, {"c", "d"}) == 0.0
        assert _similarity(set(), {"a"}) == 0.0
        assert _similarity({"a", "b", "c"}, {"a"}) == pytest.approx(1 / 3)
