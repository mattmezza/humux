"""Tests for MemoryStore.consolidate_and_cleanup."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from core.memory import MemoryStore


@pytest.fixture
async def store(tmp_path):
    """Create a MemoryStore backed by a temporary SQLite DB."""
    db_path = str(tmp_path / "memory.db")
    s = MemoryStore(db_path=db_path, long_term_limit=50)
    await s._ensure_schema()
    return s


async def _insert_short_term(
    store: MemoryStore, content: str, context: str = "", hours_until_expiry: int = 12
):
    """Helper to insert a short-term memory directly."""
    expires = datetime.now(tz=UTC) + timedelta(hours=hours_until_expiry)
    expires_str = expires.strftime("%Y-%m-%d %H:%M:%S")
    async with aiosqlite.connect(store.db_path) as db:
        await db.execute(
            "INSERT INTO short_term (content, context, expires_at) VALUES (?, ?, ?)",
            (content, context, expires_str),
        )
        await db.commit()


async def _insert_expired_short_term(store: MemoryStore, content: str, context: str = ""):
    """Helper to insert an already-expired short-term memory."""
    expired = datetime.now(tz=UTC) - timedelta(hours=1)
    expired_str = expired.strftime("%Y-%m-%d %H:%M:%S")
    async with aiosqlite.connect(store.db_path) as db:
        await db.execute(
            "INSERT INTO short_term (content, context, expires_at) VALUES (?, ?, ?)",
            (content, context, expired_str),
        )
        await db.commit()


async def _insert_long_term(store: MemoryStore, category: str, subject: str, content: str):
    """Helper to insert a long-term memory directly."""
    async with aiosqlite.connect(store.db_path) as db:
        await db.execute(
            "INSERT INTO long_term (category, subject, content, source, confidence) "
            "VALUES (?, ?, ?, 'test', 'stated')",
            (category, subject, content),
        )
        await db.commit()


async def _count_rows(store: MemoryStore, table: str) -> int:
    async with aiosqlite.connect(store.db_path) as db:
        cursor = await db.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
        row = await cursor.fetchone()
        return row[0] if row else 0


class _LLMStub:
    def __init__(self, response_json: list):
        self._response = json.dumps(response_json)
        self.generate_text = AsyncMock(return_value=self._response)
        self.last_prompt: str | None = None
        self._emit_response = self.generate_text

    async def generate_text_with_prompt(
        self, *, model: str, prompt: str, max_tokens: int = 1024
    ) -> str:
        self.last_prompt = prompt
        return await self._emit_response(model=model, prompt=prompt, max_tokens=max_tokens)


def _make_mock_llm(response_json: list) -> _LLMStub:
    """Create a mock LLM client that returns a canned JSON response."""
    llm = _LLMStub(response_json)
    llm._emit_response = llm.generate_text
    llm.generate_text = AsyncMock(side_effect=llm.generate_text_with_prompt)
    return llm


# -- Tests --


class TestConsolidateAndCleanup:
    async def test_promotes_memories_llm_selects(self, store):
        """LLM selects one memory for promotion; it appears in long-term."""
        await _insert_short_term(store, "Matteo switched to a standing desk at work")
        await _insert_short_term(store, "Matteo is at the airport right now")

        llm = _make_mock_llm(
            [
                {
                    "category": "work",
                    "subject": "matteo",
                    "content": "Uses a standing desk at work",
                },
            ]
        )

        result = await store.consolidate_and_cleanup(llm, "claude-haiku-4-5")

        assert result["active_reviewed"] == 2
        assert result["promoted_to_long_term"] == 1
        assert result["expired_deleted"] == 0

        long_term = await store.get_long_term()
        assert len(long_term) == 1
        assert long_term[0]["content"] == "Uses a standing desk at work"
        assert long_term[0]["category"] == "work"
        assert long_term[0]["subject"] == "matteo"

    async def test_deletes_expired_short_term(self, store):
        """Expired short-term rows are deleted during cleanup."""
        await _insert_expired_short_term(store, "Stale fact 1")
        await _insert_expired_short_term(store, "Stale fact 2")
        await _insert_short_term(store, "Still active fact")

        llm = _make_mock_llm([])  # nothing to promote

        result = await store.consolidate_and_cleanup(llm, "claude-haiku-4-5")

        assert result["expired_deleted"] == 2
        assert result["active_reviewed"] == 1

        # Only the active one remains
        total = await _count_rows(store, "short_term")
        assert total == 1

    async def test_empty_short_term_skips_llm_call(self, store):
        """When there are no active short-term memories, no LLM call is made."""
        await _insert_expired_short_term(store, "Already expired")

        llm = _make_mock_llm([])

        result = await store.consolidate_and_cleanup(llm, "claude-haiku-4-5")

        assert result["active_reviewed"] == 0
        assert result["promoted_to_long_term"] == 0
        assert result["expired_deleted"] == 1

        # LLM should not have been called
        llm.generate_text.assert_not_called()

    async def test_nothing_to_do(self, store):
        """No short-term memories at all — graceful no-op."""
        llm = _make_mock_llm([])

        result = await store.consolidate_and_cleanup(llm, "claude-haiku-4-5")

        assert result["active_reviewed"] == 0
        assert result["promoted_to_long_term"] == 0
        assert result["expired_deleted"] == 0
        llm.generate_text.assert_not_called()

    async def test_dedup_prevents_duplicate_promotion(self, store):
        """A promoted memory that overlaps with existing long-term is skipped."""
        await _insert_long_term(store, "work", "matteo", "Uses a standing desk at work")
        await _insert_short_term(store, "Matteo mentioned his standing desk again")

        # LLM tries to promote a duplicate
        llm = _make_mock_llm(
            [
                {
                    "category": "work",
                    "subject": "matteo",
                    "content": "Uses a standing desk at work",
                },
            ]
        )

        result = await store.consolidate_and_cleanup(llm, "claude-haiku-4-5")

        assert result["promoted_to_long_term"] == 0

        # Still only one long-term memory
        long_term = await store.get_long_term()
        assert len(long_term) == 1

    async def test_llm_returns_empty_array(self, store):
        """LLM decides nothing is worth promoting — no errors, just cleanup."""
        await _insert_short_term(store, "Matteo is having coffee")
        await _insert_expired_short_term(store, "Old fact")

        llm = _make_mock_llm([])

        result = await store.consolidate_and_cleanup(llm, "claude-haiku-4-5")

        assert result["active_reviewed"] == 1
        assert result["promoted_to_long_term"] == 0
        assert result["expired_deleted"] == 1

    async def test_llm_failure_still_cleans_expired(self, store):
        """If the LLM call fails, expired rows are still deleted."""
        await _insert_short_term(store, "Active fact")
        await _insert_expired_short_term(store, "Expired fact")

        llm = AsyncMock()
        llm.generate_text.side_effect = RuntimeError("API down")

        result = await store.consolidate_and_cleanup(llm, "claude-haiku-4-5")

        assert result["active_reviewed"] == 1
        assert result["promoted_to_long_term"] == 0
        assert result["expired_deleted"] == 1

    async def test_llm_returns_invalid_json_still_cleans(self, store):
        """If the LLM returns garbage, expired rows are still deleted."""
        await _insert_short_term(store, "Active fact")
        await _insert_expired_short_term(store, "Expired fact")

        llm = _make_mock_llm([])
        llm.generate_text = AsyncMock(return_value="not json at all")

        result = await store.consolidate_and_cleanup(llm, "claude-haiku-4-5")

        assert result["promoted_to_long_term"] == 0
        assert result["expired_deleted"] == 1

    async def test_multiple_promotions(self, store):
        """LLM promotes multiple memories in one pass."""
        await _insert_short_term(store, "Matteo prefers oat milk")
        await _insert_short_term(store, "Simge speaks Turkish and German")
        await _insert_short_term(store, "Matteo at airport right now")

        llm = _make_mock_llm(
            [
                {"category": "preference", "subject": "matteo", "content": "Prefers oat milk"},
                {
                    "category": "fact",
                    "subject": "simge",
                    "content": "Speaks Turkish and German",
                },
            ]
        )

        result = await store.consolidate_and_cleanup(llm, "claude-haiku-4-5")

        assert result["promoted_to_long_term"] == 2
        long_term = await store.get_long_term()
        assert len(long_term) == 2
        subjects = {m["subject"] for m in long_term}
        assert subjects == {"matteo", "simge"}

    async def test_existing_long_term_passed_to_prompt(self, store):
        """The LLM prompt includes existing long-term memories for deduplication."""
        await _insert_long_term(store, "preference", "matteo", "Likes espresso")
        await _insert_short_term(store, "Matteo ordered a cortado today")

        llm = _make_mock_llm([])

        await store.consolidate_and_cleanup(llm, "claude-haiku-4-5")

        # Verify the prompt contained the existing long-term memory
        prompt_text = llm.last_prompt or ""
        assert "Likes espresso" in prompt_text
        assert "cortado" in prompt_text


class TestDeleteExpiredShortTerm:
    async def test_deletes_only_expired(self, store):
        """Only expired rows are deleted, active ones remain."""
        await _insert_short_term(store, "Active", hours_until_expiry=12)
        await _insert_expired_short_term(store, "Expired 1")
        await _insert_expired_short_term(store, "Expired 2")

        count = await store._delete_expired_short_term()

        assert count == 2
        total = await _count_rows(store, "short_term")
        assert total == 1

    async def test_nothing_expired(self, store):
        """No expired rows — returns 0, no errors."""
        await _insert_short_term(store, "Still valid")

        count = await store._delete_expired_short_term()

        assert count == 0
        total = await _count_rows(store, "short_term")
        assert total == 1
