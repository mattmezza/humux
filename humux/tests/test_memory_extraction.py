"""Tests for MemoryStore.extract_memories."""

from __future__ import annotations

import json

import aiosqlite
import pytest

from core.memory import MemoryStore, _extract_json_array


@pytest.fixture
async def store(tmp_path):
    db_path = str(tmp_path / "memory.db")
    memory = MemoryStore(db_path=db_path, long_term_limit=50)
    await memory._ensure_schema()
    return memory


class _LLMStub:
    """Routes by prompt type: extraction calls return the canned array;
    update_memory calls (ADD/UPDATE/DELETE/NOOP) return ``update_response``
    (defaults to ADD so every extracted long-term fact is stored)."""

    def __init__(self, response_json, update_response=None):
        self._response = json.dumps(response_json)
        self._update_response = update_response or {"operation": "ADD"}

    async def generate_text(self, *, model: str, prompt: str, max_tokens: int = 1024) -> str:
        if "Choose exactly ONE operation" in prompt:
            return json.dumps(self._update_response)
        return self._response


def _make_mock_llm(response_json, update_response=None):
    return _LLMStub(response_json, update_response)


async def _count_rows(db_path: str, table: str) -> int:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
        row = await cursor.fetchone()
        return row[0] if row else 0


@pytest.mark.asyncio
async def test_extract_memories_stores_long_and_short_term(store) -> None:
    llm = _make_mock_llm(
        [
            {
                "tier": "LONG_TERM",
                "category": "preference",
                "subject": "matteo",
                "content": "Prefers oat milk",
            },
            {
                "tier": "SHORT_TERM",
                "content": "Working from home today",
                "context": "daily update",
                "ttl_hours": 8,
            },
        ]
    )

    stored = await store.extract_memories(
        llm,
        model="claude-haiku-4-5",
        user_msg="I like oat milk",
        agent_msg="Noted",
    )

    assert stored == 2
    assert await _count_rows(store.db_path, "long_term") == 1
    assert await _count_rows(store.db_path, "short_term") == 1


@pytest.mark.asyncio
async def test_extract_memories_skips_invalid_short_term(store) -> None:
    llm = _make_mock_llm(
        [
            {
                "tier": "SHORT_TERM",
                "content": "No ttl provided",
                "context": "oops",
            }
        ]
    )

    stored = await store.extract_memories(
        llm,
        model="claude-haiku-4-5",
        user_msg="",
        agent_msg="",
    )

    assert stored == 0
    assert await _count_rows(store.db_path, "short_term") == 0


@pytest.mark.asyncio
async def test_extract_memories_strips_markdown_code_fences(store) -> None:
    """LLMs sometimes wrap JSON in ```json ... ``` fences."""
    llm = _LLMStub(
        [
            {
                "tier": "LONG_TERM",
                "category": "fact",
                "subject": "marco",
                "content": "Email is marco@example.com",
            }
        ]
    )
    llm._response = (
        '```json\n[{"tier": "LONG_TERM", "category": "fact", "subject": "marco", '
        '"content": "Email is marco@example.com"}]\n```'
    )

    stored = await store.extract_memories(
        llm,
        model="claude-haiku-4-5",
        user_msg="My email is marco@example.com",
        agent_msg="Got it",
    )

    assert stored == 1
    assert await _count_rows(store.db_path, "long_term") == 1


@pytest.mark.asyncio
async def test_extract_memories_with_preamble_text(store) -> None:
    """LLMs sometimes add prose before/after the JSON array."""
    llm = _LLMStub([])
    llm._response = (
        "Here are the extracted memories:\n"
        '[{"tier": "LONG_TERM", "category": "preference", "subject": "matteo", '
        '"content": "Likes hiking"}]\n'
        "I hope that helps!"
    )

    stored = await store.extract_memories(
        llm,
        model="claude-haiku-4-5",
        user_msg="I love hiking",
        agent_msg="Noted",
    )

    assert stored == 1
    assert await _count_rows(store.db_path, "long_term") == 1


@pytest.mark.asyncio
async def test_extract_memories_with_indented_json(store) -> None:
    """LLMs sometimes return pretty-printed / indented JSON."""
    llm = _LLMStub([])
    llm._response = (
        "[\n"
        "    {\n"
        '        "tier": "SHORT_TERM",\n'
        '        "content": "User is handling tasks manually",\n'
        '        "context": "workflow preference",\n'
        '        "ttl_hours": 8\n'
        "    }\n"
        "]"
    )

    stored = await store.extract_memories(
        llm,
        model="claude-haiku-4-5",
        user_msg="I'm handling tasks manually",
        agent_msg="Understood",
    )

    assert stored == 1
    assert await _count_rows(store.db_path, "short_term") == 1


# -- Unit tests for _extract_json_array --


class TestExtractJsonArray:
    def test_plain_json(self):
        assert _extract_json_array('[{"a": 1}]') == [{"a": 1}]

    def test_empty_array(self):
        assert _extract_json_array("[]") == []

    def test_markdown_fenced(self):
        raw = '```json\n[{"a": 1}]\n```'
        assert _extract_json_array(raw) == [{"a": 1}]

    def test_preamble_and_trailing_text(self):
        raw = 'Here is the result:\n[{"a": 1}]\nHope that helps!'
        assert _extract_json_array(raw) == [{"a": 1}]

    def test_indented_json(self):
        raw = '[\n    {\n        "a": 1\n    }\n]'
        assert _extract_json_array(raw) == [{"a": 1}]

    def test_empty_string(self):
        assert _extract_json_array("") is None

    def test_no_json(self):
        assert _extract_json_array("No memories to extract.") is None

    def test_json_object_not_array(self):
        assert _extract_json_array('{"a": 1}') is None

    def test_brackets_in_strings(self):
        raw = '[{"content": "array [1, 2] inside"}]'
        result = _extract_json_array(raw)
        assert result == [{"content": "array [1, 2] inside"}]

    def test_fence_with_preamble(self):
        raw = 'Here are the memories:\n```json\n[{"a": 1}]\n```\nDone.'
        assert _extract_json_array(raw) == [{"a": 1}]


# -- Cooldown tests --


@pytest.mark.asyncio
async def test_cooldown_skips_rapid_extractions(store) -> None:
    """Second call within cooldown window should be skipped."""
    llm = _make_mock_llm(
        [
            {
                "tier": "LONG_TERM",
                "category": "fact",
                "subject": "matteo",
                "content": "Lives in Zurich",
            }
        ]
    )

    # First call succeeds
    stored1 = await store.extract_memories(
        llm,
        model="m",
        user_msg="I live in Zurich",
        agent_msg="Got it",
        cooldown_seconds=300,
    )
    assert stored1 == 1

    # Second call within cooldown is skipped
    stored2 = await store.extract_memories(
        llm,
        model="m",
        user_msg="I also like coffee",
        agent_msg="Noted",
        cooldown_seconds=300,
    )
    assert stored2 == 0
    # Only the first memory was stored
    assert await _count_rows(store.db_path, "long_term") == 1


@pytest.mark.asyncio
async def test_cooldown_buffers_skipped_turn_for_next_extraction(store) -> None:
    """A turn skipped by the cooldown is replayed into the next extraction."""

    class _RecordingStub:
        def __init__(self):
            self.prompts: list[str] = []

        async def generate_text(self, *, model, prompt, max_tokens=1024):
            self.prompts.append(prompt)
            return "[]"

    llm = _RecordingStub()

    # First turn runs and arms the cooldown.
    await store.extract_memories(
        llm, model="m", user_msg="turn one", agent_msg="ok", cooldown_seconds=300
    )
    # Second turn is inside the cooldown → buffered, not dropped.
    await store.extract_memories(
        llm, model="m", user_msg="buffered fact", agent_msg="reply", cooldown_seconds=300
    )
    assert store._pending_turns == [("buffered fact", "reply")]
    assert len(llm.prompts) == 1  # the buffered turn made no LLM call

    # Third turn (cooldown disabled) replays the buffered turn and clears it.
    await store.extract_memories(
        llm, model="m", user_msg="turn three", agent_msg="ok", cooldown_seconds=0
    )
    assert store._pending_turns == []
    assert "buffered fact" in llm.prompts[-1]
    assert "turn three" in llm.prompts[-1]


@pytest.mark.asyncio
async def test_pending_turns_buffer_is_capped(store) -> None:
    """The cooldown buffer never grows past _MAX_PENDING_TURNS."""

    class _Stub:
        async def generate_text(self, *, model, prompt, max_tokens=1024):
            return "[]"

    llm = _Stub()
    await store.extract_memories(
        llm, model="m", user_msg="arm", agent_msg="ok", cooldown_seconds=300
    )
    for i in range(store._MAX_PENDING_TURNS + 5):
        await store.extract_memories(
            llm, model="m", user_msg=f"turn {i}", agent_msg="ok", cooldown_seconds=300
        )
    assert len(store._pending_turns) == store._MAX_PENDING_TURNS
    # Oldest dropped, newest kept.
    assert store._pending_turns[-1][0] == f"turn {store._MAX_PENDING_TURNS + 4}"


@pytest.mark.asyncio
async def test_cooldown_zero_allows_all(store) -> None:
    """cooldown_seconds=0 should allow every call."""
    # The update pipeline rules the second (duplicate) candidate a NOOP.
    llm = _make_mock_llm(
        [
            {
                "tier": "LONG_TERM",
                "category": "fact",
                "subject": "matteo",
                "content": "Lives in Zurich",
            }
        ],
        update_response={"operation": "NOOP"},
    )

    stored1 = await store.extract_memories(
        llm,
        model="m",
        user_msg="msg1",
        agent_msg="ok",
        cooldown_seconds=0,
    )
    stored2 = await store.extract_memories(
        llm,
        model="m",
        user_msg="msg2",
        agent_msg="ok",
        cooldown_seconds=0,
    )
    # Both should succeed (dedup catches the second one, but no cooldown skip)
    assert stored1 == 1
    # Second is a duplicate by content, so dedup catches it — but cooldown didn't block
    assert stored2 == 0  # deduped, not cooldown-blocked
    assert await _count_rows(store.db_path, "long_term") == 1


# -- Per-turn cap tests --


@pytest.mark.asyncio
async def test_per_turn_cap(store) -> None:
    """At most _MAX_PER_TURN memories should be stored per call."""
    llm = _make_mock_llm(
        [
            {"tier": "LONG_TERM", "category": "fact", "subject": "a", "content": "Fact A"},
            {"tier": "LONG_TERM", "category": "fact", "subject": "b", "content": "Fact B"},
            {"tier": "LONG_TERM", "category": "fact", "subject": "c", "content": "Fact C"},
            {"tier": "LONG_TERM", "category": "fact", "subject": "d", "content": "Fact D"},
            {"tier": "LONG_TERM", "category": "fact", "subject": "e", "content": "Fact E"},
        ]
    )

    stored = await store.extract_memories(
        llm,
        model="m",
        user_msg="lots of facts",
        agent_msg="ok",
        cooldown_seconds=0,
    )

    assert stored == 3  # capped at _MAX_PER_TURN
    assert await _count_rows(store.db_path, "long_term") == 3


# -- Short-term deduplication tests --


@pytest.mark.asyncio
async def test_short_term_dedup_skips_identical(store) -> None:
    """Identical short-term memories should not be stored twice."""
    llm = _make_mock_llm(
        [
            {
                "tier": "SHORT_TERM",
                "content": "Working from home today",
                "context": "daily update",
                "ttl_hours": 8,
            }
        ]
    )

    stored1 = await store.extract_memories(
        llm,
        model="m",
        user_msg="wfh",
        agent_msg="ok",
        cooldown_seconds=0,
    )
    assert stored1 == 1

    stored2 = await store.extract_memories(
        llm,
        model="m",
        user_msg="wfh again",
        agent_msg="ok",
        cooldown_seconds=0,
    )
    assert stored2 == 0
    assert await _count_rows(store.db_path, "short_term") == 1


@pytest.mark.asyncio
async def test_short_term_dedup_skips_substring(store) -> None:
    """Short-term memory that's a substring of existing one should be skipped."""
    llm1 = _LLMStub([])
    llm1._response = json.dumps(
        [
            {
                "tier": "SHORT_TERM",
                "content": "Working from home today due to rain",
                "context": "daily update",
                "ttl_hours": 8,
            }
        ]
    )

    await store.extract_memories(
        llm1,
        model="m",
        user_msg="wfh",
        agent_msg="ok",
        cooldown_seconds=0,
    )

    llm2 = _LLMStub([])
    llm2._response = json.dumps(
        [
            {
                "tier": "SHORT_TERM",
                "content": "Working from home today",
                "context": "daily update",
                "ttl_hours": 8,
            }
        ]
    )

    stored = await store.extract_memories(
        llm2,
        model="m",
        user_msg="wfh",
        agent_msg="ok",
        cooldown_seconds=0,
    )
    assert stored == 0
    assert await _count_rows(store.db_path, "short_term") == 1
