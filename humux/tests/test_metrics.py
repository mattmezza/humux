"""Tests for token-usage metrics: per-agent aggregation, migration, formatting."""

from __future__ import annotations

import aiosqlite
import pytest

from api.admin import _fmt_tokens
from core.metrics import TokenUsageStore


@pytest.mark.asyncio
async def test_top_agents_sorted_by_total(tmp_path) -> None:
    store = TokenUsageStore(db_path=str(tmp_path / "m.db"))
    await store.record("anthropic", "opus", input_tokens=100, output_tokens=10, agent="coach")
    await store.record("anthropic", "opus", input_tokens=50, output_tokens=5, agent="coach")
    await store.record("anthropic", "opus", input_tokens=500, output_tokens=5, agent="scribe")
    await store.record("anthropic", "opus", input_tokens=1, output_tokens=1)  # no agent

    tot = await store.totals_since(24)
    agents = tot["top_agents"]
    # Only attributed agents, most-consuming first: scribe (505) > coach (165).
    # The un-attributed row (no agent) is excluded from the "top agent" ranking...
    assert [a["agent"] for a in agents] == ["scribe", "coach"]
    assert agents[0]["total"] == 505
    # ...but still counts in the grand totals (which sum the whole table).
    assert tot["total_input"] == 651


@pytest.mark.asyncio
async def test_agent_column_migrates_onto_old_schema(tmp_path) -> None:
    # A DB created before the agent column existed must gain it on next open, not
    # crash the insert.
    db = str(tmp_path / "old.db")
    async with aiosqlite.connect(db) as c:
        await c.execute(
            "CREATE TABLE token_usage (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "recorded_at DATETIME DEFAULT (datetime('now')), provider TEXT NOT NULL, "
            "model TEXT NOT NULL, input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0, "
            "cache_read_input_tokens INTEGER DEFAULT 0, "
            "cache_creation_input_tokens INTEGER DEFAULT 0)"
        )
        await c.commit()

    store = TokenUsageStore(db_path=db)
    await store.record("anthropic", "opus", input_tokens=9, agent="coach")
    tot = await store.totals_since(24)
    assert tot["top_agents"][0]["agent"] == "coach"


def test_fmt_tokens_scales() -> None:
    assert _fmt_tokens(999) == "999"
    assert _fmt_tokens(12_530) == "12.53k"
    assert _fmt_tokens(4_500_000_000) == "4.5B"
    assert _fmt_tokens(2_000_000_000_000) == "2T"
