"""Tests for token-usage metrics: per-agent aggregation, migration, formatting."""

from __future__ import annotations

from unittest.mock import AsyncMock

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


@pytest.mark.asyncio
async def test_spend_periods_include_reported_estimated_and_unpriced_calls(tmp_path, monkeypatch):
    catalog = {
        "openai": {
            "models": {
                "gpt-example": {
                    "cost": {
                        "input": 2,
                        "output": 10,
                        "cache_read": 0.1,
                        "tiers": [
                            {
                                "tier": {"type": "context", "size": 272_000},
                                "input": 4,
                                "output": 15,
                                "cache_read": 0.2,
                            }
                        ],
                    }
                }
            }
        },
    }
    monkeypatch.setattr("core.metrics.price_catalog", AsyncMock(return_value=catalog))
    store = TokenUsageStore(str(tmp_path / "m.db"))
    await store.record("openrouter", "router/model", cost_usd=0.5)
    await store.record("openrouter", "router/free", cost_usd=0)
    await store.record("typesafe", "jev", input_tokens=500_000, pricing_model="jev-latest")
    # An identically named model behind a custom endpoint has no public rate.
    await store.record("openai", "gpt-example", input_tokens=100_000)
    async with aiosqlite.connect(store.db_path) as db:
        await db.execute(
            "INSERT INTO token_usage (provider, model, recorded_at, input_tokens, output_tokens, "
            "cache_read_input_tokens, cache_creation_input_tokens) "
            "VALUES ('openai', 'gpt-example', datetime('now', '-3 days'), "
            "400000, 1000, 100000, 300000), "
            "('openai', 'gpt-example', datetime('now', '-40 days'), 1000, 0, 0, 0), "
            "('deepseek', 'unlisted', datetime('now', '-20 days'), 1000, 0, 100, 900)"
        )
        await db.commit()

    spend = await store.spend_totals()
    assert spend["24h"]["usd"] == pytest.approx(0.521)
    assert spend["24h"]["unpriced"] == 1
    assert spend["7d"]["usd"] == pytest.approx(0.521 + 1.235)
    assert spend["30d"]["unpriced"] == 2
    assert spend["All time"]["usd"] == pytest.approx(0.521 + 1.235 + 0.002)
    assert spend["All time"]["providers"] == [
        {"provider": "openai", "usd": pytest.approx(1.237), "estimated": True, "unpriced": 1},
        {"provider": "openrouter", "usd": 0.5, "estimated": False, "unpriced": 0},
        {"provider": "typesafe", "usd": pytest.approx(0.021), "estimated": True, "unpriced": 0},
        {"provider": "deepseek", "usd": None, "estimated": False, "unpriced": 1},
    ]


@pytest.mark.asyncio
async def test_reported_zero_spend_needs_no_catalog(tmp_path, monkeypatch):
    catalog = AsyncMock()
    monkeypatch.setattr("core.metrics.price_catalog", catalog)
    store = TokenUsageStore(str(tmp_path / "m.db"))
    assert (await store.spend_totals())["24h"]["usd"] is None
    await store.record("openrouter", "router/free", cost_usd=0)
    assert (await store.spend_totals())["24h"]["usd"] == 0
    catalog.assert_not_awaited()
