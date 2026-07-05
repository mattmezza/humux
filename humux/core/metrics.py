"""Token usage tracking — records LLM token counts for dashboard metrics (#199).

Stores one row per LLM completion in the config database. Provides
time-windowed aggregates for the overview dashboard.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS token_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at DATETIME DEFAULT (datetime('now')),
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_input_tokens INTEGER DEFAULT 0,
    cache_creation_input_tokens INTEGER DEFAULT 0
);
"""

# Path to the config database (co-located for persistence).
_CONFIG_DB = "data/config.db"


class TokenUsageStore:
    """Records and queries LLM token usage."""

    def __init__(self, db_path: str = _CONFIG_DB):
        self.db_path = db_path

    async def _ensure_schema(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(_SCHEMA)

    async def record(
        self,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
    ) -> None:
        """Insert a token usage row."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO token_usage "
                "(provider, model, input_tokens, output_tokens, "
                "cache_read_input_tokens, cache_creation_input_tokens) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (provider, model, input_tokens, output_tokens,
                 cache_read_input_tokens, cache_creation_input_tokens),
            )
            await db.commit()

    async def totals_since(self, hours: int) -> dict:
        """Aggregate token usage for the last *hours*.

        Returns:
            total_input, total_output, total_cache_read, total_cache_creation,
            and per-provider breakdown.
        """
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            # Grand totals
            cursor = await db.execute(
                "SELECT "
                "COALESCE(SUM(input_tokens), 0) AS total_input, "
                "COALESCE(SUM(output_tokens), 0) AS total_output, "
                "COALESCE(SUM(cache_read_input_tokens), 0) AS total_cache_read, "
                "COALESCE(SUM(cache_creation_input_tokens), 0) AS total_cache_creation "
                "FROM token_usage "
                "WHERE recorded_at >= datetime('now', ?)",
                (f"-{hours} hours",),
            )
            row = await cursor.fetchone()
            total_input = row["total_input"] if row else 0
            total_output = row["total_output"] if row else 0
            total_cache_read = row["total_cache_read"] if row else 0
            total_cache_creation = row["total_cache_creation"] if row else 0

            # Per-provider breakdown
            cursor = await db.execute(
                "SELECT provider, "
                "COALESCE(SUM(input_tokens), 0) AS total_input, "
                "COALESCE(SUM(output_tokens), 0) AS total_output, "
                "COALESCE(SUM(cache_read_input_tokens), 0) AS total_cache_read, "
                "COALESCE(SUM(cache_creation_input_tokens), 0) AS total_cache_creation "
                "FROM token_usage "
                "WHERE recorded_at >= datetime('now', ?) "
                "GROUP BY provider ORDER BY total_input DESC",
                (f"-{hours} hours",),
            )
            rows = await cursor.fetchall()
            breakdown = {}
            for r in rows:
                breakdown[r["provider"]] = {
                    "input": r["total_input"],
                    "output": r["total_output"],
                    "cache_read": r["total_cache_read"],
                    "cache_creation": r["total_cache_creation"],
                }

        return {
            "total_input": total_input,
            "total_output": total_output,
            "total_cache_read": total_cache_read,
            "total_cache_creation": total_cache_creation,
            "breakdown": breakdown,
        }


# Module-level convenience: one store instance shared across the process.
_store: TokenUsageStore | None = None


def get_store() -> TokenUsageStore:
    global _store
    if _store is None:
        _store = TokenUsageStore()
    return _store


async def record_usage(
    provider: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> None:
    """Module-level convenience for recording usage from LLM call sites."""
    try:
        await get_store().record(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
        )
    except Exception:
        log.exception("Failed to record token usage")
