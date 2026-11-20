"""Token usage and inference spend for the overview dashboard.

Stores one row per LLM completion in the config database. Provides
time-windowed aggregates for the overview dashboard.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import aiosqlite

from core.pricing import estimate_usd, price_catalog

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS token_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at DATETIME DEFAULT (datetime('now')),
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    agent TEXT DEFAULT '',
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_input_tokens INTEGER DEFAULT 0,
    cache_creation_input_tokens INTEGER DEFAULT 0,
    cost_usd REAL,
    pricing_model TEXT
);
"""

# Applied to existing DBs on open.
_MIGRATIONS = {"agent": "TEXT DEFAULT ''", "cost_usd": "REAL", "pricing_model": "TEXT"}

# Path to the config database (co-located for persistence).
_CONFIG_DB = "data/config.db"
_SPEND_PERIODS = ("24h", "7d", "30d", "All time")

# Older OpenAI-compatible rows recorded the entire prompt as input, and could
# record DeepSeek cache misses as writes. New rows store disjoint input counts.
# Group by context size to apply per-request long-context rates before summing.
_SPEND_QUERY = """
WITH usage AS (
    SELECT provider, model, pricing_model, cost_usd, output_tokens, cache_read_input_tokens,
        CASE WHEN pricing_model IS NULL AND provider != 'anthropic'
             THEN MAX(0, input_tokens - cache_read_input_tokens)
             ELSE input_tokens END AS uncached,
        CASE WHEN pricing_model IS NULL AND provider != 'anthropic'
             THEN 0 ELSE cache_creation_input_tokens END AS cache_write,
        CASE WHEN recorded_at >= datetime('now', '-24 hours') THEN 0
             WHEN recorded_at >= datetime('now', '-7 days') THEN 1
             WHEN recorded_at >= datetime('now', '-30 days') THEN 2
             ELSE 3 END AS age
    FROM token_usage
)
SELECT provider, model, pricing_model, age,
    CASE WHEN cost_usd IS NULL THEN uncached + cache_read_input_tokens + cache_write
         ELSE 0 END AS context,
    SUM(cost_usd) AS reported_cost,
    SUM(uncached) AS input_tokens, SUM(output_tokens) AS output_tokens,
    SUM(cache_read_input_tokens) AS cache_read, SUM(cache_write) AS cache_write,
    COUNT(*) AS calls
FROM usage
GROUP BY provider, model, pricing_model, age, context, cost_usd IS NULL
"""


@dataclass
class Spend:
    usd: float | None = None
    estimated: bool = False
    unpriced: int = 0

    def add(self, usd: float | None, estimated: bool, calls: int) -> None:
        if usd is None:
            self.unpriced += calls
        else:
            self.usd = (self.usd or 0.0) + usd
            self.estimated |= estimated


class TokenUsageStore:
    """Records and queries LLM token usage."""

    def __init__(self, db_path: str = _CONFIG_DB):
        self.db_path = db_path

    def _connect(self) -> aiosqlite.Connection:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        return aiosqlite.connect(self.db_path)

    async def _ensure_schema(self, db: aiosqlite.Connection) -> None:
        await db.execute(_SCHEMA)
        columns = {row[1] for row in await db.execute_fetchall("PRAGMA table_info(token_usage)")}
        for column in _MIGRATIONS.keys() - columns:
            try:
                await db.execute(
                    f"ALTER TABLE token_usage ADD COLUMN {column} {_MIGRATIONS[column]}"
                )
            except aiosqlite.OperationalError as exc:
                # A concurrent connection may have added the column.
                if "duplicate column name" not in str(exc).lower():
                    raise

    async def record(
        self,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        agent: str = "",
        *,
        cost_usd: float | None = None,
        pricing_model: str = "",
    ) -> None:
        """Insert a usage row. An empty pricing model leaves unknown costs unpriced."""
        async with self._connect() as db:
            await self._ensure_schema(db)
            await db.execute(
                "INSERT INTO token_usage "
                "(provider, model, agent, input_tokens, output_tokens, "
                "cache_read_input_tokens, cache_creation_input_tokens, cost_usd, pricing_model) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    provider,
                    model,
                    agent,
                    input_tokens,
                    output_tokens,
                    cache_read_input_tokens,
                    cache_creation_input_tokens,
                    cost_usd,
                    pricing_model,
                ),
            )
            await db.commit()

    async def totals_since(self, hours: int) -> dict:
        """Aggregate token usage for the last *hours*.

        Returns:
            total_input, total_output, total_cache_read, total_cache_creation,
            and per-provider breakdown.
        """
        async with self._connect() as db:
            await self._ensure_schema(db)
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

            # Per-agent totals (all token kinds), most-consuming first — feeds the
            # "top agent" dashboard card. Only rows with a recorded agent: usage
            # with no attribution (rows written before this column shipped, or a
            # scheduled/anonymous run outside any turn) is NOT an agent, so it must
            # not headline a "top agent" ranking — it still counts in the total /
            # cached cards, which sum the whole table.
            cursor = await db.execute(
                "SELECT agent, "
                "COALESCE(SUM(input_tokens + output_tokens + cache_read_input_tokens "
                "+ cache_creation_input_tokens), 0) AS total "
                "FROM token_usage "
                "WHERE recorded_at >= datetime('now', ?) AND agent IS NOT NULL AND agent != '' "
                "GROUP BY agent ORDER BY total DESC",
                (f"-{hours} hours",),
            )
            top_agents = [
                {"agent": r["agent"], "total": r["total"]} for r in await cursor.fetchall()
            ]

        return {
            "total_input": total_input,
            "total_output": total_output,
            "total_cache_read": total_cache_read,
            "total_cache_creation": total_cache_creation,
            "breakdown": breakdown,
            "top_agents": top_agents,
        }

    async def spend_totals(self) -> dict:
        """Sum recorded and estimated USD by provider for each overview period."""
        async with self._connect() as db:
            await self._ensure_schema(db)
            db.row_factory = aiosqlite.Row
            rows = await db.execute_fetchall(_SPEND_QUERY)

        catalog = {}
        if any(
            row["reported_cost"] is None
            and row["pricing_model"] != ""
            and row["provider"] != "typesafe"
            for row in rows
        ):
            catalog = await price_catalog()
        totals = [Spend() for _ in _SPEND_PERIODS]
        providers: list[dict[str, Spend]] = [{} for _ in _SPEND_PERIODS]
        for row in rows:
            provider = row["provider"]
            model = row["pricing_model"] if row["pricing_model"] is not None else row["model"]
            usd = row["reported_cost"]
            estimated = usd is None
            if estimated and model:
                usd = estimate_usd(
                    catalog,
                    provider,
                    model,
                    row["context"],
                    row["input_tokens"],
                    row["output_tokens"],
                    row["cache_read"],
                    row["cache_write"],
                )
            for period in range(row["age"], len(_SPEND_PERIODS)):
                totals[period].add(usd, estimated, row["calls"])
                providers[period].setdefault(provider, Spend()).add(usd, estimated, row["calls"])

        return {
            period: {
                **asdict(total),
                "providers": [
                    {"provider": provider, **asdict(spend)}
                    for provider, spend in sorted(
                        breakdown.items(), key=lambda entry: (-(entry[1].usd or 0), entry[0])
                    )
                ],
            }
            for period, total, breakdown in zip(_SPEND_PERIODS, totals, providers, strict=True)
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
    agent: str = "",
    *,
    cost_usd: float | None = None,
    pricing_model: str = "",
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
            agent=agent,
            cost_usd=cost_usd,
            pricing_model=pricing_model,
        )
    except Exception:
        log.exception("Failed to record token usage")
