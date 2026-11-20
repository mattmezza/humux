"""Public list-rate estimates for recorded inference usage."""

from __future__ import annotations

import asyncio
import math
import re
import time

import httpx

_CATALOG_URL = "https://models.dev/api.json"
_catalog: dict = {}
_refresh_at = 0.0
_catalog_lock = asyncio.Lock()


async def price_catalog() -> dict:
    """Cache the public catalog. Use the last successful fetch during outages."""
    global _catalog, _refresh_at
    if time.monotonic() < _refresh_at:
        return _catalog
    async with _catalog_lock:
        if time.monotonic() >= _refresh_at:
            try:
                async with asyncio.timeout(3), httpx.AsyncClient() as client:
                    response = await client.get(_CATALOG_URL)
                    response.raise_for_status()
                    catalog = response.json()
                if not isinstance(catalog, dict) or "openai" not in catalog:
                    raise ValueError("Invalid pricing catalog")
                _catalog = catalog
                _refresh_at = time.monotonic() + 86400
            except httpx.HTTPError, TimeoutError, ValueError:
                _refresh_at = time.monotonic() + 300
    return _catalog


def usd_amount(value: object) -> float | None:
    """Accept a nonnegative monetary amount, including an explicit zero."""
    if type(value) in (int, float) and 0 <= value < math.inf:
        return float(value)
    return None


def estimate_usd(
    catalog: dict,
    provider: str,
    model: str,
    context: int,
    input_tokens: int,
    output_tokens: int,
    cache_read: int,
    cache_write: int,
) -> float | None:
    """Apply USD per million tokens for the model and request's context tier."""
    if provider == "typesafe" and (model == "jev" or model.startswith("jev-")):
        # https://help.typesafe.ai/articles/5187202169-how-much-does-jev-cost
        return input_tokens * 0.042 / 1_000_000
    models = catalog.get("xai" if provider == "grok" else provider, {}).get("models", {})
    entry = models.get(model)
    if entry is None and provider in ("openai", "anthropic"):
        # SDKs often return the resolved, dated version of a catalog alias.
        alias = re.sub(r"-(?:\d{4}-\d{2}-\d{2}|\d{8})$", "", model)
        entry = models.get(alias)
    if entry is None or not isinstance(cost := entry.get("cost"), dict):
        return None

    threshold = 0
    rates = cost
    for candidate in cost.get("tiers") or []:
        tier = candidate.get("tier", {})
        size = tier.get("size")
        if tier.get("type") == "context" and isinstance(size, int) and threshold < size < context:
            threshold = size
            rates = cost | candidate
    if threshold == 0 and not cost.get("tiers") and context > 200_000:
        rates = cost | (cost.get("context_over_200k") or {})

    total = 0.0
    for key, tokens in (
        ("input", input_tokens),
        ("output", output_tokens),
        ("cache_read", cache_read),
        ("cache_write", cache_write),
    ):
        if tokens:
            rate = usd_amount(rates.get(key))
            if rate is None:
                return None
            total += rate * tokens
    return total / 1_000_000
