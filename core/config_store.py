"""SQLite-backed configuration store.

On first boot, imports values from config.yml + .env (the file-based seed).
After that, config.db is the source of truth.  The admin API and setup
wizard read/write this store; the agent reads from it at startup and on
reload.

Config values are stored as key-value pairs with dotted paths
(e.g. "agent.name", "channels.telegram.bot_token").  Secrets are
stored alongside plain values — the database file should live on an
encrypted volume (same as memory.db).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
from pathlib import Path
from typing import Protocol, runtime_checkable

import aiosqlite

from core.config import Config, load_config
from core.email_config import materialize_himalaya_config

log = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS setup_state (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    completed   INTEGER NOT NULL DEFAULT 0,
    current_step TEXT NOT NULL DEFAULT 'welcome'
);
INSERT OR IGNORE INTO setup_state (id, completed, current_step) VALUES (1, 0, 'welcome');
"""

# Keys that hold secrets — these are redacted in read-only API responses.
SECRET_KEYS = frozenset(
    {
        "agent.anthropic_api_key",
        "agent.openai_api_key",
        "agent.google_api_key",
        "agent.grok_api_key",
        "agent.deepseek_api_key",
        "channels.telegram.bot_token",
        "admin.api_key",
        "admin.password_hash",
        "admin.password_salt",
        "search.api_key",
        "tools.gh.token",
        "tools.imagegen.api_key",
        "calendar.providers",
        "calendar.google_oauth_client_id",
        "calendar.google_oauth_client_secret",
        "calendar.google_oauth_token",
        "contacts.providers",
        "contacts.google_oauth_token",
        "calendar.google_oauth_token",
        "calendar.google_oauth_client_id",
        "calendar.google_oauth_client_secret",
    }
)

# Keys that match these prefixes are also considered secret.
SECRET_PREFIXES = ("calendar.", "contacts.", "email.")

# Setup wizard step order.
SETUP_STEPS = [
    "welcome",
    "llm",
    "identity",
    "agent",
    "telegram",
    "email",
    "calendar",
    "search",
    "browser",
    "imagegen",
    "admin",
    "secrets",
    "done",
]


@runtime_checkable
class _HasModelDump(Protocol):
    def model_dump(self) -> dict: ...


def _flatten(obj: object, prefix: str = "") -> dict[str, str]:
    """Flatten a nested dict/Pydantic model into dotted key-value pairs."""
    items: dict[str, str] = {}
    if isinstance(obj, _HasModelDump):
        obj = obj.model_dump()
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                items.update(_flatten(v, f"{full_key}."))
            elif isinstance(v, list):
                items[full_key] = json.dumps(v)
            else:
                items[full_key] = "" if v is None else str(v)
    return items


def _unflatten(flat: dict[str, str]) -> dict:
    """Reconstruct a nested dict from dotted key-value pairs."""
    result: dict = {}
    for key, value in flat.items():
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        # Try to parse JSON for lists/bools/ints
        d[parts[-1]] = _parse_value(value)
    return result


def _parse_value(value: str) -> object:
    """Attempt to parse a string value into its native type."""
    if value == "":
        return ""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    # Try int (but not strings starting with '+' — likely phone numbers)
    if not value.startswith("+"):
        try:
            return int(value)
        except ValueError:
            pass
    # Try JSON (for lists)
    if value.startswith("[") or value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError, ValueError:
            pass
    return value


def _is_secret(key: str) -> bool:
    """Check if a config key holds a secret value."""
    if key in SECRET_KEYS:
        return True
    return any(key.startswith(p) for p in SECRET_PREFIXES)


def _redact(value: str) -> str:
    """Redact a secret value for display."""
    if not value or len(value) < 8:
        return "***" if value else ""
    return value[:4] + "***" + value[-4:]


def _email_keys_changed(keys: list | tuple | set) -> bool:
    for key in keys:
        if key == "email.himalaya.toml" or str(key).startswith("email."):
            return True
    return False


def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return base64.b64encode(derived).decode(), base64.b64encode(salt).decode()


def _verify_password(password: str, hashed: str, salt: str) -> bool:
    try:
        salt_bytes = base64.b64decode(salt.encode())
        expected = base64.b64decode(hashed.encode())
    except ValueError, TypeError:
        return False
    derived = hashlib.pbkdf2_hmac("sha256", password.encode(), salt_bytes, 200_000)
    return hmac.compare_digest(derived, expected)


class ConfigStore:
    """Async SQLite-backed config store."""

    def __init__(self, db_path: str = "data/config.db"):
        self.db_path = db_path
        self._ready = False
        # Optional infra-vault resolver (name -> str | None), attached at boot so
        # Himalaya materialisation can expand ${vault:NAME} email passwords (#110).
        self.vault_resolve = None

    async def _ensure_schema(self) -> None:
        if self._ready:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(_SCHEMA)
        self._ready = True

    # -- Setup state ---------------------------------------------------------

    async def is_setup_complete(self) -> bool:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT completed FROM setup_state WHERE id = 1")
            row = await cursor.fetchone()
            return bool(row and row[0])

    async def get_setup_step(self) -> str:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT current_step FROM setup_state WHERE id = 1")
            row = await cursor.fetchone()
            return row[0] if row else "welcome"

    async def set_setup_step(self, step: str) -> None:
        await self._ensure_schema()
        completed = 1 if step == "done" else 0
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE setup_state SET current_step = ?, completed = ? WHERE id = 1",
                (step, completed),
            )
            await db.commit()

    # -- Config CRUD ---------------------------------------------------------

    async def get(self, key: str) -> str | None:
        """Get a single config value by dotted key."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT value FROM config WHERE key = ?", (key,))
            row = await cursor.fetchone()
            return row[0] if row else None

    async def get_many(self, prefix: str = "") -> dict[str, str]:
        """Get all config values, optionally filtered by key prefix."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            if prefix:
                cursor = await db.execute(
                    "SELECT key, value FROM config WHERE key LIKE ?",
                    (f"{prefix}%",),
                )
            else:
                cursor = await db.execute("SELECT key, value FROM config")
            return {row[0]: row[1] for row in await cursor.fetchall()}

    async def set(self, key: str, value: str) -> None:
        """Set a single config value."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO config (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            await db.commit()
        if _email_keys_changed([key]):
            await materialize_himalaya_config(self)

    async def set_many(self, values: dict[str, str]) -> None:
        """Set multiple config values atomically."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            for key, value in values.items():
                await db.execute(
                    "INSERT INTO config (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, str(value)),
                )
            await db.commit()
        if _email_keys_changed(list(values.keys())):
            await materialize_himalaya_config(self)

    async def delete(self, key: str) -> bool:
        """Delete a config value. Returns True if it existed."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("DELETE FROM config WHERE key = ?", (key,))
            await db.commit()
            deleted = cursor.rowcount > 0
        if deleted and _email_keys_changed([key]):
            await materialize_himalaya_config(self)
        return deleted

    # -- Bulk operations -----------------------------------------------------

    async def import_from_config(self, config: Config) -> int:
        """Import a Config object into the store. Returns count of keys written."""
        flat = _flatten(config)
        await self.set_many(flat)
        log.info("Imported %d config keys from Config object", len(flat))
        return len(flat)

    async def import_from_yaml(self, yaml_path: str = "config.yml") -> int:
        """Load config.yml + .env, then import into the store."""
        config = load_config(yaml_path)
        return await self.import_from_config(config)

    async def export_to_config(self, vault_resolve=None) -> Config:
        """Reconstruct a Config object from the store.

        When ``vault_resolve`` (a ``name -> str | None`` callable) is supplied,
        ``${vault:NAME}`` references are resolved against the encrypted infra
        vault, with ``.env`` fallback handled by the resolver itself.
        """
        flat = await self.get_many()
        nested = _unflatten(flat)
        if vault_resolve is not None:
            from core.config import resolve_vault_vars

            nested = resolve_vault_vars(nested, vault_resolve)
        return Config.model_validate(nested)

    # -- Redacted views (for API responses) ----------------------------------

    async def get_all_redacted(self) -> dict[str, str]:
        """Get all config values with secrets redacted."""
        raw = await self.get_many()
        return {k: (_redact(v) if _is_secret(k) else v) for k, v in raw.items()}

    async def get_section_redacted(self, section: str) -> dict[str, str]:
        """Get a config section with secrets redacted."""
        raw = await self.get_many(prefix=f"{section}.")
        return {k: (_redact(v) if _is_secret(k) else v) for k, v in raw.items()}

    # -- Seed on first boot --------------------------------------------------

    async def seed_if_empty(self, yaml_path: str = "config.yml") -> bool:
        """If the store is empty, import from config.yml + .env.

        Also seeds character content from character.md if it exists and the
        config key is not already set.

        Returns True if seeding was performed.
        """
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM config")
            row = await cursor.fetchone()
            count = row[0] if row else 0

        seeded = False

        if count == 0:
            if Path(yaml_path).exists():
                imported = await self.import_from_yaml(yaml_path)
                log.info("Seeded config store from %s (%d keys)", yaml_path, imported)

                # If the YAML had meaningful content (e.g. API key is set),
                # mark setup as complete so the wizard isn't forced on users
                # who already had a working config.
                api_key = await self.get("agent.anthropic_api_key")
                if api_key:
                    await self.set_setup_step("done")
                    log.info("Existing config has API key — marking setup as complete")
            else:
                log.info("No config.yml found — config store is empty, setup wizard required")

            seeded = True

        # Seed character from its .md file if not already in the store
        existing = await self.get("agent.character")
        if not existing:
            path = Path("character.md")
            if path.exists():
                content = path.read_text().strip()
                if content:
                    await self.set("agent.character", content)
                    log.info("Seeded agent.character from character.md")
                    seeded = True

        # #98: personalia merged into character. Fold any legacy agent.personalia
        # into agent.character (prepended, nothing lost), then drop the key so this
        # runs only once.
        legacy = await self.get("agent.personalia")
        if legacy and legacy.strip():
            character = await self.get("agent.character") or ""
            await self.set("agent.character", f"{legacy.strip()}\n\n{character}".strip())
            await self.delete("agent.personalia")
            log.info("Folded legacy agent.personalia into agent.character (#98)")
            seeded = True

        # #115: persona → agent config-key renames. Carry each legacy key's value
        # to its new name (only if the new one isn't already set), then drop the
        # old key so this runs once.
        for old_key, new_key in (
            ("agent.active_persona", "agent.active_agent"),
            ("agent.personae_dir", "agent.agents_dir"),
            ("agent.personae_db_path", "agent.agents_db_path"),
            ("accounts.persona_binding_migrated", "accounts.agent_binding_migrated"),
        ):
            old_val = await self.get(old_key)
            if old_val is None:
                continue
            if await self.get(new_key) is None:
                await self.set(new_key, old_val)
            await self.delete(old_key)
            log.info("Renamed legacy config key %s → %s (#115)", old_key, new_key)
            seeded = True

        return seeded

    async def ensure_admin_password(self) -> bool:
        """Ensure the admin password hash exists; seed from env if needed."""
        await self._ensure_schema()
        existing_hash = await self.get("admin.password_hash")
        existing_salt = await self.get("admin.password_salt")
        if existing_hash and existing_salt:
            return False

        from os import getenv

        existing_api_key = await self.get("admin.api_key")
        if existing_api_key:
            hashed, salt = _hash_password(existing_api_key)
            await self.set_many({"admin.password_hash": hashed, "admin.password_salt": salt})
            await self.delete("admin.api_key")
            return True

        seed_password = getenv("ADMIN_PASSWORD") or getenv("ADMIN_API_KEY")
        if not seed_password:
            return False

        hashed, salt = _hash_password(seed_password)
        await self.set_many({"admin.password_hash": hashed, "admin.password_salt": salt})
        return True

    async def set_admin_password(self, password: str) -> None:
        """Set a new admin password hash + salt."""
        hashed, salt = _hash_password(password)
        await self.set_many({"admin.password_hash": hashed, "admin.password_salt": salt})

    async def verify_admin_password(self, password: str) -> bool:
        """Verify a password against stored hash + salt."""
        stored_hash = await self.get("admin.password_hash")
        stored_salt = await self.get("admin.password_salt")
        if not stored_hash or not stored_salt:
            return False
        return _verify_password(password, stored_hash, stored_salt)
