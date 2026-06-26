"""SQLite-backed secrets vault — storage, ACL, resolution (issue #19).

Sits on the same database file as :class:`core.config_store.ConfigStore` but owns
its own tables. Wraps the two vaults from :mod:`core.vault`:

* **infra_secrets** — encrypted with the :class:`~core.vault.InfraVault` machine
  key; resolved into config at load time via ``${vault:NAME}`` (with ``.env``
  fallback). Read by app code, not persona-scoped.
* **secrets** — encrypted with the :class:`~core.vault.PersonaVault` (envelope,
  password-derived). Used *by reference* from ``run_command`` as
  ``{{secret:NAME}}`` / ``{{secret:NAME.field}}`` after an ACL check. Values
  never enter the model's context.
* **secret_requests** — pending "agent needs a secret" requests, redeemed via a
  one-time secure web link.
* **vault_meta** — the wrapped DEK + salt for the persona vault.

Security boundary: placeholder substitution happens **only** where
:meth:`resolve_command_secrets` is called — the model-facing ``run_command``
dispatch — never in message/email tool bodies. See ``core/agent.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets as _secrets
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from core.vault import InfraVault, PersonaVault, VaultLocked, load_machine_key

log = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS infra_secrets (
    name        TEXT PRIMARY KEY,
    value       BLOB NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS secrets (
    name         TEXT PRIMARY KEY,
    value        BLOB NOT NULL,
    structured   INTEGER NOT NULL DEFAULT 0,
    shared       INTEGER NOT NULL DEFAULT 0,
    owner        TEXT NOT NULL DEFAULT '',
    description  TEXT NOT NULL DEFAULT '',
    expires_at   TEXT,
    max_uses     INTEGER,
    use_count    INTEGER NOT NULL DEFAULT 0,
    last_used_at TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS secret_requests (
    token_hash      TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    persona         TEXT NOT NULL DEFAULT '',
    reason          TEXT NOT NULL DEFAULT '',
    suggested_scope TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'pending',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vault_meta (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    wrapped_dek TEXT,
    salt        TEXT
);
"""

# A secret name: letters/digits/_/-/: (the ':' allows persona:<name>:* namespaces).
# No '.' — that is reserved as the field separator in {{secret:NAME.field}}.
_NAME_RE = re.compile(r"^[A-Za-z0-9_:-]+$")
_PLACEHOLDER_RE = re.compile(r"\{\{secret:([A-Za-z0-9_:-]+)(?:\.([A-Za-z0-9_-]+))?\}\}")


def _now() -> datetime:
    return datetime.now(UTC)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def valid_name(name: str) -> bool:
    return bool(_NAME_RE.match(name or ""))


def slugify_name(raw: str) -> str:
    """Turn an arbitrary label (e.g. a Bitwarden item name) into a valid name."""
    s = re.sub(r"[^A-Za-z0-9_:-]+", "_", (raw or "").strip()).strip("_")
    return s or "secret"


def parse_bitwarden_export(data: dict) -> list[dict]:
    """Extract login items from a Bitwarden JSON export (pure, testable).

    Returns one dict per login: ``{name, username, password, url, totp, notes}``.
    Only ``type == 1`` (login) items with a password are included.
    """
    out: list[dict] = []
    for item in data.get("items", []):
        if item.get("type") != 1:
            continue
        login = item.get("login") or {}
        password = login.get("password") or ""
        if not password:
            continue
        uris = login.get("uris") or []
        url = (uris[0].get("uri") if uris and isinstance(uris[0], dict) else "") or ""
        out.append(
            {
                "name": slugify_name(item.get("name", "")),
                "username": login.get("username") or "",
                "password": password,
                "url": url,
                "totp": login.get("totp") or "",
                "notes": item.get("notes") or "",
            }
        )
    return out


class SecretStore:
    """Vault storage + ACL + resolution on top of the config database."""

    def __init__(
        self,
        db_path: str = "data/config.db",
        infra_vault: InfraVault | None = None,
        persona_vault: PersonaVault | None = None,
    ) -> None:
        self.db_path = db_path
        self.infra = infra_vault if infra_vault is not None else InfraVault(load_machine_key())
        self.persona = persona_vault if persona_vault is not None else PersonaVault()
        self._ready = False
        self._infra_cache: dict[str, str] = {}

    async def _ensure_schema(self) -> None:
        if self._ready:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(_SCHEMA)
        self._ready = True

    # -- Persona vault lifecycle --------------------------------------------

    async def ensure_wrapped_dek(self, password: str) -> bool:
        """Create the wrapped DEK if none exists (first admin-password set).

        Returns True if a DEK was created. Also unseals the vault in-process.
        """
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute("SELECT wrapped_dek, salt FROM vault_meta WHERE id = 1")
            row = await cur.fetchone()
            if row and row[0] and row[1]:
                # Already initialised — just unseal in memory.
                self.persona.unseal(password, row[0], row[1])
                return False
            wrapped, salt = PersonaVault.create_wrapped_dek(password)
            await db.execute(
                "INSERT INTO vault_meta (id, wrapped_dek, salt) VALUES (1, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET wrapped_dek = excluded.wrapped_dek, "
                "salt = excluded.salt",
                (wrapped, salt),
            )
            await db.commit()
        self.persona.unseal(password, wrapped, salt)
        return True

    async def unseal_persona(self, password: str) -> bool:
        """Unwrap + cache the DEK using the admin password. Idempotent."""
        await self._ensure_schema()
        if self.persona.unsealed:
            return True
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute("SELECT wrapped_dek, salt FROM vault_meta WHERE id = 1")
            row = await cur.fetchone()
        if not row or not row[0] or not row[1]:
            return False
        return self.persona.unseal(password, row[0], row[1])

    async def rotate_password(self, old_password: str, new_password: str) -> None:
        """Re-wrap the DEK under a new admin password (or create it if missing)."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute("SELECT wrapped_dek, salt FROM vault_meta WHERE id = 1")
            row = await cur.fetchone()
            if row and row[0] and row[1]:
                from cryptography.fernet import InvalidToken

                try:
                    wrapped, salt = PersonaVault.rewrap(old_password, new_password, row[0], row[1])
                except InvalidToken, ValueError, TypeError:
                    # Old password wrong (or meta corrupt): cannot preserve the DEK.
                    # Leave existing secrets intact under the old wrapping rather than
                    # silently orphaning them.
                    log.warning("Vault rewrap skipped: could not unwrap DEK with old password")
                    return
            else:
                wrapped, salt = PersonaVault.create_wrapped_dek(new_password)
            await db.execute(
                "INSERT INTO vault_meta (id, wrapped_dek, salt) VALUES (1, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET wrapped_dek = excluded.wrapped_dek, "
                "salt = excluded.salt",
                (wrapped, salt),
            )
            await db.commit()
        self.persona.unseal(new_password, wrapped, salt)

    def persona_unsealed(self) -> bool:
        return self.persona.unsealed

    def lock_persona(self) -> None:
        self.persona.lock()

    # -- Persona secret CRUD -------------------------------------------------

    async def set_secret(
        self,
        name: str,
        value: str | dict,
        *,
        shared: bool = False,
        owner: str = "",
        description: str = "",
        expires_at: str | None = None,
        max_uses: int | None = None,
    ) -> None:
        """Encrypt and upsert a persona secret. ``value`` may be a scalar or dict."""
        await self._ensure_schema()
        if not valid_name(name):
            raise ValueError(f"invalid secret name: {name!r}")
        structured = isinstance(value, dict)
        plaintext = json.dumps(value) if structured else str(value)
        token = self.persona.encrypt(plaintext)  # raises VaultLocked if sealed
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO secrets "
                "(name, value, structured, shared, owner, description, expires_at, max_uses, "
                " use_count, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, datetime('now'), datetime('now')) "
                "ON CONFLICT(name) DO UPDATE SET value = excluded.value, "
                "structured = excluded.structured, shared = excluded.shared, "
                "owner = excluded.owner, description = excluded.description, "
                "expires_at = excluded.expires_at, max_uses = excluded.max_uses, "
                "use_count = 0, updated_at = datetime('now')",
                (
                    name,
                    token,
                    1 if structured else 0,
                    1 if shared else 0,
                    owner,
                    description,
                    expires_at,
                    max_uses,
                ),
            )
            await db.commit()

    async def get_secret(self, name: str) -> str | dict | None:
        """Decrypt and return a secret value (scalar or dict), or None if absent."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute("SELECT value, structured FROM secrets WHERE name = ?", (name,))
            row = await cur.fetchone()
        if not row:
            return None
        plaintext = self.persona.decrypt(row[0])  # raises VaultLocked if sealed
        return json.loads(plaintext) if row[1] else plaintext

    async def delete_secret(self, name: str) -> bool:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute("DELETE FROM secrets WHERE name = ?", (name,))
            await db.commit()
            return cur.rowcount > 0

    async def shared_names(self) -> set[str]:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute("SELECT name FROM secrets WHERE shared = 1")
            return {r[0] for r in await cur.fetchall()}

    async def list_secret_meta(self, allowed: set[str] | None = None) -> list[dict]:
        """List secret metadata — NEVER values. Optionally filter to ``allowed``."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT name, structured, shared, owner, description, expires_at, "
                "max_uses, use_count, last_used_at FROM secrets ORDER BY name"
            )
            rows = await cur.fetchall()
        out = []
        for r in rows:
            if allowed is not None and r["name"] not in allowed and not r["shared"]:
                continue
            out.append(
                {
                    "name": r["name"],
                    "structured": bool(r["structured"]),
                    "shared": bool(r["shared"]),
                    "owner": r["owner"],
                    "description": r["description"],
                    "expires_at": r["expires_at"],
                    "max_uses": r["max_uses"],
                    "use_count": r["use_count"],
                    "last_used_at": r["last_used_at"],
                }
            )
        return out

    # -- Resolution (the run_command substitution boundary) ------------------

    async def resolve_command_secrets(
        self, command: str, allowed: set[str] | None
    ) -> tuple[str, str | None]:
        """Substitute ``{{secret:NAME[.field]}}`` in ``command`` after an ACL check.

        ``allowed`` is the set of names the active persona may use; shared
        secrets are always permitted. ``allowed=None`` bypasses the ACL (used
        only for trusted, agent-constructed commands — never for model input).

        Returns ``(resolved_command, error)``. On any error the original command
        is returned unchanged and ``error`` is a message for the model. Single-use
        and audit accounting is applied only after every placeholder resolves.
        """
        matches = list(_PLACEHOLDER_RE.finditer(command))
        if not matches:
            return command, None

        if not self.persona.unsealed:
            return command, (
                "Secrets vault is locked. Ask the owner to open the admin UI to unlock it."
            )

        acl = None if allowed is None else (set(allowed) | await self.shared_names())

        # Validate + decrypt every referenced secret before substituting any.
        resolved_values: dict[tuple[str, str | None], str] = {}
        used_names: set[str] = set()
        for m in matches:
            name, field = m.group(1), m.group(2)
            if acl is not None and name not in acl:
                return command, (
                    f"Secret '{name}' is not in this persona's scope. "
                    "Request access or use a permitted secret."
                )
            value, err = await self._materialize(name)
            if err:
                return command, err
            if field is not None:
                if not isinstance(value, dict):
                    return command, f"Secret '{name}' is not structured; '.{field}' is invalid."
                if field not in value:
                    return command, f"Secret '{name}' has no field '{field}'."
                resolved_values[(name, field)] = str(value[field])
            else:
                if isinstance(value, dict):
                    return command, (
                        f"Secret '{name}' is structured; reference a field as "
                        f"{{{{secret:{name}.<field>}}}}."
                    )
                resolved_values[(name, field)] = str(value)
            used_names.add(name)

        def _sub(m: re.Match) -> str:
            return resolved_values[(m.group(1), m.group(2))]

        resolved = _PLACEHOLDER_RE.sub(_sub, command)
        await self._record_use(used_names)
        return resolved, None

    async def _materialize(self, name: str) -> tuple[str | dict | None, str | None]:
        """Fetch + decrypt a secret, honouring expiry. Returns (value, error)."""
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT value, structured, expires_at FROM secrets WHERE name = ?", (name,)
            )
            row = await cur.fetchone()
        if not row:
            return None, (
                f"Secret '{name}' is not available. Use request_secret to ask the owner for it."
            )
        expires = _parse_iso(row[2])
        if expires and _now() > expires:
            return None, f"Secret '{name}' has expired."
        try:
            plaintext = self.persona.decrypt(row[0])
        except VaultLocked:
            return None, "Secrets vault is locked."
        value = json.loads(plaintext) if row[1] else plaintext
        return value, None

    async def _record_use(self, names: set[str]) -> None:
        """Audit a successful resolution; delete single-use secrets once consumed."""
        if not names:
            return
        async with aiosqlite.connect(self.db_path) as db:
            for name in names:
                await db.execute(
                    "UPDATE secrets SET use_count = use_count + 1, "
                    "last_used_at = datetime('now') WHERE name = ?",
                    (name,),
                )
                # Single-use ("once"): drop the row once its budget is spent.
                await db.execute(
                    "DELETE FROM secrets WHERE name = ? AND max_uses IS NOT NULL "
                    "AND use_count >= max_uses",
                    (name,),
                )
            await db.commit()

    # -- Secret requests (secure-link flow) ----------------------------------

    async def create_request(
        self,
        name: str,
        persona: str = "",
        reason: str = "",
        suggested_scope: str = "",
        ttl_sec: int = 86_400,
    ) -> str:
        """Create a pending secret request and return the one-time token (plaintext).

        Only the SHA-256 of the token is stored; the plaintext goes into the
        secure link and is never persisted.
        """
        await self._ensure_schema()
        token = _secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        expires = datetime.fromtimestamp(_now().timestamp() + ttl_sec, tz=UTC).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO secret_requests "
                "(token_hash, name, persona, reason, suggested_scope, status, expires_at) "
                "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
                (token_hash, name, persona, reason, suggested_scope, expires),
            )
            await db.commit()
        return token

    async def get_request(self, token: str) -> dict | None:
        """Return a pending, unexpired request by token, or None."""
        await self._ensure_schema()
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM secret_requests WHERE token_hash = ? AND status = 'pending'",
                (token_hash,),
            )
            row = await cur.fetchone()
        if not row:
            return None
        expires = _parse_iso(row["expires_at"])
        if expires and _now() > expires:
            return None
        return dict(row)

    async def resolve_request(self, token: str) -> bool:
        await self._ensure_schema()
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "UPDATE secret_requests SET status = 'done' "
                "WHERE token_hash = ? AND status = 'pending'",
                (token_hash,),
            )
            await db.commit()
            return cur.rowcount > 0

    # -- Infra secrets (machine-key vault) -----------------------------------

    async def set_infra_secret(self, name: str, value: str, description: str = "") -> None:
        await self._ensure_schema()
        if not self.infra.available:
            raise RuntimeError("infra vault has no machine key configured")
        if not valid_name(name):
            raise ValueError(f"invalid infra secret name: {name!r}")
        token = self.infra.encrypt(value)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO infra_secrets (name, value, description, updated_at) "
                "VALUES (?, ?, ?, datetime('now')) "
                "ON CONFLICT(name) DO UPDATE SET value = excluded.value, "
                "description = excluded.description, updated_at = datetime('now')",
                (name, token, description),
            )
            await db.commit()
        self._infra_cache[name] = value

    async def get_infra_secret(self, name: str) -> str | None:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute("SELECT value FROM infra_secrets WHERE name = ?", (name,))
            row = await cur.fetchone()
        if not row:
            return None
        return self.infra.decrypt(row[0])

    async def delete_infra_secret(self, name: str) -> bool:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute("DELETE FROM infra_secrets WHERE name = ?", (name,))
            await db.commit()
        self._infra_cache.pop(name, None)
        return cur.rowcount > 0

    async def list_infra_names(self) -> list[dict]:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT name, description, updated_at FROM infra_secrets ORDER BY name"
            )
            return [dict(r) for r in await cur.fetchall()]

    async def load_infra_cache(self) -> dict[str, str]:
        """Decrypt every infra secret into memory for the sync ``${vault:}`` resolver.

        Called at boot (the infra vault is available headlessly). If the machine
        key is missing, the cache stays empty and ``${vault:}`` falls back to env.
        """
        await self._ensure_schema()
        self._infra_cache = {}
        if not self.infra.available:
            return self._infra_cache
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute("SELECT name, value FROM infra_secrets")
            rows = await cur.fetchall()
        for name, token in rows:
            try:
                self._infra_cache[name] = self.infra.decrypt(token)
            except Exception:
                log.warning("Failed to decrypt infra secret %r (wrong machine key?)", name)
        return self._infra_cache

    def infra_resolve(self, name: str) -> str | None:
        """Sync resolver for ``${vault:NAME}`` — cache, then ``.env`` fallback."""
        import os

        if name in self._infra_cache:
            return self._infra_cache[name]
        return os.environ.get(name)


if __name__ == "__main__":
    # ponytail: one runnable check exercising the full lifecycle on a temp db.
    import asyncio
    import tempfile

    def _check_bitwarden() -> None:
        export = {
            "items": [
                {
                    "type": 1,
                    "name": "ACME Portal",
                    "login": {
                        "username": "me@x.com",
                        "password": "p@ss",
                        "uris": [{"uri": "https://acme.test"}],
                        "totp": "JBSWY3DPEHPK3PXP",
                    },
                    "notes": "n",
                },
                {"type": 2, "name": "a secure note"},  # not a login -> skipped
                {"type": 1, "name": "no password", "login": {"username": "x"}},  # skipped
            ]
        }
        items = parse_bitwarden_export(export)
        assert len(items) == 1, items
        assert items[0]["name"] == "ACME_Portal"
        assert items[0]["password"] == "p@ss"
        assert items[0]["url"] == "https://acme.test"

    async def _check_store() -> None:
        with tempfile.TemporaryDirectory() as d:
            store = SecretStore(db_path=str(Path(d) / "config.db"))
            # Persona vault must be initialised + unsealed before writes.
            await store.ensure_wrapped_dek("admin-pw")
            assert store.persona_unsealed()

            # Scalar secret + ACL resolution.
            await store.set_secret("STRIPE", "sk_live_abc", owner="persona:finance")
            cmd = "curl -H 'Authorization: Bearer {{secret:STRIPE}}' https://api"
            resolved, err = await store.resolve_command_secrets(cmd, allowed={"STRIPE"})
            assert err is None and "sk_live_abc" in resolved and "{{secret" not in resolved

            # ACL denial: not in scope, not shared.
            _, err = await store.resolve_command_secrets(cmd, allowed=set())
            assert err and "scope" in err

            # Shared secret is allowed regardless of persona scope.
            await store.set_secret("GLOBAL", "g", shared=True)
            _, err = await store.resolve_command_secrets("x {{secret:GLOBAL}}", allowed=set())
            assert err is None

            # Structured secret + field reference.
            await store.set_secret("ACME", {"username": "u", "password": "pw"}, owner="persona:x")
            r, err = await store.resolve_command_secrets(
                "login {{secret:ACME.username}}:{{secret:ACME.password}}", allowed={"ACME"}
            )
            assert err is None and r == "login u:pw", r
            # Scalar referenced with a field, or struct referenced bare -> errors.
            _, err = await store.resolve_command_secrets("{{secret:STRIPE.x}}", allowed={"STRIPE"})
            assert err and "not structured" in err
            _, err = await store.resolve_command_secrets("{{secret:ACME}}", allowed={"ACME"})
            assert err and "structured" in err

            # Single-use ("once"): resolves once, then gone.
            await store.set_secret("ONCE", "one-shot", max_uses=1)
            r, err = await store.resolve_command_secrets("use {{secret:ONCE}}", allowed={"ONCE"})
            assert err is None and "one-shot" in r
            assert await store.get_secret("ONCE") is None  # consumed + deleted
            _, err = await store.resolve_command_secrets("use {{secret:ONCE}}", allowed={"ONCE"})
            assert err and "not available" in err

            # Expiry.
            past = "2000-01-01T00:00:00+00:00"
            await store.set_secret("OLD", "v", expires_at=past)
            _, err = await store.resolve_command_secrets("{{secret:OLD}}", allowed={"OLD"})
            assert err and "expired" in err

            # Audit recorded.
            meta = {m["name"]: m for m in await store.list_secret_meta()}
            assert meta["STRIPE"]["use_count"] == 1 and meta["STRIPE"]["last_used_at"]
            # list_secret_meta never leaks values.
            assert "value" not in meta["STRIPE"]

            # Requests: one-time token, hash-stored, redeemable once.
            tok = await store.create_request("NEWKEY", persona="finance", reason="need it")
            req = await store.get_request(tok)
            assert req and req["name"] == "NEWKEY" and req["status"] == "pending"
            assert await store.resolve_request(tok)
            assert await store.get_request(tok) is None  # no longer pending

            # Locked vault refuses resolution.
            store.lock_persona()
            _, err = await store.resolve_command_secrets("{{secret:GLOBAL}}", allowed={"GLOBAL"})
            assert err and "locked" in err

            # Infra vault round-trip (machine key supplied directly).
            store2 = SecretStore(
                db_path=str(Path(d) / "config2.db"), infra_vault=InfraVault("machine-key")
            )
            await store2.set_infra_secret("ANTHROPIC_API_KEY", "sk-ant-xyz")
            await store2.load_infra_cache()
            assert store2.infra_resolve("ANTHROPIC_API_KEY") == "sk-ant-xyz"
            assert store2.infra_resolve("MISSING") is None

    _check_bitwarden()
    asyncio.run(_check_store())
    print("secret_store.py self-check OK")
