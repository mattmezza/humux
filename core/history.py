"""Conversation history — persists user/assistant turns in SQLite.

Supports two modes:
- **injection**: windowed history replayed as native alternating messages.
- **session**: sticky session per (channel, user_id, chat_id) — full message array
  persisted and kept in memory for cache-friendly LLM calls.

Sessions are keyed by (channel, user_id, chat_id) so that the same user
talking in different chats (e.g. private vs. group) gets separate histories.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import aiosqlite

log = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS conversation_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel TEXT NOT NULL,
    user_id TEXT NOT NULL,
    chat_id TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at DATETIME DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_turns_lookup
    ON conversation_turns(channel, user_id, chat_id, created_at);
CREATE TABLE IF NOT EXISTS session_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel TEXT NOT NULL,
    user_id TEXT NOT NULL,
    chat_id TEXT NOT NULL DEFAULT '',
    message TEXT NOT NULL,
    created_at DATETIME DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_session_lookup
    ON session_messages(channel, user_id, chat_id, id);
CREATE TABLE IF NOT EXISTS session_system (
    channel TEXT NOT NULL,
    user_id TEXT NOT NULL,
    chat_id TEXT NOT NULL DEFAULT '',
    system TEXT NOT NULL,
    created_at DATETIME DEFAULT (datetime('now')),
    PRIMARY KEY (channel, user_id, chat_id)
);
CREATE TABLE IF NOT EXISTS chat_persona (
    channel TEXT NOT NULL,
    user_id TEXT NOT NULL,
    chat_id TEXT NOT NULL DEFAULT '',
    persona TEXT NOT NULL,
    created_at DATETIME DEFAULT (datetime('now')),
    updated_at DATETIME DEFAULT (datetime('now')),
    PRIMARY KEY (channel, user_id, chat_id)
);
"""

# Migrations applied after initial schema creation.
# Each entry is (description, SQL) — only runs if the column/index is missing.
_MIGRATIONS = [
    (
        "add chat_id to conversation_turns",
        "ALTER TABLE conversation_turns ADD COLUMN chat_id TEXT NOT NULL DEFAULT ''",
    ),
    (
        "add chat_id to session_messages",
        "ALTER TABLE session_messages ADD COLUMN chat_id TEXT NOT NULL DEFAULT ''",
    ),
    (
        "create idx_turns_lookup_v2",
        "CREATE INDEX IF NOT EXISTS idx_turns_lookup_v2 "
        "ON conversation_turns(channel, user_id, chat_id, created_at)",
    ),
    (
        "create idx_session_lookup_v2",
        "CREATE INDEX IF NOT EXISTS idx_session_lookup_v2 "
        "ON session_messages(channel, user_id, chat_id, id)",
    ),
]


class ConversationHistory:
    """Stores and retrieves conversation turns per user+channel+chat."""

    def __init__(self, db_path: str = "data/history.db", max_turns: int = 20):
        self.db_path = db_path
        self.max_turns = max_turns
        self._ready = False
        # In-memory cache for sticky sessions: {(channel, user_id, chat_id): [message_dicts]}
        self._sessions: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        # In-memory cache for the static system prompt snapshot per session.
        self._session_system: dict[tuple[str, str, str], str] = {}

    async def _ensure_schema(self) -> None:
        if self._ready:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(_SCHEMA)
            # Run migrations for existing databases that lack the chat_id column.
            for desc, sql in _MIGRATIONS:
                try:
                    await db.execute(sql)
                    await db.commit()
                except Exception:
                    pass  # Column/index already exists
        self._ready = True

    # -------------------------------------------------------------------
    # Injection mode — windowed history as native messages
    # -------------------------------------------------------------------

    async def get_messages(self, channel: str, user_id: str, chat_id: str = "") -> list[dict]:
        """Return the last *max_turns* user-assistant **pairs** with timestamps.

        Each returned dict has ``role``, ``content`` and ``created_at`` keys.
        The limit is applied to *pairs* (not individual rows) so the history
        never starts with an orphaned assistant reply.
        """
        await self._ensure_schema()
        # We select the N most-recent *user* rows by id and then grab every
        # row whose id >= the smallest of those.  Because the assistant reply
        # is always inserted right after the user message, this guarantees we
        # never slice in the middle of a pair.
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT role, content, created_at FROM conversation_turns
                WHERE channel = ? AND user_id = ? AND chat_id = ?
                  AND id >= (
                      SELECT MIN(id) FROM (
                          SELECT id
                          FROM conversation_turns
                          WHERE channel = ? AND user_id = ? AND chat_id = ? AND role = 'user'
                          ORDER BY id DESC
                          LIMIT ?
                      )
                  )
                ORDER BY id ASC
                """,
                (channel, user_id, chat_id, channel, user_id, chat_id, self.max_turns),
            )
            rows = await cursor.fetchall()

        return [
            {"role": role, "content": json.loads(content), "created_at": created_at}
            for role, content, created_at in rows
        ]

    async def add_turn(
        self, channel: str, user_id: str, role: str, content: str, chat_id: str = ""
    ) -> None:
        """Store a single message (user or assistant text)."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO conversation_turns (channel, user_id, chat_id, role, content) "
                "VALUES (?, ?, ?, ?, ?)",
                (channel, user_id, chat_id, role, json.dumps(content)),
            )
            await db.commit()

    async def clear(self, channel: str, user_id: str, chat_id: str = "") -> None:
        """Clear conversation history for a user+channel+chat triple (both modes)."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM conversation_turns WHERE channel = ? AND user_id = ? AND chat_id = ?",
                (channel, user_id, chat_id),
            )
            await db.execute(
                "DELETE FROM session_messages WHERE channel = ? AND user_id = ? AND chat_id = ?",
                (channel, user_id, chat_id),
            )
            await db.execute(
                "DELETE FROM session_system WHERE channel = ? AND user_id = ? AND chat_id = ?",
                (channel, user_id, chat_id),
            )
            await db.commit()
        # Clear in-memory session cache
        self._sessions.pop((channel, user_id, chat_id), None)
        self._session_system.pop((channel, user_id, chat_id), None)

    # -------------------------------------------------------------------
    # Session mode — sticky session per (channel, user_id, chat_id)
    # -------------------------------------------------------------------

    async def get_session(
        self, channel: str, user_id: str, chat_id: str = ""
    ) -> list[dict[str, Any]]:
        """Return the full session message array for a (channel, user_id, chat_id) triple.

        Loads from SQLite on first access, then serves from in-memory cache.
        """
        await self._ensure_schema()
        key = (channel, user_id, chat_id)
        if key not in self._sessions:
            # Load from DB
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "SELECT message FROM session_messages "
                    "WHERE channel = ? AND user_id = ? AND chat_id = ? ORDER BY id ASC",
                    (channel, user_id, chat_id),
                )
                rows = await cursor.fetchall()
            self._sessions[key] = [json.loads(row[0]) for row in rows]
        return self._sessions[key]

    async def append_session_message(
        self, channel: str, user_id: str, message: dict[str, Any], chat_id: str = ""
    ) -> None:
        """Append a message to the sticky session and persist it."""
        await self._ensure_schema()
        key = (channel, user_id, chat_id)
        # Ensure the in-memory cache is loaded
        if key not in self._sessions:
            await self.get_session(channel, user_id, chat_id)
        self._sessions[key].append(message)
        # Persist
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO session_messages (channel, user_id, chat_id, message) "
                "VALUES (?, ?, ?, ?)",
                (channel, user_id, chat_id, json.dumps(message)),
            )
            await db.commit()

    async def append_session_messages(
        self,
        channel: str,
        user_id: str,
        messages: list[dict[str, Any]],
        chat_id: str = "",
    ) -> None:
        """Append multiple messages to the sticky session and persist them."""
        if not messages:
            return
        await self._ensure_schema()
        key = (channel, user_id, chat_id)
        if key not in self._sessions:
            await self.get_session(channel, user_id, chat_id)
        self._sessions[key].extend(messages)
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(
                "INSERT INTO session_messages (channel, user_id, chat_id, message) "
                "VALUES (?, ?, ?, ?)",
                [(channel, user_id, chat_id, json.dumps(m)) for m in messages],
            )
            await db.commit()

    async def replace_session(
        self,
        channel: str,
        user_id: str,
        messages: list[dict[str, Any]],
        chat_id: str = "",
    ) -> None:
        """Atomically replace a session's messages (used by compaction).

        Rewrites both the in-memory cache and the persisted ``session_messages``
        rows. The system-prompt snapshot is left untouched.
        """
        await self._ensure_schema()
        key = (channel, user_id, chat_id)
        self._sessions[key] = list(messages)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM session_messages WHERE channel = ? AND user_id = ? AND chat_id = ?",
                (channel, user_id, chat_id),
            )
            await db.executemany(
                "INSERT INTO session_messages (channel, user_id, chat_id, message) "
                "VALUES (?, ?, ?, ?)",
                [(channel, user_id, chat_id, json.dumps(m)) for m in messages],
            )
            await db.commit()

    async def clear_session(self, channel: str, user_id: str, chat_id: str = "") -> None:
        """Clear just the sticky session for a (channel, user_id, chat_id) triple."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM session_messages WHERE channel = ? AND user_id = ? AND chat_id = ?",
                (channel, user_id, chat_id),
            )
            await db.execute(
                "DELETE FROM session_system WHERE channel = ? AND user_id = ? AND chat_id = ?",
                (channel, user_id, chat_id),
            )
            await db.commit()
        self._sessions.pop((channel, user_id, chat_id), None)
        self._session_system.pop((channel, user_id, chat_id), None)

    # -------------------------------------------------------------------
    # Session mode — static system prompt snapshot
    # -------------------------------------------------------------------

    async def get_session_system(self, channel: str, user_id: str, chat_id: str = "") -> str | None:
        """Return the cached static system prompt for a session, or None if unset.

        The system prompt is snapshotted once at the start of a session (after a
        ``/new``) and reused for every subsequent turn, so the static content is
        only built/sent once instead of being rebuilt each turn.
        """
        await self._ensure_schema()
        key = (channel, user_id, chat_id)
        if key in self._session_system:
            return self._session_system[key]
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT system FROM session_system "
                "WHERE channel = ? AND user_id = ? AND chat_id = ?",
                (channel, user_id, chat_id),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        self._session_system[key] = row[0]
        return row[0]

    async def set_session_system(
        self, channel: str, user_id: str, system: str, chat_id: str = ""
    ) -> None:
        """Persist the static system prompt snapshot for a session."""
        await self._ensure_schema()
        self._session_system[(channel, user_id, chat_id)] = system
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO session_system (channel, user_id, chat_id, system) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(channel, user_id, chat_id) DO UPDATE SET system = excluded.system",
                (channel, user_id, chat_id, system),
            )
            await db.commit()

    async def clear_session_system(self, channel: str, user_id: str, chat_id: str = "") -> None:
        """Drop just the snapshotted system prompt for a session (keep messages).

        Used when the bound persona changes mid-session so the next turn rebuilds
        the static prompt with the new identity without wiping the conversation.
        """
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM session_system WHERE channel = ? AND user_id = ? AND chat_id = ?",
                (channel, user_id, chat_id),
            )
            await db.commit()
        self._session_system.pop((channel, user_id, chat_id), None)

    # -------------------------------------------------------------------
    # Per-chat persona binding — (channel, user_id, chat_id) -> persona name
    # -------------------------------------------------------------------

    async def get_chat_persona(self, channel: str, user_id: str, chat_id: str = "") -> str | None:
        """Return the persona name bound to this triple, or None if unbound.

        Not cached: a primary-key lookup per turn is cheap, and skipping the
        cache avoids staleness when the binding is changed from the admin UI on
        a different store instance.
        """
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT persona FROM chat_persona "
                "WHERE channel = ? AND user_id = ? AND chat_id = ?",
                (channel, user_id, chat_id),
            )
            row = await cursor.fetchone()
        return row[0] if row else None

    async def set_chat_persona(
        self, channel: str, user_id: str, persona: str, chat_id: str = ""
    ) -> None:
        """Bind a (channel, user_id, chat_id) triple to a persona name (upsert)."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO chat_persona (channel, user_id, chat_id, persona) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(channel, user_id, chat_id) DO UPDATE SET "
                "persona = excluded.persona, updated_at = datetime('now')",
                (channel, user_id, chat_id, persona),
            )
            await db.commit()

    async def clear_chat_persona(self, channel: str, user_id: str, chat_id: str = "") -> None:
        """Remove a per-chat persona binding (revert to global/default identity)."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM chat_persona WHERE channel = ? AND user_id = ? AND chat_id = ?",
                (channel, user_id, chat_id),
            )
            await db.commit()

    async def bind_chat_persona(
        self, channel: str, user_id: str, chat_id: str, persona: str
    ) -> None:
        """Bind (or, with an empty name, unbind) a chat to a persona.

        Drops the snapshotted session system prompt so a new identity takes effect
        on the next turn without wiping the conversation (in injection mode there
        is no snapshot, so the clear is a harmless no-op). Call this on the running
        agent's history instance so its ``_session_system`` cache is the one that
        gets cleared.
        """
        name = (persona or "").strip()
        if name:
            await self.set_chat_persona(channel, user_id, name, chat_id)
        else:
            await self.clear_chat_persona(channel, user_id, chat_id)
        await self.clear_session_system(channel, user_id, chat_id)

    async def list_chats(self) -> list[dict[str, str]]:
        """List every known (channel, user_id, chat_id) with its bound persona.

        Drives the admin Chats page. Unions the turn/session/binding tables so a
        chat appears whichever history mode produced it (and even when it is only
        bound, e.g. a topic auto-bound before its first message). LEFT JOIN
        surfaces the binding; ``persona`` is "" when unbound.
        """
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT c.channel, c.user_id, c.chat_id, p.persona
                FROM (
                    SELECT DISTINCT channel, user_id, chat_id FROM conversation_turns
                    UNION
                    SELECT DISTINCT channel, user_id, chat_id FROM session_messages
                    UNION
                    SELECT DISTINCT channel, user_id, chat_id FROM chat_persona
                ) AS c
                LEFT JOIN chat_persona AS p
                  ON p.channel = c.channel AND p.user_id = c.user_id
                  AND p.chat_id = c.chat_id
                ORDER BY c.channel, c.user_id, c.chat_id
                """
            )
            rows = await cursor.fetchall()
        return [
            {"channel": ch, "user_id": uid, "chat_id": cid, "persona": persona or ""}
            for ch, uid, cid, persona in rows
        ]
