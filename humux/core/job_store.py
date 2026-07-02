"""Job store — single SQLite source of truth for all scheduled jobs.

Replaces the dual APScheduler-jobstore + config-store approach.
Every job (cron or one-shot) lives in a single ``jobs`` table in
``data/jobs.db``.  The scheduler reads from here; the admin UI, CLI
tool, and agent tool all write here.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import aiosqlite

log = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    type        TEXT NOT NULL DEFAULT 'agent',
    schedule    TEXT NOT NULL DEFAULT 'cron',
    cron        TEXT,
    run_at      TEXT,
    task        TEXT NOT NULL DEFAULT '',
    channel     TEXT NOT NULL DEFAULT 'telegram',
    status      TEXT NOT NULL DEFAULT 'active',
    created_by  TEXT NOT NULL DEFAULT 'admin',
    description TEXT NOT NULL DEFAULT '',
    agent     TEXT NOT NULL DEFAULT '',
    origin_user_id TEXT NOT NULL DEFAULT '',
    origin_chat_id TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

# Additive migrations for DBs created before a column existed. Each is a column
# name → ALTER statement; applied only when the column is missing.
_MIGRATIONS = {
    "agent": "ALTER TABLE jobs ADD COLUMN agent TEXT NOT NULL DEFAULT ''",
    "origin_user_id": "ALTER TABLE jobs ADD COLUMN origin_user_id TEXT NOT NULL DEFAULT ''",
    "origin_chat_id": "ALTER TABLE jobs ADD COLUMN origin_chat_id TEXT NOT NULL DEFAULT ''",
}

# Valid values. "subagent" runs the spawn_subagent primitive under ``agent``.
VALID_TYPES = ("agent", "agent_silent", "system", "memory_consolidation", "subagent")
VALID_SCHEDULES = ("cron", "once")
VALID_STATUSES = ("active", "paused", "done", "cancelled")


def _row_to_dict(row: aiosqlite.Row | sqlite3.Row) -> dict[str, Any]:
    """Convert a Row to a plain dict."""
    return dict(row)


class JobStore:
    """Async SQLite-backed job store."""

    def __init__(self, db_path: str = "data/jobs.db"):
        self.db_path = db_path
        self._ready = False

    async def _ensure_schema(self) -> None:
        if self._ready:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(_SCHEMA)
            cursor = await db.execute("PRAGMA table_info(jobs)")
            cols = {row[1] for row in await cursor.fetchall()}
            # #115: rename the legacy `persona` column to `agent` in place (keeps
            # each job's identity) instead of adding a fresh empty `agent` column.
            if "persona" in cols and "agent" not in cols:
                await db.execute("ALTER TABLE jobs RENAME COLUMN persona TO agent")
                cols.discard("persona")
                cols.add("agent")
            for col, stmt in _MIGRATIONS.items():
                if col not in cols:
                    await db.execute(stmt)
            await db.commit()
        self._ready = True

    # -- Sync helpers (for use from non-async contexts like CLI) --

    def _ensure_schema_sync(self) -> None:
        if self._ready:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as db:
            db.executescript(_SCHEMA)
            cols = {row[1] for row in db.execute("PRAGMA table_info(jobs)").fetchall()}
            # #115: rename the legacy `persona` column to `agent` in place.
            if "persona" in cols and "agent" not in cols:
                db.execute("ALTER TABLE jobs RENAME COLUMN persona TO agent")
                cols.discard("persona")
                cols.add("agent")
            for col, stmt in _MIGRATIONS.items():
                if col not in cols:
                    db.execute(stmt)
            db.commit()
        self._ready = True

    def list_jobs_sync(self, status: str | None = None, include_done: bool = False) -> list[dict]:
        """Synchronous version of list_jobs for CLI/admin use."""
        self._ensure_schema_sync()
        with sqlite3.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            if status:
                rows = db.execute(
                    "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC",
                    (status,),
                ).fetchall()
            elif include_done:
                rows = db.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM jobs WHERE status IN ('active', 'paused') "
                    "ORDER BY created_at DESC"
                ).fetchall()
        return [dict(r) for r in rows]

    def get_job_sync(self, job_id: str) -> dict | None:
        """Synchronous version of get_job."""
        self._ensure_schema_sync()
        with sqlite3.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def upsert_job_sync(
        self,
        job_id: str,
        *,
        type: str = "agent",
        schedule: str = "cron",
        cron: str | None = None,
        run_at: str | None = None,
        task: str = "",
        channel: str = "telegram",
        status: str = "active",
        created_by: str = "admin",
        description: str = "",
        agent: str = "",
        origin_user_id: str = "",
        origin_chat_id: str = "",
    ) -> dict:
        """Synchronous upsert for CLI/admin use."""
        self._ensure_schema_sync()
        with sqlite3.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            db.execute(
                """INSERT INTO jobs (id, type, schedule, cron, run_at, task, channel,
                                     status, created_by, description, agent,
                                     origin_user_id, origin_chat_id, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                   ON CONFLICT(id) DO UPDATE SET
                       type = excluded.type,
                       schedule = excluded.schedule,
                       cron = excluded.cron,
                       run_at = excluded.run_at,
                       task = excluded.task,
                       channel = excluded.channel,
                       status = excluded.status,
                       description = excluded.description,
                       -- Identity/origin are sticky (#71): a re-upsert that omits
                       -- them (admin "edit", CLI edit/cancel) must not wipe what
                       -- job creation captured. A non-empty new value still wins.
                       agent = COALESCE(NULLIF(excluded.agent, ''), agent),
                       origin_user_id =
                           COALESCE(NULLIF(excluded.origin_user_id, ''), origin_user_id),
                       origin_chat_id =
                           COALESCE(NULLIF(excluded.origin_chat_id, ''), origin_chat_id),
                       updated_at = datetime('now')
                """,
                (
                    job_id,
                    type,
                    schedule,
                    cron,
                    run_at,
                    task,
                    channel,
                    status,
                    created_by,
                    description,
                    agent,
                    origin_user_id,
                    origin_chat_id,
                ),
            )
            db.commit()
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else {}

    def delete_job_sync(self, job_id: str) -> bool:
        """Synchronous delete for CLI use."""
        self._ensure_schema_sync()
        with sqlite3.connect(self.db_path) as db:
            cursor = db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            db.commit()
            return cursor.rowcount > 0

    # -- Async CRUD --

    async def list_jobs(self, status: str | None = None, include_done: bool = False) -> list[dict]:
        """List jobs, optionally filtered by status."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            if status:
                cursor = await db.execute(
                    "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC",
                    (status,),
                )
            elif include_done:
                cursor = await db.execute("SELECT * FROM jobs ORDER BY created_at DESC")
            else:
                cursor = await db.execute(
                    "SELECT * FROM jobs WHERE status IN ('active', 'paused') "
                    "ORDER BY created_at DESC"
                )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_job(self, job_id: str) -> dict | None:
        """Get a single job by ID."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = await cursor.fetchone()
        return dict(row) if row else None

    async def upsert_job(
        self,
        job_id: str,
        *,
        type: str = "agent",
        schedule: str = "cron",
        cron: str | None = None,
        run_at: str | None = None,
        task: str = "",
        channel: str = "telegram",
        status: str = "active",
        created_by: str = "admin",
        description: str = "",
        agent: str = "",
        origin_user_id: str = "",
        origin_chat_id: str = "",
    ) -> dict:
        """Insert or update a job. Returns the job dict."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                """INSERT INTO jobs (id, type, schedule, cron, run_at, task, channel,
                                     status, created_by, description, agent,
                                     origin_user_id, origin_chat_id, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                   ON CONFLICT(id) DO UPDATE SET
                       type = excluded.type,
                       schedule = excluded.schedule,
                       cron = excluded.cron,
                       run_at = excluded.run_at,
                       task = excluded.task,
                       channel = excluded.channel,
                       status = excluded.status,
                       description = excluded.description,
                       -- Identity/origin are sticky (#71): a re-upsert that omits
                       -- them (admin "edit", CLI edit/cancel) must not wipe what
                       -- job creation captured. A non-empty new value still wins.
                       agent = COALESCE(NULLIF(excluded.agent, ''), agent),
                       origin_user_id =
                           COALESCE(NULLIF(excluded.origin_user_id, ''), origin_user_id),
                       origin_chat_id =
                           COALESCE(NULLIF(excluded.origin_chat_id, ''), origin_chat_id),
                       updated_at = datetime('now')
                """,
                (
                    job_id,
                    type,
                    schedule,
                    cron,
                    run_at,
                    task,
                    channel,
                    status,
                    created_by,
                    description,
                    agent,
                    origin_user_id,
                    origin_chat_id,
                ),
            )
            await db.commit()
            cursor = await db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = await cursor.fetchone()
        return dict(row) if row else {}

    async def update_status(self, job_id: str, status: str) -> bool:
        """Update only the status of a job."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "UPDATE jobs SET status = ?, updated_at = datetime('now') WHERE id = ?",
                (status, job_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job by ID."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            await db.commit()
            return cursor.rowcount > 0

    async def rename_agent(self, old: str, new: str) -> None:
        """Repoint jobs from a renamed agent slug (#69): the ``agent`` column
        and the ``telegram:<slug>`` channel a per-agent-bot job delivers to."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("UPDATE jobs SET agent = ? WHERE agent = ?", (new, old))
            await db.execute(
                "UPDATE jobs SET channel = ? WHERE channel = ?",
                (f"telegram:{new}", f"telegram:{old}"),
            )
            await db.commit()

    async def seed_from_config(self, jobs: list[dict]) -> int:
        """Seed jobs from config (on first boot). Only inserts, never overwrites.

        Returns count of newly inserted jobs.
        """
        await self._ensure_schema()
        inserted = 0
        async with aiosqlite.connect(self.db_path) as db:
            for job in jobs:
                cursor = await db.execute("SELECT 1 FROM jobs WHERE id = ?", (job["id"],))
                if await cursor.fetchone():
                    continue
                await db.execute(
                    """INSERT INTO jobs (id, type, schedule, cron, task, channel,
                                         status, created_by, description, agent)
                       VALUES (?, ?, 'cron', ?, ?, ?, 'active', 'config', '', ?)
                    """,
                    (
                        job["id"],
                        job.get("type", "agent"),
                        job["cron"],
                        job.get("task", ""),
                        job.get("channel", "telegram"),
                        job.get("agent", ""),
                    ),
                )
                inserted += 1
            await db.commit()
        if inserted:
            log.info("Seeded %d jobs from config", inserted)
        return inserted

    async def migrate_from_config_store(self, config_store) -> int:
        """One-time migration: move jobs from config store JSON to jobs.db.

        Reads the ``scheduler.jobs`` key from config store, inserts any
        jobs that don't already exist in jobs.db, then deletes the key.
        """
        raw = await config_store.get("scheduler.jobs")
        if not raw:
            return 0
        try:
            jobs = json.loads(raw)
        except json.JSONDecodeError, TypeError:
            return 0
        if not isinstance(jobs, list):
            return 0

        inserted = await self.seed_from_config(jobs)
        if inserted:
            log.info("Migrated %d jobs from config store to jobs.db", inserted)
        # Clean up the old key
        await config_store.delete("scheduler.jobs")
        return inserted
