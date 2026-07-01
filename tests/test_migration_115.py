"""#115 upgrade-path migrations: legacy persona-named DBs/columns/config keys
must carry their data over to the agent-named schema in place. Tests the paths
the ordinary suite (fresh DBs) never exercises."""

from __future__ import annotations

import sqlite3

from core.agents import AgentStore
from core.config_store import ConfigStore
from core.history import ConversationHistory
from core.job_store import JobStore


async def test_legacy_personae_db_file_and_tables_migrate(tmp_path):
    # Old deployment: data/personae.db with `personae` + `persona_tombstones`.
    legacy = tmp_path / "personae.db"
    con = sqlite3.connect(legacy)
    con.executescript(
        "CREATE TABLE personae (name TEXT PRIMARY KEY, agent_name TEXT DEFAULT '', "
        "role TEXT DEFAULT '', emoji TEXT DEFAULT '', voice TEXT DEFAULT '', "
        "character TEXT DEFAULT '', skills TEXT DEFAULT '', tools TEXT DEFAULT '', "
        "secrets TEXT DEFAULT '');"
        "CREATE TABLE persona_tombstones (name TEXT PRIMARY KEY);"
        "INSERT INTO personae (name, role, character) VALUES ('coach', 'Fitness', 'Be strong');"
        "INSERT INTO persona_tombstones (name) VALUES ('retired');"
    )
    con.commit()
    con.close()

    store = AgentStore(db_path=str(tmp_path / "agents.db"), seed_dir=None)
    got = await store.get("coach")
    assert got is not None and got.role == "Fitness" and got.character == "Be strong"
    # Legacy file adopted (renamed), not left behind.
    assert not legacy.exists()
    # Tombstone carried over: a tombstoned slug is not resurrected (and here seeding
    # is off anyway) — verify the row is queryable under the new table name.
    con = sqlite3.connect(tmp_path / "agents.db")
    names = {r[0] for r in con.execute("SELECT name FROM agent_tombstones").fetchall()}
    con.close()
    assert names == {"retired"}


async def test_fresh_agents_db_is_unaffected(tmp_path):
    # No legacy file → clean create, no crash from the rename attempts.
    store = AgentStore(db_path=str(tmp_path / "agents.db"), seed_dir=None)
    assert await store.list_agents() == []


async def test_legacy_chat_persona_binding_migrates(tmp_path):
    db = tmp_path / "history.db"
    con = sqlite3.connect(db)
    con.executescript(
        "CREATE TABLE chat_persona (channel TEXT NOT NULL, user_id TEXT NOT NULL, "
        "chat_id TEXT NOT NULL DEFAULT '', persona TEXT NOT NULL, "
        "PRIMARY KEY (channel, user_id, chat_id));"
        "INSERT INTO chat_persona (channel, user_id, chat_id, persona) "
        "VALUES ('telegram', 'u1', '', 'coach');"
    )
    con.commit()
    con.close()

    hist = ConversationHistory(db_path=str(db))
    assert await hist.get_chat_agent("telegram", "u1", "") == "coach"


async def test_legacy_jobs_persona_column_migrates(tmp_path):
    db = tmp_path / "jobs.db"
    con = sqlite3.connect(db)
    con.executescript(
        "CREATE TABLE jobs (id TEXT PRIMARY KEY, type TEXT NOT NULL DEFAULT 'agent', "
        "schedule TEXT NOT NULL DEFAULT 'cron', task TEXT NOT NULL DEFAULT '', "
        "channel TEXT NOT NULL DEFAULT 'telegram', status TEXT NOT NULL DEFAULT 'active', "
        "created_by TEXT NOT NULL DEFAULT 'admin', description TEXT NOT NULL DEFAULT '', "
        "persona TEXT NOT NULL DEFAULT '');"
        "INSERT INTO jobs (id, task, persona) VALUES ('j1', 'do', 'coach');"
    )
    con.commit()
    con.close()

    store = JobStore(db_path=str(db))
    job = await store.get_job("j1")
    assert job is not None and job.get("agent") == "coach"  # column renamed, value kept
    # The old column is gone.
    con = sqlite3.connect(db)
    cols = {r[1] for r in con.execute("PRAGMA table_info(jobs)").fetchall()}
    con.close()
    assert "agent" in cols and "persona" not in cols


async def test_legacy_config_keys_migrate(tmp_path):
    store = ConfigStore(db_path=str(tmp_path / "config.db"))
    await store.set("agent.active_persona", "coach")
    await store.set("agent.personae_dir", "custom/")
    await store.set("agent.personae_db_path", "custom/personae.db")
    await store.set("accounts.persona_binding_migrated", "true")
    await store.seed_if_empty(yaml_path=str(tmp_path / "nonexistent.yml"))
    assert await store.get("agent.active_agent") == "coach"
    assert await store.get("agent.active_persona") is None
    # Custom path keys fold too, so a deployment that overrode them still finds its DB.
    assert await store.get("agent.agents_dir") == "custom/"
    assert await store.get("agent.agents_db_path") == "custom/personae.db"
    assert await store.get("agent.personae_dir") is None
    assert await store.get("agent.personae_db_path") is None
    assert await store.get("accounts.agent_binding_migrated") == "true"
    assert await store.get("accounts.persona_binding_migrated") is None
