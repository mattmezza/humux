"""Admin route tests for agents: CRUD, activate, and the tab partial."""

from __future__ import annotations

from typing import Any, cast

from fastapi.testclient import TestClient

from api.admin import AgentState, create_admin_app
from core.config import Config
from core.config_store import ConfigStore

AUTH = {"Authorization": "Bearer secret"}


class _Store:
    """Config-store stub backing get/set with a dict; agents live in tmp."""

    def __init__(self, tmp_path):
        self._data = {
            "agent.agents_db_path": str(tmp_path / "agents.db"),
            "agent.agents_dir": str(tmp_path / "seed"),  # missing = no gallery
            "history.db_path": str(tmp_path / "history.db"),
            "memory.db_path": str(tmp_path / "memory.db"),
        }

    async def is_setup_complete(self) -> bool:
        return True

    async def get(self, key: str):
        if key == "admin.password_hash":
            return "hash"
        if key == "admin.password_salt":
            return "salt"
        return self._data.get(key)

    async def set(self, key: str, value: str) -> None:
        self._data[key] = value

    async def verify_admin_password(self, password: str) -> bool:
        return password == "secret"

    async def get_all_redacted(self) -> dict:
        return {}


class _SchedulerSpy:
    def __init__(self):
        self.reloads = 0

    async def load_jobs(self):
        self.reloads += 1


class _AgentStub:
    def __init__(self):
        self.config = Config()
        self.channels = {}
        self.job_store = None  # rename route reaches _get_job_store(); set per-test
        self.scheduler = _SchedulerSpy()


def _client(tmp_path):
    agent = _AgentStub()
    app, _ = create_admin_app(
        AgentState(agent=cast(Any, agent)), cast(ConfigStore, _Store(tmp_path))
    )
    return TestClient(app), agent


def test_agent_crud_and_activation(tmp_path) -> None:
    client, agent = _client(tmp_path)

    # Create via the guided fields.
    r = client.post(
        "/agents",
        json={
            "name": "coach",
            "role": "Fitness coach",
            "emoji": "🏋️",
            "voice": "en-US-GuyNeural",
            "character": "You are Forge. Direct.",
            "skills": ["memory"],
            "tools": ["run_command"],
            "secrets": ["agent:coach:*"],
        },
        headers=AUTH,
    )
    assert r.status_code == 200
    assert "Fitness coach" in r.text  # partial lists the new card

    # Read back via JSON API.
    got = client.get("/agents/coach", headers=AUTH).json()
    assert got["voice"] == "en-US-GuyNeural"
    assert got["skills"] == ["memory"] and got["tools"] == ["run_command"]

    # Activate → persisted and hot-applied to the running agent.
    r = client.post("/agents/activate", json={"name": "coach"}, headers=AUTH)
    assert r.status_code == 200
    assert "✓ Active" in r.text
    assert agent.config.agent.active_agent == "coach"

    # Deleting the active agent reverts to the default identity.
    r = client.post("/agents/delete", json={"name": "coach"}, headers=AUTH)
    assert r.status_code == 200
    assert agent.config.agent.active_agent == ""
    assert client.get("/agents/coach", headers=AUTH).status_code == 404


def test_agent_account_bindings_roundtrip(tmp_path) -> None:
    client, _ = _client(tmp_path)
    r = client.post(
        "/agents",
        json={
            "name": "coach",
            "email_accounts": [
                {"account": "coach-agent", "access_level": "read", "is_sender_identity": True},
                {"account": "personal", "access_level": "read"},
            ],
            "calendar_accounts": [{"account": "google", "access_level": "read_write"}],
            "contacts_accounts": [{"account": "agent-book", "access_level": "read_write"}],
        },
        headers=AUTH,
    )
    assert r.status_code == 200
    got = client.get("/agents/coach", headers=AUTH).json()
    # Sender identity forces read_write (you cannot send from a read-only account).
    assert got["email_accounts"][0] == {
        "account": "coach-agent",
        "access_level": "read_write",
        "is_sender_identity": True,
    }
    assert got["email_accounts"][1]["access_level"] == "read"
    assert got["calendar_accounts"] == [{"account": "google", "access_level": "read_write"}]
    assert got["contacts_accounts"] == [{"account": "agent-book", "access_level": "read_write"}]


def test_agent_editor_lists_available_accounts(tmp_path) -> None:
    client, _ = _client(tmp_path)
    client.post("/agents", json={"name": "coach"}, headers=AUTH)
    # Seed the account registry the editor reads, via the real save routes.
    client.post(
        "/email/providers",
        json={
            "providers": [
                {"name": "coach-agent", "email": "c@x.io", "imap_host": "i", "smtp_host": "s"}
            ]
        },
        headers=AUTH,
    )
    client.post(
        "/calendar/providers",
        json={"providers": [{"name": "google", "url": "u", "username": "c", "password": "p"}]},
        headers=AUTH,
    )
    page = client.get("/admin/agents/coach", headers=AUTH).text
    assert "Email, calendar &amp; contacts accounts" in page  # the binding card renders
    assert "coach-agent" in page and "google" in page  # available accounts injected


def test_agent_raw_markdown_upsert(tmp_path) -> None:
    client, _ = _client(tmp_path)
    # A legacy `personalia:` key still parses — folded into character (#98).
    raw = "---\nrole: Writer\nskills: [memory]\ntools: []\npersonalia: |\n  Editor.\n---\n"
    r = client.post("/agents", json={"name": "writer", "raw": raw}, headers=AUTH)
    assert r.status_code == 200
    got = client.get("/agents/writer", headers=AUTH).json()
    assert got["role"] == "Writer" and got["skills"] == ["memory"]
    assert "Editor." in got["character"]


def test_agent_bot_fields_persist(tmp_path) -> None:
    # Per-agent Telegram bot (#29): token + ACL survive the round-trip; the ACL
    # is parsed from a comma-separated string into ints; the token is REDACTED on
    # read (a secret, like the global Telegram token) but stored in full.
    client, _ = _client(tmp_path)
    r = client.post(
        "/agents",
        json={
            "name": "coach",
            "role": "Coach",
            "bot_token": "123456:ABC-DEF",
            "allowed_user_ids": "111, 222",
        },
        headers=AUTH,
    )
    assert r.status_code == 200
    got = client.get("/agents/coach", headers=AUTH).json()
    # Redacted on read, but head/tail prove the full value reached storage.
    assert "***" in got["bot_token"]
    assert got["bot_token"].startswith("1234") and got["bot_token"].endswith("-DEF")
    assert got["bot_token"] != "123456:ABC-DEF"
    assert "123456:ABC-DEF" not in got["markdown"]  # not leaked via the raw view either
    assert got["allowed_user_ids"] == [111, 222]


def test_agent_tool_config_persists(tmp_path) -> None:
    # Per-agent tool identity config (#93) survives the round-trip.
    client, _ = _client(tmp_path)
    tc = {"gh": {"enabled": True}, "browser": {"enabled": True, "profile": "hop"}}
    r = client.post(
        "/agents", json={"name": "hopper", "role": "Coder", "tool_config": tc}, headers=AUTH
    )
    assert r.status_code == 200
    got = client.get("/agents/hopper", headers=AUTH).json()
    assert got["tool_config"] == tc


def test_agent_gh_token_without_vault_errors(tmp_path) -> None:
    # No infra vault configured → setting a per-agent token is refused, not
    # silently dropped (so the user knows to configure a master key first).
    client, _ = _client(tmp_path)
    r = client.post("/agents", json={"name": "x", "gh_token": "ghp_abc"}, headers=AUTH)
    assert r.status_code == 400
    assert "master key" in r.text.lower()


def test_agent_gh_token_written_to_infra_vault(tmp_path) -> None:
    # With an infra vault wired, a per-agent token lands in it under the
    # namespaced name and never appears in the agent record (#93).
    import asyncio

    from core.secret_store import SecretStore
    from core.tools import gh_token_secret_name
    from core.vault import InfraVault

    store = SecretStore(db_path=str(tmp_path / "config.db"), infra_vault=InfraVault("machine-key"))
    agent = _AgentStub()
    app, _ = create_admin_app(
        AgentState(agent=cast(Any, agent)),
        cast(ConfigStore, _Store(tmp_path)),
        secret_store=store,
    )
    client = TestClient(app)
    r = client.post(
        "/agents",
        json={"name": "hopper", "tool_config": {"gh": {"enabled": True}}, "gh_token": "ghp_xyz"},
        headers=AUTH,
    )
    assert r.status_code == 200
    stored = asyncio.run(store.get_infra_secret(gh_token_secret_name("hopper")))
    assert stored == "ghp_xyz"
    got = client.get("/agents/hopper", headers=AUTH).json()
    assert "ghp_xyz" not in got["markdown"]  # token never in the agent doc


def test_agent_editor_renders_tool_identities(tmp_path) -> None:
    # The editor page renders the Tool identities card when a tool is enabled.
    client, _ = _client(tmp_path)
    client.post("/agents", json={"name": "hopper"}, headers=AUTH)
    r = client.get("/admin/agents/hopper", headers=AUTH)
    assert r.status_code == 200
    assert "Tool identities" in r.text


def test_agent_editor_offers_vault_token_source(tmp_path) -> None:
    # With gh enabled + an infra vault, the gh card offers reusing a vault secret
    # as the agent's token (#93), seeded with existing infra secret names.
    from core.secret_store import SecretStore
    from core.vault import InfraVault

    store = SecretStore(db_path=str(tmp_path / "config.db"), infra_vault=InfraVault("machine-key"))
    asyncio_run = __import__("asyncio").run
    asyncio_run(store.set_infra_secret("SHARED_PAT", "ghp_shared"))
    backing = _Store(tmp_path)
    backing._data["tools.gh.enabled"] = "true"
    agent = _AgentStub()
    app, _ = create_admin_app(
        AgentState(agent=cast(Any, agent)), cast(ConfigStore, backing), secret_store=store
    )
    client = TestClient(app)
    client.post("/agents", json={"name": "hopper"}, headers=AUTH)
    r = client.get("/admin/agents/hopper", headers=AUTH)
    assert r.status_code == 200
    assert "Token source" in r.text
    assert "SHARED_PAT" in r.text  # existing infra secret offered in the dropdown
    # Explicit PAT/App identity selector + per-agent App fields (#111).
    assert "GitHub identity" in r.text
    assert "Installation ID" in r.text


def test_activate_unknown_agent_404(tmp_path) -> None:
    client, _ = _client(tmp_path)
    assert client.post("/agents/activate", json={"name": "ghost"}, headers=AUTH).status_code == 404


def test_partial_agents_renders(tmp_path) -> None:
    client, _ = _client(tmp_path)
    r = client.get("/partials/agents", headers=AUTH)
    assert r.status_code == 200
    # Merged tab (#115): the default agent renders above the agent list.
    assert "Default agent" in r.text


def test_agent_rename_cascades(tmp_path) -> None:
    """Renaming a slug repoints the agent row, the active selection, per-chat
    bindings, the per-agent bot channel, private memory scope and jobs (#69)."""
    import asyncio

    import aiosqlite

    from core.history import ConversationHistory
    from core.job_store import JobStore
    from core.memory import MemoryStore

    client, agent = _client(tmp_path)
    history_db = str(tmp_path / "history.db")
    memory_db = str(tmp_path / "memory.db")
    agent.job_store = JobStore(db_path=str(tmp_path / "jobs.db"))

    assert (
        client.post("/agents", json={"name": "coach", "role": "Coach"}, headers=AUTH).status_code
        == 200
    )
    assert client.post("/agents/activate", json={"name": "coach"}, headers=AUTH).status_code == 200
    assert agent.config.agent.active_agent == "coach"

    async def seed() -> None:
        h = ConversationHistory(db_path=history_db)
        await h.set_chat_agent("telegram", "u1", "coach", "")  # default-bot binding
        await h.add_turn("telegram:coach", "u1", "user", "hi")  # the bot's own channel
        m = MemoryStore(db_path=memory_db)
        await m._ensure_schema()
        async with aiosqlite.connect(memory_db) as db:
            await db.execute(
                "INSERT INTO long_term (category, subject, content, scope) "
                "VALUES ('pref','s','c','coach')"
            )
            await db.commit()
        await agent.job_store.upsert_job("j1", task="t", channel="telegram:coach", agent="coach")

    asyncio.run(seed())

    r = client.post("/agents/rename", json={"old": "coach", "new": "trainer"}, headers=AUTH)
    assert r.status_code == 200
    assert client.get("/agents/coach", headers=AUTH).status_code == 404
    assert client.get("/agents/trainer", headers=AUTH).status_code == 200
    assert agent.config.agent.active_agent == "trainer"
    assert agent.scheduler.reloads == 1  # live scheduler re-registered the renamed job

    async def check() -> None:
        h = ConversationHistory(db_path=history_db)
        assert await h.get_chat_agent("telegram", "u1", "") == "trainer"
        async with aiosqlite.connect(history_db) as db:
            cur = await db.execute("SELECT channel FROM conversation_turns WHERE user_id='u1'")
            assert [row[0] for row in await cur.fetchall()] == ["telegram:trainer"]
        async with aiosqlite.connect(memory_db) as db:
            cur = await db.execute("SELECT scope FROM long_term")
            assert [row[0] for row in await cur.fetchall()] == ["trainer"]
        job = await agent.job_store.get_job("j1")
        assert job["agent"] == "trainer" and job["channel"] == "telegram:trainer"

    asyncio.run(check())


def test_agent_create_rejects_bad_slug(tmp_path) -> None:
    # The slug guard applies to the create path too, so a malformed slug can't be
    # introduced and then break channel/URL routing (#69).
    client, _ = _client(tmp_path)
    for bad in ("te:am", "my agent", "has/slash", ""):
        r = client.post("/agents", json={"name": bad, "role": "X"}, headers=AUTH)
        assert r.status_code == 400, bad


def test_agent_rename_validation(tmp_path) -> None:
    client, _ = _client(tmp_path)
    client.post("/agents", json={"name": "coach", "role": "C"}, headers=AUTH)
    client.post("/agents", json={"name": "writer", "role": "W"}, headers=AUTH)

    # Collision with an existing slug → 409.
    assert (
        client.post(
            "/agents/rename", json={"old": "coach", "new": "writer"}, headers=AUTH
        ).status_code
        == 409
    )
    # Illegal characters (the ':' would break channel routing) → 400.
    assert (
        client.post(
            "/agents/rename", json={"old": "coach", "new": "te:am"}, headers=AUTH
        ).status_code
        == 400
    )
    # Unknown source slug → 404.
    assert (
        client.post("/agents/rename", json={"old": "ghost", "new": "x"}, headers=AUTH).status_code
        == 404
    )
    # Renaming to the same slug is a harmless no-op.
    assert (
        client.post(
            "/agents/rename", json={"old": "coach", "new": "coach"}, headers=AUTH
        ).status_code
        == 200
    )
