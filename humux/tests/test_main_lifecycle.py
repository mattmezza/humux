"""Tests for main lifecycle routes on the admin app."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

import core.main as main


class _ConfigStoreStub:
    async def is_setup_complete(self) -> bool:
        return True

    async def get(self, key: str):
        if key == "admin.password_hash":
            return "hash"
        if key == "admin.password_salt":
            return "salt"
        return None

    async def verify_admin_password(self, password: str) -> bool:
        return password == "secret"


def _client(agent_state: main.AgentState) -> TestClient:
    app, auth = main.create_admin_app(agent_state, _ConfigStoreStub())
    main._attach_lifecycle_routes(app, _ConfigStoreStub(), agent_state, auth)
    return TestClient(app)


def test_agent_start_and_stop(monkeypatch) -> None:
    started_agent = SimpleNamespace(channels={"telegram": object()})
    agent_state = main.AgentState()

    async def _start_agent(_store):
        return started_agent

    async def _stop_agent(_agent):
        return None

    monkeypatch.setattr(main, "_start_agent", _start_agent)
    monkeypatch.setattr(main, "_stop_agent", _stop_agent)

    client = _client(agent_state)
    headers = {"Authorization": "Bearer secret"}

    start_resp = client.post("/agent/start", headers=headers)
    assert start_resp.status_code == 200
    assert start_resp.json() == {"status": "started", "channels": ["telegram"]}

    stop_resp = client.post("/agent/stop", headers=headers)
    assert stop_resp.status_code == 200
    assert stop_resp.json() == {"status": "stopped"}


def test_lifecycle_htmx_returns_html(monkeypatch) -> None:
    """HTMX requests should get HTML snippets instead of JSON."""
    started_agent = SimpleNamespace(channels={"telegram": object()})
    agent_state = main.AgentState()

    async def _start_agent(_store):
        return started_agent

    async def _stop_agent(_agent):
        return None

    monkeypatch.setattr(main, "_start_agent", _start_agent)
    monkeypatch.setattr(main, "_stop_agent", _stop_agent)

    client = _client(agent_state)
    headers = {"Authorization": "Bearer secret", "HX-Request": "true"}

    start_resp = client.post("/agent/start", headers=headers)
    assert start_resp.status_code == 200
    assert "text/html" in start_resp.headers["content-type"]
    assert "Started" in start_resp.text
    assert "alert-success" in start_resp.text
    assert start_resp.headers.get("HX-Trigger") == "refresh-status"

    stop_resp = client.post("/agent/stop", headers=headers)
    assert stop_resp.status_code == 200
    assert "Stopped" in stop_resp.text
    assert "alert-success" in stop_resp.text


def test_restart_agent(monkeypatch) -> None:
    """Test restart returns correct JSON and HTMX responses."""
    started_agent = SimpleNamespace(channels={"telegram": object()})
    agent_state = main.AgentState()

    async def _start_agent(_store):
        return started_agent

    async def _stop_agent(_agent):
        return None

    monkeypatch.setattr(main, "_start_agent", _start_agent)
    monkeypatch.setattr(main, "_stop_agent", _stop_agent)

    client = _client(agent_state)
    headers = {"Authorization": "Bearer secret"}

    # JSON response
    resp = client.post("/agent/restart", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "restarted"
    assert resp.json()["channels"] == ["telegram"]

    # HTMX response
    htmx_headers = {**headers, "HX-Request": "true"}
    resp = client.post("/agent/restart", headers=htmx_headers)
    assert resp.status_code == 200
    assert "Restarted" in resp.text
    assert "alert-success" in resp.text


def test_lifecycle_requires_auth() -> None:
    """Lifecycle endpoints require auth when setup is complete."""
    agent_state = main.AgentState()
    client = _client(agent_state)

    for path in ["/agent/start", "/agent/stop", "/agent/restart"]:
        resp = client.post(path)
        assert resp.status_code == 401, f"{path} should require auth"


async def test_migrate_telegram_to_default_agent(tmp_path) -> None:
    """#133: the old global Telegram bot config folds onto the default agent and
    the staged channels.telegram.* keys are cleared (self-clearing, one-shot)."""
    from core.agents import AgentStore, default_agent_from_values
    from core.config_store import ConfigStore

    cs = ConfigStore(db_path=str(tmp_path / "config.db"))
    await cs.set("channels.telegram.bot_token", "123:ABC")
    await cs.set("channels.telegram.allowed_user_ids", "111, 222")
    await cs.set("channels.telegram.group_chat.enabled", "true")
    await cs.set("channels.telegram.group_chat.reply_when_addressed_only", "false")

    store = AgentStore(
        db_path=str(tmp_path / "agents.db"),
        seed_dir=None,
        default_identity=default_agent_from_values(character="Base"),
    )
    await store.ensure_seeded()
    agent = SimpleNamespace(agents=store)

    await main._migrate_telegram_to_default_agent(cs, agent)

    d = await store.get_default()
    assert d is not None
    assert d.bot_token == "123:ABC"
    assert d.allowed_user_ids == [111, 222]
    assert d.group_chat == {
        "enabled": True,
        "reply_when_addressed_only": False,
        "ignore_bots": True,
    }
    # Staged keys are cleared, so a restart won't re-run or relaunch a global bot.
    assert not await cs.get("channels.telegram.bot_token")

    # Idempotent: a second run finds nothing staged and leaves the token in place.
    await main._migrate_telegram_to_default_agent(cs, agent)
    assert (await store.get_default()).bot_token == "123:ABC"


async def test_migrate_telegram_no_default_keeps_staged(tmp_path) -> None:
    """#133: with no flagged default agent, the staged token must NOT be discarded —
    a later boot (once a default exists) still needs to fold it on."""
    from core.agents import AgentStore
    from core.config_store import ConfigStore

    cs = ConfigStore(db_path=str(tmp_path / "config.db"))
    await cs.set("channels.telegram.bot_token", "123:ABC")
    store = AgentStore(db_path=str(tmp_path / "agents.db"), seed_dir=None, default_identity=None)
    agent = SimpleNamespace(agents=store)

    await main._migrate_telegram_to_default_agent(cs, agent)

    assert await store.get_default() is None  # nothing seeded a default
    assert await cs.get("channels.telegram.bot_token") == "123:ABC"  # preserved for later


async def test_migrate_telegram_unresolved_vault_ref_not_baked_in(tmp_path, monkeypatch) -> None:
    """#133: a vaulted token that can't resolve must not be stored as a literal
    ${vault:...}, and its staged source must be kept for a later (unsealed) boot."""
    from types import SimpleNamespace as NS

    from core.agents import AgentStore, default_agent_from_values
    from core.config_store import ConfigStore

    # An infra resolver that never resolves — simulates a sealed/empty vault.
    monkeypatch.setattr(main, "_secret_store", NS(infra_resolve=lambda name: None))

    cs = ConfigStore(db_path=str(tmp_path / "config.db"))
    await cs.set("channels.telegram.bot_token", "${vault:TELEGRAM_BOT_TOKEN}")
    store = AgentStore(
        db_path=str(tmp_path / "agents.db"),
        seed_dir=None,
        default_identity=default_agent_from_values(character="Base"),
    )
    await store.ensure_seeded()
    agent = SimpleNamespace(agents=store)

    await main._migrate_telegram_to_default_agent(cs, agent)

    assert (await store.get_default()).bot_token == ""  # not baked in as a literal ref
    assert await cs.get("channels.telegram.bot_token") == "${vault:TELEGRAM_BOT_TOKEN}"  # kept
