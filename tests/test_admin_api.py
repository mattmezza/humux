"""Admin API smoke tests for core endpoints."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from api.admin import AgentState, create_admin_app
from core.config_store import ConfigStore
from core.job_store import JobStore


class _ConfigStoreStub:
    def __init__(self, setup_complete: bool = True):
        self._setup_complete = setup_complete

    async def is_setup_complete(self) -> bool:
        return self._setup_complete

    async def get(self, key: str):
        if key == "admin.password_hash":
            return "hash"
        if key == "admin.password_salt":
            return "salt"
        return None

    async def get_all_redacted(self) -> dict:
        return {"admin.password_hash": "***", "admin.password_salt": "***"}

    async def verify_admin_password(self, password: str) -> bool:
        return password == "secret"

    async def get_section_redacted(self, section: str) -> dict:
        return {f"{section}.value": "ok"}

    async def set_many(self, values: dict) -> None:
        self._last_set = values

    async def set_admin_password(self, password: str) -> None:
        self._last_set = {"admin.password_hash": "hash", "admin.password_salt": "salt"}

    async def delete(self, key: str) -> bool:
        return key == "test.key"

    async def get_setup_step(self) -> str:
        return "welcome"

    async def set_setup_step(self, step: str) -> None:
        self._step = step


class _AgentStub:
    def __init__(self):
        self.channels = {"telegram": object()}
        self.scheduler = SimpleNamespace(scheduler=SimpleNamespace(get_jobs=lambda: [1, 2]))
        self.permissions = SimpleNamespace(rules={"run_command:jq*": "ALWAYS"})


def _client(setup_complete: bool = True) -> TestClient:
    store = _ConfigStoreStub(setup_complete=setup_complete)
    agent_state = AgentState(agent=None)
    app, _auth = create_admin_app(agent_state, cast(ConfigStore, store))
    return TestClient(app)


def test_health_reports_setup_and_running() -> None:
    store = _ConfigStoreStub(setup_complete=True)
    agent_state = AgentState(agent=cast(Any, _AgentStub()))
    app, _auth = create_admin_app(agent_state, cast(ConfigStore, store))
    client = TestClient(app)
    resp = client.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["setup_complete"] is True
    assert data["agent_running"] is True


def test_requires_auth_when_setup_complete() -> None:
    client = _client(setup_complete=True)
    resp = client.get("/config")
    assert resp.status_code == 401


def test_skips_auth_during_setup() -> None:
    client = _client(setup_complete=False)
    resp = client.get("/config")

    assert resp.status_code == 200
    assert resp.json()["admin.password_hash"] == "***"


def test_config_patch_updates_values() -> None:
    client = _client(setup_complete=True)
    headers = {"Authorization": "Bearer secret"}
    resp = client.patch("/config", json={"values": {"agent.name": "Ada"}}, headers=headers)

    assert resp.status_code == 200
    assert resp.json()["updated"] == ["agent.name"]


def test_agent_status_reports_channels_and_jobs() -> None:
    store = _ConfigStoreStub(setup_complete=True)
    agent_state = AgentState(agent=cast(Any, _AgentStub()))
    app, _auth = create_admin_app(agent_state, cast(ConfigStore, store))
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}
    resp = client.get("/agent/status", headers=headers)

    assert resp.status_code == 200
    assert resp.json() == {
        "running": True,
        "status": "STOPPED",
        "channels": ["telegram"],
        "scheduler_jobs": 2,
    }


class _JobsAgentStub:
    def __init__(self, job_store):
        self.channels = {"telegram": object()}
        self.job_store = job_store
        self.scheduler = SimpleNamespace(
            scheduler=SimpleNamespace(get_jobs=lambda: []),
            sync_job=AsyncMock(),
        )


def _toggle_checked(html: str) -> bool:
    m = re.search(r'<input id="jobs-show-completed"[^>]*>', html)
    assert m, "show-completed toggle not rendered"
    return "checked" in m.group(0)


def _jobs_client(tmp_path):
    store = JobStore(db_path=str(tmp_path / "jobs.db"))
    store.upsert_job_sync("alpha-live", cron="0 7 * * *", task="t", status="active")
    store.upsert_job_sync("omega-done", cron="0 7 * * *", task="t", status="done")
    store.upsert_job_sync("victim-live", cron="0 7 * * *", task="t", status="active")
    cfg = _ConfigStoreStub(setup_complete=True)
    agent_state = AgentState(agent=cast(Any, _JobsAgentStub(store)))
    app, _auth = create_admin_app(agent_state, cast(ConfigStore, cfg))
    return TestClient(app)


def test_partial_jobs_hides_completed_by_default(tmp_path) -> None:
    client = _jobs_client(tmp_path)
    headers = {"Authorization": "Bearer secret"}

    resp = client.get("/partials/jobs", headers=headers)
    assert resp.status_code == 200
    assert "alpha-live" in resp.text
    assert "omega-done" not in resp.text  # completed hidden by default
    assert _toggle_checked(resp.text) is False


def test_partial_jobs_show_completed_reveals_done(tmp_path) -> None:
    client = _jobs_client(tmp_path)
    headers = {"Authorization": "Bearer secret"}

    resp = client.get("/partials/jobs?show_completed=true", headers=headers)
    assert resp.status_code == 200
    assert "omega-done" in resp.text  # completed revealed
    assert _toggle_checked(resp.text) is True


def test_delete_job_preserves_show_completed_toggle(tmp_path) -> None:
    """Deleting a job while 'Show completed' is on keeps the toggle and completed jobs (#68)."""
    client = _jobs_client(tmp_path)
    headers = {"Authorization": "Bearer secret"}

    resp = client.post(
        "/jobs/delete",
        data={"job_id": "victim-live", "show_completed": "true"},
        headers=headers,
    )
    assert resp.status_code == 200
    assert "victim-live" not in resp.text  # deleted
    assert "omega-done" in resp.text  # completed still shown
    assert _toggle_checked(resp.text) is True  # toggle preserved


def test_install_log_buffer_routes_reasoning_to_buffer_not_console() -> None:
    """Model CoT reaches the admin viewer buffer but stays off the console."""
    import logging

    from api.admin import _LOG_BUFFER, _REASONING_LOGGER, install_log_buffer

    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    reasoning_logger = logging.getLogger(_REASONING_LOGGER)
    agent_logger = logging.getLogger("core.agent")
    saved_reasoning_level = reasoning_logger.level

    console_seen: list[str] = []
    console = logging.Handler()
    console.emit = lambda record: console_seen.append(record.name)  # type: ignore[method-assign]

    try:
        root.handlers = [console]
        root.setLevel(logging.INFO)
        agent_logger.setLevel(logging.INFO)
        _LOG_BUFFER.clear()

        install_log_buffer()
        reasoning_logger.info("secret chain of thought")
        agent_logger.info("ordinary log line")

        buffered = "\n".join(e["message"] for e in _LOG_BUFFER)
        assert "secret chain of thought" in buffered  # CoT surfaced in admin UI
        assert "ordinary log line" in buffered
        assert _REASONING_LOGGER not in console_seen  # but kept off the console
        assert "core.agent" in console_seen
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)
        reasoning_logger.setLevel(saved_reasoning_level)
        _LOG_BUFFER.clear()
