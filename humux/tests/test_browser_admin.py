"""Integration test for the admin browser card routes (no network/browser)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.admin import AgentState, create_admin_app
from core.config import Config
from core.config_store import ConfigStore
from core.permissions import PermissionLevel


def _client(tmp_path):
    from core.agent import AgentCore

    store = ConfigStore(db_path=str(tmp_path / "config.db"))  # fresh -> auth open
    state = AgentState(agent=AgentCore(Config()), status="RUNNING")
    app, _ = create_admin_app(state, store)
    return TestClient(app), state.agent, store


def test_tools_partial_includes_browser_card(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    client, _, _ = _client(tmp_path)
    r = client.get("/partials/tools")
    assert r.status_code == 200
    assert "Browser automation" in r.text
    assert "browserTab(" in r.text


def test_per_domain_rule_add_affects_permissions_then_delete(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    client, agent, _ = _client(tmp_path)

    r = client.post("/tools/browser/rules", json={"domain": "github.com", "level": "ALWAYS"})
    assert r.status_code == 200 and r.json()["ok"]
    rules = r.json()["rules"]
    match = [x for x in rules if x["domain"] == "github.com"]
    assert match and match[0]["level"] == "ALWAYS"

    # The rule really changes the permission decision for that domain.
    decision = agent.permissions.check(
        "run_command",
        {"command": "python3 tools/browser.py act --url https://github.com/x --steps []"},
    )
    assert decision == PermissionLevel.ALWAYS

    r = client.post("/tools/browser/rules/delete", json={"pattern": match[0]["pattern"]})
    assert r.status_code == 200 and r.json()["ok"]
    assert all(x["domain"] != "github.com" for x in r.json()["rules"])


def test_add_rule_requires_domain(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    client, _, _ = _client(tmp_path)
    r = client.post("/tools/browser/rules", json={"domain": "", "level": "ALWAYS"})
    assert r.status_code == 200 and r.json()["ok"] is False


def test_wizard_browser_step_saves_enabled(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    client, _, store = _client(tmp_path)

    # Arrive at the browser step (from search), then advance to admin enabling it.
    r = client.post("/setup/step", json={"step": "browser", "values": {}})
    assert r.status_code == 200 and "Browser automation" in r.text
    r = client.post(
        "/setup/step", json={"step": "admin", "values": {"tools.browser.enabled": "true"}}
    )
    assert r.status_code == 200

    # The value persisted to the config store.
    import asyncio

    assert asyncio.run(store.get("tools.browser.enabled")) == "true"
