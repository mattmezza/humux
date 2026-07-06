"""Inbound GitHub App webhook → agent turn (issue #210)."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

import api.admin as admin
from api.admin import AgentState, _github_event_task, _github_sig_ok, create_admin_app
from core.config_store import ConfigStore

SECRET = "s3cret"


class _ConfigStoreStub:
    async def is_setup_complete(self) -> bool:
        return True

    async def get(self, key: str):
        return None


class _Agents:
    def __init__(self, agent):
        self._agent = agent

    async def get(self, name: str):
        return self._agent if self._agent and self._agent.name == name else None


def _agent(**gh) -> SimpleNamespace:
    return SimpleNamespace(
        name="dev",
        enabled=True,
        tool_config={"gh": {"enabled": True, "webhook_secret": "WH_SECRET", **gh}},
    )


def _client(agent=None, core_extra=None) -> tuple[TestClient, SimpleNamespace]:
    fields = {
        "agents": _Agents(agent if agent is not None else _agent()),
        "process": AsyncMock(return_value=SimpleNamespace(text="[NO_UPDATES]")),
        "channels": {},
        **(core_extra or {}),
    }
    core = SimpleNamespace(**fields)
    secret_store = SimpleNamespace(
        infra_resolve=lambda n: SECRET if n == "WH_SECRET" else None,
        infra=SimpleNamespace(available=False),
    )
    app, _auth = create_admin_app(
        AgentState(agent=cast("object", core)),
        cast(ConfigStore, _ConfigStoreStub()),
        secret_store=cast("object", secret_store),
    )
    return TestClient(app), core


def _post(
    client: TestClient,
    payload: dict,
    event: str = "issues",
    sign: bool = True,
    slug: str = "dev",
    delivery: str = "",
):
    body = json.dumps(payload).encode()
    headers = {"X-GitHub-Event": event}
    if sign:
        sig = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
        headers["X-Hub-Signature-256"] = f"sha256={sig}"
    if delivery:
        headers["X-GitHub-Delivery"] = delivery
    return client.post(f"/webhooks/github/{slug}", content=body, headers=headers)


def _capture_create_task(monkeypatch) -> list:
    """Swap asyncio.create_task for a recorder (the route also registers a
    done-callback on the returned task, so return a stub that accepts one)."""
    captured: list = []

    class _FakeTask:  # hashable (goes into a set) + accepts a done-callback
        def add_done_callback(self, _cb) -> None:
            pass

    def fake(coro):
        captured.append(coro)
        return _FakeTask()

    monkeypatch.setattr(admin.asyncio, "create_task", fake)
    return captured


ISSUE_PAYLOAD = {
    "action": "opened",
    "repository": {"full_name": "acme/widgets"},
    "issue": {
        "number": 7,
        "title": "It breaks",
        "body": "Steps to reproduce…",
        "user": {"login": "alice"},
        "html_url": "https://github.com/acme/widgets/issues/7",
    },
    "sender": {"login": "alice", "type": "User"},
}


def test_sig_helper() -> None:
    body = b'{"x":1}'
    good = "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    assert _github_sig_ok(SECRET, body, good)
    assert not _github_sig_ok(SECRET, body, good[:-1] + "0")
    assert not _github_sig_ok(SECRET, body, None)
    assert not _github_sig_ok(SECRET, body, "sha1=abc")
    # Non-ASCII header must be a clean False, not a TypeError from
    # compare_digest (Starlette decodes headers as latin-1, attacker-controlled).
    assert not _github_sig_ok(SECRET, body, "sha256=ü-non-ascii")


def test_event_task_builds_thread_chat_id() -> None:
    parsed = _github_event_task("issues", ISSUE_PAYLOAD)
    assert parsed is not None
    task, chat_id, repo = parsed
    assert chat_id == "github:acme/widgets#7"
    assert repo == "acme/widgets"
    assert "acme/widgets" in task and "@alice" in task and "issues/7" in task


def test_event_task_ignores_bots_and_noise() -> None:
    bot = {**ISSUE_PAYLOAD, "sender": {"login": "dev[bot]", "type": "Bot"}}
    assert _github_event_task("issues", bot) is None  # loop guard
    edited = {**ISSUE_PAYLOAD, "action": "labeled"}
    assert _github_event_task("issues", edited) is None
    assert _github_event_task("push", ISSUE_PAYLOAD) is None


def test_webhook_rejects_bad_or_missing_signature() -> None:
    client, core = _client()
    assert _post(client, ISSUE_PAYLOAD, sign=False).status_code == 401
    body = json.dumps(ISSUE_PAYLOAD).encode()
    resp = client.post(
        "/webhooks/github/dev",
        content=body,
        headers={"X-GitHub-Event": "issues", "X-Hub-Signature-256": "sha256=" + "0" * 64},
    )
    assert resp.status_code == 401
    core.process.assert_not_called()


def test_webhook_unknown_agent_or_unconfigured_is_uniform_401() -> None:
    # Same status as a bad signature — the URL space must not leak which
    # agents exist or which have a webhook configured.
    client, _core = _client()
    assert _post(client, ISSUE_PAYLOAD, slug="ghost").status_code == 401
    client2, _core2 = _client(agent=SimpleNamespace(name="dev", enabled=True, tool_config={}))
    assert _post(client2, ISSUE_PAYLOAD).status_code == 401
    # gh tool disabled for the agent → webhook off too, same uniform 401.
    client3, _core3 = _client(agent=_agent(enabled=False))
    assert _post(client3, ISSUE_PAYLOAD).status_code == 401


def test_webhook_caps_body_size() -> None:
    client, core = _client()
    resp = client.post(
        "/webhooks/github/dev",
        content=b"x" * 26_000_001,
        headers={"X-GitHub-Event": "issues"},
    )
    assert resp.status_code == 413
    core.process.assert_not_called()


def test_webhook_failed_turn_frees_delivery_guid(monkeypatch) -> None:
    captured = _capture_create_task(monkeypatch)
    admin._GH_SEEN_DELIVERIES.clear()
    client, core = _client()
    core.process = AsyncMock(side_effect=RuntimeError("boom"))
    assert _post(client, ISSUE_PAYLOAD, delivery="guid-x").status_code == 202
    asyncio.run(captured[0])  # turn crashes; GUID must be forgotten
    assert "guid-x" not in admin._GH_SEEN_DELIVERIES
    resp = _post(client, ISSUE_PAYLOAD, delivery="guid-x")  # Redeliver works
    assert resp.status_code == 202
    for coro in captured[1:]:
        coro.close()


def test_webhook_dedupes_delivery_guid(monkeypatch) -> None:
    captured = _capture_create_task(monkeypatch)
    admin._GH_SEEN_DELIVERIES.clear()
    client, _core = _client()
    assert _post(client, ISSUE_PAYLOAD, delivery="guid-1").status_code == 202
    resp = _post(client, ISSUE_PAYLOAD, delivery="guid-1")
    assert resp.status_code == 200 and resp.text == "duplicate"
    assert len(captured) == 1
    for coro in captured:
        coro.close()


def test_webhook_ping_and_filtered_events() -> None:
    client, core = _client()
    assert _post(client, {"zen": "Keep it simple."}, event="ping").status_code == 200
    assert _post(client, {**ISSUE_PAYLOAD, "action": "labeled"}).status_code == 200
    core.process.assert_not_called()


def test_webhook_repo_allowlist_gates_events(monkeypatch) -> None:
    captured = _capture_create_task(monkeypatch)
    client, core = _client(agent=_agent(repos=["acme/other"]))
    resp = _post(client, ISSUE_PAYLOAD)
    assert resp.status_code == 200 and resp.text == "ignored"
    core.process.assert_not_called()
    # Present-but-empty list allows nothing (github_repo_violation semantics).
    client2, core2 = _client(agent=_agent(repos=[]))
    assert _post(client2, ISSUE_PAYLOAD).text == "ignored"
    core2.process.assert_not_called()
    # Case-insensitive match, like the gh --repo gate.
    client3, _core3 = _client(agent=_agent(repos=["Acme/Widgets"]))
    assert _post(client3, ISSUE_PAYLOAD).status_code == 202
    for coro in captured:
        coro.close()


def test_webhook_author_gate(monkeypatch) -> None:
    captured = _capture_create_task(monkeypatch)
    # Sender not on the list → ignored, no turn.
    client, core = _client(agent=_agent(webhook_users=["mattmezza"]))
    resp = _post(client, ISSUE_PAYLOAD)  # sender is alice
    assert resp.status_code == 200 and resp.text == "ignored"
    core.process.assert_not_called()
    # Listed sender wakes the agent; match is case-insensitive.
    client2, _core2 = _client(agent=_agent(webhook_users=["ALICE"]))
    assert _post(client2, ISSUE_PAYLOAD).status_code == 202
    # Present-but-empty list allows nobody; absent key allows anyone
    # (the anyone case is every other 202 in this file).
    client3, core3 = _client(agent=_agent(webhook_users=[]))
    assert _post(client3, ISSUE_PAYLOAD).text == "ignored"
    core3.process.assert_not_called()
    for coro in captured:
        coro.close()


def test_webhook_delivers_via_agents_own_channel(monkeypatch) -> None:
    captured = _capture_create_task(monkeypatch)
    own_bot = SimpleNamespace(send=AsyncMock(), config=SimpleNamespace(allowed_user_ids=[42]))
    default_bot = SimpleNamespace(send=AsyncMock(), config=SimpleNamespace(allowed_user_ids=[1]))
    client, core = _client(
        core_extra={"channels": {"telegram": default_bot, "telegram:dev": own_bot}}
    )
    core.process = AsyncMock(return_value=SimpleNamespace(text="triaged the issue"))
    assert _post(client, ISSUE_PAYLOAD).status_code == 202
    asyncio.run(captured[0])
    own_bot.send.assert_awaited_once_with(42, "triaged the issue")
    default_bot.send.assert_not_awaited()


def test_webhook_runs_turn_in_background(monkeypatch) -> None:
    # Capture the background coroutine instead of scheduling it, then run it
    # deterministically — the TestClient loop is gone once the request returns.
    captured = _capture_create_task(monkeypatch)
    client, core = _client()
    resp = _post(client, ISSUE_PAYLOAD)
    assert resp.status_code == 202
    assert len(captured) == 1
    asyncio.run(captured[0])
    core.process.assert_awaited_once()
    kwargs = core.process.await_args.kwargs
    assert kwargs["agent_name"] == "dev"
    assert kwargs["chat_id"] == "github:acme/widgets#7"
    assert kwargs["channel"] == "system"
