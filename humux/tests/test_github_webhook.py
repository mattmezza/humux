"""Inbound GitHub App webhook → agent turn (issue #210)."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import api.admin as admin
from api.admin import AgentState, _github_event_task, _github_sig_ok, create_admin_app
from core.config_store import ConfigStore

SECRET = "s3cret"


@pytest.fixture(autouse=True)
def _webhook_state(monkeypatch, tmp_path):
    """Isolate the module-global webhook state (#237): the per-thread rate-cap
    dict would otherwise accumulate across tests (same chat key everywhere),
    and the delivery WAL would write into the repo's real data/ dir."""
    monkeypatch.setattr(admin, "_GH_WAL_DIR", tmp_path / "wal")
    admin._GH_THREAD_TURNS.clear()
    # Tests that stub asyncio.create_task leave fake tasks in the strong-ref
    # set; drain it so a later test can gather the set safely.
    admin._GH_TURN_TASKS.clear()
    yield
    admin._GH_THREAD_TURNS.clear()


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
    # webhook_digest/webhook_ack off: these tests exercise gates/dispatch, not
    # the GitHub API extras (which need a token; covered by their own tests).
    return SimpleNamespace(
        name="dev",
        enabled=True,
        tool_config={
            "gh": {
                "enabled": True,
                "webhook_secret": "WH_SECRET",
                "webhook_digest": False,
                "webhook_ack": False,
                **gh,
            }
        },
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
    # An explicitly listed bot login passes the loop guard (trusting another
    # App is a deliberate choice); unlisted bots stay blocked.
    bot_issue = {**ISSUE_PAYLOAD, "sender": {"login": "ci-bot[bot]", "type": "Bot"}}
    client4, _core4 = _client(agent=_agent(webhook_users=["ci-bot[bot]"]))
    assert _post(client4, bot_issue).status_code == 202
    client5, core5 = _client(agent=_agent(webhook_users=["mattmezza"]))
    assert _post(client5, bot_issue).text == "ignored"
    core5.process.assert_not_called()
    # Scalar string (raw-frontmatter mistake) = one-item list, not a char set.
    client6, _core6 = _client(agent=_agent(webhook_users="Alice"))
    assert _post(client6, ISSUE_PAYLOAD).status_code == 202
    for coro in captured:
        coro.close()


def test_webhook_mention_gate(monkeypatch) -> None:
    captured = _capture_create_task(monkeypatch)
    # Handle configured but not mentioned → ignored.
    client, core = _client(agent=_agent(webhook_mention="my-agent"))
    assert _post(client, ISSUE_PAYLOAD).text == "ignored"
    core.process.assert_not_called()
    # Mention in the body (any case, with or without stored @) wakes it.
    mentioned = {
        **ISSUE_PAYLOAD,
        "issue": {**ISSUE_PAYLOAD["issue"], "body": "hey @My-Agent please look"},
    }
    assert _post(client, mentioned).status_code == 202
    client_at, _c = _client(agent=_agent(webhook_mention="@my-agent"))
    assert _post(client_at, mentioned).status_code == 202
    # Mention in the title counts too.
    titled = {
        **ISSUE_PAYLOAD,
        "issue": {**ISSUE_PAYLOAD["issue"], "title": "@my-agent: it breaks"},
    }
    assert _post(client, titled).status_code == 202
    # A longer handle sharing the prefix does NOT match (word boundary).
    other = {
        **ISSUE_PAYLOAD,
        "issue": {**ISSUE_PAYLOAD["issue"], "body": "cc @my-agent-2"},
    }
    assert _post(client, other).text == "ignored"
    # An email containing the handle is not a mention (left boundary).
    email = {
        **ISSUE_PAYLOAD,
        "issue": {**ISSUE_PAYLOAD["issue"], "body": "contact ops@my-agent for access"},
    }
    assert _post(client, email).text == "ignored"
    # Comment events match the COMMENT body only: a once-mentioning title must
    # not wake the agent for every later comment in the thread.
    comment_no_mention = {
        "action": "created",
        "repository": ISSUE_PAYLOAD["repository"],
        "issue": {**ISSUE_PAYLOAD["issue"], "title": "@my-agent: it breaks"},
        "comment": {
            "body": "thanks, me too",
            "user": {"login": "bob"},
            "html_url": "https://github.com/acme/widgets/issues/7#issuecomment-1",
        },
        "sender": {"login": "bob", "type": "User"},
    }
    assert _post(client, comment_no_mention, event="issue_comment").text == "ignored"
    comment_mention = {
        **comment_no_mention,
        "comment": {**comment_no_mention["comment"], "body": "@my-agent take a look"},
    }
    assert _post(client, comment_mention, event="issue_comment").status_code == 202
    # The App's own bot identity is dropped even when allowlisted — the
    # config-proof self-reply loop guard.
    own_echo = {**mentioned, "sender": {"login": "my-agent[bot]", "type": "Bot"}}
    client2, core2 = _client(
        agent=_agent(webhook_mention="my-agent", webhook_users=["my-agent[bot]"])
    )
    assert _post(client2, own_echo).text == "ignored"
    core2.process.assert_not_called()
    for coro in captured:
        coro.close()


def test_webhook_reply_is_discarded_not_delivered(monkeypatch) -> None:
    # #239: the turn's final reply goes nowhere — no Telegram ping, no
    # owner-DM fallback. send_message (inside the turn) is the only channel.
    captured = _capture_create_task(monkeypatch)
    own_bot = SimpleNamespace(send=AsyncMock(), config=SimpleNamespace(allowed_user_ids=[42]))
    default_bot = SimpleNamespace(send=AsyncMock(), config=SimpleNamespace(allowed_user_ids=[1]))
    client, core = _client(
        agent=_agent(webhook_chat_id="-100777:12"),  # legacy key: ignored
        core_extra={"channels": {"telegram": default_bot, "telegram:dev": own_bot}},
    )
    core.process = AsyncMock(return_value=SimpleNamespace(text="heads up"))
    assert _post(client, ISSUE_PAYLOAD).status_code == 202
    asyncio.run(captured[0])
    own_bot.send.assert_not_awaited()
    default_bot.send.assert_not_awaited()


def test_webhook_gated_delivery_stays_redeliverable(monkeypatch) -> None:
    # A delivery rejected by the author gate must NOT be deduped: after the
    # admin fixes the allowlist, GitHub's Redeliver has to work.
    captured = _capture_create_task(monkeypatch)
    admin._GH_SEEN_DELIVERIES.clear()
    blocked, _c1 = _client(agent=_agent(webhook_users=["mattmezza"]))
    assert _post(blocked, ISSUE_PAYLOAD, delivery="guid-r").text == "ignored"
    allowed, _c2 = _client(agent=_agent(webhook_users=["alice"]))
    assert _post(allowed, ISSUE_PAYLOAD, delivery="guid-r").status_code == 202
    for coro in captured:
        coro.close()


def test_event_task_states_bot_identity() -> None:
    # #239: with a mention handle configured the task tells the agent who it
    # is on GitHub (an installation token can't discover its own slug).
    mentioned = {
        **ISSUE_PAYLOAD,
        "issue": {**ISSUE_PAYLOAD["issue"], "body": "hey @my-agent please look"},
    }
    task, _, _ = _github_event_task("issues", mentioned, mention="my-agent")
    assert "You act on GitHub as @my-agent[bot]." in task
    task2, _, _ = _github_event_task("issues", ISSUE_PAYLOAD)
    assert "You act on GitHub" not in task2
    assert "discarded" in task2  # reply-goes-nowhere instruction replaces [NO_UPDATES]
    assert "NO_UPDATES" not in task2


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
    assert kwargs["chat_id"] == "github:dev:acme/widgets#7"  # agent-scoped (#244)
    assert kwargs["channel"] == "system"
    assert kwargs["steerable"] is True  # busy-thread events steer the running turn (#266)


PR_PAYLOAD = {
    "action": "opened",
    "repository": {"full_name": "acme/widgets"},
    "pull_request": {
        "number": 15,
        "title": "Add login form",
        "body": "Implements the form.\n\nCloses #7",
        "user": {"login": "alice"},
        "html_url": "https://github.com/acme/widgets/pull/15",
    },
    "sender": {"login": "alice", "type": "User"},
}


def test_event_task_threads_pr_into_linked_issue() -> None:
    # "Closes #7" in the PR body → the PR joins issue 7's conversation.
    parsed = _github_event_task("pull_request", PR_PAYLOAD)
    assert parsed is not None
    task, chat_id, _repo = parsed
    assert chat_id == "github:acme/widgets#7"
    assert "#15" in task  # the task still names the PR itself
    # All GitHub closing keywords count, case-insensitively.
    for kw in ("closes", "Closed", "fix", "Fixes", "fixed", "resolves", "Resolve"):
        p = {**PR_PAYLOAD, "pull_request": {**PR_PAYLOAD["pull_request"], "body": f"{kw} #9"}}
        assert _github_event_task("pull_request", p)[1] == "github:acme/widgets#9"
    # No closing keyword → threads under its own number.
    plain = {**PR_PAYLOAD, "pull_request": {**PR_PAYLOAD["pull_request"], "body": "WIP"}}
    assert _github_event_task("pull_request", plain)[1] == "github:acme/widgets#15"
    # A stray "fixes #N" in a plain ISSUE body must NOT re-thread the issue.
    issue = {**ISSUE_PAYLOAD, "issue": {**ISSUE_PAYLOAD["issue"], "body": "fixes #3"}}
    assert _github_event_task("issues", issue)[1] == "github:acme/widgets#7"
    # A comment on a PR (GitHub sends it as an issue whose body is the PR's)
    # threads into the linked issue too.
    pr_comment = {
        "action": "created",
        "repository": {"full_name": "acme/widgets"},
        "issue": {
            "number": 15,
            "title": "Add login form",
            "body": "Implements the form.\n\nCloses #7",
            "user": {"login": "alice"},
            "html_url": "https://github.com/acme/widgets/pull/15",
            "pull_request": {"url": "https://api.github.com/repos/acme/widgets/pulls/15"},
        },
        "comment": {
            "body": "lgtm-ish",
            "user": {"login": "bob"},
            "html_url": "https://github.com/acme/widgets/pull/15#issuecomment-1",
        },
        "sender": {"login": "bob", "type": "User"},
    }
    assert _github_event_task("issue_comment", pr_comment)[1] == "github:acme/widgets#7"


def test_event_task_new_events_are_opt_in() -> None:
    # Defaults = pre-#237 behavior: closed/synchronize/review/CI stay silent.
    closed = {**ISSUE_PAYLOAD, "action": "closed"}
    assert _github_event_task("issues", closed) is None
    sync = {**PR_PAYLOAD, "action": "synchronize"}
    assert _github_event_task("pull_request", sync) is None
    # Opted in via webhook_events → they wake the agent.
    events = admin._gh_event_filter({"issues": ["opened", "closed"]})
    parsed = _github_event_task("issues", closed, events=events)
    assert parsed is not None and "(closed)" in parsed[0]
    # …and the filter is exact: opting into closed doesn't admit reopened.
    reopened = {**ISSUE_PAYLOAD, "action": "reopened"}
    events2 = admin._gh_event_filter({"issues": ["closed"]})
    assert _github_event_task("issues", reopened, events=events2) is None
    # Unknown events/actions in the config are dropped, not honored.
    junk = admin._gh_event_filter({"issues": ["deleted"], "star": ["created"]})
    assert junk == {}
    # Malformed config (not a dict) falls back to the defaults.
    assert admin._gh_event_filter("issues") == admin._GH_DEFAULT_EVENTS


def test_event_task_pull_request_review_submitted() -> None:
    payload = {
        "action": "submitted",
        "repository": {"full_name": "acme/widgets"},
        "pull_request": {
            "number": 15,
            "title": "Add login form",
            "body": "Closes #7",
            "user": {"login": "alice"},
            "html_url": "https://github.com/acme/widgets/pull/15",
        },
        "review": {
            "state": "changes_requested",
            "body": "Please add a test for the empty-password case.",
            "user": {"login": "rev"},
            "html_url": "https://github.com/acme/widgets/pull/15#pullrequestreview-1",
        },
        "sender": {"login": "rev", "type": "User"},
    }
    events = admin._gh_event_filter({"pull_request_review": ["submitted"]})
    parsed = _github_event_task("pull_request_review", payload, events=events)
    assert parsed is not None
    task, chat_id, _ = parsed
    assert chat_id == "github:acme/widgets#7"  # threads with the PR's issue
    assert "changes_requested" in task and "empty-password" in task and "@rev" in task


def test_event_task_workflow_run_completed() -> None:
    payload = {
        "action": "completed",
        "repository": {"full_name": "acme/widgets"},
        "workflow_run": {
            "name": "ci",
            "conclusion": "failure",
            "head_branch": "issue-7-login",
            "html_url": "https://github.com/acme/widgets/actions/runs/1",
            "pull_requests": [{"number": 15}],
        },
        "sender": {"login": "alice", "type": "User"},
    }
    events = admin._gh_event_filter({"workflow_run": ["completed"]})
    parsed = _github_event_task("workflow_run", payload, events=events)
    assert parsed is not None
    task, chat_id, _ = parsed
    assert chat_id == "github:acme/widgets#15"
    assert "failure" in task and "actions/runs/1" in task
    # A run with no associated PR (push to main, cron) has no thread → ignored.
    no_pr = {**payload, "workflow_run": {**payload["workflow_run"], "pull_requests": []}}
    assert _github_event_task("workflow_run", no_pr, events=events) is None


def test_event_task_auto_events_bypass_mention_not_self_drop() -> None:
    # Mention configured, not mentioned → normally ignored…
    assert _github_event_task("pull_request", PR_PAYLOAD, mention="my-agent") is None
    # …but an auto event wakes without the mention (role trigger), by
    # event.action or bare event key.
    for auto in ({"pull_request.opened"}, {"pull_request"}):
        parsed = _github_event_task(
            "pull_request", PR_PAYLOAD, mention="my-agent", auto=frozenset(auto)
        )
        assert parsed is not None
    # The self-loop guard survives auto: the agent's own echo never wakes it.
    own = {**PR_PAYLOAD, "sender": {"login": "my-agent[bot]", "type": "Bot"}}
    assert (
        _github_event_task(
            "pull_request", own, mention="my-agent", auto=frozenset({"pull_request"})
        )
        is None
    )
    # Auto set parsing: scalar and junk are tolerated.
    assert admin._gh_auto_set("issues.closed") == frozenset({"issues.closed"})
    assert admin._gh_auto_set(None) == frozenset()
    assert admin._gh_auto_set(42) == frozenset()


def test_webhook_rate_cap(monkeypatch, tmp_path) -> None:
    captured = _capture_create_task(monkeypatch)
    monkeypatch.setattr(admin, "_GH_WAL_DIR", tmp_path / "wal")
    admin._GH_THREAD_TURNS.clear()
    client, core = _client(agent=_agent(webhook_rate_limit=2))
    assert _post(client, ISSUE_PAYLOAD).status_code == 202
    assert _post(client, ISSUE_PAYLOAD).status_code == 202
    resp = _post(client, ISSUE_PAYLOAD)
    assert resp.status_code == 200 and resp.text == "rate limited"
    # Another thread is unaffected.
    other = {**ISSUE_PAYLOAD, "issue": {**ISSUE_PAYLOAD["issue"], "number": 8}}
    assert _post(client, other).status_code == 202
    # Old timestamps age out of the window.
    key = "github:dev:acme/widgets#7"
    admin._GH_THREAD_TURNS[key] = [t - 4000 for t in admin._GH_THREAD_TURNS[key]]
    assert _post(client, ISSUE_PAYLOAD).status_code == 202
    admin._GH_THREAD_TURNS.clear()
    for coro in captured:
        coro.close()


def test_webhook_wal_written_and_cleared(monkeypatch, tmp_path) -> None:
    captured = _capture_create_task(monkeypatch)
    wal = tmp_path / "wal"
    monkeypatch.setattr(admin, "_GH_WAL_DIR", wal)
    admin._GH_SEEN_DELIVERIES.clear()
    admin._GH_THREAD_TURNS.clear()
    client, core = _client()
    assert _post(client, ISSUE_PAYLOAD, delivery="guid-wal").status_code == 202
    files = list(wal.glob("*.json"))
    assert len(files) == 1
    record = json.loads(files[0].read_text())
    assert record["agent_name"] == "dev" and record["chat_id"] == "github:dev:acme/widgets#7"
    asyncio.run(captured[0])  # turn completes → WAL entry gone
    assert list(wal.glob("*.json")) == []
    # A failed turn clears its WAL entry too (replay must be a human decision).
    core.process = AsyncMock(side_effect=RuntimeError("boom"))
    assert _post(client, ISSUE_PAYLOAD, delivery="guid-wal2").status_code == 202
    asyncio.run(captured[1])
    assert list(wal.glob("*.json")) == []
    assert "guid-wal2" not in admin._GH_SEEN_DELIVERIES  # redeliverable
    admin._GH_THREAD_TURNS.clear()


def test_webhook_wal_replay(monkeypatch, tmp_path) -> None:
    wal = tmp_path / "wal"
    wal.mkdir(parents=True)
    monkeypatch.setattr(admin, "_GH_WAL_DIR", wal)
    admin._GH_SEEN_DELIVERIES.clear()
    record = {
        "agent_name": "dev",
        "task": "do the thing",
        "chat_id": "github:acme/widgets#7",
        "repo": "acme/widgets",
        "delivery": "guid-replay",
    }
    (wal / "guid-replay.json").write_text(json.dumps(record))
    # An entry for a gone/unconfigured agent is dropped without a turn.
    (wal / "stale.json").write_text(json.dumps({**record, "agent_name": "ghost"}))
    agent = _agent()
    core = SimpleNamespace(
        agents=_Agents(agent),
        process=AsyncMock(return_value=SimpleNamespace(text="[NO_UPDATES]")),
        channels={},
    )
    secret_store = SimpleNamespace(infra_resolve=lambda n: None)

    async def run() -> int:
        n = await admin.replay_webhook_deliveries(SimpleNamespace(agent=core), secret_store)
        await asyncio.gather(*admin._GH_TURN_TASKS, return_exceptions=True)
        return n

    assert asyncio.run(run()) == 1
    core.process.assert_awaited_once()
    kwargs = core.process.await_args.kwargs
    assert kwargs["chat_id"] == "github:acme/widgets#7" and kwargs["agent_name"] == "dev"
    assert list(wal.glob("*.json")) == []  # both the replayed and the stale entry
    assert "guid-replay" in admin._GH_SEEN_DELIVERIES  # stays deduped post-replay


def test_webhook_digest_prepended(monkeypatch, tmp_path) -> None:
    captured = _capture_create_task(monkeypatch)
    monkeypatch.setattr(admin, "_GH_WAL_DIR", tmp_path / "wal")
    admin._GH_THREAD_TURNS.clear()
    from core import github_digest, tools

    monkeypatch.setattr(tools, "_agent_gh_token", lambda *a, **k: "tok-123")

    seen: dict = {}

    async def fake_digest(repo, number, token):
        seen.update(repo=repo, number=number, token=token)
        return "[Thread state] acme/widgets#7 (open): It breaks"

    monkeypatch.setattr(github_digest, "thread_digest", fake_digest)
    client, core = _client(agent=_agent(webhook_digest=True))
    core.config = SimpleNamespace()  # _agent_gh_token is stubbed; config is opaque
    assert _post(client, ISSUE_PAYLOAD).status_code == 202
    asyncio.run(captured[0])
    message = core.process.await_args.kwargs["message"]
    assert message.endswith("[Thread state] acme/widgets#7 (open): It breaks")
    assert "[GitHub] New issue" in message
    assert seen == {"repo": "acme/widgets", "number": 7, "token": "tok-123"}

    # Digest failure must not sink the turn: the bare event still dispatches.
    async def broken(repo, number, token):
        raise RuntimeError("api down")

    monkeypatch.setattr(github_digest, "thread_digest", broken)
    core.process.reset_mock()
    assert _post(client, ISSUE_PAYLOAD).status_code == 202
    asyncio.run(captured[1])
    assert core.process.await_args.kwargs["message"].startswith("[GitHub] New issue")
    admin._GH_THREAD_TURNS.clear()


def test_webhook_mention_bot_suffix_normalized(monkeypatch) -> None:
    # A webhook_mention stored as "cliobit[bot]" (easy config mistake) must
    # behave exactly like the bare slug (#244): mentions still wake the agent,
    # and the SELF-LOOP GUARD still drops the agent's own echoes — unnormalized
    # it compared the sender to "cliobit[bot][bot]" and never matched.
    captured = _capture_create_task(monkeypatch)
    mentioned = {
        **ISSUE_PAYLOAD,
        "issue": {**ISSUE_PAYLOAD["issue"], "body": "hey @cliobit[bot] please look"},
    }
    client, core = _client(agent=_agent(webhook_mention="cliobit[bot]"))
    assert _post(client, mentioned).status_code == 202
    asyncio.run(captured[0])
    message = core.process.await_args.kwargs["message"]
    assert "You act on GitHub as @cliobit[bot]." in message  # not [bot][bot]
    own_echo = {**mentioned, "sender": {"login": "cliobit[bot]", "type": "Bot"}}
    client2, core2 = _client(
        agent=_agent(webhook_mention="cliobit[bot]", webhook_users=["cliobit[bot]"])
    )
    assert _post(client2, own_echo).text == "ignored"
    core2.process.assert_not_called()


def test_gh_ack_target_mapping() -> None:
    assert (
        admin._gh_ack_target("issues", ISSUE_PAYLOAD, "acme/widgets")
        == "repos/acme/widgets/issues/7/reactions"
    )
    assert (
        admin._gh_ack_target("pull_request", PR_PAYLOAD, "acme/widgets")
        == "repos/acme/widgets/issues/15/reactions"
    )
    # Reviews have no reactions API → fall back to the PR itself.
    review = {**PR_PAYLOAD, "review": {"id": 9, "body": "x"}}
    assert (
        admin._gh_ack_target("pull_request_review", review, "acme/widgets")
        == "repos/acme/widgets/issues/15/reactions"
    )
    comment = {"comment": {"id": 33}, "issue": {"number": 7}}
    assert (
        admin._gh_ack_target("issue_comment", comment, "acme/widgets")
        == "repos/acme/widgets/issues/comments/33/reactions"
    )
    assert (
        admin._gh_ack_target("pull_request_review_comment", comment, "acme/widgets")
        == "repos/acme/widgets/pulls/comments/33/reactions"
    )
    # Nothing sensible to react to on a CI run.
    assert admin._gh_ack_target("workflow_run", {"workflow_run": {}}, "acme/widgets") == ""


def test_webhook_ack_reaction_posted(monkeypatch) -> None:
    captured = _capture_create_task(monkeypatch)
    from core import github_digest, tools

    monkeypatch.setattr(tools, "_agent_gh_token", lambda *a, **k: "tok-123")
    acked: list = []

    async def fake_ack(target, token, content="eyes"):
        acked.append({"target": target, "token": token, "content": content})
        return True

    monkeypatch.setattr(github_digest, "ack_reaction", fake_ack)
    client, core = _client(agent=_agent(webhook_ack=True))
    core.config = SimpleNamespace()  # token mint is stubbed; config is opaque
    assert _post(client, ISSUE_PAYLOAD).status_code == 202
    asyncio.run(captured[0])
    # Lifecycle (#241/#268): 👀 when picked up, 🚀 when the turn completes —
    # both on the same target, both from the executor (never the model).
    assert [a["content"] for a in acked] == ["eyes", "rocket"]
    assert all(a["target"] == "repos/acme/widgets/issues/7/reactions" for a in acked)
    assert all(a["token"] == "tok-123" for a in acked)
    core.process.assert_awaited_once()  # acks ride the turn, never replace it
    # A crashed turn closes with 😕 instead of 🚀 (#268).
    acked.clear()
    core.process = AsyncMock(side_effect=RuntimeError("boom"))
    assert _post(client, ISSUE_PAYLOAD).status_code == 202
    asyncio.run(captured[1])
    assert [a["content"] for a in acked] == ["eyes", "confused"]
    # Disabled → no reactions even though a target exists.
    acked.clear()
    client2, core2 = _client(agent=_agent(webhook_ack=False))
    assert _post(client2, ISSUE_PAYLOAD).status_code == 202
    asyncio.run(captured[2])
    assert acked == []


def test_resolve_chat_titles() -> None:
    # Live lookup + 10-min cache + bot-down fallback, all in one pass.
    admin._chat_title_cache.clear()

    class _Chat:
        title = "Family group"
        username = None
        first_name = None
        last_name = None

    calls = {"n": 0}

    async def get_chat(cid):
        calls["n"] += 1
        return _Chat()

    bot = SimpleNamespace(get_chat=get_chat)
    core = SimpleNamespace(channels={"telegram:dev": SimpleNamespace(app=SimpleNamespace(bot=bot))})
    chats = [{"chat_id": "-100777:12", "kind": "group", "channel": "telegram:dev"}]
    asyncio.run(admin._resolve_chat_titles(core, chats))
    assert chats[0]["title"] == "Family group" and calls["n"] == 1
    # Second resolve hits the TTL cache — no second API call.
    chats2 = [{"chat_id": "-100777:12", "kind": "group", "channel": "telegram:dev"}]
    asyncio.run(admin._resolve_chat_titles(core, chats2))
    assert chats2[0]["title"] == "Family group" and calls["n"] == 1
    # Unknown channel (bot down) → bare id, no crash, no title key.
    dead = [{"chat_id": "42", "kind": "dm", "channel": "telegram:gone"}]
    asyncio.run(admin._resolve_chat_titles(core, dead))
    assert "title" not in dead[0]
