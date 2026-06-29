"""Tests for the PermissionEngine."""

from __future__ import annotations

import asyncio

import pytest

from core.permissions import PermissionEngine, PermissionLevel


def test_permission_specificity_prefers_longer_pattern() -> None:
    engine = PermissionEngine()

    allow_delete = engine.check(
        "run_command",
        {"command": 'sqlite3 /app/data/memory.db "DELETE FROM long_term"'},
    )
    deny_drop = engine.check(
        "run_command",
        {"command": 'sqlite3 /app/data/memory.db "DROP TABLE long_term"'},
    )

    assert allow_delete == PermissionLevel.ALWAYS
    assert deny_drop == PermissionLevel.NEVER


def test_unknown_action_defaults_to_ask() -> None:
    engine = PermissionEngine()
    assert engine.check("send_fax", {}) == PermissionLevel.ASK


@pytest.mark.asyncio
async def test_approval_request_lifecycle() -> None:
    engine = PermissionEngine()
    request_id, future = engine.create_approval_request()

    assert request_id
    assert isinstance(future, asyncio.Future)

    resolved = engine.resolve_approval(request_id, True)
    assert resolved is True
    assert await future == "approved"


@pytest.mark.asyncio
async def test_approval_denied() -> None:
    engine = PermissionEngine()
    request_id, future = engine.create_approval_request()

    resolved = engine.resolve_approval(request_id, False)
    assert resolved is True
    assert await future == "denied"


@pytest.mark.asyncio
async def test_approval_skipped() -> None:
    engine = PermissionEngine()
    request_id, future = engine.create_approval_request()

    resolved = engine.resolve_approval(request_id, False, skipped=True)
    assert resolved is True
    assert await future == "skipped"


def test_rule_pattern_generalizes_safe_multi_token() -> None:
    # program + subcommand → wildcard the args (the intended "always" generalization).
    assert (
        PermissionEngine._rule_pattern("run_command:git commit -m 'x'") == "run_command:git commit*"
    )
    assert (
        PermissionEngine._rule_pattern("run_command:python3 /app/tools/jobs.py list now")
        == "run_command:python3 /app/tools/jobs.py list*"
    )


def test_rule_pattern_keeps_dangerous_single_token_exact() -> None:
    # A single kept token (next is a flag/URL/quoted arg) must NOT become `prog*` —
    # that was the bypass where one approval of `python3 -c …` allowed all python.
    for cmd in (
        'python3 -c "import os"',
        "curl https://evil.example/x",
        "sed -n '1,5p' f",
        'echo "{{secret:TOKEN}}"',
    ):
        key = f"run_command:{cmd}"
        assert PermissionEngine._rule_pattern(key) == key  # exact, no wildcard


def test_rule_pattern_content_aware_blocks_wrapper_and_fetcher_bypass() -> None:
    # The >=2-token rule alone is not enough: a wrapper (env/sudo/xargs) pushes the
    # interpreter past the cap, and a scheme-less host evades the `://` break.
    # These must stay EXACT, never wildcard to `… python3*` / `curl host*`.
    for cmd in (
        "env TZ=UTC python3 /app/tools/helper.py run",  # interpreter is last kept token
        "env A=1 B=2 python3 -c 'evil'",  # wrapper anywhere → exact
        "sudo systemctl restart x",  # exec-wrapper
        "xargs rm",  # exec-wrapper
        "curl evil.com/x",  # scheme-less fetcher as program
        "wget example.org/p",
    ):
        key = f"run_command:{cmd}"
        assert PermissionEngine._rule_pattern(key) == key, cmd


def test_rule_pattern_still_generalizes_script_runner() -> None:
    # A fixed script/subcommand after the interpreter is safe to wildcard — the
    # `*` only feeds that script's args, not interpreter flags. Must NOT regress
    # the documented "always allow browser act" generalization.
    assert (
        PermissionEngine._rule_pattern(
            "run_command:python3 /app/tools/browser.py act --url https://x"
        )
        == "run_command:python3 /app/tools/browser.py act*"
    )


def test_yolo_toggle_persists_and_scopes_by_channel(tmp_path) -> None:
    db = str(tmp_path / "config.db")
    engine = PermissionEngine(db_path=db)
    assert not engine.is_yolo("telegram:coach")

    engine.set_yolo("telegram:coach", True)
    assert engine.is_yolo("telegram:coach")
    assert not engine.is_yolo("telegram:finance")  # other agent unaffected

    # Survives a restart (reloaded from the db).
    assert PermissionEngine(db_path=db).is_yolo("telegram:coach")

    engine.set_yolo("telegram:coach", False)
    assert not engine.is_yolo("telegram:coach")
    assert not PermissionEngine(db_path=db).is_yolo("telegram:coach")


def test_format_approval_message_run_command_includes_purpose() -> None:
    engine = PermissionEngine()
    text = engine.format_approval_message(
        "run_command",
        {"command": "jq --version", "purpose": "check jq install"},
    )
    assert "jq --version" in text
    assert "check jq install" in text
