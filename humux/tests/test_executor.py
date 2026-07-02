"""Tests for the ToolExecutor."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from core.executor import ToolExecutor


@pytest.mark.asyncio
async def test_run_command_rejects_non_whitelisted_prefix(monkeypatch) -> None:
    executor = ToolExecutor()
    mock_exec = AsyncMock(return_value={"stdout": "", "stderr": "", "exit_code": 0})
    monkeypatch.setattr(executor, "_exec", mock_exec)

    result = await executor.run_command("rm -rf /tmp")

    assert "error" in result
    mock_exec.assert_not_called()


@pytest.mark.asyncio
async def test_run_command_allows_whitelisted_prefix(monkeypatch) -> None:
    executor = ToolExecutor()
    mock_exec = AsyncMock(return_value={"stdout": "ok", "stderr": "", "exit_code": 0})
    monkeypatch.setattr(executor, "_exec", mock_exec)

    result = await executor.run_command("jq --version")

    assert result["stdout"] == "ok"
    mock_exec.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_override_strips_leaked_managed_env(monkeypatch) -> None:
    # A GH_TOKEN present in the process env (e.g. loaded from .env) must NOT leak
    # to an agent-scoped run whose override doesn't set it — otherwise an agent
    # that switched gh off could still act as the owner (#93 security boundary).
    # GITHUB_TOKEN is gh's fallback var — it must be stripped too, not just GH_TOKEN.
    monkeypatch.setenv("GH_TOKEN", "owner-leaked")
    monkeypatch.setenv("GITHUB_TOKEN", "owner-leaked-2")
    executor = ToolExecutor(tool_env={"GH_TOKEN": "owner-leaked"})
    cmd = (
        'python3 -c "import os,sys; '
        "sys.stdout.write(os.environ.get('GH_TOKEN','NONE')+'|'+os.environ.get('GITHUB_TOKEN','NONE'))\""
    )
    resolved = executor._resolve_command(cmd)

    # agent-scoped override with no token → both gh vars stripped from the env.
    res = await executor._exec(resolved, 30, tool_env={})
    assert res["stdout"].strip() == "NONE|NONE", res

    # default path (no override) still sees the configured token.
    res2 = await executor._exec(resolved, 30)
    assert res2["stdout"].strip().startswith("owner-leaked|"), res2


def test_parse_json_output_handles_invalid_json() -> None:
    executor = ToolExecutor()
    output = executor.parse_json_output("not json")

    assert output == {"raw": "not json"}


@pytest.mark.asyncio
async def test_himalaya_command_sets_env(monkeypatch) -> None:
    executor = ToolExecutor()
    created = {}

    async def _fake_subprocess_shell(command, stdout, stderr, env, cwd=None):
        created["env"] = env
        created["cwd"] = cwd

        class _Proc:
            returncode = 0

            async def communicate(self):
                return b"", b""

        return _Proc()

    monkeypatch.setattr("core.executor.asyncio.create_subprocess_shell", _fake_subprocess_shell)
    monkeypatch.setattr("core.executor.himalaya_env", lambda: {"HIMALAYA_CONFIG": "/tmp/x"})

    await executor.run_command("himalaya envelope list")

    assert created["env"]["HIMALAYA_CONFIG"] == "/tmp/x"
