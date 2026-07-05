"""Tests for the coding harness (#76, #178): confined file tools + gating."""

from __future__ import annotations

import pytest

from core import coding
from core.agent import apply_feature_gates
from core.config import Config
from core.permissions import PermissionEngine, PermissionLevel

# ---------------------------------------------------------------------------
# Workspace confinement — the trust boundary
# ---------------------------------------------------------------------------


def test_resolve_relative_stays_in_workspace(tmp_path):
    target = coding.resolve_in_workspace(str(tmp_path), "sub/file.txt")
    assert target == tmp_path.resolve() / "sub" / "file.txt"


def test_resolve_rejects_symlinked_dir_traversal(tmp_path):
    outside = tmp_path.parent / "outside_dir"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "d").symlink_to(outside, target_is_directory=True)
    with pytest.raises(coding.WorkspaceError):
        coding.resolve_in_workspace(str(ws), "d/secret.txt")


def test_resolve_rejects_parent_escape(tmp_path):
    with pytest.raises(coding.WorkspaceError):
        coding.resolve_in_workspace(str(tmp_path), "../../etc/passwd")


def test_resolve_rejects_absolute_outside(tmp_path):
    with pytest.raises(coding.WorkspaceError):
        coding.resolve_in_workspace(str(tmp_path), "/etc/passwd")


def test_resolve_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("secret")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "link").symlink_to(outside)
    with pytest.raises(coding.WorkspaceError):
        coding.resolve_in_workspace(str(ws), "link")


def test_resolve_requires_configured_dir():
    with pytest.raises(coding.WorkspaceError):
        coding.resolve_in_workspace("", "anything")


def test_resolve_allows_root_itself(tmp_path):
    assert coding.resolve_in_workspace(str(tmp_path), ".") == tmp_path.resolve()


# ---------------------------------------------------------------------------
# read — hashline output (#178)
# ---------------------------------------------------------------------------


def test_read_file_paginates_with_hashline_anchors(tmp_path):
    lines = [f"line{i}" for i in range(10)]
    (tmp_path / "f.txt").write_text("\n".join(lines))
    out = coding.read_file(str(tmp_path), "f.txt", offset=2, limit=3)
    assert out["total_lines"] == 10
    assert out["lines_returned"] == 3
    hashes = coding.hash_lines(lines)
    expected = "\n".join(f"{i + 1}#{hashes[i]}:{lines[i]}" for i in (2, 3, 4))
    assert out["content"] == expected


def test_hashline_is_context_sensitive():
    # Identical lines get different hashes when their neighbours differ, so an
    # anchor can never silently address the wrong copy.
    h = coding.hash_lines(["x", "same", "y", "same", "z"])
    assert h[1] != h[3]
    assert all(len(x) == 2 for x in h)


def test_read_file_missing(tmp_path):
    assert "error" in coding.read_file(str(tmp_path), "nope.txt")


def test_read_file_rejects_oversize(tmp_path, monkeypatch):
    monkeypatch.setattr(coding, "MAX_READ_BYTES", 10)
    (tmp_path / "big.txt").write_text("x" * 50)
    out = coding.read_file(str(tmp_path), "big.txt")
    assert "error" in out and "too large" in out["error"]


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


def test_write_file_creates_dirs_and_overwrites(tmp_path):
    out = coding.write_file(str(tmp_path), "a/b/c.txt", "hello")
    assert out["ok"] is True
    assert (tmp_path / "a/b/c.txt").read_text() == "hello"
    coding.write_file(str(tmp_path), "a/b/c.txt", "world")
    assert (tmp_path / "a/b/c.txt").read_text() == "world"


def test_write_file_rejects_escape(tmp_path):
    with pytest.raises(coding.WorkspaceError):
        coding.write_file(str(tmp_path), "../escape.txt", "x")


# ---------------------------------------------------------------------------
# edit — multi-edit: text edits + hashline anchors (#178)
# ---------------------------------------------------------------------------


def _anchor(path, n):
    """The 'N#HH' anchor for 1-based line ``n`` of ``path``."""
    lines = path.read_text().splitlines()
    return f"{n}#{coding.hash_lines(lines)[n - 1]}"


def test_edit_unique_text_match(tmp_path):
    (tmp_path / "f.py").write_text("a = 1\nb = 2\n")
    out = coding.edit_file(str(tmp_path), "f.py", [{"oldText": "b = 2", "newText": "b = 3"}])
    assert out["replacements"] == 1
    assert (tmp_path / "f.py").read_text() == "a = 1\nb = 3\n"


def test_edit_multiple_text_edits_one_call(tmp_path):
    (tmp_path / "f.py").write_text("foo()\nbar()\n")
    out = coding.edit_file(
        str(tmp_path),
        "f.py",
        [
            {"oldText": "foo()", "newText": "FOO()"},
            {"oldText": "bar()", "newText": "BAR()"},
        ],
    )
    assert out["replacements"] == 2
    assert (tmp_path / "f.py").read_text() == "FOO()\nBAR()\n"


def test_edit_ambiguous_match_refused(tmp_path):
    (tmp_path / "f.py").write_text("x\nx\n")
    out = coding.edit_file(str(tmp_path), "f.py", [{"oldText": "x", "newText": "y"}])
    assert "error" in out
    assert (tmp_path / "f.py").read_text() == "x\nx\n"  # untouched


def test_edit_all_replaces_every_occurrence(tmp_path):
    (tmp_path / "f.py").write_text("x\nx\n")
    out = coding.edit_file(str(tmp_path), "f.py", [{"oldText": "x", "newText": "y", "all": True}])
    assert out["replacements"] == 2
    assert (tmp_path / "f.py").read_text() == "y\ny\n"


def test_edit_not_found(tmp_path):
    (tmp_path / "f.py").write_text("abc")
    assert "error" in coding.edit_file(str(tmp_path), "f.py", [{"oldText": "zzz", "newText": "q"}])


def test_edit_is_atomic_on_partial_failure(tmp_path):
    # First edit would apply, second fails → nothing is written.
    (tmp_path / "f.py").write_text("a\nb\n")
    out = coding.edit_file(
        str(tmp_path),
        "f.py",
        [{"oldText": "a", "newText": "A"}, {"oldText": "zzz", "newText": "q"}],
    )
    assert "error" in out
    assert (tmp_path / "f.py").read_text() == "a\nb\n"


def test_edit_hashline_replace_line(tmp_path):
    f = tmp_path / "f.py"
    f.write_text("a = 1\nb = 2\nc = 3\n")
    out = coding.edit_file(str(tmp_path), "f.py", [{"pos": _anchor(f, 2), "lines": ["b = 99"]}])
    assert out["ok"] is True
    assert f.read_text() == "a = 1\nb = 99\nc = 3\n"


def test_edit_hashline_replace_range_and_delete(tmp_path):
    f = tmp_path / "f.py"
    f.write_text("1\n2\n3\n4\n5\n")
    out = coding.edit_file(
        str(tmp_path), "f.py", [{"pos": _anchor(f, 2), "end": _anchor(f, 4), "lines": []}]
    )
    assert out["ok"] is True
    assert f.read_text() == "1\n5\n"


def test_edit_hashline_multiple_ranges_apply_bottom_up(tmp_path):
    f = tmp_path / "f.py"
    f.write_text("1\n2\n3\n4\n5\n")
    out = coding.edit_file(
        str(tmp_path),
        "f.py",
        [
            {"pos": _anchor(f, 1), "lines": ["one"]},
            {"pos": _anchor(f, 4), "lines": ["four"]},
        ],
    )
    assert out["replacements"] == 2
    assert f.read_text() == "one\n2\n3\nfour\n5\n"


def test_edit_hashline_stale_anchor_rejected(tmp_path):
    f = tmp_path / "f.py"
    f.write_text("a = 1\nb = 2\n")
    anchor = _anchor(f, 2)
    f.write_text("a = 1\nCHANGED\n")  # file changed since the read
    out = coding.edit_file(str(tmp_path), "f.py", [{"pos": anchor, "lines": ["b = 3"]}])
    assert "error" in out and "Stale anchor" in out["error"]
    assert f.read_text() == "a = 1\nCHANGED\n"


def test_edit_hashline_overlapping_ranges_rejected(tmp_path):
    f = tmp_path / "f.py"
    f.write_text("1\n2\n3\n4\n")
    out = coding.edit_file(
        str(tmp_path),
        "f.py",
        [
            {"pos": _anchor(f, 1), "end": _anchor(f, 3), "lines": ["x"]},
            {"pos": _anchor(f, 2), "lines": ["y"]},
        ],
    )
    assert "error" in out and "overlap" in out["error"]


def test_edit_rejects_malformed_edit(tmp_path):
    (tmp_path / "f.py").write_text("a\n")
    assert "error" in coding.edit_file(str(tmp_path), "f.py", [{"nope": 1}])
    assert "error" in coding.edit_file(str(tmp_path), "f.py", [])
    assert "error" in coding.edit_file(str(tmp_path), "f.py", "not-a-list")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Permissions + feature gating
# ---------------------------------------------------------------------------


def test_permission_defaults(tmp_path):
    p = PermissionEngine(db_path=str(tmp_path / "config.db"))
    assert p.check("read") == PermissionLevel.ALWAYS
    assert p.check("write") == PermissionLevel.ASK
    assert p.check("edit") == PermissionLevel.ASK
    assert p.check("bash") == PermissionLevel.ASK


def test_write_tools_are_write_actions(tmp_path):
    p = PermissionEngine(db_path=str(tmp_path / "config.db"))
    assert p.is_write_action("write", {"path": "x"})
    assert p.is_write_action("edit", {"path": "x"})
    assert not p.is_write_action("read", {"path": "x"})


def test_bash_inherits_never_rails(tmp_path):
    # The hard NEVER rails must hold for every bash command, allowlisted or not.
    p = PermissionEngine(db_path=str(tmp_path / "config.db"))
    drop = {"command": 'sqlite3 /app/data/x.db "DROP TABLE t"'}
    assert p.check("bash", drop) == PermissionLevel.NEVER
    # An unknown command still defaults to ASK.
    assert p.check("bash", {"command": "make test"}) == PermissionLevel.ASK


_FILE_TOOLS = {"read", "write", "edit"}


def test_feature_gate_hides_file_tools_when_off():
    from core.agent import TOOLS

    gated = apply_feature_gates(TOOLS, secrets_available=False, workspace_enabled=False)
    names = {t["name"] for t in gated}
    assert not (_FILE_TOOLS & names)
    assert "bash" in names  # bash is not workspace-gated (the old run_command)


def test_feature_gate_shows_file_tools_when_on():
    from core.agent import TOOLS

    gated = apply_feature_gates(TOOLS, secrets_available=False, workspace_enabled=True)
    assert (_FILE_TOOLS | {"bash"}) <= {t["name"] for t in gated}


def test_workspace_config_defaults_off():
    cfg = Config()
    assert cfg.workspace.enabled is False
    assert cfg.workspace.directory == ""


# --- #155: YOLO covers the coding-harness write tools ---


def _yolo_gate_agent(tmp_path):
    """Bare AgentCore wired only for the _execute_tool_inner permission gate, with
    the actual tool dispatch stubbed so nothing touches the filesystem."""
    from unittest.mock import AsyncMock, MagicMock

    from core.agent import AgentCore
    from core.executor import ToolExecutor

    agent = object.__new__(AgentCore)
    agent.permissions = PermissionEngine(db_path=str(tmp_path / "config.db"))
    agent.executor = ToolExecutor()
    agent.executor.run_in_dir = AsyncMock(return_value={"exit_code": 0})
    agent._request_approval = AsyncMock(return_value="approved")
    agent._tool_write = MagicMock(return_value={"ok": True})
    agent._tool_edit = MagicMock(return_value={"ok": True})
    agent._workspace_cwd = MagicMock(return_value=str(tmp_path))
    agent._workspace_dir = MagicMock(return_value=str(tmp_path))
    return agent


def _call(name, params):
    from core.llm import LLMToolCall

    return LLMToolCall(id="t1", name=name, arguments=params)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,params",
    [
        ("write", {"path": "a.txt", "content": "x"}),
        ("edit", {"path": "a.txt", "edits": [{"oldText": "x", "newText": "y"}]}),
        ("bash", {"command": "make test", "purpose": "t"}),
    ],
)
async def test_yolo_auto_approves_coding_tools(tmp_path, name, params):
    """Under YOLO these ASK-level write tools run without an approval prompt (#155)."""
    agent = _yolo_gate_agent(tmp_path)
    result = await agent._execute_tool_inner(_call(name, params), "telegram", "u1", {"yolo": True})
    assert "error" not in result  # dispatched, not denied
    agent._request_approval.assert_not_awaited()


@pytest.mark.asyncio
async def test_without_yolo_coding_tools_still_prompt(tmp_path):
    """Without YOLO the same tool prompts for approval as before (#155)."""
    agent = _yolo_gate_agent(tmp_path)
    await agent._execute_tool_inner(
        _call("write", {"path": "a.txt", "content": "x"}), "telegram", "u1", {"yolo": False}
    )
    agent._request_approval.assert_awaited_once()


@pytest.mark.asyncio
async def test_yolo_still_refuses_never_actions(tmp_path):
    """A NEVER-rail action is refused even under YOLO, with no prompt (#155)."""
    agent = _yolo_gate_agent(tmp_path)
    drop = {"command": 'sqlite3 /app/data/x.db "DROP TABLE t"', "purpose": "t"}
    result = await agent._execute_tool_inner(_call("bash", drop), "telegram", "u1", {"yolo": True})
    assert result == {"error": "This action is not allowed."}
    agent._request_approval.assert_not_awaited()
    agent.executor.run_in_dir.assert_not_awaited()


@pytest.mark.asyncio
async def test_unlisted_bash_runs_workspace_confined(tmp_path):
    """A non-allowlisted command routes to the workspace rail after approval (#178)."""
    agent = _yolo_gate_agent(tmp_path)
    result = await agent._execute_tool_inner(
        _call("bash", {"command": "make test", "purpose": "t"}), "telegram", "u1", {}
    )
    assert result == {"exit_code": 0}
    agent._request_approval.assert_awaited_once()
    agent.executor.run_in_dir.assert_awaited_once_with("make test", str(tmp_path))


@pytest.mark.asyncio
async def test_unlisted_bash_without_workspace_errors(tmp_path):
    """No workspace → a non-allowlisted command surfaces the allowlist error."""
    agent = _yolo_gate_agent(tmp_path)
    agent._workspace_cwd = lambda: None
    result = await agent._execute_tool_inner(
        _call("bash", {"command": "make test", "purpose": "t"}), "telegram", "u1", {"yolo": True}
    )
    assert "error" in result and "not allowed" in result["error"]
