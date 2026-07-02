"""Tests for the coding harness (issue #76): confined file tools + gating."""

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
# read_file
# ---------------------------------------------------------------------------


def test_read_file_paginates_with_line_numbers(tmp_path):
    (tmp_path / "f.txt").write_text("\n".join(f"line{i}" for i in range(10)))
    out = coding.read_file(str(tmp_path), "f.txt", offset=2, limit=3)
    assert out["total_lines"] == 10
    assert out["lines_returned"] == 3
    assert out["content"] == "3\tline2\n4\tline3\n5\tline4"


def test_read_file_missing(tmp_path):
    assert "error" in coding.read_file(str(tmp_path), "nope.txt")


def test_read_file_rejects_oversize(tmp_path, monkeypatch):
    monkeypatch.setattr(coding, "MAX_READ_BYTES", 10)
    (tmp_path / "big.txt").write_text("x" * 50)
    out = coding.read_file(str(tmp_path), "big.txt")
    assert "error" in out and "too large" in out["error"]


# ---------------------------------------------------------------------------
# write_file
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
# edit_file
# ---------------------------------------------------------------------------


def test_edit_file_unique_match(tmp_path):
    (tmp_path / "f.py").write_text("a = 1\nb = 2\n")
    out = coding.edit_file(str(tmp_path), "f.py", "b = 2", "b = 3")
    assert out["replacements"] == 1
    assert (tmp_path / "f.py").read_text() == "a = 1\nb = 3\n"


def test_edit_file_ambiguous_match_refused(tmp_path):
    (tmp_path / "f.py").write_text("x\nx\n")
    out = coding.edit_file(str(tmp_path), "f.py", "x", "y")
    assert "error" in out
    assert (tmp_path / "f.py").read_text() == "x\nx\n"  # untouched


def test_edit_file_multiple_replaces_all(tmp_path):
    (tmp_path / "f.py").write_text("x\nx\n")
    out = coding.edit_file(str(tmp_path), "f.py", "x", "y", multiple=True)
    assert out["replacements"] == 2
    assert (tmp_path / "f.py").read_text() == "y\ny\n"


def test_edit_file_not_found(tmp_path):
    (tmp_path / "f.py").write_text("abc")
    assert "error" in coding.edit_file(str(tmp_path), "f.py", "zzz", "q")


# ---------------------------------------------------------------------------
# list_dir
# ---------------------------------------------------------------------------


def test_list_dir(tmp_path):
    (tmp_path / "dir").mkdir()
    (tmp_path / "file.txt").write_text("12345")
    out = coding.list_dir(str(tmp_path), ".")
    by_name = {e["name"]: e for e in out["entries"]}
    assert by_name["dir"]["type"] == "dir"
    assert by_name["file.txt"]["type"] == "file"
    assert by_name["file.txt"]["size"] == 5


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


def test_grep_matches_with_include(tmp_path):
    (tmp_path / "a.py").write_text("import os\nx = 1\n")
    (tmp_path / "b.txt").write_text("import os\n")
    out = coding.grep(str(tmp_path), r"import", ".", include="*.py")
    assert out["count"] == 1
    assert out["matches"][0]["file"].endswith("a.py")
    assert out["matches"][0]["line"] == 1


def test_grep_invalid_regex(tmp_path):
    assert "error" in coding.grep(str(tmp_path), "(", ".")


def test_grep_skips_binary(tmp_path):
    (tmp_path / "bin").write_bytes(b"match\x00match")
    out = coding.grep(str(tmp_path), "match", ".")
    assert out["count"] == 0


def test_grep_does_not_follow_symlink_escaping_workspace(tmp_path):
    outside = tmp_path.parent / "outside_grep_secret.txt"
    outside.write_text("TOPSECRET token")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "leak.txt").symlink_to(outside)
    (ws / "real.txt").write_text("nothing here")
    out = coding.grep(str(ws), "TOPSECRET", ".")
    assert out["count"] == 0  # the escaping symlink is not read


def test_grep_skips_oversize(tmp_path, monkeypatch):
    monkeypatch.setattr(coding, "MAX_READ_BYTES", 10)
    (tmp_path / "big.txt").write_text("needle\n" * 5)
    assert coding.grep(str(tmp_path), "needle", ".")["count"] == 0


def test_list_dir_skips_escaping_symlink(tmp_path):
    outside = tmp_path.parent / "outside_listed.txt"
    outside.write_text("data")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "leak").symlink_to(outside)
    (ws / "ok.txt").write_text("x")
    names = {e["name"] for e in coding.list_dir(str(ws), ".")["entries"]}
    assert "ok.txt" in names
    assert "leak" not in names


def test_grep_caps_results(tmp_path, monkeypatch):
    monkeypatch.setattr(coding, "GREP_MAX_MATCHES", 5)
    (tmp_path / "f.txt").write_text("\n".join("hit" for _ in range(20)))
    out = coding.grep(str(tmp_path), "hit", ".")
    assert out["count"] == 5
    assert out["truncated"] is True


# ---------------------------------------------------------------------------
# Permissions + feature gating
# ---------------------------------------------------------------------------


def test_permission_defaults(tmp_path):
    p = PermissionEngine(db_path=str(tmp_path / "config.db"))
    assert p.check("read_file") == PermissionLevel.ALWAYS
    assert p.check("list_dir") == PermissionLevel.ALWAYS
    assert p.check("grep") == PermissionLevel.ALWAYS
    assert p.check("write_file") == PermissionLevel.ASK
    assert p.check("edit_file") == PermissionLevel.ASK
    assert p.check("run_command_in_dir") == PermissionLevel.ASK


def test_write_tools_are_write_actions(tmp_path):
    p = PermissionEngine(db_path=str(tmp_path / "config.db"))
    assert p.is_write_action("write_file", {"path": "x"})
    assert p.is_write_action("edit_file", {"path": "x"})
    assert p.is_write_action("run_command_in_dir", {"command": "ls"})
    assert not p.is_write_action("read_file", {"path": "x"})


def test_run_command_in_dir_inherits_run_command_never_rails(tmp_path):
    # The hard NEVER rails defined for run_command must also block run_command_in_dir,
    # not be silently downgraded to ASK (the security regression caught in review).
    p = PermissionEngine(db_path=str(tmp_path / "config.db"))
    drop = {"command": 'sqlite3 /app/data/x.db "DROP TABLE t"'}
    assert p.check("run_command", drop) == PermissionLevel.NEVER
    assert p.check("run_command_in_dir", drop) == PermissionLevel.NEVER
    # An unknown command still defaults to ASK.
    assert p.check("run_command_in_dir", {"command": "make test"}) == PermissionLevel.ASK


_FILE_TOOLS = {"read_file", "write_file", "edit_file", "list_dir", "grep", "run_command_in_dir"}


def test_feature_gate_hides_file_tools_when_off():
    from core.agent import TOOLS

    gated = apply_feature_gates(TOOLS, secrets_available=False, workspace_enabled=False)
    assert not (_FILE_TOOLS & {t["name"] for t in gated})


def test_feature_gate_shows_file_tools_when_on():
    from core.agent import TOOLS

    gated = apply_feature_gates(TOOLS, secrets_available=False, workspace_enabled=True)
    assert _FILE_TOOLS <= {t["name"] for t in gated}


def test_workspace_config_defaults_off():
    cfg = Config()
    assert cfg.workspace.enabled is False
    assert cfg.workspace.directory == ""
