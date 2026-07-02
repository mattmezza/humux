"""Coding harness — confined file read/write/edit/list/grep for the agent (issue #76).

A minimal set of file tools that let the agent operate on a real codebase
directly, instead of describing every change in chat for the user to apply by
hand. Deliberately *not* a plugin system, LSP, or AST editor — those are out of
scope for the first iteration.

Every path is resolved against a single **allowed workspace root**; anything
that escapes it (via ``..`` or a symlink) is refused. That containment is the
trust boundary, so the check is ``realpath``-based, not a string-prefix compare:
``resolve()`` follows symlinks and collapses ``..`` before we test containment.

These functions are pure (no agent/LLM state) so they are unit-testable on their
own; the thin ``_tool_*`` wrappers in ``core/agent.py`` add permission gating and
logging. Write/edit are permission-gated (ASK) by the agent; read/list/grep are
pre-approved (ALWAYS).
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

# ponytail: cap grep matches so a broad pattern can't flood the model's context;
# narrow the pattern or path to see more. Bump if it bites in practice.
GREP_MAX_MATCHES = 200
_LINE_CLIP = 400  # max chars per returned grep/line snippet
# read_file/grep read whole files into memory and are pre-approved (no prompt), so
# cap file size to avoid an accidental multi-GB OOM. Source files are far smaller;
# bigger blobs should be inspected with run_command_in_dir. ponytail: flat cap,
# stream line-by-line only if huge code files ever become a real need.
MAX_READ_BYTES = 10 * 1024 * 1024  # 10 MiB


class WorkspaceError(Exception):
    """A path escaped the workspace, or no workspace is configured."""


def _contained(root: Path, target: Path) -> bool:
    """True if ``target`` is the workspace root or a descendant of it."""
    return target == root or root in target.parents


def resolve_in_workspace(workspace: str, path: str) -> Path:
    """Resolve ``path`` to an absolute path confined to the ``workspace`` root.

    Relative paths resolve under the workspace root; absolute paths are taken
    as-is. The result must be the root itself or a descendant of it — checked
    after ``resolve()`` has followed symlinks and collapsed ``..``, so neither a
    ``../../etc/passwd`` nor a symlink pointing outside can escape. Raises
    :class:`WorkspaceError` otherwise.
    """
    if not workspace or not workspace.strip():
        raise WorkspaceError("No workspace directory is configured.")
    root = Path(workspace).expanduser().resolve()
    raw = Path(path).expanduser()
    target = (raw if raw.is_absolute() else root / raw).resolve()
    if not _contained(root, target):
        raise WorkspaceError(f"Path is outside the allowed workspace: {path}")
    return target


def read_file(workspace: str, path: str, offset: int = 0, limit: int = 100) -> dict:
    """Read up to ``limit`` lines starting at 0-indexed ``offset``, with line numbers."""
    target = resolve_in_workspace(workspace, path)
    if not target.is_file():
        return {"error": f"Not a file: {path}"}
    size = target.stat().st_size
    if size > MAX_READ_BYTES:
        return {
            "error": (
                f"File too large to read ({size} bytes > {MAX_READ_BYTES}); "
                "search it with grep or inspect it via run_command_in_dir."
            )
        }
    offset = max(0, int(offset))
    limit = max(1, int(limit))
    lines = target.read_text(errors="replace").splitlines()
    chunk = lines[offset : offset + limit]
    numbered = "\n".join(f"{offset + i + 1}\t{line}" for i, line in enumerate(chunk))
    return {
        "path": str(target),
        "offset": offset,
        "lines_returned": len(chunk),
        "total_lines": len(lines),
        "content": numbered,
    }


def write_file(workspace: str, path: str, content: str) -> dict:
    """Write ``content`` to ``path``, creating intermediate directories. Overwrites."""
    target = resolve_in_workspace(workspace, path)
    if target.is_dir():
        return {"error": f"Is a directory: {path}"}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return {"ok": True, "path": str(target), "bytes": len(content.encode())}


def edit_file(
    workspace: str, path: str, old_string: str, new_string: str, multiple: bool = False
) -> dict:
    """Find-and-replace ``old_string`` with ``new_string`` in ``path``.

    With ``multiple=False`` (default) the match must be unique — more than one
    occurrence is an error, so an ambiguous edit never silently hits the wrong
    spot. With ``multiple=True`` every occurrence is replaced.
    """
    target = resolve_in_workspace(workspace, path)
    if not target.is_file():
        return {"error": f"Not a file: {path}"}
    if not old_string:
        return {"error": "old_string must not be empty."}
    text = target.read_text()
    count = text.count(old_string)
    if count == 0:
        return {"error": "old_string not found in file."}
    if count > 1 and not multiple:
        return {
            "error": (
                f"old_string matches {count} times; add surrounding context to make it "
                "unique, or pass multiple=true to replace every occurrence."
            )
        }
    if multiple:
        new_text = text.replace(old_string, new_string)
        replacements = count
    else:
        new_text = text.replace(old_string, new_string, 1)
        replacements = 1
    target.write_text(new_text)
    return {"ok": True, "path": str(target), "replacements": replacements}


def list_dir(workspace: str, path: str = ".") -> dict:
    """List one level of ``path`` as ``[{name, type, size}]`` (size 0 for dirs)."""
    target = resolve_in_workspace(workspace, path)
    if not target.is_dir():
        return {"error": f"Not a directory: {path}"}
    entries: list[dict] = []
    for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name)):
        try:
            # A symlink pointing outside the workspace must not leak its target's
            # type/size — confine to the workspace, same boundary as every read.
            if child.is_symlink() and not _contained(target, child.resolve()):
                continue
            is_dir = child.is_dir()
            entries.append(
                {
                    "name": child.name,
                    "type": "dir" if is_dir else "file",
                    "size": 0 if is_dir else child.stat().st_size,
                }
            )
        except OSError:
            continue  # broken symlink / permission — skip, don't fail the whole listing
    return {"path": str(target), "entries": entries}


def grep(workspace: str, pattern: str, path: str = ".", include: str = "") -> dict:
    """Regex-search files under ``path`` (recursive), optionally filtered by glob.

    Returns ``[{file, line, content}]``, capped at :data:`GREP_MAX_MATCHES`.
    Binary files (those with a NUL byte) and files above :data:`MAX_READ_BYTES`
    are skipped. Symlinks resolving outside the workspace are skipped — `rglob`
    can surface them (or descend a symlinked dir), so each file is re-confined.
    """
    root = resolve_in_workspace(workspace, path)
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return {"error": f"Invalid regex: {exc}"}
    files = [root] if root.is_file() else (p for p in root.rglob("*") if p.is_file())
    matches: list[dict] = []
    for f in files:
        if include and not fnmatch.fnmatch(f.name, include):
            continue
        try:
            if not _contained(root, f.resolve()):
                continue  # symlink escaping the workspace
            if f.stat().st_size > MAX_READ_BYTES:
                continue  # too large; don't materialize it
            text = f.read_text(errors="replace")
        except OSError:
            continue
        if "\x00" in text:
            continue  # binary
        for lineno, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                matches.append({"file": str(f), "line": lineno, "content": line[:_LINE_CLIP]})
                if len(matches) >= GREP_MAX_MATCHES:
                    return {
                        "pattern": pattern,
                        "count": len(matches),
                        "matches": matches,
                        "truncated": True,
                    }
    return {"pattern": pattern, "count": len(matches), "matches": matches, "truncated": False}
