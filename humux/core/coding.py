"""Coding harness — confined file read/write/edit for the agent (#76, #178).

A minimal set of file tools that let the agent operate on a real codebase
directly. Listing and searching moved to the `bash` tool (`ls`, `rg`, `find`),
so this module is just the three structured file operations: read, write, edit.

Every path is resolved against a single **allowed workspace root**; anything
that escapes it (via ``..`` or a symlink) is refused. That containment is the
trust boundary, so the check is ``realpath``-based, not a string-prefix compare:
``resolve()`` follows symlinks and collapses ``..`` before we test containment.

Reads return **hashline**-prefixed lines (``12#KT:content``): each line carries
a 2-char hash of itself and its neighbours, so an edit can address a line or
range by anchor (``{"pos": "12#KT", "lines": [...]}``) instead of quoting the
full text back — a stale anchor (file changed since the read) is rejected.
Inspired by pi-hashline-edit; the hash only needs to be stable between a read
and the following edit, so stdlib crc32 does the job.

These functions are pure (no agent/LLM state) so they are unit-testable on
their own; the thin ``_tool_*`` wrappers in ``core/agent.py`` add permission
gating and logging. Write/edit are permission-gated (ASK); read is
pre-approved (ALWAYS).
"""

from __future__ import annotations

import re
import zlib
from pathlib import Path

# read reads whole files into memory and is pre-approved (no prompt), so cap
# file size to avoid an accidental multi-GB OOM. Source files are far smaller;
# bigger blobs should be inspected via bash. ponytail: flat cap, stream
# line-by-line only if huge code files ever become a real need.
MAX_READ_BYTES = 10 * 1024 * 1024  # 10 MiB

# 16-char hashline alphabet (same as pi-hashline-edit, so anchors look familiar).
_HASH_ALPHABET = "ZPMQVRWSNKTXJBYH"
_ANCHOR_RE = re.compile(r"^\s*(\d+)#([A-Z]{2})\s*$")


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


def _line_hash(prev: str, curr: str, nxt: str) -> str:
    """2-char hashline hash of a line in its context (prev/next neighbours).

    Context-sensitive on purpose: editing line N invalidates the anchors of
    N-1..N+1 while distant anchors stay valid, and identical lines in different
    places get different hashes so an anchor can't silently hit the wrong copy.
    """
    h = zlib.crc32(f"{prev}\0{curr}\0{nxt}".encode())
    return _HASH_ALPHABET[(h >> 4) & 15] + _HASH_ALPHABET[h & 15]


def hash_lines(lines: list[str]) -> list[str]:
    """The hashline hash for every line of a file (parallel to ``lines``)."""
    n = len(lines)
    return [
        _line_hash(lines[i - 1] if i else "", lines[i], lines[i + 1] if i + 1 < n else "")
        for i in range(n)
    ]


def read_file(workspace: str, path: str, offset: int = 0, limit: int = 100) -> dict:
    """Read up to ``limit`` lines starting at 0-indexed ``offset``.

    Each returned line is prefixed ``LINE#HH:`` — a 1-based line number plus
    its 2-char hashline hash, usable as an edit anchor (see :func:`edit_file`).
    """
    target = resolve_in_workspace(workspace, path)
    if not target.is_file():
        return {"error": f"Not a file: {path}"}
    size = target.stat().st_size
    if size > MAX_READ_BYTES:
        return {
            "error": (
                f"File too large to read ({size} bytes > {MAX_READ_BYTES}); "
                "inspect it via bash (rg, head, tail) instead."
            )
        }
    offset = max(0, int(offset))
    limit = max(1, int(limit))
    lines = target.read_text(errors="replace").splitlines()
    hashes = hash_lines(lines)
    chunk = lines[offset : offset + limit]
    numbered = "\n".join(
        f"{offset + i + 1}#{hashes[offset + i]}:{line}" for i, line in enumerate(chunk)
    )
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


def _parse_anchor(anchor: str, lines: list[str], hashes: list[str]) -> tuple[int, str]:
    """Resolve a ``"12#KT"`` anchor to a 1-based line number, or ``(0, error)``.

    The hash must match the line's CURRENT hash — a mismatch means the file
    changed since the read that produced the anchor, so the edit must not be
    applied blind.
    """
    m = _ANCHOR_RE.match(str(anchor))
    if not m:
        return 0, f"Invalid anchor {anchor!r}: expected LINE#HH, e.g. '12#KT'."
    n, want = int(m.group(1)), m.group(2)
    if not 1 <= n <= len(lines):
        return 0, f"Anchor {anchor!r} is out of range (file has {len(lines)} lines)."
    if hashes[n - 1] != want:
        return 0, (
            f"Stale anchor {anchor!r}: line {n} now hashes to #{hashes[n - 1]}. "
            "The file changed since you read it — read it again and retry."
        )
    return n, ""


def edit_file(workspace: str, path: str, edits: list[dict]) -> dict:
    """Apply one or more edits to ``path`` atomically (all-or-nothing).

    Each edit is one of:

    * ``{"oldText": ..., "newText": ...}`` — exact-substring replace.
      ``oldText`` must be unique in the file unless ``"all": true`` replaces
      every occurrence.
    * ``{"pos": "12#KT", "lines": [...]}`` — replace the anchored line (or the
      inclusive range up to ``"end": "15#BH"``) with ``lines`` (``[]`` deletes).
      Anchors come from a ``read`` and are validated against the current
      content, so a stale anchor errors instead of hitting the wrong line.

    Anchor edits are validated against one snapshot and applied bottom-up, so
    several ranges in one call don't shift each other; text edits then apply in
    order. Nothing is written unless every edit succeeds.
    """
    target = resolve_in_workspace(workspace, path)
    if not target.is_file():
        return {"error": f"Not a file: {path}"}
    if not isinstance(edits, list) or not edits:
        return {"error": "edits must be a non-empty array of edit objects."}

    text = target.read_text()
    line_ops: list[dict] = []
    text_ops: list[dict] = []
    for i, e in enumerate(edits):
        if not isinstance(e, dict):
            return {"error": f"Edit #{i + 1} is not an object."}
        if "oldText" in e:
            text_ops.append(e)
        elif "pos" in e:
            line_ops.append(e)
        else:
            return {"error": f"Edit #{i + 1} needs either oldText/newText or pos/lines."}

    replacements = 0

    # Anchor edits: validate everything against the read snapshot, then apply
    # bottom-up so earlier replacements don't shift later line numbers.
    if line_ops:
        # ponytail: assumes \n line endings (splitlines folds \r\n; reassembly
        # normalises). Fine for the code files this harness edits.
        trail = "\n" if text.endswith("\n") else ""
        lines = text.splitlines()
        hashes = hash_lines(lines)
        resolved: list[tuple[int, int, list[str]]] = []
        for e in line_ops:
            start, err = _parse_anchor(e["pos"], lines, hashes)
            if err:
                return {"error": err}
            end = start
            if e.get("end"):
                end, err = _parse_anchor(e["end"], lines, hashes)
                if err:
                    return {"error": err}
            if end < start:
                return {"error": f"Anchor range {e['pos']}..{e['end']} is reversed."}
            new_lines = e.get("lines", [])
            if not isinstance(new_lines, list):
                return {"error": "'lines' must be an array of strings."}
            resolved.append((start, end, [str(ln) for ln in new_lines]))
        resolved.sort(key=lambda r: r[0], reverse=True)
        for i in range(1, len(resolved)):
            if resolved[i][1] >= resolved[i - 1][0]:
                return {"error": "Anchor ranges overlap; split them into separate calls."}
        for start, end, new_lines in resolved:
            lines[start - 1 : end] = new_lines
            replacements += 1
        text = "\n".join(lines) + (trail if lines else "")

    for e in text_ops:
        old = str(e.get("oldText", ""))
        new = str(e.get("newText", ""))
        if not old:
            return {"error": "oldText must not be empty."}
        count = text.count(old)
        if count == 0:
            return {"error": f"oldText not found in file: {old[:120]!r}"}
        if e.get("all"):
            text = text.replace(old, new)
            replacements += count
        else:
            if count > 1:
                return {
                    "error": (
                        f"oldText matches {count} times; add surrounding context to make "
                        'it unique, or pass "all": true to replace every occurrence.'
                    )
                }
            text = text.replace(old, new, 1)
            replacements += 1

    target.write_text(text)
    return {"ok": True, "path": str(target), "replacements": replacements}
