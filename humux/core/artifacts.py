"""Serve agent-published web artifacts from the workspace (issue #82).

An artifact is no longer a dedicated tool with its own storage and TTL. It is
just files the agent writes — with the coding-harness ``write_file`` tool — under
``{workspace}/artifacts/{slug}/``. This module only resolves a *public* request
``/artifacts/{slug}/{path}`` to a servable file, keeping the same hardening the
old ``ArtifactStore`` had: traversal, dotfiles, symlinks and hardlinks are all
refused, so the no-auth public route can't be tricked into serving e.g. a
sibling ``.env``/``.git`` or anything outside the ``artifacts/`` subdir.

The slug is agent-chosen (not an unguessable id), so the route is *guessable* by
design — artifacts are public shareables; don't put secrets in one. Confinement
to the ``artifacts/`` subdir (not the whole workspace) is the real protection.
"""

from __future__ import annotations

import re
from pathlib import Path

# The fixed subdirectory of the workspace that the /artifacts/ route serves.
ARTIFACTS_SUBDIR = "artifacts"

# A slug is one path component: [A-Za-z0-9_-], so it can't traverse or hide a
# dotfile. The agent picks it; the skill tells it to stay in this charset.
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Served with every artifact response. ``sandbox`` puts the page in a unique
# opaque origin, so its (agent-authored) JS/SVG cannot read the admin origin's
# localStorage — where the admin API key lives. CDN scripts, inline scripts,
# forms and popups still work; only same-origin privilege is dropped.
ARTIFACT_CSP = "sandbox allow-scripts allow-forms allow-popups allow-modals allow-downloads"

NOT_FOUND_HTML = (
    "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
    "<title>Not found</title><meta name='viewport' "
    "content='width=device-width, initial-scale=1'>"
    "<style>body{font-family:system-ui,sans-serif;display:grid;place-items:center;"
    "min-height:100vh;margin:0;color:#444;background:#fafafa}"
    "div{text-align:center}h1{font-size:3rem;margin:0}</style></head>"
    "<body><div><h1>404</h1><p>This artifact does not exist.</p>"
    "</div></body></html>"
)


def _safe_rel(name: str) -> str | None:
    """Normalize a relative filename, or ``None`` if it is unsafe to serve.

    Rejects absolute paths, ``..`` components, control chars (incl. the null byte
    that would otherwise make ``Path.resolve`` raise a 500 on the public route),
    and any dotfile (so a planted ``.env`` stays unservable).
    """
    name = str(name).strip().replace("\\", "/").lstrip("/")
    if not name or any(ord(c) < 32 for c in name):
        return None
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." or p.startswith(".") for p in parts):
        return None
    return "/".join(parts)


def valid_id(slug: str) -> bool:
    """True if ``slug`` is a well-formed artifact slug (no traversal/control chars)."""
    return bool(_ID_RE.match(slug))


def resolve(base: Path, slug: str, file_path: str = "") -> Path | None:
    """Resolve a request to a servable file under ``base/slug/``, or ``None``.

    Blocks traversal, dotfiles, symlinks, hardlinks, and anything resolving
    outside the artifacts ``base`` — so the public route can't be tricked into
    serving a sibling source file or a planted link to ``../.env``.
    """
    if not valid_id(slug):
        return None
    rel = _safe_rel(file_path) if file_path else "index.html"
    if rel is None:
        return None
    root = base.resolve()
    target = root / slug / rel
    try:
        resolved = target.resolve()
    except OSError, ValueError:
        return None
    if target.is_symlink() or not resolved.is_relative_to(root):
        return None
    if not target.is_file():
        return None
    if target.stat().st_nlink > 1:  # hardlink to a file elsewhere on the volume
        return None
    return target


async def serving_base(config_store) -> Path | None:
    """The directory served at ``/artifacts/``, or ``None`` if serving is off.

    Artifacts live under the coding-harness workspace (issue #82), so serving is
    gated by both the workspace (which provides the agent's write path) and
    ``artifacts.enabled`` (the public no-auth route toggle). Keys may be absent on
    a store seeded before this change — fall back to the config defaults
    (workspace off, artifacts on).
    """
    ws_enabled = (await config_store.get("workspace.enabled")) == "true"
    ws_dir = (await config_store.get("workspace.directory") or "").strip()
    art_enabled = await config_store.get("artifacts.enabled")
    art_on = art_enabled is None or art_enabled == "true"
    if not (ws_enabled and ws_dir and art_on):
        return None
    return (Path(ws_dir).expanduser() / ARTIFACTS_SUBDIR).resolve()
