"""Agent-crafted web artifacts.

The agent publishes an artifact (``write_artifact`` tool) and gets back a
shareable link.  An artifact is a **directory** ``<base>/<id>/`` served read-only
at ``/artifacts/<id>/`` — so it can be a single page, a multi-file site
(``index.html`` linking sibling CSS/JS/assets), or any shareable file (PDF,
image, slides, doc).  ``<id>`` is an unguessable random string, so the route
cannot be listed or walked.

Content arrives one of two ways:
  * ``files`` — inline text the agent authored (HTML/CSS/JS/SVG/Markdown).
  * ``source_path`` — a file or directory the agent already produced on disk
    (e.g. a PDF from pandoc), copied into the artifact.

Each artifact carries its own TTL (chosen by the agent, default from config;
``0`` = keep forever) in a ``.meta.json`` sidecar; a background loop deletes
expired ones.  ``.meta.json`` and any dotfile are never served.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import shutil
import time
from pathlib import Path

log = logging.getLogger(__name__)

# How often the background loop sweeps for expired artifacts.
# ponytail: fixed hourly sweep; the per-artifact TTL is the tunable knob.
_CLEANUP_INTERVAL_S = 3600

# Hard ceiling on a single artifact (sum of its files). Generous enough for a
# PDF or a small image gallery; a cheap backstop at the write chokepoint against
# a runaway loop, since inline writes run without an approval prompt.
MAX_ARTIFACT_BYTES = 25 * 1024 * 1024  # 25 MiB

_META_NAME = ".meta.json"

# Artifact ids come from ``secrets.token_urlsafe`` → [A-Za-z0-9_-]. Validating
# against this makes id-based traversal impossible.
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
    "<body><div><h1>404</h1><p>This artifact does not exist or has expired.</p>"
    "</div></body></html>"
)


def _safe_rel(name: str) -> str | None:
    """Normalize a relative filename, or ``None`` if it is unsafe to write/serve.

    Rejects absolute paths, ``..`` components, and any dotfile (so a copied tree
    can't expose ``.git``/``.env`` and ``.meta.json`` stays internal).
    """
    name = str(name).strip().replace("\\", "/").lstrip("/")
    # Reject control chars (incl. the null byte, which makes Path.resolve raise a
    # ValueError that would otherwise surface as a 500 on the public route).
    if not name or any(ord(c) < 32 for c in name):
        return None
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." or p.startswith(".") for p in parts):
        return None
    return "/".join(parts)


def valid_id(art_id: str) -> bool:
    """True if ``art_id`` is a well-formed artifact id (no traversal/control chars)."""
    return bool(_ID_RE.match(art_id))


def _dir_size(base: Path) -> int:
    return sum(f.stat().st_size for f in base.rglob("*") if f.is_file() and not f.is_symlink())


class ArtifactStore:
    """Filesystem-backed store for web artifacts (one directory per artifact)."""

    def __init__(self, directory: str | Path, ttl_hours: int = 168, enabled: bool = True):
        self.dir = Path(directory)
        self.ttl_hours = ttl_hours  # default TTL when an artifact doesn't set its own
        self.enabled = enabled

    def create(
        self,
        *,
        files: dict[str, str] | None = None,
        source_path: str | Path | None = None,
        entrypoint: str = "index.html",
        ttl_hours: int | None = None,
        title: str = "",
    ) -> str:
        """Create a new artifact directory; return its random id.

        Provide exactly one of ``files`` (name → text) or ``source_path`` (a file
        or directory to copy in). Raises ``ValueError`` on bad input or oversize.
        """
        if bool(files) == bool(source_path):
            raise ValueError("provide exactly one of 'files' or 'source_path'")

        art_id = secrets.token_urlsafe(12)
        base = self.dir / art_id
        base.mkdir(parents=True, exist_ok=False)
        try:
            if files:
                for name, content in files.items():
                    rel = _safe_rel(name)
                    if rel is None:
                        raise ValueError(f"unsafe filename: {name!r}")
                    target = base / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(str(content), encoding="utf-8")
            else:
                src = Path(source_path)  # type: ignore[arg-type]
                if not src.exists():
                    raise ValueError(f"source_path not found: {source_path}")
                if src.is_dir():
                    # symlinks=True preserves links as links so the serve-time
                    # guard rejects them, rather than inlining their target bytes.
                    shutil.copytree(src, base, dirs_exist_ok=True, symlinks=True)
                else:
                    shutil.copy2(src, base / src.name)
                    entrypoint = src.name  # one file → it IS the entry, ignore default

            total = _dir_size(base)
            if total > MAX_ARTIFACT_BYTES:
                raise ValueError(f"artifact too large ({total} bytes, max {MAX_ARTIFACT_BYTES})")

            entrypoint = self._resolve_entrypoint(base, entrypoint)

            ttl = self.ttl_hours if ttl_hours is None else ttl_hours
            expires_at = None if ttl <= 0 else time.time() + ttl * 3600
            meta = {"expires_at": expires_at, "title": title, "entrypoint": entrypoint}
            (base / _META_NAME).write_text(json.dumps(meta), encoding="utf-8")
        except BaseException:
            shutil.rmtree(base, ignore_errors=True)
            raise

        log.info("Artifact written: %s (%d bytes, entry=%s) %s", art_id, total, entrypoint, title)
        return art_id

    @staticmethod
    def _resolve_entrypoint(base: Path, entrypoint: str) -> str:
        """Return the file served at the artifact root, or raise if none works.

        Auto-picks the sole top-level file when the default ``index.html`` is
        absent; otherwise the agent must name a valid ``entrypoint``.
        """
        rel = _safe_rel(entrypoint)
        if rel and (base / rel).is_file():
            return rel
        if entrypoint == "index.html":
            tops = [p.name for p in base.iterdir() if p.is_file() and not p.name.startswith(".")]
            if len(tops) == 1:
                return tops[0]
        raise ValueError(
            f"entrypoint {entrypoint!r} not found in the artifact; "
            "include an index.html or pass a valid 'entrypoint'"
        )

    def _meta(self, art_id: str) -> dict | None:
        try:
            return json.loads((self.dir / art_id / _META_NAME).read_text(encoding="utf-8"))
        except OSError, ValueError:
            return None

    def resolve(self, art_id: str, file_path: str = "") -> Path | None:
        """Resolve a request to a servable file, or ``None`` if unsafe/missing.

        Blocks traversal, dotfiles (incl. ``.meta.json``), symlinks, hardlinks,
        and anything resolving outside the artifact dir — so the public route
        can't be tricked into serving e.g. a planted link to ``data/master.key``.
        """
        if not _ID_RE.match(art_id):
            return None
        base = self.dir / art_id
        if not base.is_dir():
            return None
        if not file_path:
            meta = self._meta(art_id)
            file_path = (meta or {}).get("entrypoint") or "index.html"
        rel = _safe_rel(file_path)
        if rel is None:
            return None
        target = base / rel
        try:
            resolved = target.resolve()
        except OSError, ValueError:
            return None
        if target.is_symlink() or not resolved.is_relative_to(base.resolve()):
            return None
        if not target.is_file():
            return None
        if target.stat().st_nlink > 1:  # hardlink to a file elsewhere on the volume
            return None
        return target

    def cleanup(self) -> int:
        """Delete artifacts whose stored expiry has passed; return count removed.

        Per-artifact ``expires_at`` (``None`` = keep forever) lives in the
        sidecar. A dir missing/with-unreadable metadata falls back to its mtime
        plus the configured default TTL.
        """
        if not self.dir.exists():
            return 0
        now = time.time()
        removed = 0
        for child in self.dir.iterdir():
            if not child.is_dir():
                continue
            meta = self._meta(child.name)
            if meta is not None:
                expires_at = meta.get("expires_at")
            elif self.ttl_hours > 0:
                try:
                    expires_at = child.stat().st_mtime + self.ttl_hours * 3600
                except OSError:
                    continue
            else:
                expires_at = None
            if expires_at is not None and now > expires_at:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        if removed:
            log.info("Artifact cleanup removed %d expired artifact(s)", removed)
        return removed


async def store_from_config(config_store) -> ArtifactStore:
    """Build an ArtifactStore from the live config store (its source of truth).

    Reads the ``artifacts.*`` keys directly so it works even for installs seeded
    before this feature existed (missing keys → Pydantic defaults).
    """
    from core.config import ArtifactsConfig

    flat = await config_store.get_many("artifacts.")
    data = {k.split(".", 1)[1]: v for k, v in flat.items()}
    cfg = ArtifactsConfig.model_validate(data) if data else ArtifactsConfig()
    return ArtifactStore(cfg.directory, cfg.ttl_hours, cfg.enabled)


async def cleanup_loop(config_store) -> None:
    """Forever: sweep expired artifacts, then sleep. Cancel to stop."""
    while True:
        try:
            store = await store_from_config(config_store)
            if store.enabled:
                store.cleanup()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Artifact cleanup cycle failed")
        await asyncio.sleep(_CLEANUP_INTERVAL_S)
