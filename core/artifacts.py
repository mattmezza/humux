"""Agent-crafted web artifacts.

The agent writes a self-contained HTML page (``write_artifact`` tool) and gets
back a shareable link.  Pages are served read-only at ``/artifacts/<id>`` on the
admin app — no auth, but the id is an unguessable random string so the route
cannot be listed or walked.  A background loop deletes pages older than the
configured TTL.

One artifact == one standalone HTML document.  CSS goes inline in ``<style>``,
JS inline in ``<script>``, and Tailwind/Alpine load from a CDN — so a single
file covers every rung of the complexity ladder without a build step.
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import time
from pathlib import Path

log = logging.getLogger(__name__)

# How often the background loop sweeps for expired artifacts.
# ponytail: fixed hourly sweep; the TTL is the tunable knob, not the cadence.
_CLEANUP_INTERVAL_S = 3600

# Artifact ids come from ``secrets.token_urlsafe`` → [A-Za-z0-9_-]. Validating
# against this on read makes path traversal impossible: no '/', '.', or '\\'
# ever reaches the filesystem join.
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Served with every artifact response. ``sandbox`` puts the page in a unique
# opaque origin, so its (agent-authored) JS cannot read the admin origin's
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


class ArtifactStore:
    """Filesystem-backed store for self-contained HTML artifacts."""

    def __init__(self, directory: str | Path, ttl_hours: int = 168, enabled: bool = True):
        self.dir = Path(directory)
        self.ttl_hours = ttl_hours
        self.enabled = enabled

    def write(self, html: str, *, title: str = "") -> str:
        """Write ``html`` under a fresh random id; return that id."""
        self.dir.mkdir(parents=True, exist_ok=True)
        art_id = secrets.token_urlsafe(12)
        (self.dir / f"{art_id}.html").write_text(html, encoding="utf-8")
        log.info("Artifact written: %s (%d bytes) %s", art_id, len(html), title)
        return art_id

    def path_for(self, art_id: str) -> Path | None:
        """Resolve an id to its file path, or ``None`` if the id is malformed."""
        if not _ID_RE.match(art_id):
            return None
        return self.dir / f"{art_id}.html"

    def cleanup(self) -> int:
        """Delete artifacts older than ``ttl_hours``; return the count removed.

        ``ttl_hours <= 0`` keeps artifacts forever (no-op).
        """
        if self.ttl_hours <= 0 or not self.dir.exists():
            return 0
        cutoff = time.time() - self.ttl_hours * 3600
        removed = 0
        for f in self.dir.glob("*.html"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    removed += 1
            except OSError:
                log.warning("Failed to remove expired artifact: %s", f)
        if removed:
            log.info("Artifact cleanup removed %d expired file(s)", removed)
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
