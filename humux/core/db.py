"""One-shot WAL enablement for all SQLite stores (issue #168).

``journal_mode=WAL`` is a *persistent* property of a SQLite file, so flipping it
once at boot lets the live server and a concurrent ``core.cli`` process write
without hitting ``SQLITE_BUSY``. Both entrypoints (``core/main.py`` lifespan and
``core/cli.py``) call :func:`ensure_wal` at startup. Python's default 5s busy
timeout absorbs any residual contention — that is the entire concurrency story.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)


def _data_dir() -> Path:
    # Matches core/cli.py's heuristic: /app/data in Docker, ./data locally.
    p = Path("/app/data")
    return p if p.exists() else Path("data")


def ensure_wal(data_dir: str | Path | None = None) -> None:
    """Set ``journal_mode=WAL`` on every ``*.db`` in the data dir. Best-effort."""
    d = Path(data_dir) if data_dir else _data_dir()
    for db in sorted(d.glob("*.db")):
        try:
            with sqlite3.connect(db) as conn:
                mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if mode.lower() != "wal":
                log.warning("WAL not enabled for %s (mode=%s)", db.name, mode)
        except sqlite3.Error:
            log.exception("Could not enable WAL for %s", db)


def _demo() -> None:  # ponytail: self-check, not a suite
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "x.db"
        sqlite3.connect(p).execute("CREATE TABLE t(a)")
        ensure_wal(tmp)
        mode = sqlite3.connect(p).execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal", mode
    print("ok")


if __name__ == "__main__":
    _demo()
