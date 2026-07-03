"""Tests for core.db.ensure_wal (issue #168)."""

from __future__ import annotations

import sqlite3

from core.db import ensure_wal


def test_ensure_wal_flips_all_dbs(tmp_path) -> None:
    for name in ("history.db", "jobs.db"):
        sqlite3.connect(tmp_path / name).execute("CREATE TABLE t(a)")

    ensure_wal(tmp_path)

    for name in ("history.db", "jobs.db"):
        mode = sqlite3.connect(tmp_path / name).execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal", (name, mode)


def test_ensure_wal_empty_dir_is_noop(tmp_path) -> None:
    ensure_wal(tmp_path)  # no *.db files — must not raise
