"""Tests for JobStore status filtering — the contract behind the jobs UI toggle."""

from __future__ import annotations

from core.job_store import JobStore


def test_list_jobs_sync_filters_by_status(tmp_path) -> None:
    """Default lists only live jobs; include_done reveals done/cancelled (issue #68)."""
    store = JobStore(db_path=str(tmp_path / "jobs.db"))
    for jid, status in (
        ("a", "active"),
        ("p", "paused"),
        ("d", "done"),
        ("c", "cancelled"),
    ):
        store.upsert_job_sync(jid, cron="0 7 * * *", task="t", status=status)

    live = {j["id"] for j in store.list_jobs_sync()}
    assert live == {"a", "p"}

    everything = {j["id"] for j in store.list_jobs_sync(include_done=True)}
    assert everything == {"a", "p", "d", "c"}

    only_done = {j["id"] for j in store.list_jobs_sync(status="done")}
    assert only_done == {"d"}


def test_persona_persists_and_survives_omitting_edit(tmp_path) -> None:
    """A job's owning persona round-trips and is sticky across a re-upsert that
    omits it — the contract the admin/CLI 'edit' path relies on (issue #101)."""
    store = JobStore(db_path=str(tmp_path / "jobs.db"))
    store.upsert_job_sync("brief", cron="0 7 * * *", task="t", persona="coach")
    assert store.get_job_sync("brief")["persona"] == "coach"

    # Re-upsert (e.g. status change) without persona must not wipe it.
    store.upsert_job_sync("brief", cron="0 7 * * *", task="t", status="paused")
    assert store.get_job_sync("brief")["persona"] == "coach"
