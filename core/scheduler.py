"""Scheduler — APScheduler wrapper backed by JobStore.

The JobStore (``data/jobs.db``) is the single source of truth.
APScheduler runs in-memory only (no SQLAlchemy jobstore) and is
re-synced from the JobStore whenever jobs change.

Job types:
  - "agent" / "agent_silent": natural-language task -> agent.process() -> send result to channel
  - "system": raw CLI command -> executor.run_command_trusted()
  - "memory_consolidation": review short-term memories, promote worthy ones
    to long-term, delete expired entries (uses a lightweight LLM call)
  - "subagent": run agent.run_subagent() under an agent -> send result to channel (issue #15)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler

if TYPE_CHECKING:
    from core.agent import AgentCore
    from core.job_store import JobStore

log = logging.getLogger(__name__)

_AGENT_CONTEXT: AgentCore | None = None


def set_agent_context(agent: AgentCore | None) -> None:
    """Set the global agent context for persisted scheduler jobs."""
    global _AGENT_CONTEXT
    _AGENT_CONTEXT = agent


def _get_agent_context() -> AgentCore | None:
    return _AGENT_CONTEXT


async def run_agent_task(
    task: str,
    channel: str = "telegram",
    job_id: str | None = None,
    silent: bool = False,
    agent_name: str = "",
    origin_user_id: str = "",
    origin_chat_id: str = "",
) -> None:
    """Execute a natural-language task through the agent and deliver the result.

    ``agent_name`` / ``origin_chat_id`` carry the originating context (issue #71):
    the job runs as the agent that scheduled it and is delivered back to the
    chat it was scheduled in. Both fall back to the legacy behaviour (agent
    derived from a ``telegram:<agent>`` channel suffix; delivery to the owner
    DM) when empty, so jobs created before #71 are unaffected.
    """
    agent = _get_agent_context()
    if agent is None:
        log.error("Scheduler agent task dropped; agent not initialized")
        return

    lower_task = task.lower()
    is_email_check = job_id == "email_check" or (
        "email" in lower_task and "notify me if any" in lower_task
    )
    silent_mode = silent or is_email_check
    if silent_mode:
        task = (
            task
            + "\n\nIf there is nothing important or urgent to report, respond with [NO_UPDATES] "
            "only."
        )

    log.info("Scheduler running agent task: %s", task[:100])
    # A "telegram:<agent>" job is generated AS that agent (#29) so the bot
    # that delivers it also writes it — while keeping the "system" execution mode
    # (auto-approved writes, no memory/reflection). Bare channels keep the default.
    gen_agent = channel.split(":", 1)[1] if channel.startswith("telegram:") else None
    try:
        response = await agent.process(
            message=task,
            channel="system",
            user_id="scheduler",
            chat_id="scheduler",
            agent_name=agent_name or gen_agent,
        )

        # Deliver the response to the target channel
        ch = agent.channels.get(channel)
        if ch and response.text:
            text = response.text.replace("[NO_UPDATES]", "").strip()
            if silent_mode and not text:
                log.info("Scheduler silent task produced no updates; skipping send")
                return
            # Deliver to the chat the job was scheduled in (#71); fall back to
            # the owner's chat for jobs with no captured origin (pre-#71, CLI,
            # config-seeded).
            chat_id = origin_chat_id or _get_owner_chat_id(agent, channel)
            if chat_id:
                await ch.send(chat_id, text or response.text)
            else:
                log.warning("Scheduler: no owner chat ID for channel %r, response dropped", channel)
        elif not ch:
            log.warning("Scheduler: channel %r not registered, response dropped", channel)
    except Exception:
        log.exception("Scheduler agent task failed: %s", task[:100])

    # Mark one-shot jobs as done
    if job_id and agent.job_store:
        job = await agent.job_store.get_job(job_id)
        if job and job.get("schedule") == "once" and job.get("status") == "active":
            await agent.job_store.update_status(job_id, "done")
            log.info("One-shot job %r marked as done", job_id)


async def run_subagent_task(
    agent_name: str,
    task: str,
    channel: str = "telegram",
    job_id: str | None = None,
    origin_user_id: str = "",
    origin_chat_id: str = "",
) -> None:
    """Fire a scheduled subagent run and deliver its result to the channel.

    Delivers to the chat the job was scheduled in (#71), falling back to the
    owner's chat when no origin was captured (pre-#71, CLI, config-seeded).
    """
    agent = _get_agent_context()
    if agent is None:
        log.error("Scheduler subagent task dropped; agent not initialized")
        return

    owner = _get_owner_chat_id(agent, channel)
    target = origin_chat_id or owner
    log.info("Scheduler running subagent (agent=%s): %s", agent_name or "default", task[:100])
    try:
        result = await agent.run_subagent(
            task=task,
            agent_name=agent_name or "",
            origin_channel=channel,
            origin_user_id=str(origin_user_id or owner or "scheduler"),
            origin_chat_id=str(target or ""),
            background=False,
        )
        text = result.get("result") or result.get("error") or ""
        ch = agent.channels.get(channel)
        if ch and text and target:
            await ch.send(target, text)
        elif not ch:
            log.warning("Scheduler: channel %r not registered, subagent result dropped", channel)
        elif not target:
            log.warning("Scheduler: no owner chat for channel %r, subagent result dropped", channel)
    except Exception:
        log.exception("Scheduler subagent task failed: %s", task[:100])

    # Mark one-shot jobs as done
    if job_id and agent.job_store:
        job = await agent.job_store.get_job(job_id)
        if job and job.get("schedule") == "once" and job.get("status") == "active":
            await agent.job_store.update_status(job_id, "done")
            log.info("One-shot job %r marked as done", job_id)


async def run_system_command(command: str) -> None:
    """Execute a raw CLI command (e.g. memory cleanup)."""
    agent = _get_agent_context()
    if agent is None:
        log.error("Scheduler system command dropped; agent not initialized")
        return

    log.info("Scheduler running system command: %s", command[:100])
    try:
        result = await agent.executor.run_command_trusted(command)
        if result.get("exit_code", 0) != 0:
            log.warning(
                "Scheduler system command failed (exit %s): %s",
                result.get("exit_code"),
                result.get("stderr", "")[:200],
            )
    except Exception:
        log.exception("Scheduler system command failed: %s", command[:100])


async def run_memory_consolidation() -> None:
    """Review short-term memories, promote worthy ones, delete expired."""
    agent = _get_agent_context()
    if agent is None:
        log.error("Scheduler memory consolidation dropped; agent not initialized")
        return

    log.info("Scheduler running memory consolidation")
    try:
        llm = agent._memory_llm(
            agent.config.memory.consolidation_provider,
            agent.config.memory.consolidation_thinking_level,
        )
        result = await agent.memory.consolidate_and_cleanup(
            llm=llm,
            model=agent.config.memory.consolidation_model,
        )
        log.info(
            "Memory consolidation done: %d reviewed, %d promoted, %d expired deleted",
            result["active_reviewed"],
            result["promoted_to_long_term"],
            result["expired_deleted"],
        )
    except Exception:
        log.exception("Scheduler memory consolidation failed")


def _get_owner_chat_id(agent: AgentCore, channel: str) -> int | str | None:
    """Get the owner's chat ID for proactive messages.

    Works for the default agent's ``telegram`` bot and every other agent's
    ``telegram:<agent>`` bot (#29/#133): each running channel carries its own
    allowlist, so the owner is its first allowed user (None if it allows anyone).
    """
    if channel != "telegram" and not channel.startswith("telegram:"):
        return None
    ch = agent.channels.get(channel)
    ids = getattr(getattr(ch, "config", None), "allowed_user_ids", None)
    return ids[0] if ids else None


def _parse_cron(expr: str) -> dict:
    """Parse a standard 5-field cron expression into APScheduler kwargs.

    Format: minute hour day_of_month month day_of_week
    Example: "0 7 * * *" -> every day at 07:00
    """
    parts = expr.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Expected 5-field cron expression, got {len(parts)} fields: {expr!r}")

    fields = ["minute", "hour", "day", "month", "day_of_week"]
    result = {}
    for field, value in zip(fields, parts):
        if value != "*":
            result[field] = value
    return result


class AgentScheduler:
    """Wraps APScheduler with JobStore as the source of truth."""

    def __init__(self, agent: AgentCore, job_store: JobStore):
        self.agent = agent
        self.job_store = job_store
        self.tz = ZoneInfo(agent.config.agent.timezone)
        set_agent_context(agent)
        # In-memory only scheduler — no SQLAlchemy jobstore.
        # Jobs are persisted in JobStore; APScheduler just runs them.
        self.scheduler = AsyncIOScheduler(timezone=self.tz)

    async def load_jobs(self) -> None:
        """Load all active jobs from JobStore into APScheduler.

        Past one-shots (whose ``run_at`` has already elapsed, e.g. while the
        agent was down) are retired to ``done`` rather than registered, so they
        drop out of the active jobs list instead of lingering forever.
        """
        jobs = await self.job_store.list_jobs(status="active")
        for job in jobs:
            if await self._retire_if_past(job):
                continue
            self._register_job(job)

    def _resolve_run_at(self, job: dict) -> datetime | None:
        """Parse a one-shot job's ``run_at`` into a tz-aware datetime.

        Returns ``None`` when absent or unparseable. Naive datetimes are
        assumed to be in the configured timezone (e.g. a naive
        ``2026-02-20T18:00:00`` is read in the agent's timezone, not UTC).
        """
        run_at_str = job.get("run_at")
        if not run_at_str:
            return None
        try:
            run_at = datetime.fromisoformat(run_at_str)
        except ValueError:
            return None
        if run_at.tzinfo is None:
            run_at = run_at.replace(tzinfo=self.tz)
        return run_at

    async def _retire_if_past(self, job: dict) -> bool:
        """Mark an active, past one-shot job as ``done``. Returns True if retired."""
        if job.get("schedule") != "once" or job.get("status") != "active":
            return False
        run_at = self._resolve_run_at(job)
        if run_at is not None and run_at < datetime.now(self.tz):
            await self.job_store.update_status(job["id"], "done")
            log.info("One-shot job %r is in the past; marked done", job["id"])
            return True
        return False

    def _register_job(self, job: dict) -> None:
        """Register a single job dict into APScheduler."""
        job_id = job["id"]
        job_type = job["type"]
        schedule = job.get("schedule", "cron")
        task = job.get("task", "")
        channel = job.get("channel", "telegram")
        agent = job.get("agent", "")
        origin_user_id = job.get("origin_user_id", "")
        origin_chat_id = job.get("origin_chat_id", "")
        silent = job_type == "agent_silent"

        try:
            if schedule == "once":
                run_at = self._resolve_run_at(job)
                if run_at is None:
                    log.warning("One-shot job %r has no/invalid run_at; skipping", job_id)
                    return
                # Skip one-shots in the past (retired to done by _retire_if_past)
                if run_at < datetime.now(self.tz):
                    log.info("One-shot job %r is in the past; skipping", job_id)
                    return

                if job_type == "system":
                    self.scheduler.add_job(
                        run_system_command,
                        "date",
                        id=job_id,
                        run_date=run_at,
                        kwargs={"command": task},
                        replace_existing=True,
                    )
                elif job_type == "memory_consolidation":
                    self.scheduler.add_job(
                        run_memory_consolidation,
                        "date",
                        id=job_id,
                        run_date=run_at,
                        replace_existing=True,
                    )
                elif job_type == "subagent":
                    self.scheduler.add_job(
                        run_subagent_task,
                        "date",
                        id=job_id,
                        run_date=run_at,
                        kwargs={
                            "agent_name": agent,
                            "task": task,
                            "channel": channel,
                            "job_id": job_id,
                            "origin_user_id": origin_user_id,
                            "origin_chat_id": origin_chat_id,
                        },
                        replace_existing=True,
                    )
                else:
                    self.scheduler.add_job(
                        run_agent_task,
                        "date",
                        id=job_id,
                        run_date=run_at,
                        kwargs={
                            "task": task,
                            "channel": channel,
                            "job_id": job_id,
                            "silent": silent,
                            "agent_name": agent,
                            "origin_user_id": origin_user_id,
                            "origin_chat_id": origin_chat_id,
                        },
                        replace_existing=True,
                    )
            else:
                # Cron job
                cron_expr = job.get("cron")
                if not cron_expr:
                    log.warning("Cron job %r has no cron expression; skipping", job_id)
                    return
                cron_kwargs = _parse_cron(cron_expr)

                if job_type == "system":
                    self.scheduler.add_job(
                        run_system_command,
                        "cron",
                        id=job_id,
                        kwargs={"command": task},
                        replace_existing=True,
                        **cron_kwargs,
                    )
                elif job_type == "memory_consolidation":
                    self.scheduler.add_job(
                        run_memory_consolidation,
                        "cron",
                        id=job_id,
                        replace_existing=True,
                        **cron_kwargs,
                    )
                elif job_type == "subagent":
                    self.scheduler.add_job(
                        run_subagent_task,
                        "cron",
                        id=job_id,
                        kwargs={
                            "agent_name": agent,
                            "task": task,
                            "channel": channel,
                            "job_id": job_id,
                            "origin_user_id": origin_user_id,
                            "origin_chat_id": origin_chat_id,
                        },
                        replace_existing=True,
                        **cron_kwargs,
                    )
                else:
                    self.scheduler.add_job(
                        run_agent_task,
                        "cron",
                        id=job_id,
                        kwargs={
                            "task": task,
                            "channel": channel,
                            "job_id": job_id,
                            "silent": silent,
                            "agent_name": agent,
                            "origin_user_id": origin_user_id,
                            "origin_chat_id": origin_chat_id,
                        },
                        replace_existing=True,
                        **cron_kwargs,
                    )

            log.info("Registered %s job %r", schedule, job_id)

        except Exception:
            log.exception("Failed to register job %r", job_id)

    async def sync_job(self, job_id: str) -> None:
        """Re-sync a single job from JobStore into APScheduler.

        Call this after creating/editing/cancelling a job so APScheduler
        picks up the change without a full reload.
        """
        # Remove the old APScheduler job if it exists
        try:
            self.scheduler.remove_job(job_id)
        except Exception:
            pass

        # If the job is still active, re-register it
        job = await self.job_store.get_job(job_id)
        if job and job["status"] == "active":
            if await self._retire_if_past(job):
                return
            self._register_job(job)
        elif job:
            log.info("Job %r has status %r; removed from scheduler", job_id, job["status"])
        else:
            log.info("Job %r deleted; removed from scheduler", job_id)

    def add_one_shot(self, job_id: str, run_at: datetime, task: str, channel: str) -> None:
        """Schedule a one-time future task (legacy helper, used by schedule_task tool)."""
        self.scheduler.add_job(
            run_agent_task,
            "date",
            id=job_id,
            run_date=run_at,
            kwargs={"task": task, "channel": channel, "job_id": job_id},
            replace_existing=True,
        )
        log.info("Scheduled one-shot job %r at %s", job_id, run_at)

    def start(self) -> None:
        """Start the scheduler. Call after load_jobs()."""
        self.scheduler.start()
        log.info("Scheduler started with %d jobs", len(self.scheduler.get_jobs()))

    def shutdown(self) -> None:
        """Gracefully shut down the scheduler."""
        self.scheduler.shutdown(wait=False)
        log.info("Scheduler shut down")
