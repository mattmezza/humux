"""Subagents — scoped sub-loops the main agent can delegate to (issue #15).

A *subagent* is one execution primitive (``AgentCore.run_subagent``) reached by
two trigger paths: on demand via the ``spawn_subagent`` tool, or on a schedule
via a ``subagent`` job. Either way it runs the existing agent loop with **system
semantics** (no goal decomposition / memory / reflection / approval prompts) and
a persona whose tool/skill/secret scope is a *subset* of the caller's — never
wider (``narrow_scope``).

This module holds only the lifecycle bookkeeping: a small in-memory registry of
runs so list / status / cancel work from Telegram and the admin UI. Runs are
ephemeral (an ``asyncio`` task each) — a restart kills them, so an in-memory
registry is the right scope; nothing here needs to survive a reboot.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field

# Keep at most this many finished runs for after-the-fact inspection.
_MAX_FINISHED = 50


def narrow_scope(parent: list[str] | None, child: list[str] | None) -> list[str]:
    """Intersect a child scope with the parent's — inherit, never widen.

    The allowlist convention (see :class:`core.personae.Persona`) is ``[]``/``None``
    = *all*. So an empty parent means "no restriction → the child's own scope
    applies", an empty child means "unspecified → inherit the parent's", and when
    both list names the result is their intersection (the child can never gain a
    name the parent lacked).
    """
    p = parent or []
    c = child or []
    if not p:
        return list(c)
    if not c:
        return list(p)
    return [x for x in c if x in p]


@dataclass(slots=True)
class SubagentRun:
    """One subagent execution and its live status."""

    run_id: str
    persona: str
    task: str
    depth: int = 1
    background: bool = False
    status: str = "running"  # running | done | cancelled | error
    progress: str = ""
    result: str = ""
    error: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    origin_channel: str = ""
    origin_user_id: str = ""
    origin_chat_id: str = ""
    _task: asyncio.Task | None = field(default=None, repr=False, compare=False)

    @property
    def elapsed(self) -> float:
        return (self.finished_at or time.time()) - self.started_at

    @property
    def elapsed_str(self) -> str:
        secs = int(self.elapsed)
        if secs < 60:
            return f"{secs}s"
        return f"{secs // 60}m {secs % 60}s"


class SubagentRegistry:
    """In-memory registry of subagent runs for list / status / cancel."""

    def __init__(self) -> None:
        self._runs: OrderedDict[str, SubagentRun] = OrderedDict()

    def register(self, run: SubagentRun) -> None:
        self._runs[run.run_id] = run
        self._trim()

    def attach_task(self, run_id: str, task: asyncio.Task) -> None:
        run = self._runs.get(run_id)
        if run:
            run._task = task

    def get(self, run_id: str) -> SubagentRun | None:
        return self._runs.get(run_id)

    def active_count(self) -> int:
        return sum(1 for r in self._runs.values() if r.status == "running")

    def list_runs(self, active_only: bool = False) -> list[SubagentRun]:
        runs = [r for r in self._runs.values() if not active_only or r.status == "running"]
        return sorted(runs, key=lambda r: r.started_at, reverse=True)

    def running_for(self, channel: str, chat_id: str) -> list[SubagentRun]:
        """Still-running background runs spawned from one chat, oldest first.

        Surfaced in the spawning agent's turn preamble so it always knows what is
        pending. (Finished runs are folded into the chat history instead, so the
        agent remembers their results natively rather than as ephemeral status.)
        """
        out = [
            r
            for r in self._runs.values()
            if r.status == "running" and r.origin_channel == channel and r.origin_chat_id == chat_id
        ]
        return sorted(out, key=lambda r: r.started_at)

    def finish(self, run_id: str, status: str, *, result: str = "", error: str = "") -> bool:
        """Move a *running* run to a terminal state. Returns False (a no-op) when
        the run is unknown or already finished, so a terminal state is sticky —
        e.g. a late normal completion cannot overwrite a cancellation."""
        run = self._runs.get(run_id)
        if not run or run.status != "running":
            return False
        run.status = status
        run.result = result
        run.error = error
        run.finished_at = time.time()
        self._trim()
        return True

    def cancel(self, run_id: str) -> bool:
        """Request cancellation of a running subagent. Returns False if it is not
        running (unknown id or already finished)."""
        run = self._runs.get(run_id)
        if not run or run.status != "running":
            return False
        if run._task and not run._task.done():
            run._task.cancel()
        # The task's CancelledError handler flips status to "cancelled"; set it
        # eagerly too so a sync caller / the next poll sees it immediately.
        run.status = "cancelled"
        run.finished_at = time.time()
        return True

    def _trim(self) -> None:
        """Drop the oldest finished runs once we exceed the cap (running runs are
        always kept)."""
        finished = [rid for rid, r in self._runs.items() if r.status != "running"]
        excess = len(finished) - _MAX_FINISHED
        for rid in finished[:excess] if excess > 0 else []:
            self._runs.pop(rid, None)


def short_summary(text: str, limit: int = 280) -> str:
    """A one-glance summary of a subagent's result — first non-empty line, capped.

    ponytail: a truncation, not an LLM call. Add a summariser model only if the
    plain preview proves too lossy in practice.
    """
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line[:limit] + ("…" if len(line) > limit else "")
    return (text or "").strip()[:limit]
