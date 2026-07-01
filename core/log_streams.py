"""Per-agent log streams (issue #75).

The admin Logs tab used to show one flat stream, so with several agents and
subagents running at once you could not tell which line belonged to whom. The
fix is a :class:`contextvars.ContextVar` that carries the *stream* a log record
belongs to — the agent slug of the turn in flight, or ``default``.

``asyncio`` copies the context when a task is created, so a background subagent's
task inherits its spawner's stream for free; a second ContextVar marks subagent
lines so they read ``[subagent:<id>] …`` inside that one shared stream (issue
point 2: subagent logs flow into the originating agent's stream, not a separate
one). The admin log buffer reads both off the live context when a record is
emitted — logging is synchronous within the emitting task, so the values are
always the right ones.
"""

from __future__ import annotations

import contextlib
import contextvars

DEFAULT_STREAM = "default"

# Empty = "never set by a turn" (distinct from a turn that ran as the default
# identity, which sets it to "default" explicitly). The distinction matters for
# subagent_stream's fallback below.
_stream: contextvars.ContextVar[str] = contextvars.ContextVar("mpa_log_stream", default="")
_subagent: contextvars.ContextVar[str] = contextvars.ContextVar("mpa_log_subagent", default="")


def current_stream() -> str:
    """The stream the calling task's log records belong to."""
    return _stream.get() or DEFAULT_STREAM


def current_subagent() -> str:
    """The subagent label for the calling task, or ``""`` for a top-level turn."""
    return _subagent.get()


def set_stream(name: str) -> None:
    """Tag every later log record in this task with stream ``name``.

    Called once per turn after the agent is resolved. No reset: channels run
    each turn in its own ``asyncio`` task (so the value dies with the task), and
    a reused task — the REPL — overwrites it at the next turn's entry anyway.
    """
    _stream.set(name or DEFAULT_STREAM)


@contextlib.contextmanager
def subagent_stream(label: str, *, fallback: str = ""):
    """Mark records inside as a subagent's, routed into the spawner's stream.

    Keeps the inherited stream (the spawner's) when a turn set one. For a
    top-level run that inherited none — a scheduled ``subagent`` job — it adopts
    ``fallback`` (the run's own agent slug) so the lines still land in a
    sensibly named stream instead of ``default``.
    """
    stream_token = None
    if not _stream.get() and fallback:
        stream_token = _stream.set(fallback)
    sub_token = _subagent.set(label or "")
    try:
        yield
    finally:
        _subagent.reset(sub_token)
        if stream_token is not None:
            _stream.reset(stream_token)


def _demo() -> None:
    """Self-check: the stream tags follow the context, subagents keep the parent
    stream while carrying their own label, and nesting unwinds cleanly."""
    assert current_stream() == "default" and current_subagent() == ""

    set_stream("coding-helper")
    assert current_stream() == "coding-helper"

    # A spawned subagent keeps the parent stream, adds a label.
    with subagent_stream("sub_abc123", fallback="other"):
        assert current_stream() == "coding-helper"  # parent's, not the fallback
        assert current_subagent() == "sub_abc123"
    assert current_subagent() == ""  # unwound

    # A top-level subagent (no turn set a stream) adopts its fallback.
    _stream.set("")
    with subagent_stream("sub_xyz", fallback="finance-assistant"):
        assert current_stream() == "finance-assistant"
    assert current_stream() == "default"
    print("log_streams demo ok")


if __name__ == "__main__":
    _demo()
