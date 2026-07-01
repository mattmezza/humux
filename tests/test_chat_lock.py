"""Per-chat turn lock: concurrent turns of the SAME (channel, user_id, chat_id)
serialize, while different chats still run concurrently.

This guards the group-room race — two inbound messages of one chat racing the
shared history/session caches (silent-fold RMW in ``_record_inbound``, session
append) — without throttling unrelated chats.
"""

from __future__ import annotations

import asyncio

import pytest

from core.config import Config
from core.models import AgentResponse


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from core.agent import AgentCore

    return AgentCore(Config())


def _install_overlap_tracker(agent, monkeypatch):
    """Swap ``_process_impl`` for a slow coroutine that records peak overlap.

    It holds the turn across an ``await`` so a racing task overlaps whenever the
    lock does NOT serialize them — making the peak concurrent count the signal.
    """
    state = {"active": 0, "peak": 0}

    async def fake_impl(message, channel, user_id, **kw):
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.05)
        state["active"] -= 1
        return AgentResponse(text="")

    monkeypatch.setattr(agent, "_process_impl", fake_impl)
    return state


@pytest.mark.asyncio
async def test_same_chat_turns_serialize(agent, monkeypatch) -> None:
    state = _install_overlap_tracker(agent, monkeypatch)
    await asyncio.gather(
        agent.process("a", "telegram", "u", chat_id="-100"),
        agent.process("b", "telegram", "u", chat_id="-100"),
    )
    assert state["peak"] == 1  # same key → never ran at the same time


@pytest.mark.asyncio
async def test_different_chats_run_concurrently(agent, monkeypatch) -> None:
    state = _install_overlap_tracker(agent, monkeypatch)
    await asyncio.gather(
        agent.process("a", "telegram", "u", chat_id="-100"),
        agent.process("b", "telegram", "u", chat_id="-200"),
    )
    assert state["peak"] == 2  # different keys → overlapped freely
