"""Multi-message replies (#202): split marker, per-part voice, delivery list."""

import pytest

from core.config import Config
from core.models import AgentResponse, OutputMessage


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from core.agent import AgentCore

    cfg = Config()
    cfg.agent.llm_provider = "deepseek"
    cfg.agent.model = "deepseek-v4-flash"
    cfg.memory.embedding.enabled = False
    return AgentCore(cfg)  # voice pipeline unset → no TTS, split only


# --- AgentResponse.delivery_messages ---------------------------------------


def test_delivery_falls_back_to_flat_text():
    r = AgentResponse(text="hello")
    assert [m.text for m in r.delivery_messages] == ["hello"]


def test_delivery_prefers_messages_list():
    r = AgentResponse(text="a\nb", messages=[OutputMessage(text="a"), OutputMessage(text="b")])
    assert [m.text for m in r.delivery_messages] == ["a", "b"]


def test_delivery_empty_when_nothing_to_send():
    assert AgentResponse(text="").delivery_messages == []


# --- _split_reply -----------------------------------------------------------


@pytest.mark.asyncio
async def test_single_message_backward_compatible(agent):
    messages, combined = await agent._split_reply("just one message", None)
    assert [m.text for m in messages] == ["just one message"]
    assert combined == "just one message"


@pytest.mark.asyncio
async def test_split_marker_yields_multiple_messages(agent):
    messages, combined = await agent._split_reply("first[[split]]second[[split]]third", None)
    assert [m.text for m in messages] == ["first", "second", "third"]
    assert combined == "first\nsecond\nthird"


@pytest.mark.asyncio
async def test_split_swallows_surrounding_whitespace(agent):
    messages, _ = await agent._split_reply("one\n\n[[split]]\n\ntwo", None)
    assert [m.text for m in messages] == ["one", "two"]


@pytest.mark.asyncio
async def test_empty_parts_dropped(agent):
    messages, combined = await agent._split_reply("[[split]]  [[split]]only", None)
    assert [m.text for m in messages] == ["only"]
    assert combined == "only"


@pytest.mark.asyncio
async def test_react_only_turn_yields_no_messages(agent):
    messages, combined = await agent._split_reply("", None)
    assert messages == []
    assert combined == ""


# --- coalescer keeps alternation when a turn stored multiple assistant rows ----


def test_consecutive_assistant_strings_merge():
    from core.llm import _coalesce_user_messages

    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "next"},
    ]
    out = _coalesce_user_messages(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[1]["content"] == "one\n\ntwo"


def test_assistant_tool_use_blocks_not_merged():
    from core.llm import _coalesce_user_messages

    # An assistant turn carrying content blocks (tool_use) must never be merged
    # into an adjacent plain-text assistant turn — it would corrupt the pairing.
    blocks = [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]
    msgs = [
        {"role": "assistant", "content": "text"},
        {"role": "assistant", "content": blocks},
    ]
    out = _coalesce_user_messages(msgs)
    assert len(out) == 2


@pytest.mark.asyncio
async def test_voice_marker_stripped_per_part(agent):
    # No TTS pipeline configured → voice is None, but the marker must never leak.
    messages, combined = await agent._split_reply(
        "spoken bit [respond_with_voice:en][[split]]typed bit", None
    )
    assert [m.text for m in messages] == ["spoken bit", "typed bit"]
    assert "[respond_with_voice" not in combined
