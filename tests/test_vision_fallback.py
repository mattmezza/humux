"""Vision fallback: caption images on non-vision models, cache, pass through natives."""

from __future__ import annotations

import pytest

from core.config import Config
from core.llm import LLMResponse, model_supports_vision
from core.models import Attachment


def test_model_supports_vision_heuristic() -> None:
    assert not model_supports_vision("deepseek", "deepseek-v4-flash")
    assert not model_supports_vision("deepseek", "deepseek-chat")
    assert not model_supports_vision("openai", "gpt-3.5-turbo")
    assert not model_supports_vision("anthropic", "")  # unknown / empty
    assert model_supports_vision("anthropic", "claude-haiku-4-5")
    assert model_supports_vision("openai", "gpt-4o")
    assert model_supports_vision("grok", "grok-4-latest")
    assert model_supports_vision("google", "gemini-flash-latest")


class _FakeLLM:
    """Stand-in vision model: counts calls, returns a fixed caption."""

    def __init__(self, provider: str = "anthropic") -> None:
        self.provider = provider
        self.calls = 0

    async def generate(self, *, model, system, messages, tools) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text="a red bicycle leaning on a wall", tool_calls=[])


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from core.agent import AgentCore

    cfg = Config()
    cfg.agent.llm_provider = "deepseek"
    cfg.agent.model = "deepseek-v4-flash"  # no native vision
    cfg.vision.enabled = True
    return AgentCore(cfg)


def _img() -> Attachment:
    return Attachment(data=b"\x89PNG-fake-bytes", mime_type="image/png", filename="x.png")


@pytest.mark.asyncio
async def test_non_vision_model_gets_caption_injected(agent, monkeypatch) -> None:
    fake = _FakeLLM()
    monkeypatch.setattr(agent, "_vision_llm", lambda provider: fake)

    msg = await agent._build_user_message("what is this?", [_img()], "")

    # Plain text content (no image blocks), with the caption injected.
    assert isinstance(msg["content"], str)
    assert "[Image: a red bicycle leaning on a wall]" in msg["content"]
    assert "what is this?" in msg["content"]
    assert fake.calls == 1


@pytest.mark.asyncio
async def test_repeated_identical_image_hits_cache(agent, monkeypatch) -> None:
    fake = _FakeLLM()
    monkeypatch.setattr(agent, "_vision_llm", lambda provider: fake)

    await agent._build_user_message("first", [_img()], "")
    await agent._build_user_message("second", [_img()], "")

    # Same image bytes → captioned once, served from cache the second time.
    assert fake.calls == 1


@pytest.mark.asyncio
async def test_native_vision_model_passes_images_through(agent, monkeypatch) -> None:
    # Switch active model to a vision-capable one → no captioning call at all.
    agent.config.agent.llm_provider = "anthropic"
    agent.config.agent.model = "claude-4-6-sonnet"
    agent.llm.provider = "anthropic"
    fake = _FakeLLM()
    monkeypatch.setattr(agent, "_vision_llm", lambda provider: fake)

    msg = await agent._build_user_message("what is this?", [_img()], "")

    blocks = msg["content"]
    assert isinstance(blocks, list)
    assert any(b.get("type") == "image" for b in blocks)
    assert fake.calls == 0


@pytest.mark.asyncio
async def test_caption_failure_falls_back_to_raw_image(agent, monkeypatch) -> None:
    class _BoomLLM(_FakeLLM):
        async def generate(self, *, model, system, messages, tools):
            raise RuntimeError("vision model down")

    monkeypatch.setattr(agent, "_vision_llm", lambda provider: _BoomLLM())

    msg = await agent._build_user_message("what is this?", [_img()], "")

    # Captioning blew up → raw image blocks are passed through unchanged.
    assert isinstance(msg["content"], list)
    assert any(b.get("type") == "image_url" for b in msg["content"])  # deepseek→openai block
