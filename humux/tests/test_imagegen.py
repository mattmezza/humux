"""Tests for image generation (issue #55)."""

from __future__ import annotations

import base64

import pytest

from core import imagegen
from core.agent import TOOLS, apply_feature_gates
from core.config import Config
from core.imagegen import ImageBudget, ImageGenError


# ── Fake httpx client (the provider funcs take a client, so we can stub it) ──
class FakeResp:
    def __init__(self, status=200, json_data=None, content=b""):
        self.status_code = status
        self._json = json_data
        self.content = content
        self.text = "err body"

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def post(self, url, **kw):
        self.calls.append(("POST", url, kw))
        return self.responses.pop(0)

    async def get(self, url, **kw):
        self.calls.append(("GET", url, kw))
        return self.responses.pop(0)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# ── Config / feature gate ──────────────────────────────────────────────
def test_disabled_by_default():
    assert Config().tools.imagegen.enabled is False


def test_feature_gate_hides_tool_when_disabled():
    names = {
        t["name"]
        for t in apply_feature_gates(TOOLS, secrets_available=True, imagegen_enabled=False)
    }
    assert "generate_image" not in names


def test_feature_gate_shows_tool_when_enabled():
    names = {
        t["name"] for t in apply_feature_gates(TOOLS, secrets_available=True, imagegen_enabled=True)
    }
    assert "generate_image" in names


# ── Key reuse ──────────────────────────────────────────────────────────
def test_dedicated_key_wins():
    cfg = Config()
    cfg.tools.imagegen.api_key = "sk-dedicated"
    cfg.tools.imagegen.provider = "fal"
    assert imagegen.resolve_api_key(cfg) == "sk-dedicated"


def test_openai_reuses_llm_key():
    cfg = Config()
    cfg.tools.imagegen.provider = "openai"
    cfg.agent.openai_api_key = "sk-openai"
    assert imagegen.resolve_api_key(cfg) == "sk-openai"


def test_openrouter_reuse_requires_openrouter_base_url():
    cfg = Config()
    cfg.tools.imagegen.provider = "openrouter"
    cfg.agent.openai_api_key = "sk-or"
    # No openrouter base URL → not detected as an OpenRouter key.
    assert imagegen.resolve_api_key(cfg) == ""
    cfg.agent.openai_base_url = "https://openrouter.ai/api/v1"
    assert imagegen.resolve_api_key(cfg) == "sk-or"


def test_openrouter_reuses_dedicated_llm_key():
    """The dedicated OpenRouter LLM key (#128) is reused without the legacy
    openai_base_url trick, and wins over it."""
    cfg = Config()
    cfg.tools.imagegen.provider = "openrouter"
    cfg.agent.openrouter_api_key = "sk-or-dedicated"
    assert imagegen.resolve_api_key(cfg) == "sk-or-dedicated"


def test_fal_never_reuses_llm_key():
    cfg = Config()
    cfg.tools.imagegen.provider = "fal"
    cfg.agent.openai_api_key = "sk-openai"
    assert imagegen.resolve_api_key(cfg) == ""


# ── Provider request/response shapes ───────────────────────────────────
async def test_openai_parses_b64_and_sends_low_quality():
    client = FakeClient([FakeResp(200, {"data": [{"b64_json": _b64(b"PNG")}]})])
    data, mime = await imagegen._openai(client, "k", "gpt-image-1-mini", "a cat", "1024x1024")
    assert data == b"PNG" and mime == "image/png"
    body = client.calls[0][2]["json"]
    assert body["quality"] == "low"
    assert body["size"] == "1024x1024"
    assert body["model"] == "gpt-image-1-mini"


async def test_openai_dalle_forces_b64_and_omits_quality():
    client = FakeClient([FakeResp(200, {"data": [{"b64_json": _b64(b"X")}]})])
    await imagegen._openai(client, "k", "dall-e-3", "a cat", "")
    body = client.calls[0][2]["json"]
    assert "quality" not in body and "size" not in body
    # dall-e defaults to url responses → we must force b64_json so the parse works.
    assert body["response_format"] == "b64_json"


async def test_openrouter_parses_b64():
    payload = {"data": [{"b64_json": _b64(b"IMG"), "media_type": "image/webp"}]}
    client = FakeClient([FakeResp(200, payload)])
    data, mime = await imagegen._openrouter(client, "k", "google/gemini-2.5-flash-image", "a dog")
    assert data == b"IMG" and mime == "image/webp"
    assert client.calls[0][1] == "https://openrouter.ai/api/v1/images"


async def test_fal_downloads_url():
    client = FakeClient(
        [
            FakeResp(200, {"images": [{"url": "https://cdn/x.jpg", "content_type": "image/jpeg"}]}),
            FakeResp(200, content=b"JPEGBYTES"),
        ]
    )
    data, mime = await imagegen._fal(client, "k", "fal-ai/flux/schnell", "a bird")
    assert data == b"JPEGBYTES" and mime == "image/jpeg"
    assert client.calls[0][1] == "https://fal.run/fal-ai/flux/schnell"
    assert client.calls[0][2]["headers"]["Authorization"] == "Key k"  # fal uses Key, not Bearer
    assert client.calls[1] == ("GET", "https://cdn/x.jpg", {}) or client.calls[1][0] == "GET"


async def test_generate_bytes_requires_key():
    with pytest.raises(ImageGenError):
        await imagegen.generate_bytes("openai", "gpt-image-1-mini", "", "a cat")


def test_check_extracts_provider_message():
    with pytest.raises(ImageGenError, match="bad model"):
        imagegen._check(FakeResp(400, {"error": {"message": "bad model"}}))


# ── Persistence ────────────────────────────────────────────────────────
def test_save_writes_file(tmp_path):
    path = imagegen.save(b"DATA", "image/jpeg", directory=str(tmp_path / "imgs"))
    assert path.endswith(".jpg")
    from pathlib import Path

    assert Path(path).read_bytes() == b"DATA"


# ── Budget ─────────────────────────────────────────────────────────────
async def test_budget_unlimited(tmp_path):
    b = ImageBudget(db_path=str(tmp_path / "ig.db"))
    assert await b.check(0, 0) is None


async def test_budget_daily_cap(tmp_path):
    b = ImageBudget(db_path=str(tmp_path / "ig.db"))
    assert await b.check(2, 0) is None
    await b.record()
    await b.record()
    assert await b.usage() == (2, 2)
    assert await b.check(2, 0) is not None  # daily reached
    assert await b.check(5, 0) is None  # higher cap ok


async def test_budget_monthly_cap(tmp_path):
    b = ImageBudget(db_path=str(tmp_path / "ig.db"))
    await b.record()
    assert await b.check(0, 1) is not None  # monthly reached
    assert await b.check(0, 2) is None


async def test_budget_warning_near_limit(tmp_path):
    b = ImageBudget(db_path=str(tmp_path / "ig.db"))
    for _ in range(8):
        await b.record()
    assert await b.warning(0, 0) is None  # unlimited → no warning
    assert await b.warning(20, 0) is None  # 8/20 → not near
    assert await b.warning(10, 0) is not None  # 8/10 ≥ 80% → near
    assert await b.warning(8, 0) is None  # 8/8 is at cap, not "approaching"
