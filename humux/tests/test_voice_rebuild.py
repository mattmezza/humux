"""Live voice-pipeline rebuild on config save (#326).

Voice settings used to be startup-only: PATCH /config saved them, said "saved",
and the running pipeline kept the old settings until the next restart.  These
tests cover the three pieces of the fix: the config-key classification, a
pipeline rebuild that only reloads the half that changed, and the background
swap into the agent and every channel.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import voice.pipeline as vp
from api.admin import (
    PENDING_RESTART,
    VOICE_REBUILD,
    AgentState,
    _config_requires_restart,
    _install_voice_pipeline,
    _rebuild_voice_pipeline,
    _schedule_voice_rebuild,
    clear_restart_required,
    create_admin_app,
    restart_banner,
)
from core.config import Config, VoiceConfig

AUTH = {"Authorization": "Bearer secret"}


@pytest.fixture(autouse=True)
def _restore_module_state():
    """VOICE_REBUILD / PENDING_RESTART are process-wide singletons — keep each
    test's mutations from leaking into the next one."""
    before = (VOICE_REBUILD.status, VOICE_REBUILD.error, VOICE_REBUILD.generation)
    keys, err = set(PENDING_RESTART.keys), PENDING_RESTART.error
    try:
        yield
    finally:
        VOICE_REBUILD.status, VOICE_REBUILD.error, VOICE_REBUILD.generation = before
        PENDING_RESTART.keys, PENDING_RESTART.error = keys, err


# ── config-key classification ─────────────────────────────────────────────


def test_voice_keys_are_no_longer_restart_bound() -> None:
    assert _config_requires_restart({"voice.stt_api_base_url": "http://whisper:8000/v1"}) is False
    assert _config_requires_restart({"voice.tts_voice": "af_bella"}) is False
    # history.max_turns is still read once at startup.
    assert _config_requires_restart({"history.max_turns": "20"}) is True


# ── pipeline rebuild ──────────────────────────────────────────────────────


@pytest.fixture
def fake_whisper(monkeypatch):
    """Count Whisper constructions so we can tell what a rebuild reloaded."""
    loads: list[str] = []

    class _FakeWhisper:
        def __init__(self, model: str, compute_type: str = "int8") -> None:
            loads.append(model)

        def transcribe(self, _audio):  # pragma: no cover - never called here
            return [], SimpleNamespace(language="en")

    monkeypatch.setattr(vp, "WhisperModel", _FakeWhisper)
    return loads


def _cfg(**over) -> VoiceConfig:
    """A real VoiceConfig, so a new voice.* field can never be missed here."""
    return Config().voice.model_copy(update=over)


def test_build_returns_none_when_tts_disabled() -> None:
    assert vp.build_voice_pipeline(_cfg(tts_enabled=False)) is None


def test_rebuild_reuses_unchanged_components(fake_whisper) -> None:
    """A voice/URL-only change must not reload Whisper or Kokoro."""
    first = vp.build_voice_pipeline(_cfg())
    assert fake_whisper == ["base"]

    # Voice name + remote STT URL are read per call → nothing to reload.
    second = vp.build_voice_pipeline(
        _cfg(tts_voice="en-GB-SoniaNeural", stt_api_base_url="http://whisper:8000/v1"),
        previous=first,
    )
    assert fake_whisper == ["base"]  # no extra load
    assert second._whisper is first._whisper
    assert second.tts_voice == "en-GB-SoniaNeural"
    assert second.stt_api_base_url == "http://whisper:8000/v1"

    # A per-call setting (STT timeout) is picked up without a reload either.
    third = vp.build_voice_pipeline(_cfg(stt_timeout_seconds=42), previous=second)
    assert fake_whisper == ["base"]
    assert third.stt_timeout_seconds == 42

    # The whisper model itself changed → reload STT only.
    fourth = vp.build_voice_pipeline(_cfg(stt_model="small"), previous=third)
    assert fake_whisper == ["base", "small"]
    assert fourth._whisper is not third._whisper


def test_stt_change_does_not_reload_tts(fake_whisper, monkeypatch) -> None:
    """STT and TTS are independent: an stt_* edit must not touch Kokoro."""
    built: list[tuple] = []

    class _FakeKokoro:
        def __init__(self, model_path: str, voices_path: str) -> None:
            built.append((model_path, voices_path))

    monkeypatch.setitem(sys.modules, "kokoro_onnx", SimpleNamespace(Kokoro=_FakeKokoro))
    first = vp.build_voice_pipeline(_cfg(backend="kokoro"))
    assert len(built) == 1

    second = vp.build_voice_pipeline(_cfg(backend="kokoro", stt_model="small"), previous=first)
    assert len(built) == 1  # Kokoro untouched
    assert second._kokoro is first._kokoro
    assert second.stt_model == "small"


def test_failed_kokoro_load_is_not_retried_on_a_quiet_save(fake_whisper, monkeypatch) -> None:
    """A demoted backend stays demoted until the TTS settings actually change."""
    attempts: list[int] = []

    class _Boom:
        def __init__(self, *_a) -> None:
            attempts.append(1)
            raise RuntimeError("no model")

    monkeypatch.setitem(sys.modules, "kokoro_onnx", SimpleNamespace(Kokoro=_Boom))
    first = vp.build_voice_pipeline(_cfg(backend="kokoro"))
    assert attempts == [1]
    assert first.backend == "edge-tts"  # degraded, as before

    second = vp.build_voice_pipeline(_cfg(backend="kokoro", tts_voice="af_bella"), previous=first)
    assert attempts == [1]  # no retry, no second traceback
    assert second.backend == "edge-tts"


# ── swapping the pipeline in ─────────────────────────────────────────────


class _Channel:
    def __init__(self, voice=None) -> None:
        self.voice = voice


class _FakeAgent:
    def __init__(self, voice=None) -> None:
        self.voice = voice
        self.channels = {"telegram": _Channel(voice), "telegram:helper": _Channel(voice)}


def test_install_updates_agent_and_every_channel() -> None:
    old = object()
    agent = _FakeAgent(old)
    new = object()
    assert _install_voice_pipeline(agent, new) == 2
    assert agent.voice is new
    assert all(ch.voice is new for ch in agent.channels.values())


def test_rebuild_swaps_in_new_pipeline(fake_whisper) -> None:
    old = vp.build_voice_pipeline(_cfg())
    agent = _FakeAgent(old)
    state = AgentState(agent=agent)
    VOICE_REBUILD.generation = 0

    asyncio.run(_rebuild_voice_pipeline(agent, _cfg(stt_model="small"), 0, state))

    assert agent.voice is not old
    assert agent.voice.stt_model == "small"
    assert {id(ch.voice) for ch in agent.channels.values()} == {id(agent.voice)}


def test_rebuild_failure_keeps_old_pipeline_and_flags_restart(monkeypatch) -> None:
    old = object()
    agent = _FakeAgent(old)
    state = AgentState(agent=agent)

    def _boom(_cfg, _previous=None):
        raise RuntimeError("whisper model missing")

    monkeypatch.setattr("voice.pipeline.build_voice_pipeline", _boom)
    PENDING_RESTART.error = ""

    asyncio.run(_rebuild_voice_pipeline(agent, _cfg(), 0, state))

    assert agent.voice is old  # old pipeline still serving
    assert all(ch.voice is old for ch in agent.channels.values())
    assert VOICE_REBUILD.status == "failed"
    assert "whisper model missing" in VOICE_REBUILD.error
    assert "restart the agent" in PENDING_RESTART.error
    assert restart_banner()["restart_required"] is True
    clear_restart_required()
    assert restart_banner()["restart_required"] is False


def test_superseded_rebuild_is_discarded(fake_whisper) -> None:
    old = vp.build_voice_pipeline(_cfg())
    agent = _FakeAgent(old)
    state = AgentState(agent=agent)
    VOICE_REBUILD.generation = 5  # a newer save arrived meanwhile

    asyncio.run(_rebuild_voice_pipeline(agent, _cfg(stt_model="small"), 4, state))

    assert agent.voice is old  # stale rebuild did not touch the live agent


def test_schedule_supersedes_a_running_rebuild(monkeypatch) -> None:
    agent = _FakeAgent(None)
    state = AgentState(agent=agent)
    calls: list[tuple] = []

    async def _fake_rebuild(agent_, cfg, generation, state_):
        calls.append((generation, cfg))

    monkeypatch.setattr("api.admin._rebuild_voice_pipeline", _fake_rebuild)
    try:
        asyncio.run(_run_two_schedules(agent, state))
    finally:
        VOICE_REBUILD.generation = 0
    assert [c[0] for c in calls] == [2]  # only the newest rebuild ran


async def _run_two_schedules(agent, state) -> None:
    _schedule_voice_rebuild(agent, _cfg(tts_voice="af_bella"), state)
    _schedule_voice_rebuild(agent, _cfg(tts_voice="bf_emma"), state)
    await VOICE_REBUILD.task


# ── PATCH /config wiring ─────────────────────────────────────────────────


class _Store:
    """Config store stub with a real Config export (enough for patch_config)."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {"admin.password_hash": "hash", "admin.password_salt": "salt"}

    async def is_setup_complete(self) -> bool:
        return True

    async def get(self, key: str):
        return self._data.get(key)

    async def set(self, key: str, value: str) -> None:
        self._data[key] = value

    async def set_many(self, values: dict) -> None:
        self._data.update(values)

    async def verify_admin_password(self, password: str) -> bool:
        return password == "secret"

    async def get_all_redacted(self) -> dict:
        return {}

    async def export_to_config(self, vault_resolve=None):
        """Real Config, with the stored voice keys applied (that is what the
        rebuild is supposed to see)."""
        cfg = Config()
        voice = cfg.voice.model_copy(
            update={
                "stt_api_base_url": self._data.get("voice.stt_api_base_url", ""),
                "tts_voice": self._data.get("voice.tts_voice", cfg.voice.tts_voice),
                "tts_enabled": self._data.get("voice.tts_enabled", "true") == "true",
            }
        )
        return cfg.model_copy(update={"voice": voice})


class _PatchAgent:
    """The slice of AgentCore that patch_config touches."""

    def __init__(self) -> None:
        self.voice = object()
        self.channels = {"telegram": _Channel(self.voice)}
        self.config = None
        self.llm = None
        self.executor = SimpleNamespace(tool_env={})
        self.history_mode = ""
        self.memory = SimpleNamespace()
        self.search_enabled = False
        self.search_client = None

    def _build_default_accounts(self, _cfg):
        return {}

    def _build_embedder(self):
        return None


def _patch_client(agent) -> TestClient:
    app, _ = create_admin_app(AgentState(agent=agent), _Store())
    return TestClient(app)


def test_voice_save_schedules_a_rebuild_instead_of_a_restart(monkeypatch) -> None:
    scheduled: list[object] = []
    monkeypatch.setattr("api.admin._schedule_voice_rebuild", lambda a, c, s: scheduled.append(c))
    client = _patch_client(_PatchAgent())

    res = client.patch(
        "/config",
        json={"values": {"voice.stt_api_base_url": "http://whisper:8000/v1"}},
        headers=AUTH,
    ).json()

    assert res["restart_required"] is False  # no restart needed any more
    assert res["voice_rebuild"] == "scheduled"
    assert len(scheduled) == 1
    assert scheduled[0].stt_api_base_url == "http://whisper:8000/v1"


def test_non_voice_save_does_not_rebuild_voice(monkeypatch) -> None:
    scheduled: list[object] = []
    monkeypatch.setattr("api.admin._schedule_voice_rebuild", lambda a, c, s: scheduled.append(c))
    client = _patch_client(_PatchAgent())

    res = client.patch(
        "/config", json={"values": {"memory.long_term_limit": "99"}}, headers=AUTH
    ).json()

    assert res["voice_rebuild"] == "none"
    assert scheduled == []


def test_voice_save_without_a_running_agent_is_just_stored(monkeypatch) -> None:
    scheduled: list[object] = []
    monkeypatch.setattr("api.admin._schedule_voice_rebuild", lambda a, c, s: scheduled.append(c))
    client = _patch_client(None)

    res = client.patch(
        "/config", json={"values": {"voice.tts_voice": "af_bella"}}, headers=AUTH
    ).json()

    assert res["restart_required"] is False
    assert res["voice_rebuild"] == "none"
    assert scheduled == []


def test_rebuild_status_endpoint_reports_outcome() -> None:
    client = _patch_client(None)
    VOICE_REBUILD.status, VOICE_REBUILD.error = "failed", "no model"
    try:
        body = client.get("/voice/rebuild-status", headers=AUTH).json()
    finally:
        VOICE_REBUILD.status, VOICE_REBUILD.error = "idle", ""
    assert body == {"status": "failed", "error": "no model", "restart_required": False}
