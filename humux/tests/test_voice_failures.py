"""Detached Telegram handlers report failures and preserve cancellation (#325)."""

import asyncio
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import ValidationError

from core.config import VoiceConfig
from tests.test_group_chat import (
    GROUP,
    _drive_on_voice,
    _ent,
    _msg,
    _user,
    _voice_channel,
    _voice_message,
)
from voice.pipeline import VoicePipeline


@asynccontextmanager
async def _quiet(*args):
    yield


def _reply_channel():
    ch, voice = _voice_channel()
    ch._typing = _quiet
    ch._progress = _quiet
    ch._send_response = AsyncMock()
    ch.agent.process = AsyncMock(return_value=SimpleNamespace(text="reply"))
    return ch, voice


@pytest.mark.parametrize("stage", ["get_file", "download", "remote_stt", "local_stt"])
async def test_detached_voice_failure_reports_error_and_logs_context(stage, caplog):
    ch, voice = _reply_channel()
    error = RuntimeError("backend failure details")
    if stage == "get_file":
        ch.app.bot.get_file.side_effect = error
    elif stage == "download":
        ch.app.bot.get_file.return_value.download_as_bytearray.side_effect = error
    else:
        voice.stt_api_base_url = "http://whisper:8000/v1" if stage == "remote_stt" else ""
        voice.transcribe = AsyncMock(side_effect=error)

    with caplog.at_level(logging.ERROR, logger="channels.telegram"):
        await _drive_on_voice(ch, _user(), _voice_message(addressed=True))

    ch.send.assert_awaited_once_with("-100", "(could not transcribe voice)")
    ch.agent.process.assert_not_awaited()
    ch._send_response.assert_not_awaited()
    assert "chat=-100" in caplog.text
    assert "file=voice-file-1" in caplog.text
    backend = "remote" if stage == "remote_stt" else "local"
    assert f"backend={backend}" in caplog.text
    assert "backend failure details" in caplog.text
    assert "backend failure details" not in ch.send.await_args.args[1]


async def test_detached_empty_transcript_reports_error_without_inference():
    ch, voice = _reply_channel()
    voice.transcript = " \n "
    await _drive_on_voice(ch, _user(), _voice_message(addressed=True))
    ch.send.assert_awaited_once_with("-100", "(could not transcribe voice)")
    ch.agent.process.assert_not_awaited()


async def test_detached_successful_voice_still_processes_and_delivers_reply():
    ch, voice = _reply_channel()
    voice.transcript = "please summarize my inbox"
    await _drive_on_voice(ch, _user(), _voice_message(addressed=True))
    assert ch.agent.process.await_args.kwargs["message"] == (
        "[from Alice]\n[voice] please summarize my inbox"
    )
    assert ch.agent.process.await_args.kwargs["message_id"] == 77
    ch._send_response.assert_awaited_once_with("-100", ch.agent.process.return_value)
    ch.send.assert_not_awaited()


async def test_cancelled_voice_task_preserves_cancellation_without_error_notification():
    ch, voice = _reply_channel()
    started = asyncio.Event()

    async def blocked_transcription(audio):
        started.set()
        await asyncio.Event().wait()

    voice.transcribe = blocked_transcription
    task = asyncio.create_task(
        ch._run_message_handler(ch._handle_voice("file", "1", "-100", ""), "-100", "voice")
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    ch.send.assert_not_awaited()
    ch.agent.process.assert_not_awaited()


@pytest.mark.parametrize("kind", ["text", "photo", "voice"])
async def test_handler_failure_reports_generic_message(kind, caplog):
    ch, _ = _reply_channel()
    handler = AsyncMock(side_effect=RuntimeError("private details"))
    with caplog.at_level(logging.ERROR, logger="channels.telegram"):
        await ch._run_message_handler(handler(), "-100:23", kind)
    ch.send.assert_awaited_once_with(
        "-100:23", f"Sorry, I couldn't process that {kind} message. Please try again."
    )
    assert "chat=-100:23" in caplog.text
    assert "private details" in caplog.text
    assert "private details" not in ch.send.await_args.args[1]


async def test_record_only_handler_failure_stays_silent(caplog):
    ch, _ = _reply_channel()
    handler = AsyncMock(side_effect=RuntimeError("record failed"))
    with caplog.at_level(logging.ERROR, logger="channels.telegram"):
        await ch._run_message_handler(handler(), "-100", "text", notify=False)
    ch.send.assert_not_awaited()
    assert "record failed" in caplog.text


async def test_error_notification_failure_is_logged_without_escaping(caplog):
    ch, _ = _reply_channel()
    ch.send.side_effect = RuntimeError("Telegram unavailable")
    handler = AsyncMock(side_effect=RuntimeError("original failure"))
    with caplog.at_level(logging.ERROR, logger="channels.telegram"):
        await ch._run_message_handler(handler(), "-100", "photo")
    assert "original failure" in caplog.text
    assert "Could not send failure notification" in caplog.text


@pytest.mark.parametrize("configured", [None, 600])
async def test_remote_stt_uses_extended_configurable_timeout(monkeypatch, configured):
    monkeypatch.setattr("voice.pipeline.WhisperModel", None)
    kwargs = {} if configured is None else {"stt_timeout_seconds": configured}
    pipeline = VoicePipeline(stt_api_base_url="http://whisper:8000/v1", **kwargs)
    seen = []

    def receive(request):
        seen.append(request)
        return httpx.Response(200, json={"text": "transcribed long recording"})

    client_type = httpx.AsyncClient
    transport = httpx.MockTransport(receive)
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **options: client_type(transport=transport, **options)
    )
    assert await pipeline.transcribe(b"audio bytes") == "transcribed long recording"
    assert seen[0].extensions["timeout"] == {
        "connect": 10,
        "pool": 10,
        "read": configured or 300,
        "write": configured or 300,
    }
    assert seen[0].url.path == "/v1/audio/transcriptions"
    assert b"audio bytes" in seen[0].content


@pytest.mark.parametrize("value", [0, -1])
def test_stt_timeout_must_be_positive(value):
    with pytest.raises(ValidationError):
        VoiceConfig(stt_timeout_seconds=value)


@pytest.mark.parametrize("kind,addressed", [("text", True), ("text", False), ("photo", True)])
async def test_sibling_dispatch_guards_detached_failure(kind, addressed):
    ch, _ = _reply_channel()
    mention = "@coachbot" if addressed else ""
    message = _msg(mention, entities=[_ent("mention", mention)] if addressed else [])
    message.message_id = 55
    if kind == "photo":
        message.photo = [SimpleNamespace(file_id="photo-file")]
        message.document = None
    setattr(ch, f"_handle_{kind}", AsyncMock(side_effect=RuntimeError("handler failed")))
    update = SimpleNamespace(effective_user=_user(), effective_chat=GROUP, message=message)
    context = SimpleNamespace(bot=SimpleNamespace(id=999, username="coachbot"))
    before = set(asyncio.all_tasks())
    await getattr(ch, f"_on_{kind}")(update, context)
    await asyncio.gather(*(task for task in asyncio.all_tasks() if task not in before))
    if addressed:
        ch.send.assert_awaited_once_with(
            -100, f"Sorry, I couldn't process that {kind} message. Please try again."
        )
    else:
        ch.send.assert_not_awaited()
