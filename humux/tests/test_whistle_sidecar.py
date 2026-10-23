"""Exercise the sidecar's HTTP contract and real ffmpeg decoding, without model downloads."""

import importlib.util
import io
import subprocess
import sys
import threading
import time
import types
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sidecar(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "whistle_server", ROOT / "sidecars/whistle/server.py"
    )
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)

    def transcribe(path, language=None):
        with wave.open(path) as audio:
            assert audio.getframerate() == 16000
            assert audio.getnchannels() == 1
            assert audio.getsampwidth() == 2
        return {"text": "Zero, 1, 2."}

    model = Mock()
    model.transcribe.side_effect = transcribe
    monkeypatch.setitem(sys.modules, "needle", types.SimpleNamespace(Whistle=lambda: model))
    with TestClient(server.app) as client:
        yield server, client, model


def upload(client, content=None, **fields):
    if content is None:
        content = (ROOT / "voice/samples/test-stt.ogg").read_bytes()
    return client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", **fields},
        files={"file": ("../../audio.ogg", content, "audio/ogg")},
    )


def test_health_and_humux_contract(sidecar):
    _, client, model = sidecar
    assert client.get("/health").json() == {"status": "ok", "model": "whistle"}
    response = upload(client)
    assert response.status_code == 200
    assert response.json() == {"text": "Zero, 1, 2."}
    path = model.transcribe.call_args.args[0]
    assert not Path(path).exists()  # Audio is removed after inference.
    assert model.transcribe.call_args.kwargs == {"language": None}


def test_whistle_model_and_language(sidecar):
    _, client, model = sidecar
    assert upload(client, model="whistle", language="it").status_code == 200
    assert model.transcribe.call_args.kwargs == {"language": "it"}


@pytest.mark.parametrize(
    "fields",
    [
        {"model": "unknown"},
        {"language": "ja"},
        {"response_format": "text"},
    ],
)
def test_reject_unsupported_options(sidecar, fields):
    _, client, model = sidecar
    assert upload(client, **fields).status_code == 400
    model.transcribe.assert_not_called()


@pytest.mark.parametrize("content,status", [(b"", 400), (b"broken audio", 422)])
def test_reject_invalid_upload(sidecar, content, status):
    _, client, model = sidecar
    assert upload(client, content=content).status_code == status
    model.transcribe.assert_not_called()


def test_upload_limit(sidecar, monkeypatch):
    server, client, model = sidecar
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 4)
    assert upload(client, content=b"12345").status_code == 413
    model.transcribe.assert_not_called()


def test_duration_limit(sidecar, monkeypatch):
    server, client, model = sidecar
    monkeypatch.setattr(server, "MAX_AUDIO_SECONDS", 1)
    data = io.BytesIO()
    with wave.open(data, "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x00\x00" * 32000)
    assert upload(client, content=data.getvalue()).status_code == 413
    model.transcribe.assert_not_called()


def test_decode_timeout(sidecar, monkeypatch):
    server, client, model = sidecar

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("ffmpeg", 60)

    monkeypatch.setattr(server.subprocess, "run", timeout)
    assert upload(client).status_code == 422
    model.transcribe.assert_not_called()


def test_inference_failure_and_silence(sidecar):
    _, client, model = sidecar
    model.transcribe.side_effect = RuntimeError("native error")
    response = upload(client)
    assert response.status_code == 500
    assert response.json() == {"detail": "Transcription failed"}
    assert not Path(model.transcribe.call_args.args[0]).exists()
    model.transcribe.side_effect = None
    model.transcribe.return_value = {"text": ""}
    assert upload(client).json() == {"text": ""}


def test_concurrent_requests_serialize_native_inference(sidecar):
    _, client, model = sidecar
    active = 0
    peak = 0
    counter_lock = threading.Lock()

    def transcribe(*args, **kwargs):
        nonlocal active, peak
        with counter_lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with counter_lock:
            active -= 1
        return {"text": "hello"}

    model.transcribe.side_effect = transcribe
    with ThreadPoolExecutor(max_workers=4) as executor:
        responses = list(executor.map(lambda _: upload(client), range(4)))
    assert all(response.status_code == 200 for response in responses)
    assert model.transcribe.call_count == 4
    assert peak == 1
