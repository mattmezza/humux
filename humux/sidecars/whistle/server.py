"""Minimal OpenAI-compatible adapter for the CPU-only Whistle runtime."""

import logging
import subprocess
import tempfile
import threading
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_AUDIO_SECONDS = 600
LANGUAGES = {"en", "de", "fr", "es", "it", "nl", "pl"}
log = logging.getLogger(__name__)
# Whistle's native engine has process-global state and is not thread-safe.
_inference_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from needle import Whistle

    app.state.model = Whistle()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": "whistle"}


@app.post("/v1/audio/transcriptions")
def transcribe(
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Form()] = "whistle",
    language: Annotated[str | None, Form()] = None,
    response_format: Annotated[str, Form()] = "json",
):
    # whisper-1 is the model alias sent by Humux's existing remote STT client.
    if model not in {"whistle", "whisper-1"}:
        raise HTTPException(400, "Model must be whistle or whisper-1")
    if language is not None and language not in LANGUAGES:
        raise HTTPException(400, "Unsupported language; use en, de, fr, es, it, nl or pl")
    if response_format != "json":
        raise HTTPException(400, "Only JSON response format is supported")
    with tempfile.TemporaryDirectory(prefix="whistle-") as directory:
        source = Path(directory) / "input"
        target = Path(directory) / "audio.wav"
        # Ignore untrusted filenames; cap the copied input before decoding.
        with source.open("wb") as output:
            size = 0
            while chunk := file.file.read(64 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Audio upload exceeds 25 MiB")
                output.write(chunk)
        if size == 0:
            raise HTTPException(400, "Audio upload is empty")
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-t",
                    str(MAX_AUDIO_SECONDS + 1),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(target),
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(422, "Audio decoding timed out") from exc
        except subprocess.CalledProcessError as exc:
            raise HTTPException(422, "Invalid or unsupported audio") from exc
        with wave.open(str(target)) as audio:
            frames = audio.getnframes()
            if frames > MAX_AUDIO_SECONDS * 16000:
                raise HTTPException(413, "Audio exceeds 10 minutes")
            if frames == 0:
                raise HTTPException(422, "Audio contains no samples")
        try:
            with _inference_lock:
                result = app.state.model.transcribe(str(target), language=language)
        except Exception as exc:
            log.exception("Whistle transcription failed")
            raise HTTPException(500, "Transcription failed") from exc
        return {"text": result["text"]}
