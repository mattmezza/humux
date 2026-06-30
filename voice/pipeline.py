"""Voice pipeline — Whisper STT + edge-tts / Kokoro TTS."""

from __future__ import annotations

import io
import logging
import re
import subprocess
import unicodedata
import wave
from functools import partial
from typing import TYPE_CHECKING

import edge_tts
from faster_whisper import WhisperModel

if TYPE_CHECKING:
    import asyncio

log = logging.getLogger(__name__)

# Kokoro voice names encode language in the first letter (af_bella → "a" → en-us).
# kokoro-onnx needs the lang for phonemization; map the prefix, default en-us.
_KOKORO_LANG = {
    "a": "en-us",  # American English
    "b": "en-gb",  # British English
    "j": "ja",  # Japanese
    "z": "cmn",  # Mandarin Chinese
    "e": "es",  # Spanish
    "f": "fr-fr",  # French
    "h": "hi",  # Hindi
    "i": "it",  # Italian
    "p": "pt-br",  # Brazilian Portuguese
}


def _lang_for_voice(voice: str) -> str:
    return _KOKORO_LANG.get(voice[:1], "en-us")


def _pcm_to_wav(samples, sample_rate: int) -> bytes:
    """Encode float32 PCM samples (-1..1) from Kokoro into 16-bit mono WAV bytes."""
    import numpy as np

    # round (not truncate) before int16 cast — truncation biases samples toward 0
    pcm16 = np.rint(np.clip(np.asarray(samples), -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _wav_to_ogg(wav: bytes) -> bytes:
    """Transcode WAV → OGG/Opus, the format Telegram ``send_voice`` expects.

    ffmpeg already ships in the image (it decodes incoming voice messages), so
    no new dependency.  Raises if ffmpeg is missing or fails — the caller treats
    that like any Kokoro failure and falls back to edge-tts.
    """
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-c:a",
            "libopus",
            "-b:a",
            "32k",
            "-f",
            "ogg",
            "pipe:1",
        ],
        input=wav,
        capture_output=True,
        check=True,
    )
    return proc.stdout


_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)  # fenced code
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_LIST_MARKER_RE = re.compile(r"^[ \t]*[-*•‣◦]+[ \t]+", re.MULTILINE)  # leading bullets
_MD_SYMBOLS_RE = re.compile(r"[*#_~>`|]")  # markdown emphasis/heading/table chars
_WS_RE = re.compile(r"[ \t]{2,}")


def clean_for_speech(text: str) -> str:
    """Strip anything that reads badly when spoken: code, URLs, emojis, markdown.

    Voice replies should be plain speakable text — no emojis, bullets, code
    snippets, URLs, or symbols like * and #.  See issue #10.
    """
    text = _CODE_BLOCK_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _LIST_MARKER_RE.sub("", text)
    # dashes used as separators → pause; keep hyphens inside words
    text = re.sub(r"\s[-–—]+\s", ", ", text)
    text = _MD_SYMBOLS_RE.sub("", text)
    # drop emoji & other pictographic symbols (unicode category "So")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "So")
    text = _WS_RE.sub(" ", text)
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line).strip()


class VoicePipeline:
    """Speech-to-text via faster-whisper; text-to-speech via edge-tts or Kokoro."""

    def __init__(
        self,
        stt_model: str = "base",
        tts_voice: str = "en-US-AvaNeural",
        tts_enabled: bool = True,
        backend: str = "edge-tts",
        kokoro_model_path: str = "models/kokoro/kokoro-v1.0.onnx",
        kokoro_voices_path: str = "models/kokoro/voices-v1.0.bin",
        kokoro_default_voice: str = "af_bella",
    ):
        self.tts_voice = tts_voice
        self.tts_enabled = tts_enabled
        self.backend = backend
        self.kokoro_default_voice = kokoro_default_voice
        self._kokoro = None

        log.info("Loading Whisper model '%s' …", stt_model)
        self._whisper = WhisperModel(stt_model, compute_type="int8")
        log.info("Whisper model loaded.")

        if backend == "kokoro":
            # Load eagerly so a bad path/missing model degrades to edge-tts at
            # startup (logged once) rather than on every reply.  Issue #84.
            try:
                from kokoro_onnx import Kokoro

                log.info("Loading Kokoro TTS (%s) …", kokoro_model_path)
                self._kokoro = Kokoro(kokoro_model_path, kokoro_voices_path)
                log.info("Kokoro TTS loaded.")
            except Exception:
                log.exception("Kokoro TTS unavailable, falling back to edge-tts")
                self.backend = "edge-tts"

    # -- STT ----------------------------------------------------------------

    async def transcribe(
        self, audio_bytes: bytes, *, loop: asyncio.AbstractEventLoop | None = None
    ) -> str:
        """Transcribe audio bytes (OGG/WAV/MP3) to text.

        faster-whisper is synchronous and CPU-bound, so we run it in the
        default executor to avoid blocking the event loop.
        """
        import asyncio as _asyncio

        _loop = loop or _asyncio.get_running_loop()
        return await _loop.run_in_executor(None, partial(self._transcribe_sync, audio_bytes))

    def _transcribe_sync(self, audio_bytes: bytes) -> str:
        segments, info = self._whisper.transcribe(io.BytesIO(audio_bytes))
        text = " ".join(seg.text.strip() for seg in segments)
        log.info(
            "Transcribed %s audio → %d chars (lang=%s)", info.language, len(text), info.language
        )
        return text

    # -- TTS ----------------------------------------------------------------

    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        """Convert text to speech.  Returns raw audio bytes (MP3 for edge-tts,
        OGG/Opus for Kokoro) — both formats Telegram ``send_voice`` accepts.

        ``voice`` overrides the configured default (e.g. an active persona's
        own voice); empty/None falls back to the backend default.  Kokoro runs
        on-device; if it fails for any reason we fall back to edge-tts so a
        reply is never lost (issue #84).
        """
        if not self.tts_enabled:
            raise RuntimeError("TTS is disabled in config")

        text = clean_for_speech(text)
        if self._kokoro is not None:
            try:
                return await self._synthesize_kokoro(text, voice)
            except Exception:
                # ``voice`` here is a Kokoro name edge-tts won't recognise — drop
                # it (voice=None) so the fallback uses the configured edge voice.
                log.exception("Kokoro synthesis failed, falling back to edge-tts")
                return await self._synthesize_edge(text, None)
        return await self._synthesize_edge(text, voice)

    async def _synthesize_edge(self, text: str, voice: str | None) -> bytes:
        communicate = edge_tts.Communicate(text, voice or self.tts_voice)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])

        audio = buf.getvalue()
        log.info("Synthesized %d chars → %d bytes audio (edge-tts)", len(text), len(audio))
        return audio

    async def _synthesize_kokoro(self, text: str, voice: str | None) -> bytes:
        """Kokoro is synchronous and CPU-bound — run it off the event loop."""
        import asyncio as _asyncio

        name = voice or self.kokoro_default_voice
        loop = _asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self._kokoro_sync, text, name))

    def _kokoro_sync(self, text: str, voice: str) -> bytes:
        samples, sample_rate = self._kokoro.create(
            text, voice=voice, speed=1.0, lang=_lang_for_voice(voice)
        )
        audio = _wav_to_ogg(_pcm_to_wav(samples, sample_rate))
        log.info("Synthesized %d chars → %d bytes audio (kokoro/%s)", len(text), len(audio), voice)
        return audio
