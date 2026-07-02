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


def _is_kokoro_voice(voice: str | None) -> bool:
    """True for Kokoro-style names (``af_bella``, ``jm_kumo``) vs edge-tts
    names (``en-US-GuyNeural``).  Lets the edge fallback keep an agent's
    edge voice instead of dropping every voice on Kokoro failure."""
    return bool(re.match(r"[a-z][fm]_", voice or ""))


# Reply language (ISO-639-1) the agent tags onto a voice request →
# (Kokoro phoneme lang | None, default edge-tts voice).  Issue #95: an Italian
# reply must not be spoken with English phonemes.  Kokoro reads any language
# with any voice (accent is fine), so we only swap the phoneme lang; edge-tts
# voices are locale-locked, so we must swap to a voice in that language.  A
# ``None`` Kokoro lang means Kokoro can't speak it (e.g. German) → use edge.
_TTS_LANG: dict[str, tuple[str | None, str]] = {
    "en": ("en-us", "en-US-AvaNeural"),
    "it": ("it", "it-IT-ElsaNeural"),
    "fr": ("fr-fr", "fr-FR-DeniseNeural"),
    "es": ("es", "es-ES-ElviraNeural"),
    "pt": ("pt-br", "pt-BR-FranciscaNeural"),
    "hi": ("hi", "hi-IN-SwaraNeural"),
    "ja": ("ja", "ja-JP-NanamiNeural"),
    "zh": ("cmn", "zh-CN-XiaoxiaoNeural"),
    "de": (None, "de-DE-KatjaNeural"),
}


# Kokoro v1.0 voice names, grouped by language, for the admin voice picker.
# Free-text fields still accept any name; this is the suggestion/selection list.
KOKORO_VOICES: tuple[str, ...] = (
    # English (US)
    "af_bella",
    "af_heart",
    "af_nicole",
    "af_nova",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    # English (UK)
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    # French
    "ff_siwis",
    # Italian
    "if_sara",
    "im_nicola",
    # Japanese
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jm_kumo",
    # Mandarin Chinese
    "zf_xiaobei",
    "zf_xiaoxiao",
    "zm_yunjian",
    "zm_yunxi",
    # Spanish
    "ef_dora",
    "em_alex",
    # Hindi
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
    # Brazilian Portuguese
    "pf_dora",
    "pm_alex",
)

# Kokoro phonemization languages (code, label) for the admin preview dropdown.
# The language is independent of the voice, so any voice can read any language.
KOKORO_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("en-us", "English (US)"),
    ("en-gb", "English (UK)"),
    ("it", "Italian"),
    ("fr-fr", "French"),
    ("es", "Spanish"),
    ("pt-br", "Portuguese (BR)"),
    ("hi", "Hindi"),
    ("ja", "Japanese"),
    ("cmn", "Mandarin"),
)


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
        timeout=30,  # bound the blocking call so a stuck ffmpeg can't starve the pool
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

    async def synthesize(
        self, text: str, voice: str | None = None, lang: str | None = None
    ) -> bytes:
        """Convert text to speech.  Returns raw audio bytes (MP3 for edge-tts,
        OGG/Opus for Kokoro) — both formats Telegram ``send_voice`` accepts.

        ``voice`` overrides the configured default (e.g. an active agent's
        own voice); empty/None falls back to the backend default.  ``lang`` is
        the ISO-639-1 language the agent wrote the reply in (issue #95): Kokoro
        keeps the voice and reads with that language's phonemes (accent ok),
        while edge-tts swaps to a locale-matched voice since its voices are
        locale-locked.  Kokoro runs on-device; if it fails for any reason we
        fall back to edge-tts so a reply is never lost (issue #84).
        """
        if not self.tts_enabled:
            raise RuntimeError("TTS is disabled in config")

        text = clean_for_speech(text)
        known = _TTS_LANG.get(lang) if lang else None
        kokoro_lang = known[0] if known else None
        # Skip Kokoro only when the reply language is one it KNOWS it can't
        # phonemize (e.g. German) — then go straight to a matching edge-tts voice.
        # An unknown/untagged language just lets Kokoro derive lang from the voice.
        skip_kokoro = known is not None and known[0] is None
        if self._kokoro is not None and not skip_kokoro:
            try:
                return await self._synthesize_kokoro(text, voice, kokoro_lang)
            except Exception:
                log.exception("Kokoro synthesis failed, falling back to edge-tts")
        # edge-tts path — primary backend, or the Kokoro fallback. edge-tts can't
        # speak a Kokoro voice name (e.g. an agent configured with af_bella while
        # the backend is edge-tts), so drop it to the configured default; keep a
        # real edge voice so its preference survives.
        return await self._synthesize_edge(text, self._edge_voice(voice, lang))

    def _edge_voice(self, voice: str | None, lang: str | None) -> str | None:
        """Resolve the edge-tts voice.  A Kokoro voice name can't drive edge-tts,
        so drop it to the configured default.  edge voices are locale-locked, so
        if the reply language (``lang``) doesn't match the chosen voice's locale,
        swap to a default voice for that language so the audio is intelligible
        (issue #95)."""
        base = None if _is_kokoro_voice(voice) else voice
        if lang:
            want = _TTS_LANG.get(lang, (None, None))[1]
            current = base or self.tts_voice
            if want and not current.lower().startswith(f"{lang}-"):
                return want
        return base

    async def _synthesize_edge(self, text: str, voice: str | None) -> bytes:
        communicate = edge_tts.Communicate(text, voice or self.tts_voice)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])

        audio = buf.getvalue()
        log.info("Synthesized %d chars → %d bytes audio (edge-tts)", len(text), len(audio))
        return audio

    async def _synthesize_kokoro(
        self, text: str, voice: str | None, lang: str | None = None
    ) -> bytes:
        """Kokoro is synchronous and CPU-bound — run it off the event loop.

        ``lang`` overrides the phonemization language (independent of the voice,
        so an Italian voice can read English); ``None`` derives it from the voice.
        """
        import asyncio as _asyncio

        name = voice or self.kokoro_default_voice
        loop = _asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self._kokoro_sync, text, name, lang))

    def _kokoro_sync(self, text: str, voice: str, lang: str | None = None) -> bytes:
        lang = lang or _lang_for_voice(voice)
        samples, sample_rate = self._kokoro.create(text, voice=voice, speed=1.0, lang=lang)
        audio = _wav_to_ogg(_pcm_to_wav(samples, sample_rate))
        log.info(
            "Synthesized %d chars → %d bytes audio (kokoro/%s/%s)",
            len(text),
            len(audio),
            voice,
            lang,
        )
        return audio

    async def preview(self, text: str, voice: str, lang: str = "") -> tuple[bytes, str]:
        """Synthesize a short sample with a SPECIFIC voice for the admin voice
        picker.  The engine is chosen from the voice name (Kokoro-style →
        Kokoro, else edge-tts), independent of the configured backend.  ``lang``
        (Kokoro only) overrides the pronunciation language.  Returns
        ``(audio_bytes, mime_type)``.
        """
        text = clean_for_speech(text) or text
        if _is_kokoro_voice(voice):
            if self._kokoro is None:
                raise RuntimeError(
                    "Kokoro model isn't loaded — switch the TTS backend to "
                    "'kokoro' and restart to preview Kokoro voices."
                )
            return await self._synthesize_kokoro(text, voice, lang or None), "audio/ogg"
        return await self._synthesize_edge(text, voice), "audio/mpeg"
