"""Kokoro TTS backend: lang mapping, WAV encoding, OGG transcode, fallback (#84).

Builds a bare VoicePipeline via ``object.__new__`` so we never load Whisper or
the ONNX model — only the new branch logic is under test.  ffmpeg is stubbed so
the dispatch tests don't shell out; one guarded test exercises real ffmpeg.
"""

import asyncio
import io
import shutil
import wave

import numpy as np
import pytest

from voice.pipeline import (
    VoicePipeline,
    _is_kokoro_voice,
    _lang_for_voice,
    _pcm_to_wav,
    _wav_to_ogg,
)


def test_lang_for_voice_prefix():
    assert _lang_for_voice("af_bella") == "en-us"
    assert _lang_for_voice("bm_george") == "en-gb"
    assert _lang_for_voice("jf_alpha") == "ja"
    assert _lang_for_voice("ff_siwis") == "fr-fr"
    assert _lang_for_voice("if_sara") == "it"
    assert _lang_for_voice("xx_unknown") == "en-us"  # default


def test_is_kokoro_voice():
    assert _is_kokoro_voice("af_bella")
    assert _is_kokoro_voice("jm_kuma")
    assert not _is_kokoro_voice("en-US-GuyNeural")  # edge-tts name
    assert not _is_kokoro_voice("")
    assert not _is_kokoro_voice(None)


def test_pcm_to_wav_roundtrip():
    samples = np.array([0.0, 1.0, -1.0, 2.0, -2.0, 0.5], dtype=np.float32)  # ±2.0 clip
    data = _pcm_to_wav(samples, 24000)
    assert data[:4] == b"RIFF"
    with wave.open(io.BytesIO(data), "rb") as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    assert frames[1] == 32767 and frames[2] == -32767  # clipped to int16 range
    assert frames[3] == 32767 and frames[4] == -32767
    assert frames[5] == 16384  # 0.5 rounded; truncation would give 16383


def _bare_pipeline():
    p = object.__new__(VoicePipeline)
    p.tts_enabled = True
    p.tts_voice = "en-US-AvaNeural"
    p.kokoro_default_voice = "af_bella"
    p._kokoro = None
    return p


class _FakeKokoro:
    def __init__(self, raise_it=False):
        self.raise_it = raise_it
        self.calls = []

    def create(self, text, voice, speed, lang):
        self.calls.append((text, voice, lang))
        if self.raise_it:
            raise RuntimeError("onnx boom")
        return np.zeros(10, dtype=np.float32), 24000


def test_synthesize_uses_kokoro_when_loaded(monkeypatch):
    monkeypatch.setattr("voice.pipeline._wav_to_ogg", lambda wav: b"OGG:" + wav[:4])
    p = _bare_pipeline()
    p._kokoro = _FakeKokoro()
    out = asyncio.run(p.synthesize("hello", voice="jf_alpha"))
    assert out == b"OGG:RIFF"  # kokoro WAV transcoded via _wav_to_ogg
    assert p._kokoro.calls == [("hello", "jf_alpha", "ja")]


def test_synthesize_falls_back_to_edge_on_kokoro_error(monkeypatch):
    p = _bare_pipeline()
    p._kokoro = _FakeKokoro(raise_it=True)
    seen = {}

    async def fake_edge(text, voice):
        seen["voice"] = voice
        return b"EDGE-AUDIO"

    monkeypatch.setattr(p, "_synthesize_edge", fake_edge)
    out = asyncio.run(p.synthesize("hello", voice="af_bella"))
    assert out == b"EDGE-AUDIO"  # kokoro raised → edge fallback
    assert seen["voice"] is None  # Kokoro voice name dropped so edge-tts accepts it (#84)


def test_synthesize_fallback_keeps_edge_voice(monkeypatch):
    """A persona's valid edge-tts voice must survive a Kokoro failure (#84)."""
    p = _bare_pipeline()
    p._kokoro = _FakeKokoro(raise_it=True)
    seen = {}

    async def fake_edge(text, voice):
        seen["voice"] = voice
        return b"EDGE-AUDIO"

    monkeypatch.setattr(p, "_synthesize_edge", fake_edge)
    asyncio.run(p.synthesize("hello", voice="en-US-GuyNeural"))
    assert seen["voice"] == "en-US-GuyNeural"  # edge voice preserved, not dropped


def test_synthesize_default_voice_when_none(monkeypatch):
    monkeypatch.setattr("voice.pipeline._wav_to_ogg", lambda wav: wav)
    p = _bare_pipeline()
    p._kokoro = _FakeKokoro()
    asyncio.run(p.synthesize("hi", voice=None))
    assert p._kokoro.calls[0][1] == "af_bella"  # fell back to kokoro_default_voice


def test_wav_to_ogg_real_ffmpeg():
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    ogg = _wav_to_ogg(_pcm_to_wav(np.zeros(2400, dtype=np.float32), 24000))
    assert ogg[:4] == b"OggS"  # valid Ogg stream
