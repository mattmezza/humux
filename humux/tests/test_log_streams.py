"""Per-agent log streams (issue #75): context tagging, buffering, filtering."""

from __future__ import annotations

import logging

from api.admin import _BufferHandler, _filter_log_entries, _stream_hue
from core.log_streams import current_stream, set_stream, subagent_stream


def _emit(logger_name: str, level: int = logging.INFO, msg: str = "hi") -> dict:
    """Emit one record through a fresh _BufferHandler and return the entry."""
    handler = _BufferHandler()
    rec = logging.LogRecord(logger_name, level, __file__, 1, msg, None, None)
    captured: list[dict] = []
    # The handler appends to the module deque; intercept by swapping it.
    import api.admin as admin

    saved = admin._LOG_BUFFER
    admin._LOG_BUFFER = captured  # type: ignore[assignment]
    try:
        handler.emit(rec)
    finally:
        admin._LOG_BUFFER = saved
    return captured[0] if captured else {}


def test_record_tagged_with_current_stream() -> None:
    set_stream("coding-helper")
    entry = _emit("core.agent")
    assert entry["stream"] == "coding-helper"
    assert entry["message"] == "hi"
    set_stream("default")  # restore


def test_subagent_lines_prefixed_and_keep_parent_stream() -> None:
    set_stream("coding-helper")
    with subagent_stream("sub_abc", fallback="ignored"):
        # parent stream kept; line marked as the subagent's
        assert current_stream() == "coding-helper"
        entry = _emit("core.agent", msg="thinking")
    assert entry["stream"] == "coding-helper"
    assert entry["message"] == "[subagent:sub_abc] thinking"
    set_stream("default")


def test_non_core_loggers_not_captured() -> None:
    assert _emit("httpx") == {}  # outside the include list → dropped


def _entries() -> list[dict]:
    def e(ts, levelno, level, stream, name, message):
        return {
            "ts": ts,
            "levelno": levelno,
            "level": level,
            "stream": stream,
            "name": name,
            "message": message,
        }

    return [
        e(100.0, 20, "INFO", "default", "core.a", "alpha"),
        e(200.0, 30, "WARNING", "coding-helper", "core.b", "beta"),
        e(300.0, 40, "ERROR", "finance-assistant", "core.c", "[subagent:sub_x] gamma"),
    ]


def test_filter_by_stream_regex() -> None:
    out = _filter_log_entries(_entries(), stream="coding|finance")
    assert {e["stream"] for e in out} == {"coding-helper", "finance-assistant"}


def test_filter_by_min_level() -> None:
    out = _filter_log_entries(_entries(), level="WARNING")
    assert [e["message"] for e in out] == ["beta", "[subagent:sub_x] gamma"]


def test_filter_by_text() -> None:
    assert [e["message"] for e in _filter_log_entries(_entries(), q="alpha")] == ["alpha"]
    # subagent prefix is searchable text
    assert len(_filter_log_entries(_entries(), q="subagent")) == 1


def test_filter_by_time_range() -> None:
    # 150..250 epoch → only the 200.0 entry. datetime-local strings map to epoch.
    import datetime as dt

    def local(ts: float) -> str:
        return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")

    out = _filter_log_entries(_entries(), since=local(150), until=local(250))
    assert [e["message"] for e in out] == ["beta"]


def test_bad_regex_is_noop_not_crash() -> None:
    # an unfinished regex must not blank the viewer
    assert len(_filter_log_entries(_entries(), stream="[")) == 3


def test_stream_hue_is_stable() -> None:
    assert _stream_hue("default") == _stream_hue("default")
    assert 0 <= _stream_hue("coding-helper") < 360
