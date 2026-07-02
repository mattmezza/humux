"""Offline checks for the `explore` verb's pure logic (no browser, no LLM)."""

import pytest

from tools.browser import _apply_action, _format_state


def test_format_state_lists_indexed_elements():
    out = _format_state(
        "https://x.test/",
        "Title",
        "body text",
        [{"idx": 0, "tag": "button", "type": "", "label": "Book"}],
    )
    assert "URL: https://x.test/" in out
    assert "[0] button 'Book'" in out


def test_format_state_handles_empty():
    assert "(none found)" in _format_state("u", "t", "x", [])


class _FakePage:
    def __init__(self):
        self.calls = []

    def click(self, sel, timeout=None, force=False):
        self.calls.append(("click", sel))

    def fill(self, sel, text, timeout=None):
        self.calls.append(("fill", sel, text))


def test_apply_action_dispatches_into_owning_frame():
    frame = _FakePage()
    # frame_map routes index 3 to its owning frame; selector is frame-local.
    note = _apply_action({3: frame}, {"action": "click", "index": 3}, 1000)
    assert frame.calls == [("click", '[data-bu-idx="3"]')]
    assert note == "clicked [3]"


def test_apply_action_rejects_unknown_verb():
    with pytest.raises(ValueError):
        _apply_action({0: _FakePage()}, {"action": "teleport", "index": 0}, 1000)


def test_format_state_marks_frames():
    out = _format_state(
        "u",
        "t",
        "x",
        [{"idx": 5, "tag": "input", "type": "text", "label": "Card", "frame": "js.stripe.com"}],
    )
    assert "in frame: js.stripe.com" in out
    assert "[5] input(text) 'Card'" in out
