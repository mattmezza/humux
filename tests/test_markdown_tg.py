"""Tests for channels/markdown_tg.py."""

from channels.markdown_tg import to_telegram_html


def test_bold():
    assert to_telegram_html("**hi**") == "<b>hi</b>"
    assert to_telegram_html("__hi__") == "<b>hi</b>"


def test_italic():
    assert to_telegram_html("*hi*") == "<i>hi</i>"
    assert to_telegram_html("_hi_") == "<i>hi</i>"


def test_bold_then_italic_mix():
    assert to_telegram_html("**a** and *b*") == "<b>a</b> and <i>b</i>"


def test_strike():
    assert to_telegram_html("~~gone~~") == "<s>gone</s>"


def test_inline_code_protected_from_markup():
    # markup-looking chars inside code must not be converted
    assert to_telegram_html("`**x**`") == "<code>**x**</code>"


def test_html_special_chars_escaped():
    assert to_telegram_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"


def test_code_escapes_html():
    assert to_telegram_html("`<tag>`") == "<code>&lt;tag&gt;</code>"


def test_fenced_block():
    out = to_telegram_html("```python\nprint(1)\n```")
    assert out == "<pre>print(1)</pre>"


def test_fenced_block_escapes():
    out = to_telegram_html("```\na < b\n```")
    assert out == "<pre>a &lt; b</pre>"


def test_link():
    assert to_telegram_html("[site](https://x.com)") == '<a href="https://x.com">site</a>'


def test_header_becomes_bold():
    assert to_telegram_html("# Title") == "<b>Title</b>"
    assert to_telegram_html("### Sub") == "<b>Sub</b>"


def test_bullets():
    assert to_telegram_html("- one\n- two") == "\u2022 one\n\u2022 two"


def test_underscore_in_word_not_italic():
    assert to_telegram_html("foo_bar_baz") == "foo_bar_baz"


def test_plain_text_unchanged():
    assert to_telegram_html("just text") == "just text"


def test_combined():
    src = "# Report\n\n**Total:** 5 items\n\n- `a`\n- *b*\n\nSee [docs](https://d.io)"
    out = to_telegram_html(src)
    assert "<b>Report</b>" in out
    assert "<b>Total:</b>" in out
    assert "<code>a</code>" in out
    assert "<i>b</i>" in out
    assert '<a href="https://d.io">docs</a>' in out
    assert "\u2022 " in out


def test_table_converted_to_bullets():
    """ASCII/Markdown tables should become bullet lists on Telegram."""
    src = "| Item | Count |\n|---|---|\n| Apples | 5 |\n| Oranges | 3 |"
    out = to_telegram_html(src)
    assert "\u2022" in out
    assert "Apples" in out
    assert "Oranges" in out
    assert "|" not in out  # pipes should be gone


def test_table_with_header():
    """Table header row should be preserved as a bullet."""
    src = "| Step | Meaning |\n|------|--------|\n| run | Execute |"
    out = to_telegram_html(src)
    assert "Step" in out
    assert "Meaning" in out
    assert "run" in out


def test_pipe_in_inline_code_not_mistaken_for_table():
    """A pipe inside inline code should not trigger table conversion."""
    src = f"Use `|` as a pipe."
    out = to_telegram_html(src)
    assert "|" in out


def test_single_pipe_not_a_table():
    """A lone pipe without matched delimiters should stay as text."""
    assert to_telegram_html("a | b") == "a | b"
