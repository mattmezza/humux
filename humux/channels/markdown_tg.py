"""Convert common Markdown to Telegram-flavored HTML.

LLMs emit Markdown (**bold**, *italic*, fenced code, ...). Telegram renders a
small HTML subset, not Markdown. This module does a deterministic conversion so
agent responses display formatted instead of showing raw markup.

Supported Telegram HTML tags: b, i, u, s, code, pre, a, blockquote, tg-spoiler.
Anything not mappable is left as escaped text.
"""

from __future__ import annotations

import re

__all__ = ["to_telegram_html"]

_FENCE_RE = re.compile(r"```(\w+)?\n?(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+?)`")
_BOLD_RE = re.compile(r"(?<!\*)\*\*(?!\s)(.+?)(?<!\s)\*\*(?!\*)", re.DOTALL)
_BOLD_ALT_RE = re.compile(r"(?<!_)__(?!\s)(.+?)(?<!\s)__(?!_)", re.DOTALL)
_ITALIC_RE = re.compile(r"(?<![\*\w])\*(?!\s)([^\*\n]+?)(?<!\s)\*(?![\*\w])")
_ITALIC_ALT_RE = re.compile(r"(?<![_\w])_(?!\s)([^_\n]+?)(?<!\s)_(?![_\w])")
_STRIKE_RE = re.compile(r"~~(?!\s)(.+?)(?<!\s)~~", re.DOTALL)
_LINK_RE = re.compile(r"\[([^\]]+?)\]\((https?://[^\s)]+)\)")
_HEADER_RE = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*#*$", re.MULTILINE)
_BULLET_RE = re.compile(r"^([ \t]*)[\*\-][ \t]+", re.MULTILINE)

# Detect ASCII/Markdown tables: a line containing | and a separator line with -+-
_TABLE_ROW_RE = re.compile(r"^[ \t]*\|[^\n]+\|[ \t]*$", re.MULTILINE)
_TABLE_SEP_RE = re.compile(r"^[ \t]*\|[-:| \t]+\|[ \t]*$", re.MULTILINE)

_PLACEHOLDER = "\x00{}\x00"


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _strip_tables(text: str) -> str:
    """Replace ASCII/Markdown tables with a bullet list of their data rows.

    Tables are unreadable on Telegram. This strips the separator row and
    converts each data row into a bullet point with pipe-separated cells
    rendered as a natural-language list. When no data rows remain the
    table is removed entirely.
    """
    lines = text.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _TABLE_ROW_RE.match(line):
            rows: list[str] = [line]
            j = i + 1
            while j < len(lines) and _TABLE_ROW_RE.match(lines[j]):
                rows.append(lines[j])
                j += 1
            sep_idx = next((k for k, r in enumerate(rows) if _TABLE_SEP_RE.match(r)), None)
            # Include header row (before separator) as first bullet, then data rows
            data_rows = rows[:sep_idx] + rows[sep_idx + 1 :] if sep_idx is not None else rows
            if data_rows:
                for row in data_rows:
                    cells = [c.strip() for c in row.strip(" \t|").split("|")]
                    cell_text = " — ".join(c for c in cells if c)
                    result.append(f"• {cell_text}")
            i = j
            continue
        result.append(line)
        i += 1
    return "\n".join(result)


def to_telegram_html(text: str) -> str:
    """Render Markdown ``text`` as Telegram-safe HTML.

    Code spans/blocks are protected before escaping so their contents are never
    interpreted as markup. Returns a string suitable for ``parse_mode="HTML"``.

    Tables are automatically converted to bullet lists before processing, since
    Telegram does not render them readably.
    """
    # Strip tables first -- they are unreadable on Telegram.
    text = _strip_tables(text)

    stash: list[str] = []

    def _stash(html: str) -> str:
        stash.append(html)
        return _PLACEHOLDER.format(len(stash) - 1)

    def _fence(m: re.Match[str]) -> str:
        body = _esc(m.group(2).rstrip("\n"))
        return _stash(f"<pre>{body}</pre>")

    text = _FENCE_RE.sub(_fence, text)

    text = _INLINE_CODE_RE.sub(lambda m: _stash(f"<code>{_esc(m.group(1))}</code>"), text)

    text = _esc(text)

    text = _HEADER_RE.sub(lambda m: f"<b>{m.group(1)}</b>", text)
    text = _BULLET_RE.sub(r"\1• ", text)

    text = _LINK_RE.sub(lambda m: _stash(f'<a href="{_esc(m.group(2))}">{m.group(1)}</a>'), text)

    text = _BOLD_RE.sub(r"<b>\1</b>", text)
    text = _BOLD_ALT_RE.sub(r"<b>\1</b>", text)
    text = _STRIKE_RE.sub(r"<s>\1</s>", text)
    text = _ITALIC_RE.sub(r"<i>\1</i>", text)
    text = _ITALIC_ALT_RE.sub(r"<i>\1</i>", text)

    for i, html in enumerate(stash):
        text = text.replace(_PLACEHOLDER.format(i), html)

    return text
