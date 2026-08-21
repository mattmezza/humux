"""Typed transcript entries for the read-only web view (``/weburl``).

Flattens raw stored history into entries a dumb frontend can render without
knowing anything about providers or history modes:

    {"kind": "text",        "role": "user"|"assistant", "text": ...}
    {"kind": "reasoning",   "text": ...}
    {"kind": "tool_call",   "name": ..., "args": {...}}
    {"kind": "tool_result", "name": ..., "text": ...}

Injection mode stores plain text turns, so it only ever yields ``text``.
Session mode stores the provider-shaped message array, so Anthropic content
blocks (``thinking`` / ``tool_use`` / ``tool_result``) and OpenAI-family raw
dicts (``reasoning_content``, ``tool_calls``, ``role: tool``) also yield the
disclosure kinds. Anything unrecognised is dropped rather than guessed at.

The system prompt is stored in its own table and never reaches this module.
"""

from __future__ import annotations

import json
import re
from typing import Any

# The per-turn preamble the agent prepends to the user's text (live date/time,
# skills index, memories, accounts, execution plan). It is context for the model,
# not something the user typed, so it is stripped from the displayed message.
# Every block is either a bracketed line or a <tag>…</tag> wrapper, and the whole
# preamble always opens with the date stamp — which is what gates the strip.
_PREAMBLE_MARKER = "[Current date & time:"
_PREAMBLE_BLOCK = re.compile(r"\A(?:\[[^\]]*\]|<([a-z_]+)>.*?</\1>)\s*", re.S)

# Tool results can be megabytes (a file read, a search dump). Cap what we ship;
# the page shows the head behind a "show more" anyway.
_MAX_RESULT_CHARS = 8000


def strip_preamble(text: str) -> str:
    """Drop the injected per-turn preamble from a stored user message.

    ponytail: heuristic — a user message that itself opens with a bracketed line
    right after a real preamble would lose that line. Cheaper than threading the
    raw message through the whole persistence path just for this view.
    """
    if not text.startswith(_PREAMBLE_MARKER):
        return text
    while match := _PREAMBLE_BLOCK.match(text):
        text = text[match.end() :]
    return text.strip()


def _truncate(text: str) -> str:
    if len(text) <= _MAX_RESULT_CHARS:
        return text
    return text[:_MAX_RESULT_CHARS] + "\n… (truncated)"


def _block_text(content: Any) -> str:
    """Flatten a tool_result ``content`` (str, or a list of text blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, default=str)


def _parse_args(raw: Any) -> Any:
    """Tool arguments as a JSON-able value (OpenAI ships them as a JSON string)."""
    if isinstance(raw, str):
        try:
            return json.loads(raw or "{}")
        except ValueError:
            return raw
    return raw if raw is not None else {}


def _session_entries(msg: dict, names: dict[str, str]) -> list[dict]:
    """Entries for one stored session message. ``names`` maps tool-use id → name,
    accumulated across the batch so a result can be labelled with its call."""
    role = str(msg.get("role") or "")
    if role == "system":
        return []  # never surfaced, defence in depth
    if role == "tool":  # OpenAI-family tool result
        call_id = str(msg.get("tool_call_id") or "")
        return [
            {
                "kind": "tool_result",
                "name": names.get(call_id, ""),
                "text": _truncate(_block_text(msg.get("content"))),
            }
        ]

    out: list[dict] = []
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if isinstance(reasoning, str) and reasoning.strip():
        out.append({"kind": "reasoning", "text": reasoning.strip()})

    content = msg.get("content")
    text_parts: list[str] = []
    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(str(block.get("text") or ""))
            elif btype == "thinking":
                thought = str(block.get("thinking") or "").strip()
                if thought:
                    out.append({"kind": "reasoning", "text": thought})
            elif btype == "redacted_thinking":
                out.append({"kind": "reasoning", "text": "[redacted reasoning]"})
            elif btype == "tool_use":
                name = str(block.get("name") or "")
                names[str(block.get("id") or "")] = name
                out.append({"kind": "tool_call", "name": name, "args": block.get("input") or {}})
            elif btype == "tool_result":
                call_id = str(block.get("tool_use_id") or "")
                out.append(
                    {
                        "kind": "tool_result",
                        "name": names.get(call_id, ""),
                        "text": _truncate(_block_text(block.get("content"))),
                    }
                )
            elif btype in ("image", "input_image", "image_url"):
                text_parts.append("[image]")

    for call in msg.get("tool_calls") or []:  # OpenAI-family assistant tool calls
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        name = str(fn.get("name") or "")
        names[str(call.get("id") or "")] = name
        out.append({"kind": "tool_call", "name": name, "args": _parse_args(fn.get("arguments"))})

    text = "\n\n".join(p for p in text_parts if p).strip()
    if role == "user":
        text = strip_preamble(text)
    if text:
        out.append({"kind": "text", "role": role or "assistant", "text": text})
    return out


def to_entries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten :meth:`ConversationHistory.transcript_rows` output into entries.

    Each entry carries the row id it came from plus its index within that row, so
    the frontend has a stable key, and the row's ``created_at`` as ``at``.
    """
    names: dict[str, str] = {}
    entries: list[dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload")
        if row.get("source") == "session":
            row_entries = _session_entries(payload, names) if isinstance(payload, dict) else []
        elif isinstance(payload, str):
            role = str(row.get("role") or "assistant")
            text = strip_preamble(payload) if role == "user" else payload
            row_entries = [{"kind": "text", "role": role, "text": text}] if text.strip() else []
        else:
            row_entries = []
        for index, entry in enumerate(row_entries):
            entry["id"] = f"{row['id']}.{index}"
            entry["at"] = row.get("created_at") or ""
            entries.append(entry)
    return entries
