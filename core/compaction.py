"""Conversation compaction for session history mode.

When a sticky session grows close to the model's context window, the oldest
turns are summarised by a (cheap) LLM into a single synthetic exchange, while
the most recent turns are kept verbatim. This keeps long conversations going
without blowing the context window, and — unlike provider-specific server-side
compaction — works across every provider MPA supports.

The trigger is the *real* token usage reported by the provider after the turn
(see ``LLMResponse.usage``), so no local tokenizer is needed.
"""

from __future__ import annotations

import logging
from typing import Any

from core.config import CompactionConfig
from core.llm import LLMClient

log = logging.getLogger(__name__)

# Known context-window sizes (tokens). Used for percent-mode thresholds.
# Unknown models fall back to CompactionConfig.context_window.
CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-4-8": 1_000_000,
    "claude-opus-4-7": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    "claude-opus-4-5": 200_000,
    "claude-opus-4-1": 200_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-4-6-sonnet": 1_000_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-4-5-haiku": 200_000,
}

_SUMMARY_PROMPT = """\
You are compacting the earlier part of a conversation between a user and their
personal AI assistant so it fits in a smaller context window. Write a dense,
factual summary that preserves everything needed to continue the conversation
seamlessly. Include:
- The user's goals, requests, and any decisions made.
- Concrete facts, names, identifiers, numbers, file paths, and preferences.
- Results of actions already taken (emails sent, events created, lookups done).
- Open threads or pending follow-ups.

Do NOT include pleasantries or restate this instruction. Be concise but complete.
Write the summary as plain prose / bullet points.

<conversation>
{transcript}
</conversation>

Summary:"""


def effective_window(config: CompactionConfig, model: str) -> int:
    """Return the context window (tokens) to use for percent-mode thresholds."""
    key = (model or "").strip().lower()
    return CONTEXT_WINDOWS.get(key, config.context_window)


def compaction_threshold_tokens(config: CompactionConfig, model: str) -> int:
    """Return the absolute token count at which compaction should trigger."""
    if config.threshold_type == "tokens":
        return config.threshold_tokens
    window = effective_window(config, model)
    return int(window * config.threshold_percent / 100)


def should_compact(config: CompactionConfig, context_tokens: int, model: str) -> bool:
    """Return True if the current context size warrants compaction."""
    if not config.enabled or not context_tokens:
        return False
    return context_tokens >= compaction_threshold_tokens(config, model)


def _is_real_user_turn(message: dict[str, Any]) -> bool:
    """True if this is a genuine user turn (not a tool_result carrier message)."""
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if isinstance(content, str):
        return True
    if isinstance(content, list):
        # Tool results are delivered as user messages whose blocks are all
        # tool_result; a real user turn has at least one text/image block.
        return any(isinstance(b, dict) and b.get("type") != "tool_result" for b in content)
    return True


def _block_text(content: Any) -> str:
    """Flatten a message's content into plain text for summarisation."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append(str(block.get("text", "")))
        elif btype == "tool_use":
            parts.append(f"[tool_use {block.get('name', '')}: {block.get('input', {})}]")
        elif btype == "tool_result":
            parts.append(f"[tool_result: {block.get('content', '')}]")
        elif btype == "image":
            parts.append("[image]")
    return "\n".join(p for p in parts if p)


def _render_transcript(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "?")
        text = _block_text(msg.get("content")).strip()
        if text:
            lines.append(f"{role.upper()}: {text}")
    return "\n\n".join(lines)


async def compact_messages(
    llm: LLMClient,
    model: str,
    messages: list[dict[str, Any]],
    keep_recent_turns: int,
) -> tuple[list[dict[str, Any]], str] | None:
    """Summarise the oldest turns of a session, keeping the recent ones verbatim.

    Returns ``(new_messages, summary)`` or ``None`` if there is nothing worth
    compacting (too few turns). The rebuilt session is:

        [user(<summary>), assistant(<ack>), *recent_turns_verbatim]

    The cut is made at a real user-turn boundary so a ``tool_use`` block is
    never split from its ``tool_result``.
    """
    boundaries = [i for i, m in enumerate(messages) if _is_real_user_turn(m)]
    # Need more turns than we intend to keep, otherwise there's nothing to fold.
    if len(boundaries) <= keep_recent_turns:
        return None

    cut = boundaries[len(boundaries) - keep_recent_turns]
    prefix = messages[:cut]
    tail = messages[cut:]
    if not prefix:
        return None

    transcript = _render_transcript(prefix)
    if not transcript.strip():
        return None

    summary = await llm.generate_text(
        model=model,
        prompt=_SUMMARY_PROMPT.format(transcript=transcript),
        max_tokens=2048,
    )
    summary = (summary or "").strip()
    if not summary:
        return None

    new_messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "Here is a summary of the earlier part of our conversation "
                "(it was compacted to save space):\n\n"
                f"<conversation_summary>\n{summary}\n</conversation_summary>"
            ),
        },
        {
            "role": "assistant",
            "content": "Understood — I'll continue with that summarized context in mind.",
        },
        *tail,
    ]
    return new_messages, summary
