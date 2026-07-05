"""Reply decision — should the agent reply to this message at all? (#36)

In a shared group chat with multiple bots and people, not every message
warrants a reply. A naive "always reply" agent caught in a chat with another
bot produces an infinite reaction loop (bot A replies to bot B replies to
bot A...). This adds a cheap one-shot LLM gate that filters out messages the
agent should stay quiet on:

- messages clearly addressed to someone else (another bot/person),
- self-referential bot-to-bot loops,
- messages the agent has nothing useful to add to.

The gate is advisory and *fails open*: any error returns True (reply), so a
classifier hiccup never silently drops a real user message. A separate hard
rate-limit backstop in AgentCore guarantees loop termination regardless.

The judgement is only as good as its inputs, so the caller feeds it the
recent conversation (so follow-ups / pronouns / reply-threads are readable)
and the names of the sibling assistants in the room (so "addressed to someone
else" is anchored to real agents, not a guess) — see ``AgentCore``.
"""

from __future__ import annotations

import logging

from core.llm import LLMClient

log = logging.getLogger(__name__)

# Only the last few messages matter for "is this turn for me"; a long history
# just dilutes the signal and costs tokens on a call meant to be cheap.
_HISTORY_TAIL = 6
_CONTENT_CLIP = 300

_DECIDE_PROMPT = """\
You are a reply filter for {identity}, taking part in a shared group chat \
that may contain several bots and several people.
{others}
Decide whether {identity} should reply to the LATEST message below, using the \
recent conversation for context.

Answer SKIP (do not reply) when ANY of these hold:
- The latest message is clearly addressed to someone else — another assistant \
or a person named/mentioned that is not {identity}.
- The message is part of a bot-to-bot back-and-forth that adds nothing \
(a reaction loop) — e.g. two assistants echoing pleasantries or acknowledgements.
- {identity} has nothing genuinely useful or relevant to contribute.

Answer REPLY when the message is a real question, request, or remark that \
{identity} can help with, or clearly continues a conversation {identity} is part of.

When in doubt about a message that looks like it came from a person, answer \
REPLY — only answer SKIP when you are confident a reply is unwanted.

Recent conversation:
{history}

Latest message:
{message}

Respond with ONLY one word: REPLY or SKIP"""


def _format_history(history: list[dict] | None) -> str:
    """Render the last few turns as a compact transcript for the gate."""
    if not history:
        return "(none)"
    lines = []
    for turn in history[-_HISTORY_TAIL:]:
        role = turn.get("role", "?")
        content = turn.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        content = content.strip().replace("\n", " ")
        if len(content) > _CONTENT_CLIP:
            content = content[:_CONTENT_CLIP] + "…"
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(none)"


def _format_others(others: list[str] | None, identity: str) -> str:
    """A line naming the sibling assistants so 'addressed to someone else' is
    concrete. Empty when none are known — the prompt then omits the hint."""
    names = [n for n in (others or []) if n and n != identity]
    if not names:
        return ""
    return (
        f"Other assistants that may be in this chat: {', '.join(names)}. "
        f"A message aimed at one of them (and not {identity}) should be SKIPped.\n"
    )


async def should_reply(
    llm: LLMClient,
    model: str,
    message: str,
    identity: str = "the assistant",
    *,
    history: list[dict] | None = None,
    others: list[str] | None = None,
    max_tokens: int = 2048,
) -> bool:
    """Return True if the agent should reply to ``message``.

    ``history`` is the recent conversation (``{role, content}`` dicts, oldest
    first) and ``others`` the display-names of sibling assistants in the room;
    both sharpen the judgement and are optional (an empty gate still works).

    ``max_tokens`` is generous on purpose: when ``thinking_level`` is enabled
    the reasoning tokens count against this budget, so a tight cap (the old 8)
    truncates before the REPLY/SKIP word ever lands — which, failing open, then
    replies to everything. We only read the first word, so the model stopping
    early after emitting it keeps the call cheap regardless.

    Fails open: returns True on an empty model response or any error, so a
    classifier failure never drops a genuine message. Only an explicit SKIP
    suppresses the reply.
    """
    text = message.strip()
    if not text:
        return True  # nothing to classify — let the normal path handle it

    prompt = _DECIDE_PROMPT.format(
        identity=identity,
        others=_format_others(others, identity),
        history=_format_history(history),
        message=text,
    )
    try:
        raw = await llm.generate_text(model=model, prompt=prompt, max_tokens=max_tokens)
    except Exception:
        log.exception("Reply decision LLM call failed; defaulting to reply")
        return True

    decision = raw.strip().upper()
    if decision.startswith("SKIP"):
        log.info("Reply decision: SKIP (%s)", text[:80])
        return False
    return True
