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
"""

from __future__ import annotations

import logging

from core.llm import LLMClient

log = logging.getLogger(__name__)

_DECIDE_PROMPT = """\
You are a reply filter for {identity}, taking part in a shared group chat \
that may contain several bots and several people.

Decide whether {identity} should reply to the LATEST message below.

Answer SKIP (do not reply) when ANY of these hold:
- The message is clearly addressed to someone else — another bot or a person \
named/mentioned that is not {identity}.
- The message is part of a bot-to-bot back-and-forth that adds nothing \
(a reaction loop) — e.g. two assistants echoing pleasantries or acknowledgements.
- {identity} has nothing genuinely useful or relevant to contribute.

Answer REPLY when the message is a real question, request, or remark that \
{identity} can help with, or clearly continues a conversation {identity} is part of.

When in doubt about a message that looks like it came from a person, answer \
REPLY — only answer SKIP when you are confident a reply is unwanted.

Latest message:
{message}

Respond with ONLY one word: REPLY or SKIP"""


async def should_reply(
    llm: LLMClient,
    model: str,
    message: str,
    identity: str = "the assistant",
) -> bool:
    """Return True if the agent should reply to ``message``.

    Fails open: returns True on an empty model response or any error, so a
    classifier failure never drops a genuine message. Only an explicit SKIP
    suppresses the reply.
    """
    text = message.strip()
    if not text:
        return True  # nothing to classify — let the normal path handle it

    prompt = _DECIDE_PROMPT.format(identity=identity, message=text)
    try:
        raw = await llm.generate_text(model=model, prompt=prompt, max_tokens=8)
    except Exception:
        log.exception("Reply decision LLM call failed; defaulting to reply")
        return True

    decision = raw.strip().upper()
    if decision.startswith("SKIP"):
        log.info("Reply decision: SKIP (%s)", text[:80])
        return False
    return True
