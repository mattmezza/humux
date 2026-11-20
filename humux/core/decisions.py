"""Typed decisions for incoming messages and reply delivery."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

from core import metrics
from core.config import DecisionConfig

log = logging.getLogger(__name__)

_TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
_HISTORY_TAIL = 6
_CONTENT_CLIP = 300
_REACTIONS = {
    "none": "An ordinary question, instruction, or message where an emoji adds nothing.",
    "thumbsup": "A lightweight acknowledgment, agreement, or an explicit thumbs-up request.",
    "heart": "Warm appreciation, encouragement, or an explicit heart request.",
    "party": "Congratulations, good news, or an explicit celebration request.",
    "laugh": "A joke or an explicit laughter reaction request.",
    "eyes": "An acknowledgment of something shared for review, or an explicit eyes request.",
    "cry": "Empathy about sad news, or an explicit crying reaction request.",
    "fire": "Enthusiastic praise or an explicit fire reaction request.",
    "thumbsdown": "An explicit thumbs-down request.",
}


@dataclass(frozen=True)
class IncomingDecision:
    reply: bool = True
    reaction: str | None = None
    reaction_only: bool = False


def _noul(instructions: str) -> dict:
    return {"type": "noul", "instructions": instructions}


def _format_history(history: list[dict]) -> str:
    """Render the last few messages as a compact transcript."""
    lines = []
    for turn in history[-_HISTORY_TAIL:]:
        role = turn.get("role", "?")
        content = str(turn.get("content", "")).strip().replace("\n", " ")
        if len(content) > _CONTENT_CLIP:
            content = content[:_CONTENT_CLIP] + "…"
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(none)"


def _probability(answer: object) -> float | None:
    if not isinstance(answer, dict) or answer.get("type") != "noul":
        return None
    value = answer.get("noul")
    if type(value) in (float, int) and 0 <= value <= 1:
        return float(value)
    return None


def typesafe_api_key(configured: str) -> str:
    """Use the resolved config value, falling back to the server environment."""
    if configured and not configured.startswith("${"):
        return configured
    return os.environ.get("TYPESAFE_API_KEY", "")


class DecisionClient:
    """Evaluate the configured decision model using typed questions."""

    def __init__(self, config: DecisionConfig):
        self.model = config.model
        self._key = typesafe_api_key(config.api_key)

    async def _evaluate(self, state: dict, questions: dict) -> dict:
        if not self._key or self._key.startswith("${"):
            log.warning("TypeSafe decision model has no API key")
            return {}
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                result = await client.post(
                    _TYPESAFE_URL,
                    headers={"Authorization": f"Bearer {self._key}"},
                    json={"model": self.model, "state": state, "questions": questions},
                )
                result.raise_for_status()
                body = result.json()
            if isinstance(body, dict) and isinstance(body.get("answers"), dict):
                usage = body.get("usage")
                if isinstance(usage, dict):
                    input_tokens = usage.get("input_tokens")
                    output_tokens = usage.get("output_tokens")
                    if (
                        type(input_tokens) is int
                        and input_tokens >= 0
                        and type(output_tokens) is int
                        and output_tokens >= 0
                    ):
                        model = body.get("model")
                        await metrics.record_usage(
                            provider="typesafe",
                            model=self.model,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            pricing_model=(
                                model if isinstance(model, str) and model else self.model
                            ),
                        )
                return body["answers"]
        except httpx.HTTPStatusError as exc:
            log.warning("TypeSafe decision endpoint returned HTTP %d", exc.response.status_code)
            return {}
        except (httpx.RequestError, ValueError) as exc:
            log.warning("TypeSafe decision request failed (%s)", type(exc).__name__)
            return {}
        log.warning("TypeSafe decision endpoint returned an invalid answer envelope")
        return {}

    async def incoming(
        self,
        message: str,
        identity: str,
        *,
        history: list[dict],
        others: list[str],
        gate_reply: bool,
        can_react: bool,
    ) -> IncomingDecision:
        questions = {}
        if gate_reply:
            questions["reply"] = _noul(
                "Should the named assistant respond to the latest message? Say yes for a real "
                "question, request, follow-up, or relevant remark it can help with. Say no for "
                "messages addressed to someone else, repetitive bot-to-bot acknowledgments, "
                "or messages it cannot contribute to. Favor replying when a person may need help."
            )
        if can_react:
            questions["reaction"] = {
                "type": "choice",
                "instructions": (
                    "Which single emoji, if any, should the named assistant react with to the "
                    "latest message? Prefer none unless a reaction fits clearly or was requested."
                ),
                "criteria": _REACTIONS,
            }
            questions["reaction_only"] = _noul(
                "Did the user explicitly ask the named assistant to acknowledge this message "
                "with only an emoji reaction, with no substantive question or task to answer?"
            )
        if not questions or not message.strip():
            return IncomingDecision()
        answers = await self._evaluate(
            {
                "assistant": identity,
                "other_assistants": others,
                "recent_conversation": _format_history(history),
                "latest_message": message[-4000:],
            },
            questions,
        )
        reply = _probability(answers.get("reply")) if gate_reply else None
        if reply is not None and reply < 0.35:
            return IncomingDecision(reply=False)
        reaction = answers.get("reaction")
        if can_react and isinstance(reaction, dict) and reaction.get("type") == "choice":
            choice = reaction.get("choice")
            probabilities = reaction.get("probabilities")
            if isinstance(choice, str) and choice in _REACTIONS and choice != "none":
                probability = probabilities.get(choice) if isinstance(probabilities, dict) else None
                if type(probability) in (int, float) and 0.65 <= probability <= 1:
                    only = _probability(answers.get("reaction_only"))
                    return IncomingDecision(
                        reaction=choice, reaction_only=only is not None and only >= 0.85
                    )
        return IncomingDecision()

    async def spoken_parts(self, message: str, parts: list[str]) -> set[int]:
        """Choose the parts to synthesize, batching them in one request."""
        questions = {
            f"part_{i}": _noul(
                f"Should reply_parts[{i}] be delivered as a voice note? Honor requests for "
                "text or audio. Favor voice for short conversational replies or replies to a "
                "[voice] message. Keep text for code, links, commands, structured answers, and "
                "details the user may need to copy. When uncertain choose text."
            )
            for i, part in enumerate(parts)
            if part and len(part) <= 1400
        }
        if not questions:
            return set()
        answers = await self._evaluate(
            {"latest_message": message[-4000:], "reply_parts": [part[:1400] for part in parts]},
            questions,
        )
        return {
            int(key.removeprefix("part_"))
            for key in questions
            if (probability := _probability(answers.get(key))) is not None and probability >= 0.65
        }
