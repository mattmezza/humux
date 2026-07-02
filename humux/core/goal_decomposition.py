"""Goal decomposition — breaks complex user requests into structured sub-goals.

When a user sends a complex, multi-faceted request (e.g. "Plan my trip to
Tokyo"), the decomposer uses a cheap/fast LLM call to break it into ordered
sub-goals *before* the main inference model processes the request.  The
decomposed plan is injected into the system prompt so the main model can
follow it step-by-step.

The decomposition is optional and gated by a classifier: simple requests
(greetings, single questions, quick lookups) skip decomposition entirely.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from core.llm import LLMClient

log = logging.getLogger(__name__)

_CLASSIFY_PROMPT = """\
You are a request classifier for a personal AI assistant.

Determine whether the following user message requires goal decomposition
(i.e. it is a complex, multi-step request) or is simple enough to handle
directly.

User message: {user_msg}

Rules:
- COMPLEX: multi-step tasks, planning requests, requests involving multiple
  actions or decisions (e.g. "Plan my trip to Tokyo", "Set up my new laptop",
  "Help me prepare for the interview next week", "Organise my finances for
  the month").
- SIMPLE: greetings, single questions, factual lookups, status checks,
  single-action commands (e.g. "Hi", "What's the weather?", "Send an email
  to Marco", "What time is it in Tokyo?", "Read my latest email").
- When in doubt, classify as SIMPLE — decomposition adds latency and should
  only be used when it genuinely helps.

Respond with ONLY one word: COMPLEX or SIMPLE"""

_DECOMPOSE_PROMPT = """\
You are a goal decomposition engine for a personal AI assistant.

Break the following user request into a structured plan of 2-6 ordered
sub-goals. Each sub-goal should be concrete and actionable.

User message: {user_msg}

Return a JSON object with:
{{
  "goal": "<one-line summary of the overall goal>",
  "steps": [
    {{"id": 1, "title": "<short title>", "description": "<what to do>", \
"depends_on": []}},
    {{"id": 2, "title": "<short title>", "description": "<what to do>", \
"depends_on": [1]}},
    ...
  ]
}}

Rules:
- Keep it to 2-6 steps. Don't over-decompose.
- Each step should be independently actionable by an AI assistant with
  access to tools (email, calendar, web search, messaging, memory, CLI).
- Use depends_on to indicate ordering constraints. If step 3 needs results
  from step 1, set depends_on: [1].
- Be practical and specific. Avoid vague steps like "Research options".
  Instead: "Search for direct flights from Zurich to Tokyo in March".
- The assistant has access to: email, calendar, web search, messaging
  (Telegram/WhatsApp), memory database, and CLI tools.

Respond with ONLY the JSON object, no other text."""

# Regex to match a JSON object in the response
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


@dataclass
class SubGoal:
    id: int
    title: str
    description: str
    depends_on: list[int] = field(default_factory=list)


@dataclass
class DecomposedGoal:
    goal: str
    steps: list[SubGoal]

    def format_for_prompt(self) -> str:
        """Format the decomposed goal as a block for the system prompt."""
        lines = [f"Overall goal: {self.goal}", ""]
        for step in self.steps:
            deps = ""
            if step.depends_on:
                plural = "s" if len(step.depends_on) > 1 else ""
                ids = ", ".join(str(d) for d in step.depends_on)
                deps = f" (after step{plural} {ids})"
            lines.append(f"  {step.id}. {step.title}{deps}")
            lines.append(f"     {step.description}")
        return "\n".join(lines)


def _extract_json_object(raw: str) -> dict | None:
    """Best-effort extraction of a JSON object from an LLM response."""
    raw = raw.strip()
    if not raw:
        return None

    # Try direct parse
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try code fence
    fence_match = _FENCE_RE.search(raw)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1).strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Find outermost { ... }
    start = raw.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        end = -1
        for i in range(start, len(raw)):
            ch = raw[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            try:
                result = json.loads(raw[start : end + 1])
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

    return None


async def classify_complexity(
    llm: LLMClient,
    model: str,
    user_msg: str,
) -> bool:
    """Return True if the user message is complex enough to decompose.

    Uses a fast LLM call to classify the message. Returns False (simple)
    on any error — decomposition is always optional.
    """
    # Quick heuristic: very short messages are almost never complex
    if len(user_msg.strip()) < 20:
        return False

    prompt = _CLASSIFY_PROMPT.format(user_msg=user_msg)
    try:
        raw = await llm.generate_text(model=model, prompt=prompt, max_tokens=16)
        return raw.strip().upper().startswith("COMPLEX")
    except Exception:
        log.exception("Goal complexity classification failed")
        return False


async def decompose_goal(
    llm: LLMClient,
    model: str,
    user_msg: str,
) -> DecomposedGoal | None:
    """Decompose a complex user request into sub-goals.

    Returns None if decomposition fails or the LLM returns invalid output.
    """
    prompt = _DECOMPOSE_PROMPT.format(user_msg=user_msg)
    try:
        raw = await llm.generate_text(model=model, prompt=prompt, max_tokens=2048)
    except Exception:
        log.exception("Goal decomposition LLM call failed")
        return None

    parsed = _extract_json_object(raw)
    if not parsed:
        log.warning("Goal decomposition returned non-JSON: %s", raw[:200])
        return None

    goal = parsed.get("goal", "")
    steps_raw = parsed.get("steps", [])
    if not isinstance(steps_raw, list) or not steps_raw:
        log.warning("Goal decomposition returned no steps")
        return None

    steps: list[SubGoal] = []
    for s in steps_raw[:6]:  # Cap at 6 steps
        if not isinstance(s, dict):
            continue
        steps.append(
            SubGoal(
                id=s.get("id", len(steps) + 1),
                title=s.get("title", ""),
                description=s.get("description", ""),
                depends_on=s.get("depends_on", []),
            )
        )

    if not steps:
        return None

    result = DecomposedGoal(goal=goal, steps=steps)
    log.info(
        "Decomposed goal into %d steps: %s",
        len(steps),
        goal[:80],
    )
    return result
