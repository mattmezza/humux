"""Task reflection — post-task introspection and learning.

After the agent completes a task (tool-use loop finishes), a background
reflection step analyses what happened: did tools fail? Were there retries?
Did the plan work? The resulting "lesson" is stored in a SQLite database
and injected into future system prompts so the agent can learn from past
mistakes.

This mirrors the memory system's architecture: a cheap/fast model runs
in the background, and results are persisted in SQLite for prompt injection.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import aiosqlite

from core.llm import LLMClient
from core.memory import _similarity, _tokens  # reuse lexical sim — no new dep (#156)

log = logging.getLogger(__name__)

# Lessons that steer toward a binary that isn't installed are worse than none —
# they send the agent at a missing tool. rg is allowlisted but absent on the host
# (see the executor allowlist issue #156), so drop advice recommending it.
# ponytail: extend this set if other phantom-tool lessons surface.
_PHANTOM_TOOL_RE = re.compile(r"\b(rg|ripgrep)\b", re.IGNORECASE)

# Two lessons this similar (Jaccard over content tokens) are the same advice
# reworded — the store fills with near-duplicate variants otherwise (#156).
_NEAR_DUP_THRESHOLD = 0.6

_SCHEMA_FILE = Path(__file__).resolve().parent.parent / "schema" / "reflections.sql"

_REFLECTION_PROMPT = """\
You are a reflection engine for a personal AI assistant. After the assistant
completes a task, you analyse the execution to extract lessons learned.

## User request
{user_msg}

## Assistant response
{agent_msg}

## Tool calls executed
{tool_log}

Analyse the execution and determine:
1. Was the task completed successfully, partially, or did it fail?
2. Were there any tool errors, API timeouts, or unexpected issues?
3. Is there a lesson worth remembering for future tasks?

Return a JSON object:
{{
  "outcome": "success" | "partial" | "failure",
  "task_summary": "<one-line summary of what was attempted>",
  "lesson": "<concise lesson learned, or empty string if nothing noteworthy>",
  "tool_issues": [
    {{"tool": "<tool_name>", "issue": "<what went wrong>", \
"suggestion": "<what to try next time>"}}
  ],
  "category": "general" | "tool" | "api" | "planning" | "communication"
}}

Rules:
- Most tasks complete successfully with nothing noteworthy. In that case,
  set lesson to "" (empty string) and tool_issues to [].
- Only record lessons that would genuinely help in future tasks.
  Examples of good lessons:
  - "Weather API timed out; should retry with a different provider"
  - "User prefers concise email replies, not lengthy ones"
  - "Calendar event creation requires end time; estimate 1 hour if not specified"
  - "himalaya list --folder flag is case-sensitive; use 'INBOX' not 'inbox'"
- Do NOT record trivial observations like "task completed successfully"
  or "user said thank you".
- Do NOT recommend switching to a different tool or binary ("use X instead
  of Y"); the runtime's command allowlist already defines what is available.
- Be concise. Lessons should be 1-2 sentences max.

Respond with ONLY the JSON object, no other text."""

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def _extract_json_object(raw: str) -> dict | None:
    """Best-effort extraction of a JSON object from an LLM response."""
    raw = raw.strip()
    if not raw:
        return None

    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    fence_match = _FENCE_RE.search(raw)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1).strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

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


class ReflectionStore:
    """SQLite-backed storage for task reflections.

    Handles schema initialisation, storing new reflections, and
    formatting past lessons for system prompt injection.
    """

    def __init__(self, db_path: str = "data/reflections.db", max_reflections: int = 12):
        self.db_path = db_path
        self.max_reflections = max_reflections
        self._ready = False

    async def _ensure_schema(self) -> None:
        if self._ready:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        schema = _SCHEMA_FILE.read_text()
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(schema)
        self._ready = True

    async def get_recent_reflections(self) -> list[dict]:
        """Retrieve recent reflections with non-empty lessons for prompt injection."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT category, task_summary, lesson, outcome, tool_issues "
                "FROM reflections "
                "WHERE lesson != '' "
                "ORDER BY created_at DESC LIMIT ?",
                (self.max_reflections,),
            )
            return [dict(row) for row in await cursor.fetchall()]

    async def format_for_prompt(self) -> str:
        """Format recent reflections into a block for the system prompt."""
        reflections = await self.get_recent_reflections()
        if not reflections:
            return ""

        lines: list[str] = []
        # #156: drop phantom-tool advice and near-duplicate lessons (the store
        # accumulates reworded repeats), not just byte-identical ones.
        kept_tokens: list[set[str]] = []
        for r in reflections:
            lesson = r["lesson"]
            if _PHANTOM_TOOL_RE.search(lesson):
                continue
            toks = _tokens(lesson)
            if any(_similarity(toks, prev) >= _NEAR_DUP_THRESHOLD for prev in kept_tokens):
                continue
            kept_tokens.append(toks)
            outcome_marker = ""
            if r["outcome"] != "success":
                outcome_marker = f" [{r['outcome']}]"
            lines.append(f"- [{r['category']}]{outcome_marker} {r['lesson']}")

            # Include tool-specific issues if present
            if r.get("tool_issues"):
                try:
                    issues = (
                        json.loads(r["tool_issues"])
                        if isinstance(r["tool_issues"], str)
                        else r["tool_issues"]
                    )
                    if isinstance(issues, list):
                        for issue in issues:
                            if isinstance(issue, dict) and issue.get("suggestion"):
                                lines.append(f"  -> {issue['tool']}: {issue['suggestion']}")
                except json.JSONDecodeError, TypeError:
                    pass

        return "## Lessons from past tasks\n" + "\n".join(lines) if lines else ""

    async def reflect_on_task(
        self,
        llm: LLMClient,
        model: str,
        user_msg: str,
        agent_msg: str,
        tool_log: list[dict],
    ) -> bool:
        """Run reflection on a completed task and store the result.

        Returns True if a meaningful lesson was stored.
        """
        # Skip reflection if no tools were used (simple Q&A)
        if not tool_log:
            return False

        # Format tool log for the prompt
        tool_log_str = self._format_tool_log(tool_log)

        prompt = _REFLECTION_PROMPT.format(
            user_msg=user_msg,
            agent_msg=agent_msg,
            tool_log=tool_log_str,
        )

        try:
            raw = await llm.generate_text(model=model, prompt=prompt, max_tokens=1024)
        except Exception:
            log.exception("Task reflection LLM call failed")
            return False

        parsed = _extract_json_object(raw)
        if not parsed:
            log.warning("Task reflection returned non-JSON: %s", raw[:200])
            return False

        lesson = parsed.get("lesson", "").strip()
        if not lesson:
            log.debug("Task reflection found nothing noteworthy")
            return False

        # Store the reflection
        return await self._store_reflection(parsed)

    @staticmethod
    def _format_tool_log(tool_log: list[dict]) -> str:
        """Format a list of tool call results into a readable log."""
        if not tool_log:
            return "(no tools used)"

        lines: list[str] = []
        for entry in tool_log:
            name = entry.get("name", "unknown")
            result = entry.get("result", {})
            # Truncate long results
            result_str = json.dumps(result)
            if len(result_str) > 500:
                result_str = result_str[:500] + "..."
            status = "OK" if not result.get("error") else f"ERROR: {result.get('error', '')[:100]}"
            lines.append(f"- {name}: {status}")
            if result.get("error"):
                lines.append(f"  Detail: {result_str}")
        return "\n".join(lines)

    async def _store_reflection(self, parsed: dict) -> bool:
        """Store a reflection in the database."""
        task_summary = parsed.get("task_summary", "")
        outcome = parsed.get("outcome", "success")
        lesson = parsed.get("lesson", "")
        tool_issues = parsed.get("tool_issues", [])
        category = parsed.get("category", "general")

        if not lesson:
            return False

        # Drop advice that points at a tool that isn't installed (#156).
        if _PHANTOM_TOOL_RE.search(lesson):
            log.debug("Skipping reflection recommending an uninstalled tool: %s", lesson[:80])
            return False

        # Validate outcome
        if outcome not in ("success", "partial", "failure"):
            outcome = "success"

        # Validate category
        valid_categories = ("general", "tool", "api", "planning", "communication")
        if category not in valid_categories:
            category = "general"

        tool_issues_str = json.dumps(tool_issues) if tool_issues else None

        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            # Skip exact OR near-duplicate lessons — the store fills with reworded
            # variants of the same advice otherwise (#156).
            cursor = await db.execute("SELECT lesson FROM reflections")
            new_tokens = _tokens(lesson)
            for (existing,) in await cursor.fetchall():
                if existing == lesson or _similarity(new_tokens, _tokens(existing)) >= (
                    _NEAR_DUP_THRESHOLD
                ):
                    log.debug("Skipping duplicate reflection: %s", lesson[:80])
                    return False

            await db.execute(
                "INSERT INTO reflections (task_summary, outcome, lesson, tool_issues, category) "
                "VALUES (?, ?, ?, ?, ?)",
                (task_summary, outcome, lesson, tool_issues_str, category),
            )
            await db.commit()
            log.info("Stored task reflection: [%s] %s", category, lesson[:80])

            # Prune old reflections beyond the limit
            await db.execute(
                "DELETE FROM reflections WHERE id NOT IN "
                "(SELECT id FROM reflections ORDER BY created_at DESC LIMIT ?)",
                (self.max_reflections * 2,),  # Keep some buffer
            )
            await db.commit()

        return True
