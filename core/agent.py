"""Agent core — LLM call with agentic tool-use loop."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import shlex
import uuid
from collections import OrderedDict, deque
from datetime import datetime
from typing import Any, cast
from zoneinfo import ZoneInfo

from tavily import TavilyClient

from core.compaction import compact_messages, should_compact
from core.config import Config
from core.embeddings import LOCAL_PROVIDERS, EmbeddingClient, LocalEmbeddingClient
from core.executor import ToolExecutor
from core.goal_decomposition import DecomposedGoal, classify_complexity, decompose_goal
from core.history import ConversationHistory
from core.job_store import JobStore
from core.llm import LLMClient, LLMToolCall, model_supports_vision
from core.memory import MemoryStore
from core.models import AgentResponse, Attachment
from core.permissions import PermissionEngine, PermissionLevel, format_approval_message
from core.personae import Persona, PersonaStore
from core.prompt_builder import build_prompt_sections
from core.scheduler import AgentScheduler
from core.secret_store import SecretStore
from core.skills import SkillsEngine
from core.subagents import (
    FILE_HANDOFF_INSTRUCTION,
    RESULT_FOR_AGENT_INSTRUCTION,
    SubagentRegistry,
    SubagentRun,
    fallback_summary,
    narrow_scope,
    normalize_effort,
    resolve_cap,
    short_summary,
    summarize_batch,
)
from core.task_reflection import ReflectionStore
from core.tools import tool_env
from voice.pipeline import VoicePipeline

log = logging.getLogger(__name__)

# Vision fallback caption cache cap (per process). Captions are keyed by image
# hash so repeated identical images don't re-hit the vision model.
_VISION_CACHE_MAX = 256


def _shell_quote(s: str) -> str:
    """Quote a string for safe shell interpolation."""
    return shlex.quote(s)


# -- Tool definitions the LLM can call --

TOOLS = [
    # Generic CLI executor — the LLM constructs commands using skill knowledge
    {
        "name": "run_command",
        "description": (
            "Execute a CLI command. Use skill documentation to construct correct syntax. "
            "Returns stdout, stderr, and exit_code."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The full CLI command to run"},
                "purpose": {
                    "type": "string",
                    "description": "Brief explanation of what this command does",
                },
            },
            "required": ["command", "purpose"],
        },
    },
    # Structured tools for write actions (permission-gated via PermissionEngine)
    {
        "name": "send_email",
        "description": "Send a new email on behalf of the user.",
        "input_schema": {
            "type": "object",
            "properties": {
                "account": {
                    "type": "string",
                    "description": "Email account name (e.g. 'personal', 'work')",
                },
                "from": {
                    "type": "string",
                    "description": "Sender email address (must match the account)",
                },
                "to": {
                    "type": "string",
                    "description": "Recipient email address(es), comma-separated",
                },
                "cc": {"type": "string", "description": "CC recipient(s), comma-separated"},
                "bcc": {"type": "string", "description": "BCC recipient(s), comma-separated"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["account", "to", "subject", "body"],
        },
    },
    {
        "name": "reply_email",
        "description": "Reply to an existing email by message ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "account": {
                    "type": "string",
                    "description": "Email account name (e.g. 'personal', 'work')",
                },
                "message_id": {
                    "type": "string",
                    "description": "The ID of the message to reply to",
                },
                "body": {"type": "string", "description": "The reply body text"},
                "reply_all": {
                    "type": "boolean",
                    "description": "Reply to all recipients (default: false)",
                },
                "folder": {
                    "type": "string",
                    "description": "Folder the message is in (default: INBOX)",
                },
            },
            "required": ["account", "message_id", "body"],
        },
    },
    {
        "name": "send_message",
        "description": "Send a message to a contact via Telegram or WhatsApp.",
        "input_schema": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "enum": ["telegram", "whatsapp"],
                    "description": "Which messaging channel to use",
                },
                "to": {"type": "string", "description": "Recipient identifier (chat ID or phone)"},
                "text": {"type": "string", "description": "Message text"},
            },
            "required": ["channel", "to", "text"],
        },
    },
    {
        "name": "create_calendar_event",
        "description": "Create a calendar event or send an invite.",
        "input_schema": {
            "type": "object",
            "properties": {
                "calendar": {
                    "type": "string",
                    "description": "Calendar name (e.g. 'google', 'icloud')",
                },
                "summary": {"type": "string", "description": "Event title"},
                "start": {"type": "string", "description": "ISO datetime with timezone"},
                "end": {"type": "string", "description": "ISO datetime with timezone"},
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of attendee email addresses",
                },
            },
            "required": ["calendar", "summary", "start", "end"],
        },
    },
    # Read-only / utility tools
    {
        "name": "web_search",
        "description": "Search the web for information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "load_skill",
        "description": "Load a named skill document by name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name to load"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "recall_memory",
        "description": (
            "Search your FULL long-term memory by meaning for facts about the user that "
            "aren't already shown to you. Only the few most-relevant memories are injected "
            "into each turn; call this when you suspect a relevant stored fact exists beyond "
            "them — it searches the whole store, including older archived memories, and ranks "
            "matches by relevance. Pass a natural-language query describing the fact you're "
            "after (e.g. 'dietary restrictions and food allergies'), not just keywords."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language description of the fact(s) to recall",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max memories to return (default 10).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "manage_jobs",
        "description": (
            "Create, list, or cancel scheduled jobs. "
            "Use action='create' to schedule a one-time or recurring task. "
            "Use action='list' to see all active jobs. "
            "Use action='cancel' to stop a job from running."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "cancel"],
                    "description": (
                        "What to do: create a new job, list existing jobs, or cancel a job"
                    ),
                },
                "job_id": {
                    "type": "string",
                    "description": (
                        "For create: a short unique identifier (lowercase, dashes ok). "
                        "For cancel: the ID of the job to cancel."
                    ),
                },
                "task": {
                    "type": "string",
                    "description": (
                        "What the agent should do when the job runs (natural language instruction)"
                    ),
                },
                "run_at": {
                    "type": "string",
                    "description": (
                        "For one-time jobs: ISO datetime with timezone offset when the task "
                        "should run (e.g. '2026-02-21T09:00:00+01:00'). "
                        "If no offset is provided, the user's configured timezone is assumed."
                    ),
                },
                "cron": {
                    "type": "string",
                    "description": (
                        "For recurring jobs: 5-field cron expression "
                        "(minute hour day month weekday). "
                        "Example: '30 7 * * 1-5' = weekdays at 07:30"
                    ),
                },
                "channel": {
                    "type": "string",
                    "description": "Channel to deliver the result on (default: telegram)",
                },
                "description": {
                    "type": "string",
                    "description": "Short human-readable description of this job",
                },
            },
            "required": ["action"],
        },
    },
    # Subagents (issue #15) — delegate a scoped subtask to a sub-loop.
    {
        "name": "spawn_subagent",
        "description": (
            "Delegate a self-contained subtask to a subagent. The subagent runs "
            "the full agent loop under a persona, with a tool/skill/secret scope "
            "that is never wider than yours, and returns a structured result. It "
            "has NO memory of this conversation — put everything it needs in "
            "'task'.\n"
            "Persona: by DEFAULT omit 'persona' — the subagent runs as YOU (your "
            "identity, tools, scope). This is almost always what you want. Set "
            "'persona' ONLY when the user explicitly asked for a named specialist, "
            "or the subtask plainly belongs to a different one. Never pick a "
            "persona just because the roster lists some.\n"
            "Sizing: by default the subagent runs at the configured ceilings. Size "
            "it to the job with 'max_steps', 'token_budget', and 'thinking_effort' "
            "— smaller for a quick lookup, larger / 'high' effort for hard "
            "multi-step work. Requested values are capped at the configured maxima.\n"
            "Files: you share a filesystem with the subagent, so it reports the "
            "absolute paths of any files it creates in its result — you can then "
            "read or send them.\n"
            "Use background=true for long-running work: you get a run id "
            "immediately and the result is posted to this chat when done (monitor "
            "or cancel it with /jobs or the admin Jobs page). Use background=false "
            "(default) to block and get the result back in this turn.\n"
            "Subagents are depth-limited, so prefer one focused delegation over "
            "deep nesting."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "The complete instruction for the subagent. Be specific and "
                        "self-contained — it cannot see this conversation."
                    ),
                },
                "persona": {
                    "type": "string",
                    "description": (
                        "Persona name to run as. OMIT THIS by default — the subagent "
                        "then runs as you (same identity, tools, scope), which is "
                        "almost always correct. Only set it when the user explicitly "
                        "named a specialist or the subtask clearly belongs to one."
                    ),
                },
                "max_steps": {
                    "type": "integer",
                    "description": (
                        "Tool-call rounds the subagent may run before a hard stop. "
                        "Omit for the configured default; capped at the maximum. "
                        "Lower it for quick tasks, raise it for thorough ones."
                    ),
                },
                "token_budget": {
                    "type": "integer",
                    "description": (
                        "Approximate token ceiling for the whole run (minimum 1000). "
                        "Omit for the configured default; capped at the maximum."
                    ),
                },
                "thinking_effort": {
                    "type": "string",
                    "enum": ["off", "low", "medium", "high"],
                    "description": (
                        "How hard the subagent reasons each step. Omit to inherit "
                        "your own level. Use 'high' for tricky reasoning, 'off'/'low' "
                        "for simple mechanical work."
                    ),
                },
                "background": {
                    "type": "boolean",
                    "description": (
                        "Run asynchronously (default false). True returns a run id "
                        "now and posts the result back to this chat when done."
                    ),
                },
            },
            "required": ["task"],
        },
    },
    # Secrets vault (issue #19) — discover + request secrets by NAME only.
    {
        "name": "list_secrets",
        "description": (
            "List the names of stored secrets you may use (with descriptions). "
            "Returns NAMES ONLY — never values. Use a listed name by reference as "
            "{{secret:NAME}} inside run_command."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "request_secret",
        "description": (
            "Ask the owner to provide a secret you need but don't have (e.g. a website "
            "login). Sends the owner a secure web link to enter the value; you never "
            "handle the value yourself. Use when a needed {{secret:NAME}} is not listed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name for the secret (letters, digits, _ - : only)",
                },
                "reason": {
                    "type": "string",
                    "description": "Why you need it / what you'll do with it",
                },
                "suggested_scope": {
                    "type": "string",
                    "description": "Optional hint: which persona(s) should be able to use it",
                },
            },
            "required": ["name", "reason"],
        },
    },
    {
        "name": "write_artifact",
        "description": (
            "Publish a web artifact and get back a shareable link (e.g. "
            "https://host/artifacts/AbC123xy/). Use this whenever the answer is "
            "richer than chat can show: reports, dashboards, charts, comparison "
            "tables, interactive checklists/trackers, slide decks, or any 'give "
            "me a mini-site / document for X'. The link serves a whole directory.\n"
            "Provide EXACTLY ONE of:\n"
            "- 'html': a full standalone HTML document (becomes index.html). Best "
            "for a single page. Inline CSS in <style> and JS in <script>. Climb "
            "only as high as needed: plain semantic HTML for a quick report; a "
            "classless CSS framework (MVP.css / Water.css via CDN) for clean docs; "
            "custom CSS or TailwindCSS v4 (browser build "
            "<script src='https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4'>"
            "</script>) for designed pages; JS or Alpine.js (CDN) only when "
            "interactive.\n"
            "- 'files': a {relative_path: text} map for a MULTI-FILE site — e.g. "
            "{'index.html': '…', 'style.css': '…', 'app.js': '…'}. Link them with "
            "relative URLs (href='style.css'). Must include index.html (or set "
            "'entrypoint').\n"
            "- 'source_path': an absolute path to a file or directory you already "
            "produced on disk — a PDF, image, slides, doc, or a prebuilt site dir. "
            "Use this for binary/generated outputs (e.g. write a PDF with pandoc to "
            "/tmp, then publish it). Publishing an on-disk file asks the owner for "
            "approval first.\n"
            "Pick 'ttl_hours' to fit the content: a small number for things that go "
            "stale fast (a daily report), a large number or 0 (keep forever) for "
            "lasting references. Omit to use the configured default. After writing, "
            "give the returned link to the user."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "html": {
                    "type": "string",
                    "description": "A complete HTML document; published as index.html.",
                },
                "files": {
                    "type": "object",
                    "description": (
                        "Map of relative filename → text content for a multi-file "
                        "artifact (e.g. index.html + style.css + app.js)."
                    ),
                    "additionalProperties": {"type": "string"},
                },
                "source_path": {
                    "type": "string",
                    "description": (
                        "Absolute path to a file or directory to copy in (PDF, "
                        "image, slides, doc, or a prebuilt site). Asks for approval."
                    ),
                },
                "entrypoint": {
                    "type": "string",
                    "description": "File served at the root URL. Default 'index.html'.",
                },
                "ttl_hours": {
                    "type": "integer",
                    "description": (
                        "How long to keep this artifact, in hours. 0 = forever. "
                        "Omit for the configured default."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": "Optional short title, for your reference and the logs.",
                },
            },
        },
    },
]


def _persona_scope(persona: Persona | None) -> str:
    """The memory scope key for an active persona (#42).

    A persona's own name is its private scope; no persona (default identity) =
    ``""`` = shared only.
    """
    return persona.name if persona else ""


def scoped_tools(persona: Persona | None) -> list[dict]:
    """Filter the function-tool schemas by the active persona's tool scope.

    ``load_skill`` is always retained — it is the core mechanic personae rely on
    to read their allowlisted skills. An empty scope (or no persona) = all tools.
    """
    if persona is None or not persona.tools:
        return TOOLS
    # ``load_skill`` and the vault discovery/request tools are always retained:
    # they are the mechanics personae rely on to read skills and obtain secrets.
    # ``recall_memory`` too — memory is injected for every persona (scope-filtered),
    # so its on-demand counterpart exposes nothing extra and stays available (#47).
    _always = {"load_skill", "recall_memory", "list_secrets", "request_secret"}
    return [t for t in TOOLS if persona.allows_tool(t["name"]) or t["name"] in _always]


def apply_feature_gates(
    tools: list[dict],
    *,
    secrets_available: bool,
    artifacts_enabled: bool,
    subagents_enabled: bool = True,
) -> list[dict]:
    """Drop tools whose backing feature is unavailable/disabled, so the model is
    never offered a capability it can't use (defence in depth — the tool handlers
    also refuse). Disabling ``artifacts`` here means no persona can call it."""
    out = tools
    if not secrets_available:
        out = [t for t in out if t["name"] not in ("list_secrets", "request_secret")]
    if not artifacts_enabled:
        out = [t for t in out if t["name"] != "write_artifact"]
    if not subagents_enabled:
        out = [t for t in out if t["name"] != "spawn_subagent"]
    return out


class AgentCore:
    def __init__(self, config: Config, secret_store: SecretStore | None = None):
        self.config = config
        # Secrets vault (issue #19). Shared, process-wide so the persona DEK
        # unsealed by an admin login is visible to the agent at runtime.
        self.secret_store = secret_store
        self.llm: LLMClient = LLMClient.from_agent_config(config.agent)
        self.skills = SkillsEngine(
            db_path=config.agent.skills_db_path,
            seed_dir=config.agent.skills_dir,
        )
        self.personae = PersonaStore(
            db_path=config.agent.personae_db_path,
            seed_dir=config.agent.personae_dir,
        )
        self.executor = ToolExecutor(tool_env=tool_env(config))
        self.history = ConversationHistory(
            db_path=config.history.db_path,
            max_turns=config.history.max_turns,
        )
        self.history_mode = config.history.mode  # "injection" or "session"
        mem_cfg = config.memory
        self.memory = MemoryStore(
            db_path=mem_cfg.db_path,
            long_term_limit=mem_cfg.long_term_limit,
            embedder=self._build_embedder(),
            injection_top_k=mem_cfg.embedding.injection_top_k,
            recall_top_k=mem_cfg.embedding.recall_top_k,
            default_importance=mem_cfg.default_importance,
            archive_after_days=mem_cfg.archive_after_days,
            archive_max_importance=mem_cfg.archive_max_importance,
            archive_min_idle_days=mem_cfg.archive_min_idle_days,
            hygiene_enabled=mem_cfg.hygiene_enabled,
            hygiene_similarity_threshold=mem_cfg.hygiene_similarity_threshold,
        )
        self.reflections = ReflectionStore(
            db_path=config.task_reflection.db_path,
            max_reflections=config.task_reflection.max_reflections,
        )
        self.channels: dict = {}
        self.voice: VoicePipeline | None = None
        self.job_store = JobStore(db_path="data/jobs.db")
        self.scheduler = AgentScheduler(self, self.job_store)
        # Live registry of subagent runs (issue #15) — list/status/cancel.
        self.subagents = SubagentRegistry()
        config_db = "data/config.db"
        self.permissions = PermissionEngine(db_path=config_db)
        self.prompt_capture: deque[dict[str, str]] = deque(maxlen=20)
        # Vision fallback caption cache (image hash -> "[Image: ...]"), LRU-bounded.
        self._vision_cache: OrderedDict[str, str] = OrderedDict()

        # Web search (Tavily)
        if config.search.enabled and config.search.api_key:
            self.search_client: TavilyClient | None = TavilyClient(
                api_key=config.search.api_key,
            )
            log.info("Web search enabled (provider: %s)", config.search.provider)
        else:
            self.search_client = None
            log.info("Web search disabled (no API key or not enabled)")

    async def process(
        self,
        message: str,
        channel: str,
        user_id: str,
        attachments: list[Attachment] | None = None,
        chat_id: str = "",
        persona_name: str | None = None,
    ) -> AgentResponse:
        """Process an incoming message through the LLM with tool-use loop.

        ``chat_id`` distinguishes different chats for the same user (e.g.
        a private Telegram chat vs. a group chat).  Each unique
        (channel, user_id, chat_id) triple gets its own conversation history,
        preventing context leakage across chats.

        ``persona_name`` forces the identity instead of resolving it from the
        channel/binding ladder — used by the scheduler so a ``telegram:<persona>``
        job is generated *as* that persona while keeping the ``system`` execution
        mode (auto-approved writes, no memory/reflection) (#29).
        """

        # Handle /new (alias /clear) command — clear conversational context.
        if message.strip().lower() in ("/new", "/clear"):
            if self.history_mode == "session":
                await self.history.clear_session(channel, user_id, chat_id)
            else:
                await self.history.clear(channel, user_id, chat_id)
            log.info(
                "Conversation cleared by user (channel=%s, user=%s, chat=%s)",
                channel,
                user_id,
                chat_id,
            )
            return AgentResponse(text="Conversation cleared.")

        # Goal decomposition — classify and (if complex) decompose the request.
        # The resulting plan is request-specific, so it is injected per turn
        # (in the user-message preamble), not baked into the static prompt.
        decomposed_goal: DecomposedGoal | None = None
        if self.config.goal_decomposition.enabled and channel != "system":
            decomposed_goal = await self._maybe_decompose(message)

        # Resolve the active persona (its identity, skills + tool scope) — a
        # per-chat binding wins over the globally selected persona (#14). An
        # explicit override (scheduler) skips the ladder (#29).
        if persona_name:
            persona = await self._load_persona(persona_name)
        else:
            persona = await self._resolve_persona(channel, user_id, chat_id)

        # Per-turn preamble: live date/time + fresh memory/reflections + skills
        # index + plan. Memory is scoped to the active persona (#42): shared +
        # its private. Skills index is scoped to the persona's allowlist (#46).
        session_key = (channel, user_id, chat_id) if self.history_mode == "session" else None
        preamble = await self._turn_preamble(
            decomposed_goal,
            query=message,
            scope=_persona_scope(persona),
            persona=persona,
            session_key=session_key,
            offer_personae=True,
        )
        # Append the status of still-running background subagents from this chat,
        # so the agent always knows what is pending (their results are folded into
        # the conversation history when they finish). (#15)
        if channel != "system":
            note = self._subagent_status_note(channel, chat_id)
            if note:
                preamble = f"{preamble}\n\n{note}"

        tools = apply_feature_gates(
            scoped_tools(persona),
            secrets_available=self.secret_store is not None,
            artifacts_enabled=self.config.artifacts.enabled,
            subagents_enabled=self.config.subagents.enabled,
        )

        # Static system prompt. In session mode it is snapshotted once at the
        # start of the session and reused for every turn (so the static content
        # is only built once, not rebuilt and re-sent each turn). In injection
        # mode the prompt is windowed/stateless, so it is rebuilt per call.
        if self.history_mode == "session":
            system = await self._session_system_prompt(channel, user_id, chat_id, persona=persona)
        else:
            system = await self._build_system_prompt(persona=persona)

        if self.config.admin.capture_prompts:
            self._record_system_prompt(
                channel=channel,
                user_id=user_id,
                chat_id=chat_id,
                prompt=system,
            )

        if self.history_mode == "session":
            return await self._process_session(
                system, preamble, message, channel, user_id, attachments, chat_id, tools, persona
            )
        return await self._process_injection(
            system, preamble, message, channel, user_id, attachments, chat_id, tools, persona
        )

    async def _resolve_persona(self, channel: str, user_id: str, chat_id: str) -> Persona | None:
        """Resolve the active persona for this request, in precedence order:

        0. a per-persona bot — a ``"telegram:<name>"`` channel binds straight to
           persona ``<name>``: the bot that received the message *is* the persona (#29),
        1. the per-chat binding for ``(channel, user_id, chat_id)`` (#14),
        2. the globally-selected persona (``config.agent.active_persona``, #13),
        3. the default identity (``None``).
        """
        # 0. Bot-per-persona: the channel name carries the persona (e.g. "telegram:coach").
        _, sep, persona_name = channel.partition(":")
        if sep and persona_name:
            persona = await self._load_persona(persona_name)
            if persona:
                return persona
            # Unknown/deleted persona — fall through to the ordinary ladder.

        # 1. Per-chat binding.
        bound = await self.history.get_chat_persona(channel, user_id, chat_id)
        if bound:
            persona = await self._load_persona(bound)
            if persona:
                return persona

        # 2. Globally-selected persona.
        name = (self.config.agent.active_persona or "").strip()
        if name:
            return await self._load_persona(name)

        # 3. Default identity.
        return None

    async def _load_persona(self, name: str) -> Persona | None:
        """Load a persona by name, returning ``None`` if it is missing/broken."""
        try:
            return await self.personae.get(name)
        except Exception:
            log.exception("Failed to load persona %r", name)
            return None

    async def bind_chat_persona(
        self, channel: str, user_id: str, chat_id: str, persona_name: str
    ) -> None:
        """Bind (or, with an empty name, unbind) a chat to a persona.

        Thin pass-through to the history store, which also drops the snapshotted
        session system prompt so the new identity takes effect on the next turn.
        """
        await self.history.bind_chat_persona(channel, user_id, chat_id, persona_name)

    async def bind_chat_persona_by_label(
        self, channel: str, user_id: str, chat_id: str, label: str
    ) -> str | None:
        """Auto-bind a chat to the persona matching ``label`` (case-insensitive).

        Matches the label against each persona's ``name``, ``agent_name`` and
        ``role``. Only binds when the chat is not already bound, so a manual
        rebind is never clobbered. Returns the bound persona name, or ``None``.
        """
        label = (label or "").strip()
        if not label:
            return None
        if await self.history.get_chat_persona(channel, user_id, chat_id):
            return None  # already bound — don't override a manual choice
        target = label.lower()
        try:
            personae = await self.personae.list_personae()
        except Exception:
            log.exception("Failed to list personae for topic auto-bind")
            return None
        for p in personae:
            labels = {p.name.lower(), (p.agent_name or "").lower(), (p.role or "").lower()}
            if target in labels - {""}:
                await self.bind_chat_persona(channel, user_id, chat_id, p.name)
                return p.name
        return None

    async def _turn_preamble(
        self,
        decomposed_goal: DecomposedGoal | None,
        query: str | None = None,
        scope: str = "",
        persona: Persona | None = None,
        session_key: tuple[str, str, str] | None = None,
        offer_personae: bool = False,
    ) -> str:
        """Build the per-turn preamble prepended to the current user message.

        Always carries the live date/time (so the agent knows 'now' every turn);
        also carries fresh, query-relevant memory + reflections, the live skills
        index, and the execution plan when the request was decomposed.

        Memory/reflections/skills live here, not in the static system prompt: in
        session mode that prompt is snapshotted once and would freeze any
        mid-session change out of view until ``/new`` (#41, #46) — e.g. a skill
        added via the skill-creator stayed invisible. The preamble is rebuilt
        every turn and rides on the new (uncached) user message, so it costs only
        the block's own tokens and is also relevance-ranked per turn.

        ``scope`` is the active persona's memory scope (#42): ``""`` = shared
        only, ``"<persona>"`` = shared + that persona's private memory.
        ``persona`` scopes the skills index to its allowlist. ``session_key``
        gates skills re-injection (see below); ``None`` = always inject.
        """
        now = datetime.now(ZoneInfo(self.config.agent.timezone))
        stamp = now.strftime("%A, %B %d, %Y %H:%M %Z")
        preamble = f"[Current date & time: {stamp}]"

        # Skills index, scoped to the persona's allowlist. Rebuilt fresh per turn
        # so a skill added mid-session (e.g. via skill-creator) is immediately
        # visible without a /new (#46). Cheap: a local DB read, like memory.
        #
        # In session mode (``session_key`` set) the preamble is persisted into the
        # growing history, so re-sending an unchanged index every turn would just
        # accumulate identical copies. We skip it only when the exact block is
        # ALREADY present in the replayed history (so the model still sees it).
        # Gating on the real history — not a side cache — keeps it correct by
        # construction across /new, compaction, persona rebind and concurrent
        # turns: any of those that drop or change the block simply won't find it,
        # and the failure direction is a harmless re-send, never a blind turn.
        # Injection mode and tests pass ``None`` → always include.
        try:
            skills_index = await self.skills.get_index_block(
                allow=persona.skills if persona else None
            )
            if skills_index:
                block = f"<available_skills>\n{skills_index}\n</available_skills>"
                if session_key is None or not await self._skills_block_in_history(
                    session_key, block
                ):
                    preamble += f"\n\n{block}"
        except Exception:
            log.exception("Failed to load skills index for turn preamble")

        # ponytail: in session mode this now runs a query embed + cosine scan +
        # reinforce-write every turn (was once per session). Intended — that is
        # what makes injection fresh and per-turn relevant — and cheap for a
        # personal store. If the store grows huge, gate behind the recall_memory
        # tool (issue #41 phase 2) instead of always-injecting top-k.
        try:
            memories = await self.memory.format_for_prompt(query=query, scope=scope)
            if memories:
                preamble += f"\n\n<memories>\n{memories}\n</memories>"
        except Exception:
            log.exception("Failed to load memories for turn preamble")

        # Roster of personae the agent can delegate to via spawn_subagent, so its
        # choice is informed rather than guessed (#15). Only on the main turn —
        # selection stays user-led (omit persona = run as yourself / the bound one).
        if offer_personae:
            roster = await self._personae_roster_block(persona)
            if roster:
                preamble += f"\n\n{roster}"

        if self.config.task_reflection.enabled:
            try:
                reflections = await self.reflections.format_for_prompt()
                if reflections:
                    preamble += f"\n\n<task_reflections>\n{reflections}\n</task_reflections>"
            except Exception:
                log.exception("Failed to load task reflections for turn preamble")

        if decomposed_goal:
            preamble += (
                "\n\n<execution_plan>\n"
                "Your request has been analysed and broken into the following sub-goals.\n"
                "Follow this plan step-by-step, completing each sub-goal in order "
                "(respecting dependencies). Report progress as you go.\n\n"
                f"{decomposed_goal.format_for_prompt()}\n"
                "</execution_plan>"
            )
        return preamble

    async def _personae_roster_block(self, persona: Persona | None) -> str:
        """Compact `name — role` roster of personae the agent can delegate to (#15).

        Makes specialist delegation an informed choice instead of a guess, while
        leaving selection user-led: omitting ``persona`` runs the subagent as the
        caller itself. Returns "" (nothing injected) when subagents are disabled,
        the active persona can't spawn, or there is no one to delegate to.
        """
        if not self.config.subagents.enabled:
            return ""
        if persona is not None and not persona.allows_tool("spawn_subagent"):
            return ""
        try:
            personae = await self.personae.list_personae()
        except Exception:
            log.exception("Failed to list personae for the subagent roster")
            return ""
        current = persona.name if persona else ""
        lines = []
        for p in personae:
            role = p.role.strip().splitlines()[0].strip() if (p.role or "").strip() else ""
            tag = " (you)" if p.name == current else ""
            lines.append(f"- {p.name}{tag}" + (f" — {role}" if role else ""))
        if not lines:
            return ""
        body = "\n".join(lines)
        return (
            "<personae>\n"
            "These personae exist ONLY so you can honour an explicit request for a "
            "specialist. By default, spawn_subagent with NO 'persona' so the "
            "subagent runs as you — do not assign one of these unless the user "
            "asked for it or the subtask plainly belongs to it.\n"
            f"{body}\n"
            "</personae>"
        )

    async def _skills_block_in_history(self, session_key: tuple[str, str, str], block: str) -> bool:
        """True if the exact ``<available_skills>`` block is already present in the
        replayed session history — so the model still sees it and we needn't
        re-send it this turn (#46 follow-up).

        Reads the same message array that will be sent to the model, so the
        decision is correct by construction: after a /new or compaction the block
        is gone (→ re-send), a persona rebind or new skill changes the block (→
        re-send), and concurrent turns that haven't yet persisted both re-send
        (harmless). Cheap: a substring scan over the (compaction-bounded) history.
        """
        try:
            messages = await self.history.get_session(*session_key)
        except Exception:
            return False  # safe direction: re-send rather than risk a blind turn
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                if block in content:
                    return True
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and block in str(part.get("text", "")):
                        return True
        return False

    def _subagent_status_note(self, channel: str, chat_id: str) -> str:
        """List background subagents from this chat that are *still running*, for
        the turn preamble — so the agent knows what is pending. When the whole
        batch finishes you'll be prompted to answer the user with their results,
        so finished runs need no mention here. (#15)
        """
        runs = self.subagents.running_for(channel, chat_id)
        if not runs:
            return ""
        lines = []
        for r in runs:
            who = f"- [{r.run_id}] {r.persona or 'default'} — running ({r.elapsed_str})"
            lines.append(f"{who}; {r.progress}" if r.progress else who)
        body = "\n".join(lines)
        return (
            "<background_subagents>\n"
            "Background helpers you spawned from this chat are still running. When "
            "they finish you'll be prompted to fold their results into a reply, so "
            "don't pre-empt or invent their results now, and don't claim one is "
            "finished while it shows here.\n"
            f"{body}\n"
            "</background_subagents>"
        )

    async def _session_system_prompt(
        self,
        channel: str,
        user_id: str,
        chat_id: str,
        persona: Persona | None = None,
    ) -> str:
        """Return the session's static system prompt, building it once if needed.

        Built fresh after a ``/new`` (when no snapshot exists), then reused for
        the lifetime of the session so the static content is sent only once.
        The prompt is purely static now — memory/reflections are injected per
        turn in the preamble (#41), so the snapshot never goes stale.
        """
        cached = await self.history.get_session_system(channel, user_id, chat_id)
        if cached is not None:
            return cached
        system = await self._build_system_prompt(persona=persona)
        await self.history.set_session_system(channel, user_id, system, chat_id)
        return system

    async def _maybe_compact(
        self, channel: str, user_id: str, chat_id: str, response: Any
    ) -> str | None:
        """Compact the session if the context exceeds the configured threshold.

        Returns a user-facing notice when compaction happened, else ``None``.
        Failures are logged and swallowed — compaction must never break a turn.
        """
        cfg = self.config.compaction
        if self.history_mode != "session" or not cfg.enabled:
            return None
        usage = getattr(response, "usage", None) or {}
        context_tokens = int(usage.get("context_tokens") or 0)
        if not should_compact(cfg, context_tokens, self.config.agent.model):
            return None

        session = await self.history.get_session(channel, user_id, chat_id)
        try:
            llm = self._background_llm(cfg.provider, cfg.thinking_level)
            result = await compact_messages(llm, cfg.model, session, cfg.keep_recent_turns)
        except Exception:
            log.exception("Conversation compaction failed")
            return None
        if not result:
            return None

        new_messages, _summary = result
        await self.history.replace_session(channel, user_id, new_messages, chat_id)
        log.info(
            "Compacted session %s/%s/%s: %d → %d messages (~%d ctx tokens)",
            channel,
            user_id,
            chat_id,
            len(session),
            len(new_messages),
            context_tokens,
        )
        return (
            f"🗜️ Our conversation was getting large (~{context_tokens:,} tokens). "
            "I summarized the earlier part to free up space; recent messages are kept as-is."
        )

    async def _build_user_message(
        self,
        message: str,
        attachments: list[Attachment] | None = None,
        preamble: str = "",
    ) -> dict:
        """Build the user message dict, handling multimodal content.

        ``preamble`` (live date/time + optional execution plan) is prepended to
        the message text so the agent always knows 'now' for the current turn.

        When the active model can't see images and a vision fallback is
        configured, images are captioned by a secondary model and the text is
        injected in place of the image blocks so the model can still "see".
        """
        text = f"{preamble}\n\n{message}" if preamble else message
        image_attachments = [a for a in (attachments or []) if a.is_image]
        if image_attachments:
            if self._vision_fallback_active():
                captions = await self._caption_images(image_attachments, message)
                if captions:
                    text = "\n\n".join([text, *captions]) if text else "\n\n".join(captions)
                    return {"role": "user", "content": text}
                # Captioning failed entirely — fall through to native image blocks.
            content_blocks: list[dict] = []
            if text:
                content_blocks.append({"type": "text", "text": text})
            for att in image_attachments:
                if self.llm.provider == "anthropic":
                    content_blocks.append(att.to_anthropic_block())
                else:
                    content_blocks.append(att.to_openai_block())
            return {"role": "user", "content": content_blocks}
        return {"role": "user", "content": text}

    def _vision_fallback_active(self) -> bool:
        """True when the active model lacks vision and a fallback is enabled."""
        return self.config.vision.enabled and not model_supports_vision(
            self.llm.provider, self.config.agent.model
        )

    async def _caption_images(self, images: list[Attachment], user_text: str) -> list[str]:
        """Caption each image with a task-aware prompt, returning ``[Image: ...]``
        strings. Returns ``[]`` on any failure so the caller can fall back to
        passing the raw image blocks through. Captions are cached by image hash.
        """
        vis = self.config.vision
        llm = self._vision_llm(vis.provider)
        out: list[str] = []
        for att in images:
            key = hashlib.sha256(att.data).hexdigest()
            cached = self._vision_cache.get(key)
            if cached is not None:
                self._vision_cache.move_to_end(key)
                out.append(cached)
                continue
            try:
                caption = await self._caption_one(llm, vis.model, att, user_text)
            except Exception:
                log.exception("Vision fallback captioning failed")
                return []
            entry = f"[Image: {caption}]"
            self._vision_cache[key] = entry
            # ponytail: bounded LRU, drop oldest past the cap — fine for a single
            # process; swap for a shared store only if multi-instance dedup matters.
            if len(self._vision_cache) > _VISION_CACHE_MAX:
                self._vision_cache.popitem(last=False)
            out.append(entry)
        return out

    async def _caption_one(
        self, llm: LLMClient, model: str, att: Attachment, user_text: str
    ) -> str:
        """Caption a single image via the vision model. Task-aware: the user's
        message steers what to extract (e.g. OCR vs. scene description)."""
        block = att.to_anthropic_block() if llm.provider == "anthropic" else att.to_openai_block()
        system = (
            "You caption images for a model that cannot see them. "
            "Describe the image so the reader understands it, and transcribe any "
            "visible text verbatim (OCR). Be concise but complete."
        )
        ask = "Describe this image."
        if user_text.strip():
            ask += f' The user sent it with this message: "{user_text.strip()}" — '
            ask += "focus on what is relevant to that."
        messages = [{"role": "user", "content": [block, {"type": "text", "text": ask}]}]
        response = await llm.generate(model=model, system=system, messages=messages, tools=[])
        return response.text.strip() or "(no description available)"

    def _vision_llm(self, provider: str) -> LLMClient:
        """Return an LLM client for image captioning (mirrors ``_memory_llm``)."""
        return self._background_llm(provider)

    async def _process_injection(
        self,
        system: str,
        preamble: str,
        message: str,
        channel: str,
        user_id: str,
        attachments: list[Attachment] | None = None,
        chat_id: str = "",
        tools: list[dict] | None = None,
        persona: Persona | None = None,
    ) -> AgentResponse:
        """Injection mode: replay windowed history as native alternating messages."""
        tools = tools if tools is not None else TOOLS
        history = await self.history.get_messages(channel, user_id, chat_id)
        messages: list[dict] = []

        if history:
            # Replay history as proper alternating user/assistant messages
            for turn in history:
                messages.append({"role": turn["role"], "content": turn["content"]})

        # The actual current request — always the last user message.
        messages.append(await self._build_user_message(message, attachments, preamble))

        log.info(
            "Processing message (injection) from %s/%s/%s: %s",
            channel,
            user_id,
            chat_id,
            message[:100],
        )

        # Initial LLM call
        response = await self.llm.generate(
            model=self.config.agent.model,
            max_tokens=4096,
            system=system,
            messages=messages,
            tools=cast(Any, tools),
        )

        # Agentic loop — keep going while the LLM wants to call tools
        request_state = self._new_request_state(
            persona, origin={"channel": channel, "user_id": user_id, "chat_id": chat_id}
        )
        tool_log: list[dict] = []
        while response.tool_calls:
            await self._batch_approve_writes(response.tool_calls, channel, user_id, request_state)
            tool_results = []
            for call in response.tool_calls:
                result = await self._execute_tool(call, channel, user_id, request_state)
                tool_log.append({"name": call.name, "args": call.arguments, "result": result})
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": json.dumps(result),
                    }
                )

            # Feed tool results back to the LLM
            messages.append(self.llm.assistant_message(response))
            messages.extend(self.llm.tool_result_messages(tool_results))
            response = await self.llm.generate(
                model=self.config.agent.model,
                max_tokens=4096,
                system=system,
                messages=messages,
                tools=cast(Any, tools),
            )
        final_text = response.text
        log.info("Response: %s", final_text[:200])

        # Check if the LLM wants to respond with voice
        voice_bytes = await self._maybe_synthesize_voice(
            final_text, voice=persona.voice if persona else None
        )
        if voice_bytes:
            final_text = final_text.replace("[respond_with_voice]", "").strip()

        # Persist the turn (user message + final assistant text only)
        history_message = self._history_message_text(message, attachments)
        await self.history.add_turn(channel, user_id, "user", history_message, chat_id)
        await self.history.add_turn(channel, user_id, "assistant", final_text, chat_id)

        # Automatic memory extraction
        if channel != "system":
            asyncio.create_task(
                self._extract_memories(message, final_text, persona),
                name=f"memory-extract-{user_id}",
            )

        # Automatic task reflection (when tools were used)
        if channel != "system" and self.config.task_reflection.enabled and tool_log:
            asyncio.create_task(
                self._reflect_on_task(message, final_text, tool_log),
                name=f"task-reflect-{user_id}",
            )

        return AgentResponse(text=final_text, voice=voice_bytes)

    async def _process_session(
        self,
        system: str,
        preamble: str,
        message: str,
        channel: str,
        user_id: str,
        attachments: list[Attachment] | None = None,
        chat_id: str = "",
        tools: list[dict] | None = None,
        persona: Persona | None = None,
    ) -> AgentResponse:
        """Session mode: sticky session per (channel, user_id, chat_id).

        The full message array is kept in memory and persisted to SQLite.
        New messages are appended, giving the LLM full conversational
        continuity with a cache-friendly prefix.
        """
        tools = tools if tools is not None else TOOLS

        # Load existing session (from memory cache or DB)
        session = await self.history.get_session(channel, user_id, chat_id)

        # Append the new user message (with the live date/time preamble)
        user_msg = await self._build_user_message(message, attachments, preamble)
        await self.history.append_session_message(channel, user_id, user_msg, chat_id)

        log.info(
            "Processing message (session) from %s/%s/%s: %s",
            channel,
            user_id,
            chat_id,
            message[:100],
        )

        # Initial LLM call with the full session
        response = await self.llm.generate(
            model=self.config.agent.model,
            max_tokens=4096,
            system=system,
            messages=session,
            tools=cast(Any, tools),
        )

        # Agentic loop — keep going while the LLM wants to call tools
        new_messages: list[dict] = []
        request_state = self._new_request_state(
            persona, origin={"channel": channel, "user_id": user_id, "chat_id": chat_id}
        )
        tool_log: list[dict] = []
        while response.tool_calls:
            await self._batch_approve_writes(response.tool_calls, channel, user_id, request_state)
            tool_results = []
            for call in response.tool_calls:
                result = await self._execute_tool(call, channel, user_id, request_state)
                tool_log.append({"name": call.name, "args": call.arguments, "result": result})
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": json.dumps(result),
                    }
                )

            # Append tool exchange to session
            assistant_msg = self.llm.assistant_message(response)
            tool_result_msgs = self.llm.tool_result_messages(tool_results)

            new_messages.append(assistant_msg)
            new_messages.extend(tool_result_msgs)

            # The in-memory session list is mutated by append_session_messages
            # so the next generate() call sees the updated messages.
            await self.history.append_session_messages(
                channel, user_id, [assistant_msg, *tool_result_msgs], chat_id
            )

            response = await self.llm.generate(
                model=self.config.agent.model,
                max_tokens=4096,
                system=system,
                messages=session,
                tools=cast(Any, tools),
            )

        # Append the final assistant response to the session
        final_assistant_msg = {"role": "assistant", "content": response.text}
        await self.history.append_session_message(channel, user_id, final_assistant_msg, chat_id)

        final_text = response.text
        log.info("Response: %s", final_text[:200])

        # Compaction — if the context has grown past the configured threshold,
        # summarise the oldest turns. ``response.usage`` reflects the full
        # session that was just sent, so it's the authoritative context size.
        system_notice = await self._maybe_compact(channel, user_id, chat_id, response)

        # Check if the LLM wants to respond with voice
        voice_bytes = await self._maybe_synthesize_voice(
            final_text, voice=persona.voice if persona else None
        )
        if voice_bytes:
            final_text = final_text.replace("[respond_with_voice]", "").strip()

        # Automatic memory extraction
        if channel != "system":
            asyncio.create_task(
                self._extract_memories(message, final_text, persona),
                name=f"memory-extract-{user_id}",
            )

        # Automatic task reflection (when tools were used)
        if channel != "system" and self.config.task_reflection.enabled and tool_log:
            asyncio.create_task(
                self._reflect_on_task(message, final_text, tool_log),
                name=f"task-reflect-{user_id}",
            )

        return AgentResponse(text=final_text, voice=voice_bytes, system_notice=system_notice)

    @staticmethod
    def _history_message_text(message: str, attachments: list[Attachment] | None = None) -> str:
        """Build the text to store in history for a user message."""
        image_attachments = [a for a in (attachments or []) if a.is_image]
        if image_attachments:
            n = len(image_attachments)
            label = "image" if n == 1 else f"{n} images"
            suffix = f" [{label} attached]"
            return (message + suffix) if message else suffix.strip()
        return message

    async def _maybe_synthesize_voice(self, text: str, voice: str | None = None) -> bytes | None:
        """Synthesize voice if requested by the LLM, using the persona's voice
        when one is set (else the configured default)."""
        if "[respond_with_voice]" in text and self.voice:
            clean_text = text.replace("[respond_with_voice]", "").strip()
            try:
                return await self.voice.synthesize(clean_text, voice=voice)
            except Exception:
                log.exception("TTS synthesis failed, sending text only")
        return None

    async def _execute_tool(
        self,
        tool_call: LLMToolCall,
        channel: str,
        user_id: str,
        request_state: dict | None = None,
    ) -> dict:
        """Dispatch a tool call from the LLM, with permission checks."""
        name = tool_call.name
        params = tool_call.arguments

        if request_state is None:
            request_state = self._new_request_state()

        is_write_action = self.permissions.is_write_action(name, params)
        # Write-state is tracked per distinct action (tool + params), so a
        # failure, skip, or completion of one write never blocks a different one.
        write_sig = self._write_signature(name, params) if is_write_action else None
        executed_writes = request_state.setdefault("executed_writes", set())
        write_decisions = request_state.setdefault("write_decisions", {})
        # ``manage_jobs`` is exempt: job creation is idempotent and guarded on
        # job id + status inside the tool, so an earlier write in the same turn
        # must never block a (re)create — that was the "already fulfilled" bug
        # against brand-new job ids (issue #11). ``spawn_subagent`` is likewise
        # exempt: each spawn is a distinct run (its own run id), so the agent may
        # legitimately fan out the same task more than once in a turn (#15).
        if (
            is_write_action
            and name not in ("manage_jobs", "spawn_subagent")
            and write_sig in executed_writes
        ):
            return {
                "error": (
                    "This exact action was already completed in this request; not repeating it."
                )
            }
        if is_write_action and write_decisions.get(write_sig) == "denied":
            return {"error": "Action denied by user."}
        if is_write_action and write_decisions.get(write_sig) == "skipped":
            return {
                "error": (
                    "User skipped this action. "
                    "Do not retry this exact action — "
                    "move on to something else."
                )
            }

        # --- Permission check ---
        level = self.permissions.check(name, params)

        if level == PermissionLevel.NEVER:
            log.warning("Permission DENIED (NEVER): %s — %s", name, params)
            return {"error": "This action is not allowed."}

        if level == PermissionLevel.ASK and channel != "system":
            match_key = self.permissions.match_key(name, params)
            approvals = request_state.get("approvals", {})
            if is_write_action and write_sig in write_decisions:
                # Same write asked earlier in this turn — reuse that decision
                # rather than prompting again, but only for the identical action.
                decision = write_decisions[write_sig]
            elif not is_write_action and isinstance(approvals, dict) and match_key in approvals:
                decision = approvals[match_key]
            else:
                decision = await self._request_approval(name, params, channel, user_id)
                if is_write_action:
                    write_decisions[write_sig] = decision
                elif isinstance(approvals, dict):
                    approvals[match_key] = decision
                    request_state["approvals"] = approvals
            if decision == "skipped":
                log.info("Permission SKIPPED by user: %s", name)
                return {
                    "error": (
                        "User skipped this action. "
                        "Do not retry this action or attempt similar alternatives — "
                        "move on to something else."
                    )
                }
            if decision != "approved":
                log.info("Permission DENIED (user rejected): %s", name)
                return {"error": "Action denied by user."}

            if not is_write_action:
                self.permissions.add_rule(
                    self.permissions.match_key(name, params),
                    PermissionLevel.ALWAYS,
                )

        # --- Dispatch ---
        if name == "run_command":
            log.info("Tool call: run_command — %s", params.get("purpose", ""))
            command = params["command"]
            # Secret substitution boundary (issue #19): {{secret:NAME}} is resolved
            # ONLY here, for the model's generic command tool, after an ACL check.
            # Structured tools (send_email/send_message/…) build their commands
            # elsewhere and never pass through this path, so a secret cannot be
            # exfiltrated through a message/email body.
            if self.secret_store is not None:
                allowed = set(request_state.get("persona_secrets") or [])
                command, serr = await self.secret_store.resolve_command_secrets(command, allowed)
                if serr:
                    return {"error": serr}
            return await self.executor.run_command(command)

        if name == "send_email":
            result = await self._tool_send_email(params)
            if is_write_action and self._is_tool_success(result):
                executed_writes.add(write_sig)
            return result

        if name == "reply_email":
            result = await self._tool_reply_email(params)
            if is_write_action and self._is_tool_success(result):
                executed_writes.add(write_sig)
            return result

        if name == "send_message":
            result = await self._tool_send_message(params)
            if is_write_action and self._is_tool_success(result):
                executed_writes.add(write_sig)
            return result

        if name == "create_calendar_event":
            result = await self._tool_create_calendar_event(params)
            if is_write_action and self._is_tool_success(result):
                executed_writes.add(write_sig)
            return result

        if name == "web_search":
            log.info("Tool call: web_search — %s", params.get("query", ""))
            return await self._tool_web_search(params)

        if name == "load_skill":
            skill_name = str(params.get("name", "")).strip()
            if not skill_name:
                return {"error": "Missing skill name."}
            allowed = (request_state or {}).get("allowed_skills")
            if allowed and skill_name not in allowed:
                return {"error": f"Skill '{skill_name}' is not available to the active persona."}
            content = await self.skills.get_skill_content(skill_name)
            if not content:
                return {"error": f"Skill not found: {skill_name}"}
            return {"name": skill_name, "content": content}

        if name == "recall_memory":
            return await self._tool_recall_memory(params, request_state)

        if name == "manage_jobs":
            log.info("Tool call: manage_jobs — %s", params.get("action", ""))
            result = await self._tool_manage_jobs(params)
            if is_write_action and self._is_tool_success(result):
                executed_writes.add(write_sig)
            return result

        if name == "list_secrets":
            if self.secret_store is None:
                return {"error": "Secrets vault is not configured."}
            allowed = set(request_state.get("persona_secrets") or [])
            allowed |= await self.secret_store.shared_names()
            meta = await self.secret_store.list_secret_meta(allowed=allowed)
            return {
                "secrets": [
                    {
                        "name": m["name"],
                        "description": m["description"],
                        "shared": m["shared"],
                        "structured": m["structured"],
                        "last_used_at": m["last_used_at"],
                    }
                    for m in meta
                ]
            }

        if name == "spawn_subagent":
            result = await self._tool_spawn_subagent(params, channel, user_id, request_state)
            if is_write_action and self._is_tool_success(result):
                executed_writes.add(write_sig)
            return result

        if name == "request_secret":
            return await self._tool_request_secret(params, channel, user_id, request_state)

        if name == "write_artifact":
            result = self._tool_write_artifact(params)
            if is_write_action and self._is_tool_success(result):
                executed_writes.add(write_sig)
            return result

        return {"error": f"Unknown tool: {name}"}

    @staticmethod
    def _new_request_state(
        persona: Persona | None = None,
        *,
        depth: int = 0,
        origin: dict | None = None,
        run_id: str | None = None,
    ) -> dict:
        """Fresh per-turn state tracking write actions and approval decisions.

        ``allowed_skills`` carries the active persona's skill allowlist so
        ``load_skill`` can refuse skills outside scope (defence in depth — the
        index already hides them). ``depth``/``origin``/``persona_obj`` carry the
        context a ``spawn_subagent`` call needs to narrow scope, cap recursion,
        and post a background result back to the originating chat (issue #15).
        """
        return {
            "executed_writes": set(),
            "write_decisions": {},
            "approvals": {},
            "allowed_skills": persona.skills if persona else None,
            # Secret scope for {{secret:}} ACL in run_command (issue #19).
            "persona_secrets": list(persona.secrets) if persona else [],
            "persona_name": persona.name if persona else "",
            # Subagent plumbing (issue #15).
            "persona_obj": persona,
            "depth": depth,
            "origin": origin or {},
            "run_id": run_id,
        }

    @staticmethod
    def _write_signature(name: str, params: dict) -> str:
        """Stable signature for a write action, keyed on tool name + arguments.

        Two calls share a signature only when they would perform the identical
        write, so deduplication and remembered skip/deny decisions apply per
        action rather than blocking every write after the first.
        """
        try:
            payload = json.dumps(params, sort_keys=True, default=str)
        except Exception:
            payload = repr(params)
        return f"{name}:{payload}"

    @staticmethod
    def _is_tool_success(result: dict) -> bool:
        if not isinstance(result, dict):
            return False
        if "error" in result:
            return False
        if "exit_code" in result:
            return result.get("exit_code") == 0
        if "ok" in result:
            return result.get("ok") is True
        return True

    # -- Structured tool implementations --

    async def _tool_send_email(self, params: dict) -> dict:
        """Send an email via himalaya CLI."""
        account = params["account"]
        to = params["to"]
        subject = params["subject"]
        body = params["body"]
        cc = params.get("cc")
        bcc = params.get("bcc")
        from_addr = params.get("from")
        log.info("Tool call: send_email — to=%s subject=%s", to, subject)

        # Build MML message headers
        headers = []
        if from_addr:
            headers.append(f"From: {from_addr}")
        headers.append(f"To: {to}")
        if cc:
            headers.append(f"Cc: {cc}")
        if bcc:
            headers.append(f"Bcc: {bcc}")
        headers.append(f"Subject: {subject}")
        mml = "\n".join(headers) + "\n\n" + body

        command = (
            f"printf %s {_shell_quote(mml)} | himalaya -a {_shell_quote(account)} message send"
        )
        return await self.executor.run_command_trusted(command)

    async def _tool_reply_email(self, params: dict) -> dict:
        """Reply to an email via himalaya CLI."""
        account = params["account"]
        message_id = params["message_id"]
        body = params["body"]
        reply_all = params.get("reply_all", False)
        folder = params.get("folder")
        log.info("Tool call: reply_email — account=%s message=%s", account, message_id)

        cmd_parts = [f"printf %s {_shell_quote(body)} | himalaya -a {_shell_quote(account)}"]
        if folder:
            cmd_parts.append(f"--folder {_shell_quote(folder)}")
        cmd_parts.append("message reply")
        if reply_all:
            cmd_parts.append("--all")
        cmd_parts.append(_shell_quote(message_id))

        return await self.executor.run_command_trusted(" ".join(cmd_parts))

    async def _tool_send_message(self, params: dict) -> dict:
        """Send a message via a registered channel."""
        channel_name = params["channel"]
        to = params["to"]
        text = params["text"]
        log.info("Tool call: send_message — channel=%s to=%s", channel_name, to)

        channel = self.channels.get(channel_name)
        if not channel:
            return {"error": f"Channel '{channel_name}' is not enabled."}

        try:
            await channel.send(to, text)
            return {"ok": True, "channel": channel_name, "to": to}
        except Exception as exc:
            return {"error": str(exc)}

    def _base_url(self) -> str:
        import os

        return os.getenv("MPA_BASE_URL", f"http://localhost:{self.config.admin.port}")

    def _tool_write_artifact(self, params: dict) -> dict:
        """Publish a web artifact (page, multi-file site, or file); return its URL."""
        from core.artifacts import ArtifactStore

        cfg = self.config.artifacts
        if not cfg.enabled:
            return {"error": "Web artifacts are disabled in config (artifacts.enabled)."}

        files = params.get("files") or None
        if files is not None and not isinstance(files, dict):
            return {"error": "'files' must be a map of filename → text content."}
        html = params.get("html")
        if html and not files:
            files = {"index.html": str(html)}
        source_path = params.get("source_path") or None
        if not files and not source_path:
            return {"error": "Provide one of 'html', 'files', or 'source_path'."}
        if files and source_path:
            return {"error": "Provide only one of 'html'/'files' or 'source_path'."}

        entrypoint = str(params.get("entrypoint") or "index.html")
        ttl_hours = params.get("ttl_hours")
        if ttl_hours is not None:
            try:
                ttl_hours = int(ttl_hours)
            except TypeError, ValueError:
                return {"error": "'ttl_hours' must be an integer (0 = keep forever)."}
        title = str(params.get("title", "")).strip()

        store = ArtifactStore(cfg.directory, cfg.ttl_hours)
        try:
            art_id = store.create(
                files=files,
                source_path=source_path,
                entrypoint=entrypoint,
                ttl_hours=ttl_hours,
                title=title,
            )
        except (ValueError, OSError) as exc:
            return {"error": str(exc)}
        url = f"{self._base_url()}/artifacts/{art_id}/"
        log.info("Tool call: write_artifact — %s (%s)", url, title or "untitled")
        return {"ok": True, "url": url, "title": title}

    async def _notify_secret_request(
        self, channel: str, user_id: str, name: str, reason: str, link: str
    ) -> None:
        """Best-effort: push the owner a secure link to provide a requested secret.

        Link only — the value is NEVER entered or shown over the chat channel.
        """
        text = (
            f"🔑 I need the secret '{name}' to continue"
            + (f" ({reason})" if reason else "")
            + ".\nAdd it securely via this link (no value over chat):\n"
            + link
        )
        ch = self.channels.get(channel)
        if ch is None:
            return
        try:
            await ch.send(user_id, text)
        except Exception:
            log.exception("Failed to send secret-request link via %s", channel)

    async def _tool_request_secret(
        self, params: dict, channel: str, user_id: str, request_state: dict | None
    ) -> dict:
        """Create a pending secret request and send the owner a secure fill link."""
        if self.secret_store is None:
            return {"error": "Secrets vault is not configured."}
        from core.secret_store import valid_name

        sname = str(params.get("name", "")).strip()
        if not valid_name(sname):
            return {"error": "Invalid secret name (use letters, digits, _ - : only)."}
        reason = str(params.get("reason", "")).strip()
        scope = str(params.get("suggested_scope", "")).strip()
        persona_name = (request_state or {}).get("persona_name", "")
        log.info("Tool call: request_secret — %s (persona=%s)", sname, persona_name)
        token = await self.secret_store.create_request(
            sname, persona=persona_name, reason=reason, suggested_scope=scope
        )
        link = f"{self._base_url()}/vault/fill/{token}"
        await self._notify_secret_request(channel, user_id, sname, reason, link)
        return {
            "status": "requested",
            "secure_link": link,
            "message": (
                f"Sent the owner a secure link to provide '{sname}'. It is not available "
                "yet — let the user know, then continue once they confirm it's added."
            ),
        }

    async def _tool_create_calendar_event(self, params: dict) -> dict:
        """Create a calendar event via the CalDAV helper script."""
        calendar = params["calendar"]
        summary = params["summary"]
        start = params["start"]
        end = params["end"]
        attendees = params.get("attendees", [])
        log.info("Tool call: create_calendar_event — %s on %s", summary, calendar)

        cmd_parts = [
            "python3 /app/tools/calendar_write.py",
            f"--calendar {_shell_quote(calendar)}",
            f"--summary {_shell_quote(summary)}",
            f"--start {_shell_quote(start)}",
            f"--end {_shell_quote(end)}",
        ]
        for addr in attendees:
            cmd_parts.append(f"--attendee {_shell_quote(addr)}")

        return await self.executor.run_command(" ".join(cmd_parts))

    async def _tool_manage_jobs(self, params: dict) -> dict:
        """Create, list, or cancel scheduled jobs via the JobStore."""
        action = params.get("action", "")

        if action == "list":
            jobs = await self.job_store.list_jobs()
            return {
                "ok": True,
                "jobs": [
                    {
                        "id": j["id"],
                        "type": j["type"],
                        "schedule": j["schedule"],
                        "cron": j.get("cron"),
                        "run_at": j.get("run_at"),
                        "task": j["task"],
                        "channel": j["channel"],
                        "status": j["status"],
                        "description": j.get("description", ""),
                        "created_by": j.get("created_by", ""),
                    }
                    for j in jobs
                ],
            }

        if action == "cancel":
            job_id = params.get("job_id", "").strip()
            if not job_id:
                return {"error": "Missing job_id for cancel action."}
            existing = await self.job_store.get_job(job_id)
            if not existing:
                return {"error": f"Job not found: {job_id}"}
            await self.job_store.update_status(job_id, "cancelled")
            await self.scheduler.sync_job(job_id)
            return {"ok": True, "cancelled": job_id}

        if action == "create":
            task = params.get("task", "").strip()
            if not task:
                return {"error": "Missing 'task' for create action."}

            job_id = params.get("job_id", "").strip()
            if not job_id:
                job_id = f"agent_{uuid.uuid4().hex[:8]}"
            else:
                # Block only when this explicit id is already live (active or
                # paused). Done/cancelled ids may be recreated; auto-generated
                # ids are unique by construction. (issue #11)
                existing = await self.job_store.get_job(job_id)
                if existing and existing["status"] in ("active", "paused"):
                    return {
                        "error": (
                            f"Job already exists and is {existing['status']}: {job_id}. "
                            "Cancel it first or use a different id."
                        )
                    }

            channel = params.get("channel", "telegram")
            description = params.get("description", "")
            cron_expr = params.get("cron")
            run_at_str = params.get("run_at")

            if cron_expr:
                # Recurring cron job
                from core.scheduler import _parse_cron

                try:
                    _parse_cron(cron_expr)
                except ValueError as exc:
                    return {"error": str(exc)}

                await self.job_store.upsert_job(
                    job_id=job_id,
                    type="agent",
                    schedule="cron",
                    cron=cron_expr,
                    task=task,
                    channel=channel,
                    status="active",
                    created_by="agent",
                    description=description,
                )
                await self.scheduler.sync_job(job_id)
                return {
                    "ok": True,
                    "job_id": job_id,
                    "schedule": "cron",
                    "cron": cron_expr,
                    "task": task,
                    "channel": channel,
                }

            elif run_at_str:
                # One-shot job
                try:
                    run_at = datetime.fromisoformat(run_at_str)
                except ValueError:
                    return {"error": f"Invalid datetime format: {run_at_str!r}. Use ISO format."}

                # Treat naive datetimes as being in the configured timezone
                if run_at.tzinfo is None:
                    tz = ZoneInfo(self.config.agent.timezone)
                    run_at = run_at.replace(tzinfo=tz)

                await self.job_store.upsert_job(
                    job_id=job_id,
                    type="agent",
                    schedule="once",
                    run_at=run_at.isoformat(),
                    task=task,
                    channel=channel,
                    status="active",
                    created_by="agent",
                    description=description,
                )
                await self.scheduler.sync_job(job_id)
                return {
                    "ok": True,
                    "job_id": job_id,
                    "schedule": "once",
                    "run_at": run_at.isoformat(),
                    "task": task,
                    "channel": channel,
                }
            else:
                return {"error": "Must specify 'cron' for recurring or 'run_at' for one-time jobs."}

        return {"error": f"Unknown action: {action!r}. Use 'create', 'list', or 'cancel'."}

    async def _tool_recall_memory(self, params: dict, request_state: dict | None = None) -> dict:
        """Deliberate semantic search over the full long-term memory store (#47).

        Scoped to the active persona (#42): ``persona_name`` on the per-turn
        request state is the persona's private memory scope (``""`` = the default
        identity's shared-only view), so recall never crosses into another
        persona's private memories — same boundary the injection readers enforce.
        """
        query = str(params.get("query", "")).strip()
        if not query:
            return {"error": "Missing 'query'."}
        limit = params.get("limit")
        limit = limit if isinstance(limit, int) and not isinstance(limit, bool) else None
        scope = (request_state or {}).get("persona_name") or ""
        try:
            memories = await self.memory.recall(query, limit, scope=scope)
        except Exception:
            log.exception("recall_memory failed for query: %s", query)
            return {"error": "Memory recall failed."}
        log.info("Tool call: recall_memory — %r (%d hits)", query, len(memories))
        return {"query": query, "count": len(memories), "memories": memories}

    # -- Subagents (issue #15) ------------------------------------------------

    async def _tool_spawn_subagent(
        self, params: dict, channel: str, user_id: str, request_state: dict
    ) -> dict:
        """``spawn_subagent`` tool: delegate a scoped subtask to a sub-loop."""
        task = str(params.get("task", "")).strip()
        if not task:
            return {"error": "Missing 'task' for spawn_subagent."}
        origin = request_state.get("origin") or {}
        return await self.run_subagent(
            task=task,
            persona_name=str(params.get("persona", "")).strip(),
            origin_channel=origin.get("channel", channel),
            origin_user_id=str(origin.get("user_id", user_id)),
            origin_chat_id=str(origin.get("chat_id", "")),
            parent_state=request_state,
            background=bool(params.get("background", False)),
            max_steps=params.get("max_steps"),
            token_budget=params.get("token_budget"),
            thinking_effort=params.get("thinking_effort"),
        )

    async def run_subagent(
        self,
        *,
        task: str,
        persona_name: str = "",
        origin_channel: str = "",
        origin_user_id: str = "",
        origin_chat_id: str = "",
        parent_state: dict | None = None,
        background: bool = False,
        max_steps: object = None,
        token_budget: object = None,
        thinking_effort: str | None = None,
    ) -> dict:
        """Run a subagent — the one primitive behind both the tool and scheduled
        ``subagent`` jobs. Scope is narrowed from the caller (inherit-never-widen);
        recursion depth and per-run budgets are enforced.

        ``max_steps`` / ``token_budget`` / ``thinking_effort`` let the caller size
        the run; each defaults to the configured value and is clamped to it as a
        ceiling (``thinking_effort`` defaults to inheriting the caller's level).
        """
        cfg = self.config.subagents
        if not cfg.enabled:
            return {"error": "Subagents are disabled."}
        parent_state = parent_state or {}
        parent_depth = int(parent_state.get("depth", 0) or 0)
        if parent_depth >= cfg.recursion_depth:
            return {
                "error": (
                    f"Max subagent recursion depth ({cfg.recursion_depth}) reached; "
                    "do this work directly instead of spawning another subagent."
                )
            }

        # Resolve + narrow the persona. A name must exist; with no name the child
        # inherits the caller's identity and scope.
        if persona_name:
            requested = await self._load_persona(persona_name)
            if requested is None:
                try:
                    names = [p.name for p in await self.personae.list_personae()]
                except Exception:
                    names = []
                hint = f" Available: {', '.join(names)}." if names else ""
                return {
                    "error": (
                        f"Persona not found: {persona_name}.{hint} "
                        "Omit 'persona' to run as yourself."
                    )
                }
        else:
            requested = parent_state.get("persona_obj")
        child_persona = self._narrow_persona(requested, parent_state) if requested else None

        run_id = f"sub_{uuid.uuid4().hex[:8]}"
        child_state = self._new_request_state(
            child_persona,
            depth=parent_depth + 1,
            origin={
                "channel": origin_channel,
                "user_id": origin_user_id,
                "chat_id": origin_chat_id,
            },
            run_id=run_id,
        )
        run = SubagentRun(
            run_id=run_id,
            persona=child_persona.name if child_persona else "",
            task=task,
            depth=parent_depth + 1,
            background=background,
            max_steps=resolve_cap(max_steps, cfg.max_steps),
            token_budget=resolve_cap(token_budget, cfg.token_budget, floor=1000),
            effort=normalize_effort(thinking_effort),
            origin_channel=origin_channel,
            origin_user_id=origin_user_id,
            origin_chat_id=origin_chat_id,
        )

        if background:
            if self.subagents.active_count() >= cfg.max_concurrent:
                return {
                    "error": (
                        f"Too many subagents running (max {cfg.max_concurrent}). "
                        "Wait for one to finish or cancel it via /jobs."
                    )
                }
            self.subagents.register(run)
            bg = asyncio.create_task(
                self._run_subagent_background(run, child_persona, child_state),
                name=f"subagent-{run_id}",
            )
            self.subagents.attach_task(run_id, bg)
            log.info(
                "Spawned background subagent %s (persona=%s)", run_id, run.persona or "default"
            )
            return {
                "ok": True,
                "run_id": run_id,
                "background": True,
                "status": "running",
                "persona": run.persona,
                "note": (
                    "Running in the background; its result is posted to this chat "
                    "automatically when done — you don't relay it. Each later turn "
                    "shows this run's status until it finishes."
                ),
            }

        # Synchronous: run to completion and return the result to the caller.
        self.subagents.register(run)
        log.info("Running subagent %s (persona=%s)", run_id, run.persona or "default")
        try:
            text = await self._run_subagent_loop(task, child_persona, child_state, run)
        except Exception as exc:
            log.exception("Subagent %s failed", run_id)
            self.subagents.finish(run_id, "error", error=str(exc))
            return {"error": f"Subagent failed: {exc}", "run_id": run_id}
        self.subagents.finish(run_id, "done", result=text)
        return {
            "ok": True,
            "run_id": run_id,
            "persona": run.persona,
            "summary": short_summary(text),
            "result": text,
        }

    def _narrow_persona(self, requested: Persona, parent_state: dict) -> Persona:
        """Build a child persona whose scopes are a subset of the caller's."""
        parent: Persona | None = parent_state.get("persona_obj")
        p_skills = parent.skills if parent else []
        p_tools = parent.tools if parent else []
        p_secrets = parent.secrets if parent else []
        return Persona(
            name=requested.name,
            agent_name=requested.agent_name,
            role=requested.role,
            emoji=requested.emoji,
            voice=requested.voice,
            personalia=requested.personalia,
            character=requested.character,
            skills=narrow_scope(p_skills, requested.skills),
            tools=narrow_scope(p_tools, requested.tools),
            secrets=narrow_scope(p_secrets, requested.secrets),
        )

    async def _run_subagent_loop(
        self, task: str, child_persona: Persona | None, child_state: dict, run: SubagentRun
    ) -> str:
        """The subagent's agentic loop — system semantics, budgeted and depth-capped.

        Mirrors the main injection loop but runs from a clean slate (no history),
        skips approval/decomposition/memory/reflection (channel='system'), and
        stops at this run's step/token budget (sized by the spawning agent).
        """
        cfg = self.config.subagents
        tools = apply_feature_gates(
            scoped_tools(child_persona),
            secrets_available=self.secret_store is not None,
            artifacts_enabled=self.config.artifacts.enabled,
            subagents_enabled=cfg.enabled,
        )
        # At the depth ceiling a subagent may not spawn further — don't even offer it.
        if child_state["depth"] >= cfg.recursion_depth:
            tools = [t for t in tools if t["name"] != "spawn_subagent"]

        system = await self._build_system_prompt(persona=child_persona)
        system = f"{system}\n\n{RESULT_FOR_AGENT_INSTRUCTION}\n\n{FILE_HANDOFF_INSTRUCTION}"
        # Memory/reflections inject per-turn via the preamble (#41), scoped to the
        # child persona (#42); query=task keeps the injection relevant.
        preamble = await self._turn_preamble(None, query=task, scope=_persona_scope(child_persona))
        messages: list[dict] = [await self._build_user_message(task, None, preamble)]

        # effort None = inherit the main client's level; otherwise an effort-scoped
        # clone (same provider/connection, overridden thinking level).
        llm = self.llm
        if run.effort is not None:
            llm = self._background_llm(self.llm.provider, run.effort)
        response = await llm.generate(
            model=self.config.agent.model,
            max_tokens=4096,
            system=system,
            messages=messages,
            tools=cast(Any, tools),
        )
        steps = 0
        tokens = self._usage_total(response.usage)
        while response.tool_calls and steps < run.max_steps and tokens < run.token_budget:
            steps += 1
            run.progress = f"step {steps}: {', '.join(c.name for c in response.tool_calls)}"[:120]
            tool_results = []
            for call in response.tool_calls:
                result = await self._execute_tool(
                    call, "system", run.origin_user_id or "subagent", child_state
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": json.dumps(result),
                    }
                )
            messages.append(llm.assistant_message(response))
            messages.extend(llm.tool_result_messages(tool_results))
            response = await llm.generate(
                model=self.config.agent.model,
                max_tokens=4096,
                system=system,
                messages=messages,
                tools=cast(Any, tools),
            )
            tokens += self._usage_total(response.usage)

        text = response.text or ""
        if response.tool_calls:
            text = (text + "\n\n[subagent stopped: reached its step/token budget]").strip()
        run.progress = "done"
        return text

    async def _run_subagent_background(
        self, run: SubagentRun, child_persona: Persona | None, child_state: dict
    ) -> None:
        """Run a subagent off-turn. When the chat's whole batch of background runs
        has finished, the spawning agent ingests their results and writes one reply
        — the user never sees raw subagent output; it works for the agent (#15)."""
        try:
            text = await self._run_subagent_loop(run.task, child_persona, child_state, run)
        except asyncio.CancelledError:
            # User-initiated stop (registry.cancel already flipped the status, so
            # finish() is a no-op here). Mark it synthesised so a sibling's batch
            # doesn't report on a run the user deliberately cancelled.
            self.subagents.finish(run.run_id, "cancelled")
            run.synthesized = True
            # This may have been the last running sibling: a done/error run that
            # deferred earlier would otherwise be orphaned (its reply lost), since
            # cancellation is the one terminal path that never re-checks the
            # barrier. Release it before unwinding. (Safe to await here: the
            # cancellation was already delivered once and won't re-fire.)
            await self._maybe_deliver_subagent_batch(run)
            raise
        except Exception as exc:
            log.exception("Background subagent %s failed", run.run_id)
            if self.subagents.finish(run.run_id, "error", error=str(exc)):
                await self._maybe_deliver_subagent_batch(run)
            return
        # finish() returns False if a late cancellation already finalised the run,
        # in which case this completion must not also trigger a reply.
        if self.subagents.finish(run.run_id, "done", result=text):
            await self._maybe_deliver_subagent_batch(run)

    async def _maybe_deliver_subagent_batch(self, run: SubagentRun) -> None:
        """Once every background run for this chat is done, distil the batch into a
        chat notification + a context digest and deliver them. The barrier collapses
        a fan-out of parallel spawns into a single delivery (#15)."""
        channel, user_id, chat_id = run.origin_channel, run.origin_user_id, run.origin_chat_id
        if not chat_id or channel == "system":
            return  # scheduler / system-origin runs have no user chat to answer
        # Barrier — race-free because there is no await between this check and
        # marking the batch below: while another background run for the chat is
        # still running, defer; the last finisher delivers. (Sync runs are ignored:
        # they return inline and never reach this path.)
        runs = self.subagents.list_runs()
        if any(
            r.background
            and r.status == "running"
            and r.origin_channel == channel
            and r.origin_chat_id == chat_id
            for r in runs
        ):
            return
        batch = [
            r
            for r in runs
            if r.background
            and not r.synthesized
            and r.origin_channel == channel
            and r.origin_chat_id == chat_id
            and r.status in ("done", "error")
        ]
        if not batch:
            return
        for r in batch:
            r.synthesized = True
        await self._summarize_and_deliver(channel, user_id, chat_id, batch)

    async def _summarize_and_deliver(
        self, channel: str, user_id: str, chat_id: str, batch: list[SubagentRun]
    ) -> None:
        """Distil a finished batch into a one-line chat notification + a concise
        context digest, then deliver: notification → the user, digest → the agent's
        context. The raw subagent output reaches neither the user nor the context.
        """
        notification, digest = await self._summarize_subagent_batch(batch)
        # The user only ever saw the notification; the agent's context keeps the
        # concise digest (so it can answer follow-ups) — never the raw output.
        framed = notification
        if digest and digest.strip() and digest.strip() != notification.strip():
            framed = f"{notification}\n\n<subagent_digest>\n{digest}\n</subagent_digest>"
        await self._record_subagent_context(channel, user_id, chat_id, framed)
        ch = self.channels.get(channel)
        if ch and chat_id and notification:
            try:
                await ch.send(chat_id, notification)
            except Exception:
                log.exception("Failed to deliver subagent notification (chat=%s)", chat_id)

    async def _summarize_subagent_batch(self, batch: list[SubagentRun]) -> tuple[str, str]:
        """(notification, digest) for a finished batch via the summary inference,
        falling back to truncation when it is disabled or the inference fails."""
        items = [
            (
                r.task,
                r.result if r.status == "done" else f"[failed: {r.error or 'unknown error'}]",
                r.persona or "",
                r.status,
            )
            for r in batch
        ]
        cfg = self.config.subagent_summary
        if cfg.enabled:
            try:
                llm = self._background_llm(cfg.provider, cfg.thinking_level)
                return await summarize_batch(llm, cfg.model, items)
            except Exception:
                log.exception("Subagent summary inference failed; using truncation fallback")
        return fallback_summary(items)

    async def _record_subagent_context(
        self, channel: str, user_id: str, chat_id: str, framed: str
    ) -> None:
        """Record a background batch's notification + digest as an assistant turn —
        merged into the trailing assistant turn so replayed history stays strictly
        alternating for providers that require it (#15)."""
        if not chat_id or channel == "system":
            return
        try:
            if self.history_mode == "session":
                merged = await self.history.append_to_last_session_message(
                    channel, user_id, f"\n\n{framed}", chat_id
                )
                if not merged:
                    await self.history.append_session_message(
                        channel, user_id, {"role": "assistant", "content": framed}, chat_id
                    )
            else:
                merged = await self.history.append_to_last_turn(
                    channel, user_id, "assistant", f"\n\n{framed}", chat_id
                )
                if not merged:
                    await self.history.add_turn(channel, user_id, "assistant", framed, chat_id)
        except Exception:
            log.exception("Failed to record subagent context (chat=%s)", chat_id)

    @staticmethod
    def _usage_total(usage: dict | None) -> int:
        """Best-effort token count for budgeting (0 when the provider omits usage)."""
        if not usage:
            return 0
        return int(usage.get("input_tokens", 0) or 0) + int(usage.get("output_tokens", 0) or 0)

    async def _tool_web_search(self, params: dict) -> dict:
        """Search the web via Tavily API."""
        if not self.search_client:
            return {"error": "Web search is not configured. Set search.api_key in config."}

        query = params.get("query", "").strip()
        if not query:
            return {"error": "Empty search query."}

        max_results = self.config.search.max_results

        try:
            response = await asyncio.to_thread(
                self.search_client.search,
                query=query,
                max_results=max_results,
            )
        except Exception as exc:
            log.exception("Tavily search failed for query: %s", query)
            return {"error": f"Search failed: {exc}"}

        # Format results for the LLM
        results = []
        for item in response.get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                }
            )

        return {
            "query": query,
            "results": results,
        }

    async def _request_approval(
        self, tool_name: str, params: dict, channel: str, user_id: str
    ) -> str:
        """Ask the user to approve a single tool call via their channel.

        Returns one of ``"approved"``, ``"denied"``, or ``"skipped"``.
        """
        return await self._await_approval(
            format_approval_message(tool_name, params),
            channel,
            user_id,
            tool_name,
            params,
        )

    async def _batch_approve_writes(
        self,
        tool_calls: list,
        channel: str,
        user_id: str,
        request_state: dict,
    ) -> None:
        """Approve a turn's pending write actions with a single prompt.

        The LLM can emit several write tool calls in one response (e.g. "set
        reminders for the next 5 days"). Prompting for each separately forces
        the user to approve one-at-a-time. Instead, collect every write that
        still needs a decision, ask once, and record the decision per action
        so :meth:`_execute_tool` reuses it instead of prompting again.

        A lone write is left to the per-call path — batching only helps when
        there are two or more. The decision is all-or-nothing across the batch.
        """
        if channel == "system":
            return
        write_decisions = request_state.setdefault("write_decisions", {})
        pending: list[tuple[str, str]] = []  # (signature, description)
        seen: set[str] = set()
        for call in tool_calls:
            if not self.permissions.is_write_action(call.name, call.arguments):
                continue
            if self.permissions.check(call.name, call.arguments) != PermissionLevel.ASK:
                continue
            sig = self._write_signature(call.name, call.arguments)
            if sig in write_decisions or sig in seen:
                continue
            seen.add(sig)
            pending.append((sig, format_approval_message(call.name, call.arguments)))
        if len(pending) < 2:
            return
        lines = "\n\n".join(f"{i}. {desc}" for i, (_, desc) in enumerate(pending, 1))
        description = f"Approve these {len(pending)} actions?\n\n{lines}"
        decision = await self._await_approval(description, channel, user_id)
        for sig, _ in pending:
            write_decisions[sig] = decision

    def _approval_image(self, tool_name: str | None, params: dict | None) -> str | None:
        """Screenshot to attach to a browser `act` approval (mobile follow-along).

        The agent is told to screenshot the page before acting, which writes the
        per-profile preview; we surface it so the user sees the page next to the
        Approve/Deny buttons. Returns None for non-browser actions or no preview.
        """
        if tool_name != "run_command" or not isinstance(params, dict):
            return None
        cmd = params.get("command", "")
        if "browser.py act" not in cmd:
            return None
        from tools.browser import _preview_path

        parts = shlex.split(cmd)
        profile = "default"
        if "--profile" in parts:
            i = parts.index("--profile")
            if i + 1 < len(parts):
                profile = parts[i + 1]
        path = _preview_path(profile)
        return str(path) if path.exists() else None

    async def _await_approval(
        self,
        description: str,
        channel: str,
        user_id: str,
        tool_name: str | None = None,
        params: dict | None = None,
    ) -> str:
        """Send an approval prompt to the channel and wait for the response.

        Creates a pending approval future, sends the prompt, and waits.
        Returns one of ``"approved"``, ``"denied"``, or ``"skipped"``.
        """
        ch = self.channels.get(channel)
        if not ch:
            # No channel available to ask — auto-approve (e.g. admin API)
            log.warning("No channel %r for approval, auto-approving", channel)
            return "approved"

        request_id, future = self.permissions.create_approval_request(tool_name, params)

        # Send the approval prompt via the channel
        try:
            await ch.send_approval_request(
                user_id,
                request_id,
                description,
                image_path=self._approval_image(tool_name, params),
            )
        except AttributeError:
            # Channel doesn't support approval requests — auto-approve
            log.warning("Channel %r doesn't support approvals, auto-approving", channel)
            self.permissions.resolve_approval(request_id, True)
            return "approved"
        except Exception:
            log.exception("Failed to send approval request")
            self.permissions.resolve_approval(request_id, True)
            return "approved"

        # Wait for the user's response (timeout after 2 minutes)
        try:
            return await asyncio.wait_for(future, timeout=120)
        except TimeoutError:
            log.info("Approval request %s timed out", request_id)
            self.permissions._pending.pop(request_id, None)
            return "skipped"

    async def _extract_memories(
        self, user_msg: str, agent_msg: str, persona: Persona | None = None
    ) -> None:
        """Run automatic memory extraction in the background.

        Uses a cheap/fast model to identify facts worth remembering
        from the conversation turn, then stores them in the memory DB.
        Exceptions are logged and swallowed — this must never crash the
        main agent loop.

        ``persona`` scopes what is written (#42): facts the extractor marks
        private land in that persona's scope, everything else stays shared.
        """
        try:
            llm = self._memory_llm(
                self.config.memory.extraction_provider,
                self.config.memory.extraction_thinking_level,
            )
            stored = await self.memory.extract_memories(
                llm=llm,
                model=self.config.memory.extraction_model,
                user_msg=user_msg,
                agent_msg=agent_msg,
                cooldown_seconds=self.config.memory.extraction_cooldown_seconds,
                persona_scope=_persona_scope(persona),
            )
            if stored:
                log.info("Background memory extraction stored %d memories", stored)
        except Exception:
            log.exception("Background memory extraction failed")

    def _memory_llm(self, provider: str, thinking_level: str = "") -> LLMClient:
        """Return an LLM client for memory operations.

        If the requested provider matches the main inference provider the
        existing client is reused; otherwise a new one is created using the
        API key / base-URL already stored in the agent config.
        """
        return self._background_llm(provider, thinking_level)

    def _background_llm(self, provider: str, thinking_level: str = "") -> LLMClient:
        """Return an LLM client for background tasks (memory, reflection, etc.).

        Background tasks carry their own thinking level, independent of the
        main inference one. When the provider matches the main client we clone
        it (sharing the underlying SDK connection) and override only the level;
        otherwise a fresh client is built from the stored credentials.
        """
        if provider == self.llm.provider:
            clone = copy.copy(self.llm)
            clone.thinking_level = (thinking_level or "").strip().lower()
            return clone
        cfg = self.config.agent
        return LLMClient(
            provider=provider,
            api_key=getattr(cfg, f"{provider}_api_key", ""),
            base_url=getattr(cfg, f"{provider}_base_url", None),
            thinking_level=thinking_level,
        )

    def _build_embedder(self):
        """Construct the embedding client for semantic memory, if enabled.

        For ``provider: local`` a lazy on-device fastembed client is returned
        (no model load until first use, so this stays cheap). For API providers
        credentials fall back to the matching agent provider key / base URL.
        Returns None when disabled or unusable (the store then runs on Tier-1
        lexical retrieval).
        """
        emb = self.config.memory.embedding
        if not emb.enabled:
            return None

        if emb.provider in LOCAL_PROVIDERS:
            try:
                return LocalEmbeddingClient(model=emb.model, cache_dir=emb.cache_dir)
            except Exception:
                log.exception("Failed to build local embedder; disabling semantic memory")
                return None

        cfg = self.config.agent
        api_key = emb.api_key or getattr(cfg, f"{emb.provider}_api_key", "")
        base_url = emb.base_url or getattr(cfg, f"{emb.provider}_base_url", "") or None
        if not api_key:
            log.warning("Memory embeddings enabled but no API key for provider %s", emb.provider)
            return None
        try:
            return EmbeddingClient(
                provider=emb.provider,
                api_key=api_key,
                model=emb.model,
                base_url=base_url,
                dimensions=emb.dimensions,
            )
        except Exception:
            log.exception("Failed to build embedding client; disabling semantic memory")
            return None

    async def _maybe_decompose(self, message: str) -> DecomposedGoal | None:
        """Classify and optionally decompose a user message into sub-goals.

        Returns None if the message is simple or decomposition fails/is disabled.
        """
        gd_cfg = self.config.goal_decomposition
        llm = self._background_llm(gd_cfg.provider, gd_cfg.thinking_level)

        try:
            is_complex = await classify_complexity(llm, gd_cfg.model, message)
        except Exception:
            log.exception("Goal complexity classification failed")
            return None

        if not is_complex:
            log.debug("Message classified as SIMPLE, skipping decomposition")
            return None

        log.info("Message classified as COMPLEX, decomposing...")
        try:
            return await decompose_goal(llm, gd_cfg.model, message)
        except Exception:
            log.exception("Goal decomposition failed")
            return None

    async def _reflect_on_task(self, user_msg: str, agent_msg: str, tool_log: list[dict]) -> None:
        """Run task reflection in the background after tool-use.

        Uses a cheap/fast model to analyse the execution and extract
        lessons learned. Exceptions are logged and swallowed — this must
        never crash the main agent loop.
        """
        try:
            tr_cfg = self.config.task_reflection
            llm = self._background_llm(tr_cfg.provider, tr_cfg.thinking_level)
            stored = await self.reflections.reflect_on_task(
                llm=llm,
                model=tr_cfg.model,
                user_msg=user_msg,
                agent_msg=agent_msg,
                tool_log=tool_log,
            )
            if stored:
                log.info("Background task reflection stored a lesson")
        except Exception:
            log.exception("Background task reflection failed")

    async def _build_system_prompt(
        self,
        decomposed_goal: DecomposedGoal | None = None,
        persona: Persona | None = None,
    ) -> str:
        # Memory, reflections AND the skills index are NOT baked into the static
        # prompt: in session mode it is snapshotted once and would freeze stale —
        # a skill added mid-session stayed invisible until /new (#41, #46). All
        # three are injected fresh per turn in the preamble instead (see
        # _turn_preamble), which also makes memory query-relevant every turn.
        sections = build_prompt_sections(
            config=self.config,
            history_mode=self.history_mode,
            skills_index="",
            memories="",
            reflections="",
            decomposed_goal=decomposed_goal,
            persona=persona,
            secrets_available=self.secret_store is not None,
            include_memories=False,
            include_reflections=False,
            include_skills=False,
        )
        return sections.full_prompt

    def _record_system_prompt(
        self, *, channel: str, user_id: str, chat_id: str, prompt: str
    ) -> None:
        """Record generated prompts in a ring buffer for admin debugging."""
        user_hash = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]
        chat_hash = hashlib.sha256(chat_id.encode("utf-8")).hexdigest()[:12] if chat_id else ""
        self.prompt_capture.appendleft(
            {
                "captured_at": datetime.now(ZoneInfo(self.config.agent.timezone)).isoformat(),
                "channel": channel,
                "user_hash": user_hash,
                "chat_hash": chat_hash,
                "prompt": prompt,
            }
        )

    def get_recent_system_prompts(self) -> list[dict[str, str]]:
        """Return recent captured system prompts for admin debug endpoints."""
        return list(self.prompt_capture)

    def _extract_text(self, response) -> str:
        """Deprecated: retained for backward compatibility."""
        return response.text if hasattr(response, "text") else ""
