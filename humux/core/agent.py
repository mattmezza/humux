"""Agent core — LLM call with agentic tool-use loop."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import re
import shlex
import time
import uuid
from collections import OrderedDict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

from tavily import TavilyClient

from core import coding, imagegen
from core.agents import Agent, AgentStore, default_agent_from_values, topic_base_id
from core.compaction import compact_messages, should_compact
from core.config import Config, search_ready
from core.embeddings import LOCAL_PROVIDERS, EmbeddingClient, LocalEmbeddingClient
from core.executor import ToolExecutor
from core.goal_decomposition import DecomposedGoal, classify_complexity, decompose_goal
from core.history import ConversationHistory
from core.imagegen import ImageBudget
from core.job_store import JobStore
from core.llm import (
    LLMClient,
    LLMToolCall,
    model_supports_vision,
    reset_capture_context,
    reset_usage_agent,
    set_capture_context,
    set_usage_agent,
)
from core.log_streams import set_stream, subagent_stream
from core.memory import MemoryStore
from core.models import IMAGE_MIME_TYPES, AgentResponse, Attachment, OutputMessage
from core.permissions import PermissionEngine, PermissionLevel, format_approval_message
from core.prompt_builder import build_prompt_sections
from core.reply_decision import should_reply
from core.scheduler import AgentScheduler
from core.secret_store import SecretStore
from core.skills import SkillsEngine
from core.subagents import (
    FILE_HANDOFF_INSTRUCTION,
    RESULT_FOR_AGENT_INSTRUCTION,
    SubagentRegistry,
    SubagentRun,
    fallback_summary,
    narrow_accounts,
    narrow_scope,
    normalize_effort,
    resolve_cap,
    short_summary,
    summarize_batch,
)
from core.task_reflection import ReflectionStore
from core.tools import _gh_app_configured, effective_tool_env, github_repo_violation
from voice.pipeline import VoicePipeline

log = logging.getLogger(__name__)


def _narrow_gh_repos(parent: Agent | None, requested_tc: dict) -> dict:
    """Copy a child's tool identity verbatim but intersect its GitHub repo
    allowlist with the parent's — inherit-never-widen for repos (#111).

    ``repos`` empty/absent = unrestricted, so this only tightens: a child can
    never gain a repo the parent lacked. Every other tool-identity field
    (tokens, profiles) travels unchanged.
    """
    tc = dict(requested_tc or {})
    child_gh = tc.get("gh")
    if not isinstance(child_gh, dict):
        return tc
    parent_gh = (parent.tool_setting("gh") if parent else None) or {}
    if not (child_gh.get("repos") or parent_gh.get("repos")):
        return tc  # neither restricts repos → nothing to narrow
    merged = dict(child_gh)
    merged["repos"] = narrow_scope(parent_gh.get("repos"), child_gh.get("repos"))
    return {**tc, "gh": merged}


# Vision fallback caption cache cap (per process). Captions are keyed by image
# hash so repeated identical images don't re-hit the vision model.
_VISION_CACHE_MAX = 256

# Max characters a single folded run of silent group turns (#30) may reach
# before a fresh turn is started, so a busy never-addressed room can't grow one
# history row without bound. ponytail: generous char cap; raise if one
# un-addressed run legitimately needs more context than this.
_SILENT_FOLD_MAX_CHARS = 16000

# When the model's response is cut off at the output-token limit (issue #77),
# any tool call in it has truncated/empty arguments. Instead of running the
# half-built call (which returns a misleading "missing parameter" error and
# sends the model into a retry loop), feed back this notice so it produces a
# smaller output. ponytail: cap consecutive truncations so a model that keeps
# overflowing can't loop forever — the repeat-failure breaker (#78) generalises this.
_TRUNCATION_NOTICE = (
    "Your previous response was cut off at the output token limit before this "
    "tool call's arguments were complete, so the call was NOT run. Produce a "
    "smaller output: write large content (HTML, files) to disk incrementally — "
    "e.g. in chunks via bash — or split the work across turns, rather "
    "than passing it all in one tool argument."
)
_MAX_TRUNCATION_RETRIES = 3
# Shown to the user when the truncation cap trips and the model produced no
# usable text (only a cut-off tool call) — better than a blank reply.
_TRUNCATION_GIVEUP_MESSAGE = (
    "I couldn't fit my response within the output limit. Try narrowing the "
    "request, or ask me to produce the result in smaller parts."
)


def _truncation_tool_results(response) -> list[dict]:
    """Error tool_results for a truncated response's pending calls (issue #77).

    The half-built calls are not executed; each tool_use still needs a paired
    tool_result for the next turn, so emit the truncation notice for each.
    """
    return [
        {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": json.dumps({"error": _TRUNCATION_NOTICE}),
        }
        for call in response.tool_calls
    ]


# A malformed tool call (e.g. bash with no `command`) used to raise out of
# the loop and kill the turn/subagent (#78). The agentic loop is also unbounded —
# a model can repeat the same failing call until the token budget is gone. Both
# are handled at the one point every tool call routes through (_execute_tool):
# convert any exception into a recoverable error, and refuse a call whose exact
# signature has already failed this many times.
_MAX_REPEAT_FAILURES = 3
_REPEAT_FAILURE_NOTICE = (
    "You have already called this exact tool with these exact arguments and it "
    "kept failing. Stop retrying it — change the arguments, take a different "
    "approach, or report the problem to the user."
)
_LOOP_ABORT_MESSAGE = (
    "I had to stop — I made too many tool calls without reaching an answer. "
    "Could you rephrase, or break the request into smaller steps?"
)


def _failure_signature(name: str, params: object) -> str:
    """Stable key identifying a tool call, for the repeat-failure breaker (#78)."""
    try:
        return f"{name}:{json.dumps(params, sort_keys=True, default=str)}"
    except TypeError, ValueError:
        return f"{name}:{params!r}"


def _shell_quote(s: str) -> str:
    """Quote a string for safe shell interpolation."""
    return shlex.quote(s)


def _as_int(value: object, default: int) -> int:
    """Coerce an LLM-supplied value to int, falling back to ``default``."""
    try:
        return int(value)  # type: ignore[arg-type]
    except TypeError, ValueError:
        return default


# Control marker the LLM appends to request a spoken reply (see the <voice> prompt
# block). It is internal signalling and must never reach the user, whether or not
# synthesis ran. An optional ":lang" suffix (e.g. [respond_with_voice:it]) tells
# TTS the language the reply is written in so it isn't spoken with the wrong
# phonemes (issue #95). VOICE_MARKER stays the canonical bare form for prompts.
VOICE_MARKER = "[respond_with_voice]"
# Match the marker with ANY ":suffix" (or none) so it is always stripped — even
# when the model writes a malformed code like ":english" or ":it-IT". A strict
# suffix pattern would fail to match those and leak the raw marker to the user;
# voice_request_lang validates the code separately.
_VOICE_MARKER_RE = re.compile(r"\[respond_with_voice(?::([^\]]*))?\]")

# Multi-message replies (#202): the model separates sequential messages with this
# marker; each part becomes its own delivered message and may carry its own voice
# marker, so a turn can mix text and voice bubbles. Absent = one message
# (unchanged behaviour). Split swallows surrounding whitespace so blank lines
# around the marker don't leak into a bubble.
SPLIT_MARKER = "[[split]]"
_SPLIT_MARKER_RE = re.compile(r"\s*\[\[split\]\]\s*", re.IGNORECASE)

# Cap an approval prompt's text on the fail-closed retry so an over-long
# description (e.g. a huge bash command) fits a channel's message limit. Well
# under Telegram's 4096 even with the channel's "Permission request:" prefix.
# ponytail: fixed cap; the channel-layer delivery fix (#77-style) may supersede.
_APPROVAL_TEXT_CAP = 3500


def _truncate_approval(text: str) -> str:
    """Clip an approval prompt to a length channels can deliver (see cap above)."""
    return text if len(text) <= _APPROVAL_TEXT_CAP else text[: _APPROVAL_TEXT_CAP - 1] + "…"


# When the approval prompt can't be delivered we fail closed (skip the action),
# but the user never saw it — so the model must NOT be told "user skipped", or
# it wrongly concludes the owner is stepping in and replies to a phantom.
_UNDELIVERABLE_ERROR = {
    "error": (
        "The approval prompt could not be delivered to the user (channel/delivery "
        "error). The user did NOT see this request and did NOT skip or intervene. "
        "Do not retry the same action; if it matters, say you couldn't reach the "
        "user for approval."
    )
}


def strip_voice_marker(text: str) -> str:
    """Remove the voice control marker (bare or with a ``:lang`` suffix) so it
    never leaks into a user-visible reply."""
    return _VOICE_MARKER_RE.sub("", text).strip()


def voice_request_lang(text: str) -> str | None:
    """ISO-639-1 language tagged on the voice marker (``[respond_with_voice:it]``
    → ``"it"``), or ``None`` when the marker is bare, absent, or carries a code
    we can't read as a 2-letter language (issue #95). Tolerates a region suffix
    (``it-IT`` → ``it``) and a full name's first two letters (``english`` → ``en``)
    while rejecting junk (``123``, ``-``) so a bad tag degrades to default voice."""
    m = _VOICE_MARKER_RE.search(text)
    if not m:
        return None
    code = (m.group(1) or "").strip().lower()[:2]
    return code if re.fullmatch(r"[a-z]{2}", code) else None


def _strip_command_suffix(message: str) -> str:
    """Normalise a slash command for matching: lower-cased and with any
    ``@botname`` suffix removed (Telegram appends it to group commands, e.g.
    ``/new@coach``). Non-commands are returned lower-cased and stripped, so a
    normal message is matched verbatim by the caller."""
    text = message.strip()
    if text.startswith("/"):
        text = text.split("@", 1)[0]
    return text.lower()


_CONTROL_COMMANDS = ("/yolo-on", "/yolo-off", "/new", "/clear", "/stop")

# Shown as the reply of an aborted turn AND stored as its assistant history entry,
# so the next turn's model sees it was interrupted mid-work (#146).
_STOPPED_MESSAGE = (
    "⏹️ Stopped at your request — I halted this turn before finishing, so some "
    "actions may already have run and others were skipped. Tell me how to proceed."
)

# Telegram command names (setMyCommands) may only be [a-z0-9_], so the hyphenated
# /yolo-on is menu-registered under a /yolo_on underscore alias; canonicalise it
# back here so the runtime treats both forms identically.
_COMMAND_ALIASES = {"/yolo_on": "/yolo-on", "/yolo_off": "/yolo-off"}


def _normalize_command(token: str) -> str:
    """Strip the ``@bot`` suffix and fold a menu underscore alias to its canonical
    hyphenated command."""
    stripped = _strip_command_suffix(token)
    return _COMMAND_ALIASES.get(stripped, stripped)


def _control_command(message: str) -> str:
    """The control command a message *is*, tolerant of the ``@bot`` suffix, the
    ``/yolo_on`` menu alias, and duplicate/merged self-repetition (#154).

    Inbound coalescing or a redelivering client can hand the runtime a command as
    ``"/yolo-on\\n\\n/yolo-on"`` or ``"/yolo-on@bot\\n\\n/yolo-on"``; a bare
    exact-match misses those and the command is processed as ordinary text (YOLO
    never actually toggles). Returns the command only when *every* whitespace-
    separated token normalises to the same control command — mixed text (e.g.
    ``"/new please"``) returns ``""`` so a command is never silently pulled out of
    a real message."""
    tokens = message.split()
    if not tokens:
        return ""
    normalised = {_normalize_command(tok) for tok in tokens}
    if len(normalised) == 1:
        (only,) = tuple(normalised)
        if only in _CONTROL_COMMANDS:
            return only
    return ""


# -- Tool definitions the LLM can call --

TOOLS = [
    # Generic CLI executor — the LLM constructs commands using skill knowledge.
    # One tool, two rails (#178): allowlisted commands run as before (reads
    # pre-approved via rules); any other command asks the owner and runs
    # confined to the coding workspace (the old run_command_in_dir).
    {
        "name": "bash",
        "description": (
            "Execute a shell command — reads/queries, CLI writes that have no "
            "dedicated structured tool, listing/searching files (ls, find, rg), and "
            "builds/tests/linters in the workspace. Allowlisted commands run "
            "pre-approved; anything else asks the owner and requires the coding "
            "workspace. Use skill documentation to construct correct syntax. "
            "Returns stdout, stderr, and exit_code."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The full shell command to run"},
                "purpose": {
                    "type": "string",
                    "description": "Brief explanation of what this command does",
                },
                "workdir": {
                    "type": "string",
                    "description": (
                        "Working directory inside the workspace (default: the workspace root)"
                    ),
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
                    "description": "Email account name (e.g. 'personal', 'work'). Optional — "
                    "defaults to the active agent's sender identity. Only accounts the "
                    "agent is allowed to send from may be used.",
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
            "required": ["to", "subject", "body"],
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
                    "description": "Email account name (e.g. 'personal', 'work'). Optional — "
                    "defaults to the active agent's sender identity.",
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
            "required": ["message_id", "body"],
        },
    },
    {
        "name": "send_message",
        "description": "Send a message to a contact via Telegram. "
        "For WhatsApp, use the wacli CLI via bash (`wacli send text`).",
        "input_schema": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "enum": ["telegram"],
                    "description": "Which messaging channel to use",
                },
                "to": {"type": "string", "description": "Recipient identifier (chat ID or phone)"},
                "text": {"type": "string", "description": "Message text"},
            },
            "required": ["channel", "to", "text"],
        },
    },
    {
        "name": "set_reaction",
        "description": (
            "React to the user's message with an emoji instead of sending a text "
            "reply — a fast, non-verbal acknowledgement (Telegram only). Defaults to "
            "the message that triggered this turn, so usually you pass only `emoji`. "
            "Use it for lightweight acks where a sentence would just be clutter: "
            "thumbsup for 'got it'/done, heart for thanks, eyes for 'I see your "
            "photo/file', party for good news, cry or pray for bad news, laugh for "
            "something funny, check/cross for approving/denying. To acknowledge with "
            "ONLY a reaction, call this tool and then end your turn with no text at all "
            "— an empty reply sends nothing, so the reaction stands on its own. When you "
            "actually have information to convey, reply with text (you may still react in "
            "addition). Reactions on messages older than 24h silently no-op."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "emoji": {
                    "type": "string",
                    "enum": [
                        "thumbsup",
                        "thumbsdown",
                        "heart",
                        "fire",
                        "party",
                        "laugh",
                        "cry",
                        "surprise",
                        "pray",
                        "100",
                        "thinking",
                        "eyes",
                        "check",
                        "cross",
                        "star",
                        "rocket",
                        "clap",
                        "muscle",
                        "handshake",
                        "target",
                    ],
                },
                "chat_id": {
                    "type": "string",
                    "description": "Override target chat (defaults to the current chat).",
                },
                "message_id": {
                    "type": "integer",
                    "description": "Override target message (defaults to the triggering message).",
                },
            },
            "required": ["emoji"],
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
                    "description": "Calendar name (e.g. 'google', 'icloud'). Optional — defaults "
                    "to the active agent's writable calendar. Only calendars the agent "
                    "has read_write access to may be used.",
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
            "required": ["summary", "start", "end"],
        },
    },
    {
        "name": "create_contact",
        "description": "Add a contact to a CardDAV address book.",
        "input_schema": {
            "type": "object",
            "properties": {
                "account": {
                    "type": "string",
                    "description": "Contacts account name. Optional — defaults to the active "
                    "identity's writable contacts account. Only accounts with read_write "
                    "access may be used.",
                },
                "name": {"type": "string", "description": "Full name of the contact"},
                "email": {"type": "string", "description": "Email address (optional)"},
                "phone": {"type": "string", "description": "Phone number (optional)"},
                "organization": {"type": "string", "description": "Organization (optional)"},
            },
            "required": ["name"],
        },
    },
    # Read-only / utility tools
    {
        "name": "search_contacts",
        "description": "Search or list contacts in a CardDAV address book. Omit 'query' to "
        "list all contacts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "account": {
                    "type": "string",
                    "description": "Contacts account name. Optional — defaults to the active "
                    "identity's bound contacts account.",
                },
                "query": {
                    "type": "string",
                    "description": "Name/email substring to match; omit to list all.",
                },
            },
            "required": [],
        },
    },
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
        "name": "generate_image",
        "description": (
            "Generate an image from a text prompt and deliver it to the user as a "
            "native photo in the chat. Use when the user asks for a picture, "
            "illustration, diagram, concept art, logo, or any visual. The image is "
            "sent to the user automatically — do NOT put the file path or base64 in "
            "your reply, just briefly say what you made. Load the 'image_generation' "
            "skill for prompting tips. A daily/monthly image budget may apply (resets "
            "00:00 UTC); if it's reached you get an error — tell the owner instead of "
            "retrying."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed description of the image to generate",
                },
                "size": {
                    "type": "string",
                    "description": (
                        "Optional WIDTHxHEIGHT, e.g. '1024x1024'. Honored by OpenAI; "
                        "other providers use the model's default aspect."
                    ),
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "remember",
        "description": (
            "Save a durable long-term memory — a fact, preference, or relationship about "
            "the owner or their contacts. Use it proactively whenever you learn something "
            "worth keeping. Do NOT store transient action-confirmations (e.g. 'filed issue "
            "#12', 'created PR #40') — those are not durable facts. Reading is automatic: "
            "relevant memories are injected each turn, and recall_memory searches the rest."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The fact to remember, as a clear standalone sentence.",
                },
                "subject": {
                    "type": "string",
                    "description": (
                        "Who or what it is about, e.g. 'matteo' or a contact's name. Leave "
                        "empty rather than repeating the category (not 'work'/'fact')."
                    ),
                },
                "category": {
                    "type": "string",
                    "enum": ["fact", "preference", "relationship", "work"],
                    "description": "Kind of memory. Defaults to 'fact'.",
                },
            },
            "required": ["content"],
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
            "Manage the full lifecycle of scheduled jobs. "
            "action='create' schedules a one-time or recurring task. "
            "action='list' shows jobs (active by default). "
            "action='get' returns one job. "
            "action='update' edits a job's fields; set status='paused' to pause "
            "or status='active' to resume. "
            "action='cancel' permanently stops a job. "
            "When the job runs, its result is delivered back to THIS chat by "
            "default — the one you're in right now — so a reminder lands where it "
            "was asked for. The user does not need to say 'in this chat'; leave "
            "'channel' unset unless they explicitly want it delivered elsewhere."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "get", "update", "cancel"],
                    "description": ("create, list, get, update, or cancel a job"),
                },
                "job_id": {
                    "type": "string",
                    "description": (
                        "For create: a short unique identifier (lowercase, dashes ok). "
                        "For get/update/cancel: the ID of the target job."
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
                    "description": (
                        "Usually omit this. By default the result is delivered to "
                        "the chat the job was created in (this conversation). Only "
                        "set it to override that with a specific delivery channel."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "Short human-readable description of this job",
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "paused"],
                    "description": (
                        "update only: 'paused' halts the job without deleting it; "
                        "'active' resumes it."
                    ),
                },
                "filter_status": {
                    "type": "string",
                    "description": "list only: only return jobs with this status.",
                },
                "include_done": {
                    "type": "boolean",
                    "description": "list only: also include done/cancelled jobs.",
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
            "the full agent loop under an agent, with a tool/skill/secret scope "
            "that is never wider than yours, and returns a structured result. It "
            "has NO memory of this conversation — put everything it needs in "
            "'task'.\n"
            "Agent: by DEFAULT omit 'agent' — the subagent runs as YOU (your "
            "identity, tools, scope). This is almost always what you want. Set "
            "'agent' ONLY when the user explicitly asked for a named specialist, "
            "or the subtask plainly belongs to a different one. Never pick a "
            "agent just because the roster lists some.\n"
            "Sizing: by default the subagent runs at the configured ceilings. Size "
            "it to the job with 'max_steps', 'token_budget', and 'thinking_effort' "
            "— smaller for a quick lookup, larger / 'high' effort for hard "
            "multi-step work. Requested values are capped at the configured maxima.\n"
            "Files: you share a filesystem with the subagent, so it reports the "
            "absolute paths of any files it creates in its result — you can then "
            "read or send them.\n"
            "Use background=true for long-running work: you get a run id "
            "immediately and the result is posted to this chat when done (monitor "
            "or cancel it via /subagents or the admin Jobs page). Use background=false "
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
                "agent": {
                    "type": "string",
                    "description": (
                        "Agent name to run as. OMIT THIS by default — the subagent "
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
            "{{secret:NAME}} inside an allowlisted bash command (substitution runs "
            "only for allowlisted commands)."
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
                    "description": "Optional hint: which agent(s) should be able to use it",
                },
            },
            "required": ["name", "reason"],
        },
    },
    # Coding harness (#76, #178) — direct file ops confined to the configured
    # workspace. Offered only when workspace.enabled and a directory is set. All
    # paths are relative to the workspace root (or absolute inside it); escaping
    # it is blocked. Listing/searching goes through bash (ls, find, rg).
    {
        "name": "read",
        "description": (
            "Read a file (or a slice of it) from the workspace. Each line comes "
            "back as 'LINE#HH:content' — the LINE#HH prefix is an anchor you can "
            "pass to edit. Large files are paginated — use 'offset' and 'limit' "
            "to page through. Paths are relative to the workspace root."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path within the workspace"},
                "offset": {
                    "type": "integer",
                    "description": "Starting line, 0-indexed (default 0)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max lines to return (default 100)",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write",
        "description": (
            "Write content to a file in the workspace, creating intermediate "
            "directories as needed. Overwrites the file if it exists. Asks the "
            "owner for approval first. Prefer edit for small changes to an "
            "existing file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path within the workspace"},
                "content": {"type": "string", "description": "Full file content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit",
        "description": (
            "Apply one or more edits to a workspace file in a single call "
            "(all-or-nothing). Each edit is either {oldText, newText} — replace an "
            'exact substring, unique in the file unless "all": true — or '
            "{pos, end?, lines} — replace the line (or inclusive range) anchored by "
            "a 'LINE#HH' prefix from read with the given lines ([] deletes). Stale "
            "anchors (file changed since the read) are rejected. Asks the owner "
            "for approval first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path within the workspace"},
                "edits": {
                    "type": "array",
                    "description": "Edits to apply atomically (all succeed or none apply)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {
                                "type": "string",
                                "description": "Exact text to replace (text edit)",
                            },
                            "newText": {"type": "string", "description": "Replacement text"},
                            "all": {
                                "type": "boolean",
                                "description": "Replace every occurrence of oldText",
                            },
                            "pos": {
                                "type": "string",
                                "description": "Anchor 'LINE#HH' of the (first) line to replace",
                            },
                            "end": {
                                "type": "string",
                                "description": "Anchor of the last line of the range (inclusive)",
                            },
                            "lines": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Replacement lines for the anchored range",
                            },
                        },
                    },
                },
            },
            "required": ["path", "edits"],
        },
    },
]


def _agent_scope(agent: Agent | None) -> str:
    """The memory scope key for an active agent (#42).

    An agent's own name is its private scope; no agent (default identity) =
    ``""`` = shared only.
    """
    return agent.name if agent else ""


def scoped_tools(agent: Agent | None) -> list[dict]:
    """Filter the function-tool schemas by the active agent's tool scope.

    An empty scope (or no agent) = all tools.
    """
    if agent is None or not agent.tools:
        return TOOLS
    # The memory and vault discovery/request tools are always retained: they are
    # the mechanics agents rely on to recall facts and obtain secrets.
    _always = {
        "recall_memory",
        "remember",
        "list_secrets",
        "request_secret",
    }
    return [t for t in TOOLS if agent.allows_tool(t["name"]) or t["name"] in _always]


def apply_feature_gates(
    tools: list[dict],
    *,
    secrets_available: bool,
    subagents_enabled: bool = True,
    imagegen_enabled: bool = False,
    workspace_enabled: bool = False,
    search_enabled: bool = False,
) -> list[dict]:
    """Drop tools whose backing feature is unavailable/disabled, so the model is
    never offered a capability it can't use (defence in depth — the tool handlers
    also refuse)."""
    out = tools
    if not secrets_available:
        out = [t for t in out if t["name"] not in ("list_secrets", "request_secret")]
    if not subagents_enabled:
        out = [t for t in out if t["name"] != "spawn_subagent"]
    if not imagegen_enabled:
        out = [t for t in out if t["name"] != "generate_image"]
    if not search_enabled:
        out = [t for t in out if t["name"] != "web_search"]
    if not workspace_enabled:
        out = [t for t in out if t["name"] not in ("read", "write", "edit")]
    return out


class AgentCore:
    def __init__(self, config: Config, secret_store: SecretStore | None = None):
        self.config = config
        # Secrets vault (issue #19). Shared, process-wide so the agent DEK
        # unsealed by an admin login is visible to the agent at runtime.
        self.secret_store = secret_store
        self.llm: LLMClient = LLMClient.from_agent_config(config.agent)
        self.llm.temperature = config.agent.temperature  # #12: configurable sampling temp
        self.skills = SkillsEngine(
            db_path=config.agent.skills_db_path,
            seed_dir=config.agent.skills_dir,
            installed_dir=config.agent.skills_installed_dir,
        )
        self.agents = AgentStore(
            db_path=config.agent.agents_db_path,
            seed_dir=config.agent.agents_dir,
            # Seed the base ``default`` agent row from config on first boot (#115
            # follow-up: the default identity is a real agent row now).
            default_identity=default_agent_from_values(
                character=config.agent.character,
                agent_name=config.agent.name,
                email_accounts=config.agent.email_accounts,
                calendar_accounts=config.agent.calendar_accounts,
                contacts_accounts=config.agent.contacts_accounts,
            ),
        )
        # Account bindings for the base default when it is the *implicit* carrier
        # (a subagent spawned with no parent identity). The main flow resolves the
        # default agent row directly; this is the fallback for that one edge (#110).
        self._default_accounts = self._build_default_accounts(config)
        resolve = self.secret_store.infra_resolve if self.secret_store else (lambda _n: None)
        agent_env = effective_tool_env(self.config, self.agents.default_identity, resolve)
        self.executor = ToolExecutor(tool_env=agent_env)
        # Image-generation usage guardrail (issue #55). Cheap to construct; the
        # SQLite table is created lazily on first use.
        self.image_budget = ImageBudget(config.tools.imagegen.db_path)
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
        # Reply-decision rate-limit backstop (#36): recent auto-reply timestamps
        # per (channel, chat_id). In-memory, resets on restart — a runaway loop
        # is transient, so persistence would be over-engineering.
        # ponytail: unbounded keys if you have thousands of distinct chats;
        # prune oldest keys if that ever shows up in memory.
        self._reply_times: dict[tuple[str, str], list[float]] = {}

        # Web search — Tavily needs an SDK client; SearXNG is just an async HTTP
        # GET (no client). self.search_enabled gates the tool for both providers.
        self.search_client: TavilyClient | None = None
        self.search_enabled = search_ready(config.search)
        if self.search_enabled and config.search.provider == "tavily":
            self.search_client = TavilyClient(api_key=config.search.api_key)
        if config.search.enabled:
            log.info(
                "Web search %s (provider: %s)",
                "enabled" if self.search_enabled else "misconfigured",
                config.search.provider,
            )
        else:
            log.info("Web search disabled")

    async def process(
        self,
        message: str,
        channel: str,
        user_id: str,
        attachments: list[Attachment] | None = None,
        chat_id: str = "",
        agent_name: str | None = None,
        respond: bool = True,
        addressed: bool = True,
        message_id: int | None = None,
    ) -> AgentResponse:
        """Serialize concurrent turns of one chat, then run the turn.

        A single ``(channel, user_id, chat_id)`` is one conversation, and its
        turns share in-memory history/session caches. When two inbound messages
        of the same chat arrive as separate tasks (a crowded group room), their
        read-modify-writes can interleave — the silent-fold RMW in
        ``_record_inbound`` drops a line of ambient context, and the session
        cache can lose an append. The per-chat lock makes turns of one chat run
        one at a time; different chats and agents get different keys, so
        cross-chat concurrency is unchanged. The turn logic lives in
        ``_process_impl``.
        """
        # /stop — abort the active turn for this chat. Handled BEFORE the chat lock
        # (#146): a running turn HOLDS that lock, so a queued /stop would otherwise
        # wait for the very turn it means to cancel. Gated on ``respond`` so an
        # unaddressed "/stop@otherbot" in a group doesn't act. No-op (brief note)
        # when nothing is running.
        if respond and _control_command(message) == "/stop":
            if self.request_stop(channel, user_id, chat_id):
                # The aborting turn delivers _STOPPED_MESSAGE itself; stay silent
                # here so the user isn't answered twice.
                return AgentResponse(text="")
            return AgentResponse(text="Nothing to stop — no turn is running.")

        # Mid-turn steering (#145): an addressed follow-up that arrives while a turn
        # is already running for this chat is buffered and injected into that turn
        # between tool rounds — the user redirects without waiting for the whole loop.
        # Detected BEFORE the lock (the running turn holds it). Control commands
        # (/new, /yolo…) and the system/scheduler path are never steers. The entry
        # still falls through to the lock: if the running turn drained it (consumed)
        # this task returns silently; if not (turn ended first), it runs as its own
        # turn so a late steer is never lost.
        key = (channel, user_id, chat_id)
        steer_entry: dict | None = None
        if (
            respond
            and addressed
            and channel != "system"
            and not _control_command(message)
            and self._active_turns_map().get(key) is not None
        ):
            steer_entry = {"text": message, "message_id": message_id, "consumed": False}
            self._steer_map().setdefault(key, []).append(steer_entry)

        async with self._chat_lock(channel, user_id, chat_id):
            if steer_entry is not None:
                self._discard_steer(key, steer_entry)
                if steer_entry["consumed"]:
                    return AgentResponse(text="")
            return await self._process_impl(
                message,
                channel,
                user_id,
                attachments=attachments,
                chat_id=chat_id,
                agent_name=agent_name,
                respond=respond,
                addressed=addressed,
                message_id=message_id,
            )

    def _active_turns_map(self) -> dict:
        """Conversation key → abort Event for the turn currently running there.

        Lazy like ``_chat_locks`` so it also works on a bare-``__new__`` instance
        (the test pattern). A turn registers its Event in ``_process_impl`` and
        pops it on exit; ``request_stop`` sets it to break the tool loop (#146)."""
        turns = getattr(self, "_active_turns", None)
        if turns is None:
            turns = self._active_turns = {}
        return turns

    def request_stop(self, channel: str, user_id: str, chat_id: str) -> bool:
        """Ask the running turn of one chat to abort at the next tool round.

        Returns True if a turn was active (its abort Event was set), False if the
        chat was idle. Scoped per conversation key, so a stop never crosses into
        another chat (#146). The turn checks the flag between rounds/tool calls, so
        it takes effect after the in-flight LLM call or tool returns, not instantly.
        """
        ev = self._active_turns_map().get((channel, user_id, chat_id))
        if ev is None:
            return False
        ev.set()
        return True

    def _steer_map(self) -> dict:
        """Conversation key → list of pending steer entries (#145).

        Lazy like ``_active_turns_map`` so a bare-``__new__`` test instance works
        too. A follow-up message that arrives mid-turn deposits ``{"text", ...,
        "consumed"}`` here; the running turn drains it between tool rounds and
        injects it before the next LLM call. Per conversation key, so a steer from
        one chat never reaches another. In-memory, resets on restart.
        """
        m = getattr(self, "_steer_buffers", None)
        if m is None:
            m = self._steer_buffers = {}
        return m

    def _drain_steer(self, channel: str, user_id: str, chat_id: str) -> list[dict]:
        """Pop and return this chat's pending steer entries (does NOT mark consumed).

        Draining is split from marking so ``consumed`` flips only once the steer has
        actually reached the model — see ``_commit_steer``. If the ``generate()`` that
        was to carry the steer raises, the entries stay unconsumed and their depositing
        tasks run them as their own turns instead of being silently dropped (#145)."""
        return self._steer_map().pop((channel, user_id, chat_id), [])

    def _steer_message(self, entries: list[dict]) -> dict:
        """Build one labelled user message from drained steer entries (#145). ponytail:
        text only — a steer's attachments are dropped; if it wasn't consumed its own
        turn still carries them, and mid-turn image steers are rare enough to defer."""
        joined = "\n\n".join(e["text"] for e in entries)
        text = (
            "[The user sent a follow-up while you were working:]\n"
            f"<steering_message>\n{joined}\n</steering_message>\n"
            "Continue your current task taking this into account."
        )
        return {"role": "user", "content": text}

    async def _commit_steer(self, channel: str, chat_id: str, entries: list[dict]) -> None:
        """Confirm steers the model has now received: mark each consumed (so its
        depositing task returns silently instead of re-running it) and ack 👀. Called
        only AFTER the carrying ``generate()`` returns, so a failed call leaves the
        steer unconsumed and it falls through to run as its own turn (#145)."""
        for e in entries:
            e["consumed"] = True
            await self._ack_steer(channel, chat_id, e.get("message_id"))

    def _discard_steer(self, key: tuple, entry: dict) -> None:
        """Remove one still-pending entry (turn ended before it was drained), so it
        can't steer the very turn it is about to run as (#145)."""
        lst = self._steer_map().get(key)
        if lst and entry in lst:
            lst.remove(entry)
            if not lst:
                self._steer_map().pop(key, None)

    async def _ack_steer(self, channel: str, chat_id: str, message_id) -> None:
        """React 👀 on a steering message so the user sees it was picked up, without
        the model spending a reply on it (#145). Best-effort: channels without
        reactions (or a failing call) are silently skipped."""
        if not (chat_id and message_id):
            return
        ch = self.channels.get(channel)
        react = getattr(ch, "react", None) if ch else None
        if not callable(react):
            return
        try:
            await react(chat_id, int(message_id), "👀")
        except Exception as exc:  # cosmetic — never break the turn over an ack
            log.debug("Steer ack reaction failed: %s", exc)

    def _chat_lock(self, channel: str, user_id: str, chat_id: str) -> asyncio.Lock:
        """The turn lock for one chat key, created on first use.

        Built lazily rather than in ``__init__`` so ``process()`` also works on an
        instance created via ``object.__new__`` (the bare-agent test pattern) and
        on any future alternate constructor — the lock map has a single owner here.
        In-memory, resets on restart. ponytail: one lock per distinct chat key,
        never evicted — a lock is tiny; prune stale keys only if a real deployment
        ever shows growth.
        """
        locks = getattr(self, "_chat_locks", None)
        if locks is None:
            locks = self._chat_locks = {}
        key = (channel, user_id, chat_id)
        lock = locks.get(key)
        if lock is None:
            lock = locks[key] = asyncio.Lock()
        return lock

    async def _process_impl(
        self,
        message: str,
        channel: str,
        user_id: str,
        attachments: list[Attachment] | None = None,
        chat_id: str = "",
        agent_name: str | None = None,
        respond: bool = True,
        addressed: bool = True,
        message_id: int | None = None,
    ) -> AgentResponse:
        """Process an incoming message through the LLM with tool-use loop.

        ``chat_id`` distinguishes different chats for the same user (e.g.
        a private Telegram chat vs. a group chat).  Each unique
        (channel, user_id, chat_id) triple gets its own conversation history,
        preventing context leakage across chats.

        ``agent_name`` forces the identity instead of resolving it from the
        channel/binding ladder — used by the scheduler so a ``telegram:<agent>``
        job is generated *as* that agent while keeping the ``system`` execution
        mode (auto-approved writes, no memory/reflection) (#29).

        ``respond=False`` records the message into history for context but
        generates no reply — the respond-gate for group rooms (#30): a bot stays
        silent for messages not addressed to it (and for other bots' messages),
        yet still sees them as inbound turns when it is later addressed. No
        agent, preamble, or LLM call runs on this path.

        ``addressed`` is whether the message was explicitly directed at THIS bot
        (@mention / reply / ``/cmd@bot``). It usually tracks ``respond``, but
        diverges when ``reply_when_addressed_only=False`` makes a bot reply to
        everything — there ``respond`` is True for unaddressed messages too. The
        YOLO toggle keys off ``addressed`` (not ``respond``) so a bare ``/yolo-on``
        never flips every bot in a room. Defaults True for non-group channels.

        ``message_id`` is the inbound message's id, carried into request_state so
        the ``set_reaction`` tool can react to the triggering message without the
        model having to know its id (#70). Channel-specific; None off Telegram.
        """

        # Respond-gate (#30): record the turn for context, but do not reply. Runs
        # before everything else so a suppressed message costs only a DB write.
        if not respond:
            await self._record_inbound(channel, user_id, chat_id, message, attachments)
            return AgentResponse(text="")

        command = _control_command(message)

        # Resolve the agent for this turn — a per-chat binding wins over the default
        # (#14); an explicit override (scheduler) skips the ladder (#29).
        if agent_name:
            agent = await self._load_agent(agent_name)
        else:
            agent = await self._resolve_agent(channel, user_id, chat_id)

        # Kill-switch (#115 flw): a disabled agent processes NOTHING — no reply, no
        # LLM, no tools, not even a /new or /yolo command. Its Telegram bot goes
        # silent. Gated BEFORE the command shortcuts below; applies to every channel
        # and to scheduler overrides too.
        if agent is not None and not agent.enabled:
            log.info("Agent %r is disabled (kill-switch) — dropping turn", agent.name)
            return AgentResponse(text="")

        # YOLO toggle — /yolo-on grants this agent a free pass (ASK actions run
        # without a prompt); /yolo-off restores prompting. Require explicit
        # addressing, not just respond=True: with reply_when_addressed_only=False
        # every bot in a group responds to a bare "/yolo-on", and we must not flip
        # them all — `addressed` is the same "directed at THIS bot" signal that
        # routes a reply, so "/yolo-on@thatbot" targets exactly one agent. Scoped
        # per (channel, chat_id): a bot's free pass is confined to the chat it was
        # granted in, never silently extended to its other chats.
        if command in ("/yolo-on", "/yolo-off"):
            # Ignore unless explicitly addressed, and never on the system/scheduler
            # path (channel='system' is exempt from prompts anyway, so a YOLO scope
            # there would be a dead, unread row).
            if not addressed or channel == "system":
                return AgentResponse(text="")
            on = command == "/yolo-on"
            self.permissions.set_yolo(self._yolo_scope(channel, chat_id), on)
            log.warning(
                "YOLO %s by user=%s channel=%s chat=%s",
                "ON" if on else "OFF",
                user_id,
                channel,
                chat_id,
            )
            if on:
                return AgentResponse(
                    text=(
                        "🔓 YOLO mode ON for this chat — I'll run actions without "
                        "asking. Only a short hard-blocked list is still refused "
                        "(e.g. direct DB drops/alters); everything else, including "
                        "file and network commands, runs unprompted. Send /yolo-off "
                        "to restore approvals."
                    )
                )
            return AgentResponse(text="🔒 YOLO mode OFF — I'll ask before risky actions again.")

        # Handle /new (alias /clear) command — clear conversational context. In a
        # group the command arrives as "/new@botname"; strip the @-suffix so the
        # addressed bot still honours it.
        if command in ("/new", "/clear"):
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

        # Tag this turn's log records with the agent's stream (#75) so the admin
        # Logs tab can filter per agent. Subagents spawned below inherit it (the
        # ContextVar is copied into their task) and add their own label.
        set_stream(agent.name if agent else "default")

        # Reply decision (#36): in a shared/group chat, stay quiet for messages
        # aimed at someone else or caught in a bot-to-bot reaction loop. Off by
        # default; never gates 1:1 chats (group_only) or scheduler/system turns.
        # A hard per-chat rate cap backstops the LLM gate so a runaway loop
        # always terminates even if the gate keeps voting "reply". Runs before
        # goal decomposition so a suppressed message costs only this one cheap
        # call, never a decompose pass. Returns an empty response → no send.
        rd_cfg = self.config.reply_decision
        if (
            rd_cfg.enabled
            and not addressed  # explicit @mention of THIS bot always replies (#185)
            and channel != "system"
            and (not rd_cfg.group_only or self._is_group_chat(user_id, chat_id))
        ):
            # Reserve a slot BEFORE the awaited LLM call so concurrent messages
            # in the same chat (each its own task) see the reservation and trip
            # the cap — closing the check-then-act race that would otherwise let
            # a burst sail past the cap. A SKIP releases its slot below.
            # The cap is per GROUP, not per topic (#183): a folded topic id would
            # give a bot-loop one fresh budget per topic, reopening the storm the
            # cap exists to close (#36/#53).
            cap_chat = topic_base_id(chat_id) or chat_id
            reserved = self._reserve_reply(channel, cap_chat, rd_cfg)
            if reserved is None:
                log.warning(
                    "Reply suppressed: rate cap %d/%ds hit for chat=%s channel=%s",
                    rd_cfg.max_replies_per_window,
                    rd_cfg.window_seconds,
                    chat_id,
                    channel,
                )
                return AgentResponse(text="")
            identity = (agent.agent_name or agent.name) if agent else "the assistant"
            llm = self._background_llm(rd_cfg.provider, rd_cfg.thinking_level)
            # Feed the gate the recent turns (so follow-ups / pronouns read) and
            # the sibling bots' names (so "addressed to someone else" is anchored
            # to real agents). Both best-effort — the gate works without them.
            history = await self.history.get_messages(channel, user_id, chat_id)
            others = [
                a.agent_name or a.name
                for a in await self.agents.list_agents()
                if a.enabled and a.bot_token and a.name != (agent.name if agent else "")
            ]
            if not await should_reply(
                llm, rd_cfg.model, message, identity, history=history, others=others
            ):
                self._release_reply(channel, cap_chat, reserved)
                return AgentResponse(text="")

        # Goal decomposition — classify and (if complex) decompose the request.
        # The resulting plan is request-specific, so it is injected per turn
        # (in the user-message preamble), not baked into the static prompt.
        decomposed_goal: DecomposedGoal | None = None
        if self.config.goal_decomposition.enabled and channel != "system":
            decomposed_goal = await self._maybe_decompose(message)

        # Per-turn preamble: live date/time + fresh memory/reflections + skills
        # index + plan. Memory is scoped to the active agent (#42): shared +
        # its private. Skills index is scoped to the agent's allowlist (#46).
        session_key = (channel, user_id, chat_id) if self.history_mode == "session" else None
        preamble = await self._turn_preamble(
            decomposed_goal,
            query=message,
            scope=_agent_scope(agent),
            agent=agent,
            session_key=session_key,
            offer_agents=True,
        )
        # Append the status of still-running background subagents from this chat,
        # so the agent always knows what is pending (their results are folded into
        # the conversation history when they finish). (#15)
        if channel != "system":
            note = self._subagent_status_note(channel, chat_id)
            if note:
                preamble = f"{preamble}\n\n{note}"

        tools = self._tools_for_turn(agent)

        # Static system prompt. In session mode it is snapshotted once at the
        # start of the session and reused for every turn (so the static content
        # is only built once, not rebuilt and re-sent each turn). In injection
        # mode the prompt is windowed/stateless, so it is rebuilt per call.
        if self.history_mode == "session":
            system = await self._session_system_prompt(channel, user_id, chat_id, agent=agent)
        else:
            system = await self._build_system_prompt(agent=agent)

        if self.config.admin.capture_prompts:
            self._record_system_prompt(
                channel=channel,
                user_id=user_id,
                chat_id=chat_id,
                prompt=system,
            )

        # Register this turn's abort Event so /stop and the Stop button can reach it
        # (#146). The chat lock serialises turns of one key, so there's never a live
        # entry to clobber; popped in finally whether the turn returns or raises.
        turn_key = (channel, user_id, chat_id)
        self._active_turns_map()[turn_key] = asyncio.Event()

        # Record every generate() this turn under this context so the admin
        # Inspect tab can show the exact last-sent payload (#99). Reset in finally
        # so a context never leaks onto an unrelated later turn on the same task.
        cap_token = set_capture_context((channel, user_id, chat_id))
        # Attribute this turn's token usage to the serving agent (#199 flw).
        agent_token = set_usage_agent(agent.name if agent else "")
        try:
            if self.history_mode == "session":
                return await self._process_session(
                    system,
                    preamble,
                    message,
                    channel,
                    user_id,
                    attachments,
                    chat_id,
                    tools,
                    agent,
                    message_id,
                )
            return await self._process_injection(
                system,
                preamble,
                message,
                channel,
                user_id,
                attachments,
                chat_id,
                tools,
                agent,
                message_id,
            )
        finally:
            reset_capture_context(cap_token)
            reset_usage_agent(agent_token)
            self._active_turns_map().pop(turn_key, None)

    async def _resolve_agent(self, channel: str, user_id: str, chat_id: str) -> Agent | None:
        """Resolve the agent for this request, in precedence order:

        0. a per-agent bot — a ``"telegram:<name>"`` channel binds straight to
           agent ``<name>``: the bot that received the message *is* the agent (#29),
        1. the per-chat binding for ``(channel, user_id, chat_id)`` (#14),
        2. the agent flagged as the default (``is_default``) — or ``None``.
        """
        # 0. Bot-per-agent: the channel name carries the agent (e.g. "telegram:coach").
        _, sep, agent_name = channel.partition(":")
        if sep and agent_name:
            agent = await self._load_agent(agent_name)
            if agent:
                return agent
            # Unknown/deleted agent — fall through to the ordinary ladder.

        # 1. Per-chat binding. A forum-topic id ("<chat>:<thread>", #183) with no
        # binding of its own inherits the group's binding, so a group bound to an
        # agent serves that agent in every topic; a topic binding still wins.
        bound = await self.history.get_chat_agent(channel, user_id, chat_id)
        if not bound:
            base = topic_base_id(chat_id)
            if base is not None:
                bound = await self.history.get_chat_agent(channel, user_id, base)
        if bound:
            agent = await self._load_agent(bound)
            if agent:
                return agent

        # 2. The agent flagged as the default — a normal agent (its own scope/tools/
        # accounts) that answers when nothing else is selected. None if none is
        # flagged, which reproduces the pre-row config base.
        try:
            return await self.agents.get_default()
        except Exception:
            log.exception("Failed to resolve the default agent")
            return None

    async def allowed_skills_for(
        self, channel: str, user_id: str = "", chat_id: str = ""
    ) -> list[str] | None:
        """The skill allowlist of the agent serving this context (#178).

        Same resolution the turn uses: a per-agent bot binds to its own agent, a
        chat binding wins, else the default. ``None`` or an empty list means *no
        restriction* (all skills). With empty ids (startup) it resolves the bot's
        own agent.
        """
        agent = await self._resolve_agent(channel, user_id, chat_id)
        return agent.skills if agent else None

    async def record_delivered_message(
        self, channel: str, user_id: str, chat_id: str, text: str, role: str = "assistant"
    ) -> None:
        """Fold an out-of-band message into a chat's history so it's in context.

        Used when the scheduler delivers a reminder/notification straight to the
        chat it was scheduled in (#71 follow-up): ``ch.send`` alone doesn't touch
        history, so without this the agent wouldn't see, on the user's next turn,
        what it already told them. ``(channel, user_id, chat_id)`` is the same key
        the originating turn used (the captured job origin), so the message lands
        in the right conversation. Respects the configured history mode.
        """
        if not text:
            return
        if self.history_mode == "session":
            await self.history.append_session_message(
                channel, user_id, {"role": role, "content": text}, chat_id
            )
        else:
            await self.history.add_turn(channel, user_id, role, text, chat_id)

    async def may_act_in_chat(
        self, channel: str, user_id: str, chat_id: str, sender_id: int
    ) -> bool:
        """Per-chat trigger/DM gate (#129): may ``sender_id`` make the resolved
        agent reply in this chat?

        Consults the agent that owns this chat (same resolution the turn uses) and
        its ``chat_settings``. No agent, or no setting for the chat, means allowed —
        so unconfigured chats behave exactly as before.
        """
        agent = await self._resolve_agent(channel, user_id, chat_id)
        if agent is None:
            return True
        return agent.chat_permits(str(chat_id), sender_id)

    async def _load_agent(self, name: str) -> Agent | None:
        """Load an agent by name, returning ``None`` if it is missing/broken."""
        try:
            return await self.agents.get(name)
        except Exception:
            log.exception("Failed to load agent %r", name)
            return None

    async def bind_chat_agent(
        self, channel: str, user_id: str, chat_id: str, agent_name: str
    ) -> None:
        """Bind (or, with an empty name, unbind) a chat to an agent.

        Thin pass-through to the history store, which also drops the snapshotted
        session system prompt so the new identity takes effect on the next turn.
        """
        await self.history.bind_chat_agent(channel, user_id, chat_id, agent_name)

    def _tools_for_turn(self, agent: Agent | None) -> list[dict]:
        """The function-tool schemas offered to the model this turn: the agent's
        tool scope, with feature-gated tools dropped."""
        return apply_feature_gates(
            scoped_tools(agent),
            secrets_available=self.secret_store is not None,
            subagents_enabled=self.config.subagents.enabled,
            imagegen_enabled=self.config.tools.imagegen.enabled,
            workspace_enabled=self.config.workspace.enabled
            and bool(self.config.workspace.directory.strip()),
            search_enabled=self.search_enabled,
        )

    async def _turn_preamble(
        self,
        decomposed_goal: DecomposedGoal | None,
        query: str | None = None,
        scope: str = "",
        agent: Agent | None = None,
        session_key: tuple[str, str, str] | None = None,
        offer_agents: bool = False,
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

        ``scope`` is the active agent's memory scope (#42): ``""`` = shared
        only, ``"<agent>"`` = shared + that agent's private memory.
        ``agent`` scopes the skills index to its allowlist. ``session_key``
        gates skills re-injection (see below); ``None`` = always inject.
        """
        now = datetime.now(ZoneInfo(self.config.agent.timezone))
        stamp = now.strftime("%A, %B %d, %Y %H:%M %Z")
        preamble = f"[Current date & time: {stamp}]"

        # Web artifacts (#82): the workspace 'artifacts/' folder is published to the
        # public internet with no auth. The agent can write anywhere in the
        # workspace, so it must KNOW this folder is special before it (or a request)
        # drops something private there — and it needs the base URL to share a link,
        # which isn't otherwise visible to the model. One always-on preamble line
        # carries both, gated to when artifacts are actually servable (workspace
        # harness on + public route on) so the warning only shows when it's true.
        if self._workspace_dir() and self.config.artifacts.enabled:
            preamble += (
                "\n[The workspace 'artifacts/' folder is PUBLIC: anything you write under "
                f"artifacts/<slug>/ is served at {self._base_url()}/artifacts/<slug>/ with no "
                "login. Write there only to share deliberately — never private data.]"
            )

        # Skills index, scoped to the agent's allowlist. Rebuilt fresh per turn
        # so a skill added mid-session (e.g. via skill-creator) is immediately
        # visible without a /new (#46). Cheap: a local DB read, like memory.
        #
        # In session mode (``session_key`` set) the preamble is persisted into the
        # growing history, so re-sending an unchanged index every turn would just
        # accumulate identical copies. We skip it only when the exact block is
        # ALREADY present in the replayed history (so the model still sees it).
        # Gating on the real history — not a side cache — keeps it correct by
        # construction across /new, compaction, agent rebind and concurrent
        # turns: any of those that drop or change the block simply won't find it,
        # and the failure direction is a harmless re-send, never a blind turn.
        # Injection mode and tests pass ``None`` → always include.
        try:
            skills_index = await self.skills.get_index_block(allow=agent.skills if agent else None)
            block = (
                f"<available_skills>\n{skills_index}\n</available_skills>" if skills_index else ""
            )
            if block and (
                session_key is None or not await self._skills_block_in_history(session_key, block)
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

        # Roster of agents the agent can delegate to via spawn_subagent, so its
        # choice is informed rather than guessed (#15). Only on the main turn —
        # selection stays user-led (omit agent = run as yourself / the bound one).
        if offer_agents:
            roster = await self._agents_roster_block(agent)
            if roster:
                preamble += f"\n\n{roster}"

        # Which email/calendar/contacts accounts this identity may use, so the
        # tools route without guessing account names (#110). Names and access
        # levels only — credentials never enter the prompt. Falls back to the
        # default-agent bindings when no agent is active.
        accounts_identity = agent if agent is not None else self._default_accounts
        if accounts_identity is not None:
            note = self._account_note(accounts_identity)
            if note:
                preamble += f"\n\n{note}"

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

    async def _agents_roster_block(self, agent: Agent | None) -> str:
        """Compact `name — role` roster of agents the agent can delegate to (#15).

        Makes specialist delegation an informed choice instead of a guess, while
        leaving selection user-led: omitting ``agent`` runs the subagent as the
        caller itself. Returns "" (nothing injected) when subagents are disabled,
        the active agent can't spawn, or there is no one to delegate to.
        """
        if not self.config.subagents.enabled:
            return ""
        if agent is not None and not agent.allows_tool("spawn_subagent"):
            return ""
        try:
            agents = [a for a in await self.agents.list_agents() if a.enabled]
        except Exception:
            log.exception("Failed to list agents for the subagent roster")
            return ""
        current = agent.name if agent else ""
        lines = []
        for p in agents:
            role = p.role.strip().splitlines()[0].strip() if (p.role or "").strip() else ""
            tag = " (you)" if p.name == current else ""
            lines.append(f"- {p.name}{tag}" + (f" — {role}" if role else ""))
        if not lines:
            return ""
        body = "\n".join(lines)
        return (
            "<agents>\n"
            "These agents exist ONLY so you can honour an explicit request for a "
            "specialist. By default, spawn_subagent with NO 'agent' so the "
            "subagent runs as you — do not assign one of these unless the user "
            "asked for it or the subtask plainly belongs to it.\n"
            f"{body}\n"
            "</agents>"
        )

    async def _skills_block_in_history(self, session_key: tuple[str, str, str], block: str) -> bool:
        """True if the exact ``<available_skills>`` block is already present in the
        replayed session history — so the model still sees it and we needn't
        re-send it this turn (#46 follow-up).

        Reads the same message array that will be sent to the model, so the
        decision is correct by construction: after a /new or compaction the block
        is gone (→ re-send), an agent rebind or new skill changes the block (→
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
            who = f"- [{r.run_id}] {r.agent or 'default'} — running ({r.elapsed_str})"
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
        agent: Agent | None = None,
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
        system = await self._build_system_prompt(agent=agent)
        await self.history.set_session_system(channel, user_id, system, chat_id)
        return system

    async def _maybe_compact(
        self, channel: str, user_id: str, chat_id: str, response: Any, model: str = ""
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
        # Threshold % is judged against the model the turn actually ran on
        # (a per-agent override may have a different context window).
        if not should_compact(cfg, context_tokens, model or self.config.agent.model):
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
        llm: LLMClient | None = None,
        model: str = "",
    ) -> dict:
        """Build the user message dict, handling multimodal content.

        ``llm``/``model`` are the client and model this turn actually runs on
        (a per-agent override may differ from the globals) — they pick the
        image-block format and decide whether the vision fallback engages.

        ``preamble`` (live date/time + optional execution plan) is prepended to
        the message text so the agent always knows 'now' for the current turn.

        When the active model can't see images and a vision fallback is
        configured, images are captioned by a secondary model and the text is
        injected in place of the image blocks so the model can still "see".
        """
        llm = llm or self.llm
        model = model or self.config.agent.model
        text = f"{preamble}\n\n{message}" if preamble else message
        image_attachments = [a for a in (attachments or []) if a.is_image]
        if image_attachments:
            if self.config.vision.enabled and not model_supports_vision(llm.provider, model):
                captions = await self._caption_images(image_attachments, message)
                if captions:
                    text = "\n\n".join([text, *captions]) if text else "\n\n".join(captions)
                    return {"role": "user", "content": text}
                # Captioning failed entirely — fall through to native image blocks.
            content_blocks: list[dict] = []
            if text:
                content_blocks.append({"type": "text", "text": text})
            for att in image_attachments:
                if llm.provider == "anthropic":
                    content_blocks.append(att.to_anthropic_block())
                else:
                    content_blocks.append(att.to_openai_block())
            return {"role": "user", "content": content_blocks}
        return {"role": "user", "content": text}

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
        agent: Agent | None = None,
        message_id: int | None = None,
    ) -> AgentResponse:
        """Injection mode: replay windowed history as native alternating messages."""
        tools = tools if tools is not None else TOOLS
        llm, model, max_tokens = self._agent_llm(agent)
        history = await self.history.get_messages(channel, user_id, chat_id)
        messages: list[dict] = []

        if history:
            # Replay history as proper alternating user/assistant messages
            for turn in history:
                messages.append({"role": turn["role"], "content": turn["content"]})

        # The actual current request — always the last user message.
        messages.append(await self._build_user_message(message, attachments, preamble, llm, model))

        log.info(
            "Processing message (injection) from %s/%s/%s: %s",
            channel,
            user_id,
            chat_id,
            message[:100],
        )

        # Initial LLM call
        response = await llm.generate(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=cast(Any, tools),
        )

        # Agentic loop — keep going while the LLM wants to call tools
        request_state = self._new_request_state(
            agent,
            origin={
                "channel": channel,
                "user_id": user_id,
                "chat_id": chat_id,
                "message_id": message_id,
            },
        )
        # Resolve the YOLO grant once per turn (channel+chat_id are in scope here
        # but not in _execute_tool); every tool call this turn reads the cached flag.
        request_state["yolo"] = self.permissions.is_yolo(self._yolo_scope(channel, chat_id))
        tool_log: list[dict] = []
        truncations = 0
        rounds = 0
        abort = self._active_turns_map().get((channel, user_id, chat_id))
        stopped = False
        while response.tool_calls and rounds < self.config.agent.max_tool_rounds:
            rounds += 1
            if abort and abort.is_set():  # /stop or the Stop button fired (#146)
                stopped = True
                break
            if response.truncated:
                truncations += 1
                if truncations > _MAX_TRUNCATION_RETRIES:
                    log.warning("Giving up after %d truncated responses in a row", truncations)
                    break
                tool_results = _truncation_tool_results(response)
            else:
                truncations = 0
                await self._batch_approve_writes(
                    response.tool_calls, channel, user_id, request_state
                )
                tool_results = []
                for call in response.tool_calls:
                    if abort and abort.is_set():
                        # Stop pressed mid-batch — let the in-flight call finish,
                        # skip the rest, and break the loop on the next check (#146).
                        stopped = True
                        break
                    result = await self._execute_tool(call, channel, user_id, request_state)
                    tool_log.append({"name": call.name, "args": call.arguments, "result": result})
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": call.id,
                            "content": json.dumps(result),
                        }
                    )
                if stopped:
                    break

            # Feed tool results back to the LLM
            messages.append(llm.assistant_message(response))
            messages.extend(llm.tool_result_messages(tool_results))
            # Mid-turn steering (#145): fold any follow-up the user sent while we
            # worked into the next call. Appended after the tool_result user turn;
            # generate() merges the consecutive user turns, so alternation holds.
            # Committed (consumed + 👀) only after generate() returns, so a failed
            # call leaves the steer to run as its own turn rather than vanish.
            steer = self._drain_steer(channel, user_id, chat_id)
            if steer:
                messages.append(self._steer_message(steer))
            response = await llm.generate(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                tools=cast(Any, tools),
            )
            if steer:
                await self._commit_steer(channel, chat_id, steer)
        if stopped:
            final_text = _STOPPED_MESSAGE
        else:
            final_text = response.text
            if response.truncated and not final_text:
                final_text = _TRUNCATION_GIVEUP_MESSAGE
            elif response.tool_calls and not final_text:
                final_text = _LOOP_ABORT_MESSAGE
        log.info("Response: %s", final_text[:200])

        # Split into one or more delivery messages (#202); each part may be voice.
        # ``final_text`` becomes the combined marker-free text used for history,
        # memory extraction, and the backward-compat AgentResponse.text.
        messages, final_text = await self._split_reply(final_text, agent)

        # Persist the turn (user message + one assistant turn per delivered message).
        # A react-only turn (or any reply that sends nothing) yields no messages —
        # don't store an empty assistant turn: some providers reject empty content on
        # the next replay, and the coalescer folds the resulting adjacent user turns (#70).
        history_message = self._history_message_text(message, attachments)
        await self.history.add_turn(channel, user_id, "user", history_message, chat_id)
        for msg in messages:
            if msg.text:
                await self.history.add_turn(channel, user_id, "assistant", msg.text, chat_id)

        # Automatic memory extraction
        if channel != "system":
            asyncio.create_task(
                self._extract_memories(message, final_text, agent),
                name=f"memory-extract-{user_id}",
            )

        # Automatic task reflection (when tools were used)
        if channel != "system" and self.config.task_reflection.enabled and tool_log:
            asyncio.create_task(
                self._reflect_on_task(message, final_text, tool_log),
                name=f"task-reflect-{user_id}",
            )

        return AgentResponse(
            text=final_text,
            voice=next((m.voice for m in messages if m.voice), None),
            attachments=request_state.get("pending_attachments", []),
            messages=messages,
        )

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
        agent: Agent | None = None,
        message_id: int | None = None,
    ) -> AgentResponse:
        """Session mode: sticky session per (channel, user_id, chat_id).

        The full message array is kept in memory and persisted to SQLite.
        New messages are appended, giving the LLM full conversational
        continuity with a cache-friendly prefix.
        """
        tools = tools if tools is not None else TOOLS
        llm, model, max_tokens = self._agent_llm(agent)

        # Load existing session (from memory cache or DB)
        session = await self.history.get_session(channel, user_id, chat_id)

        # Append the new user message (with the live date/time preamble)
        user_msg = await self._build_user_message(message, attachments, preamble, llm, model)
        await self.history.append_session_message(channel, user_id, user_msg, chat_id)

        log.info(
            "Processing message (session) from %s/%s/%s: %s",
            channel,
            user_id,
            chat_id,
            message[:100],
        )

        # Initial LLM call with the full session
        response = await llm.generate(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=session,
            tools=cast(Any, tools),
        )

        # Agentic loop — keep going while the LLM wants to call tools
        new_messages: list[dict] = []
        request_state = self._new_request_state(
            agent,
            origin={
                "channel": channel,
                "user_id": user_id,
                "chat_id": chat_id,
                "message_id": message_id,
            },
        )
        # Resolve the YOLO grant once per turn (channel+chat_id are in scope here
        # but not in _execute_tool); every tool call this turn reads the cached flag.
        request_state["yolo"] = self.permissions.is_yolo(self._yolo_scope(channel, chat_id))
        tool_log: list[dict] = []
        truncations = 0
        rounds = 0
        abort = self._active_turns_map().get((channel, user_id, chat_id))
        stopped = False
        while response.tool_calls and rounds < self.config.agent.max_tool_rounds:
            rounds += 1
            if abort and abort.is_set():  # /stop or the Stop button fired (#146)
                stopped = True
                break
            if response.truncated:
                truncations += 1
                if truncations > _MAX_TRUNCATION_RETRIES:
                    log.warning("Giving up after %d truncated responses in a row", truncations)
                    break
                tool_results = _truncation_tool_results(response)
            else:
                truncations = 0
                await self._batch_approve_writes(
                    response.tool_calls, channel, user_id, request_state
                )
                tool_results = []
                for call in response.tool_calls:
                    if abort and abort.is_set():
                        # Stop pressed mid-batch — let the in-flight call finish,
                        # skip the rest, and break the loop on the next check (#146).
                        stopped = True
                        break
                    result = await self._execute_tool(call, channel, user_id, request_state)
                    tool_log.append({"name": call.name, "args": call.arguments, "result": result})
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": call.id,
                            "content": json.dumps(result),
                        }
                    )
                if stopped:
                    # Don't persist a half-batch: the assistant tool_use turn below
                    # would name calls whose results we're dropping, leaving an
                    # orphan tool_use in the session. Break with only whole rounds
                    # recorded; _STOPPED_MESSAGE becomes the assistant turn (#146).
                    break

            # Append tool exchange to session
            assistant_msg = llm.assistant_message(response)
            tool_result_msgs = llm.tool_result_messages(tool_results)

            new_messages.append(assistant_msg)
            new_messages.extend(tool_result_msgs)

            # The in-memory session list is mutated by append_session_messages
            # so the next generate() call sees the updated messages.
            await self.history.append_session_messages(
                channel, user_id, [assistant_msg, *tool_result_msgs], chat_id
            )

            # Mid-turn steering (#145): fold any follow-up the user sent while we
            # worked into the next call, and persist it so the sticky session keeps a
            # faithful record. generate() merges it with the tool_result user turn.
            # Committed (consumed + 👀) only after generate() returns; on a failed
            # call the steer stays unconsumed and runs as its own turn (no double
            # action — no tool runs between injecting the steer and this generate).
            steer = self._drain_steer(channel, user_id, chat_id)
            if steer:
                steer_msg = self._steer_message(steer)
                new_messages.append(steer_msg)
                await self.history.append_session_messages(channel, user_id, [steer_msg], chat_id)

            response = await llm.generate(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=session,
                tools=cast(Any, tools),
            )
            if steer:
                await self._commit_steer(channel, chat_id, steer)

        if stopped:
            final_text = _STOPPED_MESSAGE
        else:
            final_text = response.text
            if response.truncated and not final_text:
                final_text = _TRUNCATION_GIVEUP_MESSAGE
            elif response.tool_calls and not final_text:
                final_text = _LOOP_ABORT_MESSAGE

        # Split into one or more delivery messages (#202); each part may be voice.
        # ``final_text`` becomes the combined marker-free text — one assistant turn
        # in the session (multiple assistant messages would break user/assistant
        # alternation), and the backward-compat AgentResponse.text.
        messages, final_text = await self._split_reply(final_text, agent)

        # Append the final assistant response to the session. Skip an empty final
        # (a react-only turn sends nothing): the reaction is already recorded as the
        # assistant tool_use turn above, and an empty assistant message is dead weight
        # that some providers reject on the next call (#70).
        if final_text:
            final_assistant_msg = {"role": "assistant", "content": final_text}
            await self.history.append_session_message(
                channel, user_id, final_assistant_msg, chat_id
            )

        log.info("Response: %s", final_text[:200])

        # Compaction — if the context has grown past the configured threshold,
        # summarise the oldest turns. ``response.usage`` reflects the full
        # session that was just sent, so it's the authoritative context size.
        system_notice = await self._maybe_compact(channel, user_id, chat_id, response, model)

        # Automatic memory extraction
        if channel != "system":
            asyncio.create_task(
                self._extract_memories(message, final_text, agent),
                name=f"memory-extract-{user_id}",
            )

        # Automatic task reflection (when tools were used)
        if channel != "system" and self.config.task_reflection.enabled and tool_log:
            asyncio.create_task(
                self._reflect_on_task(message, final_text, tool_log),
                name=f"task-reflect-{user_id}",
            )

        return AgentResponse(
            text=final_text,
            voice=next((m.voice for m in messages if m.voice), None),
            attachments=request_state.get("pending_attachments", []),
            system_notice=system_notice,
            messages=messages,
        )

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

    async def _split_reply(
        self, final_text: str, agent: Agent | None
    ) -> tuple[list[OutputMessage], str]:
        """Split the model's final reply into ordered delivery messages (#202).

        Parts are separated by SPLIT_MARKER; each part is independently checked
        for the voice marker, so one turn can send several text bubbles, a voice
        note, or a mix. No marker → a single message (unchanged behaviour).
        Returns the message list and the combined marker-free text (for history,
        logging, and the backward-compat ``AgentResponse.text``).
        """
        voice_name = (agent.voice or None) if agent else None
        parts = [p for p in (seg.strip() for seg in _SPLIT_MARKER_RE.split(final_text)) if p]
        messages: list[OutputMessage] = []
        for part in parts:
            voice_bytes = await self._maybe_synthesize_voice(part, voice=voice_name)
            messages.append(OutputMessage(text=strip_voice_marker(part), voice=voice_bytes))
        combined = "\n".join(m.text for m in messages if m.text)
        return messages, combined

    async def _maybe_synthesize_voice(self, text: str, voice: str | None = None) -> bytes | None:
        """Synthesize voice if requested by the LLM, using the agent's voice
        when one is set (else the configured default)."""
        if _VOICE_MARKER_RE.search(text) and self.voice:
            clean_text = strip_voice_marker(text)
            lang = voice_request_lang(text)
            try:
                return await self.voice.synthesize(clean_text, voice=voice, lang=lang)
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
        """Run a tool call, never letting a malformed call crash the turn (#78).

        Single choke point for every tool call: convert any unexpected exception
        into a recoverable error result, and refuse a call whose identical
        signature has already failed ``_MAX_REPEAT_FAILURES`` times this turn so
        the model can't burn the budget looping on the same broken call.
        """
        if request_state is None:
            request_state = self._new_request_state()
        sig = _failure_signature(tool_call.name, tool_call.arguments)
        failures = request_state.setdefault("failure_counts", {})
        if failures.get(sig, 0) >= _MAX_REPEAT_FAILURES:
            return {"error": _REPEAT_FAILURE_NOTICE}
        try:
            result = await self._execute_tool_inner(tool_call, channel, user_id, request_state)
        except Exception as exc:
            log.exception("Tool %r raised", tool_call.name)
            result = {
                "error": (
                    f"The '{tool_call.name}' tool failed unexpectedly: {exc}. "
                    "Don't repeat the same call — check the arguments, try a different "
                    "approach, or tell the user if it can't be done."
                )
            }
        # Count every error toward the breaker, transient ones (e.g. a command
        # timeout) included: the failure mode we stop is a model retrying the
        # *identical* call many times — varying the args resets the count.
        if isinstance(result, dict) and result.get("error"):
            failures[sig] = failures.get(sig, 0) + 1
        return result

    async def _execute_tool_inner(
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

        # Permission rules are scoped to the active agent (#100); "" = default.
        agent_scope = request_state.get("agent_name") or ""

        is_write_action = self.permissions.is_write_action(name, params)
        # A bash command outside the executor's prefix allowlist runs workspace-
        # confined under per-call approval (the old run_command_in_dir, folded in
        # by #178): treat it as a write action so it is never exact-learned into
        # an ALWAYS rule — builds/tests re-ask each time.
        unlisted_bash = name == "bash" and not self.executor.command_allowed(
            str(params.get("command") or "")
        )
        if unlisted_bash:
            is_write_action = True
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
        # A non-allowlisted bash command is exempt too: re-running tests/builds
        # in a turn is legitimate (it is a write action only so it always asks).
        if (
            is_write_action
            and name not in ("manage_jobs", "spawn_subagent")
            and not unlisted_bash
            and write_sig in executed_writes
        ):
            return {
                "error": (
                    "This exact action was already completed in this request; not repeating it."
                )
            }
        if is_write_action and write_decisions.get(write_sig) == "denied":
            return {"error": "Action denied by user."}
        if is_write_action and write_decisions.get(write_sig) == "undeliverable":
            return _UNDELIVERABLE_ERROR
        if is_write_action and write_decisions.get(write_sig) == "skipped":
            return {
                "error": (
                    "User skipped this action. "
                    "Do not retry this exact action — "
                    "move on to something else."
                )
            }

        # --- Permission check ---
        level = self.permissions.check(name, params, scope=agent_scope)

        if level == PermissionLevel.NEVER:
            log.warning("Permission DENIED (NEVER): %s — %s", name, params)
            return {"error": "This action is not allowed."}

        # A non-allowlisted bash command runs workspace-confined with no executor
        # prefix guard, so it must ASK every time — a wildcard ALWAYS rule (the
        # #148 read rules `bash:cat*`, `bash:ls*`, …) must NOT auto-approve it, or
        # `cat ~/.ssh/id_rsa` would read outside the workspace with no prompt.
        # NEVER still blocks (checked above); YOLO still applies (explicit opt-in).
        if unlisted_bash and level == PermissionLevel.ALWAYS:
            level = PermissionLevel.ASK

        # YOLO bypass: when the owner put this agent+chat in YOLO, skip the approval
        # prompt for ASK actions — auto-approved without persisting a rule. The
        # decision is computed once per turn (request_state["yolo"], keyed by
        # channel+chat_id where chat_id is in scope) so it can't leak across chats.
        # Runs after the NEVER check so hard rails still hold even in YOLO.
        if level == PermissionLevel.ASK and channel != "system" and request_state.get("yolo"):
            log.warning("YOLO auto-approve: %s on channel=%s", name, channel)
            level = PermissionLevel.ALWAYS

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
                decision = await self._request_approval(
                    name,
                    params,
                    channel,
                    user_id,
                    scope=agent_scope,
                    chat_id=str(request_state.get("origin", {}).get("chat_id") or ""),
                )
                if is_write_action:
                    write_decisions[write_sig] = decision
                elif isinstance(approvals, dict):
                    approvals[match_key] = decision
                    request_state["approvals"] = approvals
            if decision == "undeliverable":
                log.warning("Permission UNDELIVERABLE (fail-closed): %s", name)
                return _UNDELIVERABLE_ERROR
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
                # Learn an exact-command ALWAYS rule so this read auto-approves
                # next time — but never from a degenerate key. A bare
                # `bash` (command arg missing) would whitelist every
                # command and nullify the allowlist (#79); learn_always_rule
                # refuses those and keeps asking.
                self.permissions.learn_always_rule(
                    self.permissions.match_key(name, params),
                    generalize=False,
                    scope=agent_scope,
                )

        # --- Dispatch ---
        if name == "bash":
            log.info("Tool call: bash — %s", params.get("purpose", ""))
            command = params.get("command")
            if not command:
                return {"error": "bash requires a non-empty 'command' argument."}
            # Optional working directory — must live inside the workspace (#178).
            workspace = self._workspace_cwd()
            cwd = workspace
            workdir = str(params.get("workdir") or "").strip()
            if workdir and workdir != ".":
                if workspace is None:
                    return {"error": "workdir requires the coding workspace to be enabled."}
                try:
                    resolved = coding.resolve_in_workspace(workspace, workdir)
                except coding.WorkspaceError as exc:
                    return {"error": str(exc)}
                if not resolved.is_dir():
                    return {"error": f"Not a directory: {workdir}"}
                cwd = str(resolved)
            if unlisted_bash:
                # Outside the prefix allowlist: runs ONLY workspace-confined, under
                # the per-call approval that already happened above (the old
                # run_command_in_dir path, #76/#178). No secret substitution here —
                # the {{secret:}} boundary stays on the allowlisted rail.
                if workspace is None:
                    # No workspace → surface the executor's allowlist guidance.
                    return await self.executor.run_command(command)
                return await self.executor.run_in_dir(command, cwd)
            # Secret substitution boundary (issue #19): {{secret:NAME}} is resolved
            # ONLY here, for the model's generic command tool, after an ACL check.
            # Structured tools (send_email/send_message/…) build their commands
            # elsewhere and never pass through this path, so a secret cannot be
            # exfiltrated through a message/email body.
            if self.secret_store is not None:
                allowed = set(request_state.get("agent_secrets") or [])
                command, serr = await self.secret_store.resolve_command_secrets(command, allowed)
                if serr:
                    return {"error": serr}
            # Per-agent tool identity (#93): an agent runs `gh`/`browser` with its
            # own credentials/profile, never the owner's. No agent → the shared
            # default env (unchanged path).
            agent = request_state.get("agent_obj")
            # Per-agent GitHub repo allowlist (#111) — block before running.
            bad_repo = github_repo_violation(agent, command)
            if bad_repo:
                return {
                    "error": (
                        f"Agent '{agent.name}' is not allowed to use the GitHub "
                        f"repo '{bad_repo}'. Allowed repos are set on the agent's "
                        "GitHub tool identity."
                    )
                }
            agent_env = None
            # Build the per-turn tool env when an agent is active OR a GitHub App
            # is configured — the latter so its rotating installation token (#111)
            # is minted fresh per command instead of the stale one cached at
            # construction. A static PAT doesn't rotate, so the no-agent/PAT case
            # keeps using the executor's shared default (unchanged).
            if agent is not None or _gh_app_configured(self.config):
                store = self.secret_store
                resolve = store.infra_resolve if store else (lambda _n: None)
                agent_env = effective_tool_env(self.config, agent, resolve)
            return await self.executor.run_command(command, tool_env=agent_env, cwd=cwd)

        if name == "send_email":
            result = await self._tool_send_email(params, request_state)
            if is_write_action and self._is_tool_success(result):
                executed_writes.add(write_sig)
            return result

        if name == "reply_email":
            result = await self._tool_reply_email(params, request_state)
            if is_write_action and self._is_tool_success(result):
                executed_writes.add(write_sig)
            return result

        if name == "send_message":
            result = await self._tool_send_message(params)
            if is_write_action and self._is_tool_success(result):
                executed_writes.add(write_sig)
            return result

        if name == "set_reaction":
            return await self._tool_set_reaction(params, request_state)

        if name == "create_calendar_event":
            result = await self._tool_create_calendar_event(params, request_state)
            if is_write_action and self._is_tool_success(result):
                executed_writes.add(write_sig)
            return result

        if name == "search_contacts":
            return await self._tool_search_contacts(params, request_state)

        if name == "create_contact":
            result = await self._tool_create_contact(params, request_state)
            if is_write_action and self._is_tool_success(result):
                executed_writes.add(write_sig)
            return result

        if name == "web_search":
            log.info("Tool call: web_search — %s", params.get("query", ""))
            return await self._tool_web_search(params)

        if name == "generate_image":
            return await self._tool_generate_image(params, request_state)

        if name == "remember":
            return await self._tool_remember(params, request_state)

        if name == "recall_memory":
            return await self._tool_recall_memory(params, request_state)

        if name == "manage_jobs":
            log.info("Tool call: manage_jobs — %s", params.get("action", ""))
            result = await self._tool_manage_jobs(params, request_state)
            if is_write_action and self._is_tool_success(result):
                executed_writes.add(write_sig)
            return result

        if name == "list_secrets":
            if self.secret_store is None:
                return {"error": "Secrets vault is not configured."}
            allowed = set(request_state.get("agent_secrets") or [])
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

        # Coding harness (#76, #178)
        if name == "read":
            return self._tool_read(params)
        if name == "write":
            result = self._tool_write(params)
            if is_write_action and self._is_tool_success(result):
                executed_writes.add(write_sig)
            return result
        if name == "edit":
            result = self._tool_edit(params)
            if is_write_action and self._is_tool_success(result):
                executed_writes.add(write_sig)
            return result

        return {"error": f"Unknown tool: {name}"}

    @staticmethod
    def _yolo_scope(channel: str, chat_id: str) -> str:
        """Key for a YOLO grant: the agent (channel) within one chat (chat_id), so
        a free pass is confined to where it was granted. ``\\x1f`` can't appear in a
        channel name or chat id, so it's an unambiguous separator."""
        return f"{channel}\x1f{chat_id}"

    @staticmethod
    def _new_request_state(
        agent: Agent | None = None,
        *,
        depth: int = 0,
        origin: dict | None = None,
        run_id: str | None = None,
    ) -> dict:
        """Fresh per-turn state tracking write actions and approval decisions.

        ``depth``/``origin``/``agent_obj`` carry the
        context a ``spawn_subagent`` call needs to narrow scope, cap recursion,
        and post a background result back to the originating chat (issue #15).
        """
        return {
            "executed_writes": set(),
            "write_decisions": {},
            "approvals": {},
            # Media produced mid-turn (e.g. generate_image) to deliver natively (#55).
            "pending_attachments": [],
            # Secret scope for {{secret:}} ACL in bash commands (issue #19).
            "agent_secrets": list(agent.secrets) if agent else [],
            "agent_name": agent.name if agent else "",
            # Subagent plumbing (issue #15).
            "agent_obj": agent,
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

    @staticmethod
    def _build_default_accounts(config: Config) -> Agent | None:
        """Build the default identity's account carrier from config (#110).

        When no agent is active, ``send_email``/``create_calendar_event``/etc.
        route through this. It is a :class:`Agent` holding ONLY the account
        bindings (no skills/tools/secrets/character), so it never affects scoping,
        memory, or identity — only account routing. Returns ``None`` when the
        default agent has no bindings, preserving the legacy unscoped behaviour.
        """
        from core.agents import _as_account_list

        email = _as_account_list(config.agent.email_accounts, sender=True)
        calendar = _as_account_list(config.agent.calendar_accounts)
        contacts = _as_account_list(config.agent.contacts_accounts)
        if not email and not calendar and not contacts:
            return None
        return Agent(
            name="",
            email_accounts=email,
            calendar_accounts=calendar,
            contacts_accounts=contacts,
        )

    def _accounts_identity(self, request_state: dict | None) -> Agent | None:
        """The identity whose account bindings apply this turn: the active agent,
        else the default-agent carrier (#110). ``None`` = unscoped (legacy)."""
        agent = (request_state or {}).get("agent_obj")
        return agent if agent is not None else self._default_accounts

    @staticmethod
    def _account_note(agent: Agent) -> str:
        """Preamble block naming the identity's bound email/calendar/contacts
        accounts and access levels (#110). Names + levels only — never credentials."""
        lines: list[str] = []
        if agent.email_accounts:
            parts = []
            for e in agent.email_accounts:
                tag = e["access_level"] + (", sender" if e.get("is_sender_identity") else "")
                parts.append(f"{e['account']} ({tag})")
            lines.append("Email accounts: " + ", ".join(parts))
        if agent.calendar_accounts:
            parts = [f"{e['account']} ({e['access_level']})" for e in agent.calendar_accounts]
            lines.append("Calendar accounts: " + ", ".join(parts))
        if agent.contacts_accounts:
            parts = [f"{e['account']} ({e['access_level']})" for e in agent.contacts_accounts]
            lines.append("Contacts accounts: " + ", ".join(parts))
        if not lines:
            return ""
        body = "\n".join(lines)
        return (
            "<accounts>\n"
            "The only email/calendar/contacts accounts you may use, with your access "
            "level. read = read only; read_write = read plus send / create. send_email "
            "and reply_email default to your sender identity; create_calendar_event "
            "defaults to your writable calendar; create_contact to your writable "
            "contacts account.\n"
            f"{body}\n"
            "</accounts>"
        )

    @staticmethod
    def _resolve_email_send(agent: Agent | None, params: dict) -> tuple[str | None, dict | None]:
        """Route + authorise an email account for a send/reply (#110).

        Returns ``(account, error)``. With no active agent the agent runs
        unscoped (legacy single-user behaviour) and the requested account is used
        verbatim. With an agent, the email_accounts bindings are the allowlist:
        the account defaults to the agent's send identity, an unbound account is
        refused, and sending on a read-only binding is refused. Credentials are
        never touched here — only the account *name* is resolved.
        """
        account = str(params.get("account") or "").strip()
        if agent is None:
            if not account:
                return None, {"error": "The 'account' parameter is required."}
            return account, None
        if not account:
            account = agent.sender_identity() or ""
            if not account:
                return None, {
                    "error": "No send email identity is bound. Bind an email account as "
                    "the sender identity in the admin UI (agent editor or Assistant tab)."
                }
        level = agent.email_access(account)
        if level is None:
            return None, {
                "error": f"Not allowed to use the '{account}' email account. "
                "Grant it access in the admin UI."
            }
        if level != "read_write":
            return None, {"error": f"Read-only access to '{account}' — cannot send email from it."}
        return account, None

    @staticmethod
    def _resolve_calendar_write(
        agent: Agent | None, params: dict
    ) -> tuple[str | None, dict | None]:
        """Route + authorise a calendar for an event write, mirroring
        :meth:`_resolve_email_send` (#110). Defaults to the agent's first
        writable calendar; refuses unbound or read-only calendars."""
        calendar = str(params.get("calendar") or "").strip()
        if agent is None:
            if not calendar:
                return None, {"error": "The 'calendar' parameter is required."}
            return calendar, None
        if not calendar:
            calendar = next(
                (
                    e["account"]
                    for e in agent.calendar_accounts
                    if e.get("access_level") == "read_write"
                ),
                "",
            )
            if not calendar:
                return None, {
                    "error": "No writable calendar is bound. Grant read_write on a "
                    "calendar account in the admin UI."
                }
        level = agent.calendar_access(calendar)
        if level is None:
            return None, {"error": f"Not allowed to use the '{calendar}' calendar."}
        if level != "read_write":
            return None, {
                "error": f"Read-only access to '{calendar}' — cannot create events on it."
            }
        return calendar, None

    @staticmethod
    def _resolve_contacts_access(
        agent: Agent | None, params: dict, *, need_write: bool
    ) -> tuple[str | None, dict | None]:
        """Route + authorise a contacts account, mirroring the email/calendar
        resolvers (#110). ``need_write`` gates create_contact on a read_write
        binding; read tools pass ``need_write=False``. Defaults to the first bound
        account (first writable one when ``need_write``)."""
        account = str(params.get("account") or "").strip()
        if agent is None:
            if not account:
                return None, {"error": "The 'account' parameter is required."}
            return account, None
        if not account:
            if need_write:
                account = next(
                    (
                        e["account"]
                        for e in agent.contacts_accounts
                        if e.get("access_level") == "read_write"
                    ),
                    "",
                )
                if not account:
                    return None, {
                        "error": "No writable contacts account is bound. Grant read_write "
                        "on a contacts account in the admin UI."
                    }
            else:
                account = agent.contacts_accounts[0]["account"] if agent.contacts_accounts else ""
                if not account:
                    return None, {
                        "error": "No contacts account is bound. Grant access to a contacts "
                        "account in the admin UI."
                    }
        level = agent.contacts_access(account)
        if level is None:
            return None, {"error": f"Not allowed to use the '{account}' contacts account."}
        if need_write and level != "read_write":
            return None, {
                "error": f"Read-only access to '{account}' — cannot create contacts on it."
            }
        return account, None

    async def _tool_send_email(self, params: dict, request_state: dict | None = None) -> dict:
        """Send an email via himalaya CLI."""
        account, err = self._resolve_email_send(self._accounts_identity(request_state), params)
        if err:
            return err
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

        # himalaya v1.2.0: -a/--account is an OPTION on the subcommand, so it must
        # follow `message send` — a leading `himalaya -a … message send` is rejected
        # as an unexpected argument. The body is piped in as MML on stdin.
        command = (
            f"printf %s {_shell_quote(mml)} | himalaya message send -a {_shell_quote(account)}"
        )
        return await self.executor.run_command_trusted(command)

    async def _tool_reply_email(self, params: dict, request_state: dict | None = None) -> dict:
        """Reply to an email via himalaya CLI."""
        account, err = self._resolve_email_send(self._accounts_identity(request_state), params)
        if err:
            return err
        message_id = params["message_id"]
        body = params["body"]
        reply_all = params.get("reply_all", False)
        folder = params.get("folder")
        log.info("Tool call: reply_email — account=%s message=%s", account, message_id)

        # himalaya v1.2.0: `message reply` opens $EDITOR (not automation-safe), so
        # build the reply template non-interactively and pipe it to `template send`.
        # -a/-A/--folder are OPTIONS on the subcommand (a leading -a is rejected);
        # <ID> then [BODY] are positional.
        reply = ["himalaya template reply", "-a", _shell_quote(account)]
        if reply_all:
            reply.append("-A")
        if folder:
            reply += ["--folder", _shell_quote(folder)]
        reply += [_shell_quote(message_id), _shell_quote(body)]
        command = " ".join(reply) + f" | himalaya template send -a {_shell_quote(account)}"

        return await self.executor.run_command_trusted(command)

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

    async def _tool_set_reaction(self, params: dict, request_state: dict) -> dict:
        """React to a message with an emoji (#70).

        The channel/chat/message default to the turn's origin (the message that
        triggered this turn), so the model normally supplies only ``emoji`` — it
        has no way to know a message id otherwise. Reactions are cosmetic and
        pre-approved (ALWAYS), so this never prompts.
        """
        origin = (request_state or {}).get("origin") or {}
        channel_name = origin.get("channel") or ""
        chat_id = params.get("chat_id") or origin.get("chat_id")
        message_id = params.get("message_id") or origin.get("message_id")
        emoji = str(params.get("emoji", "")).strip()
        log.info("Tool call: set_reaction — %s on %s/%s", emoji, channel_name, message_id)
        if not emoji:
            return {"error": "Missing 'emoji'."}
        if not (chat_id and message_id):
            return {"error": "No message to react to in this context."}
        channel = self.channels.get(channel_name)
        react = getattr(channel, "react", None) if channel else None
        if not callable(react):
            return {"error": f"Channel '{channel_name}' does not support reactions."}
        try:
            await react(chat_id, int(message_id), emoji)
            return {"ok": True, "emoji": emoji}
        except Exception as exc:
            return {"error": str(exc)}

    def _base_url(self) -> str:
        import os

        return os.getenv("HUMUX_BASE_URL", f"http://localhost:{self.config.admin.port}")

    # -- Coding harness (issue #76) ------------------------------------------

    def _workspace_dir(self) -> str | None:
        """The configured workspace root, or None if the harness is off/unset.

        Defence in depth: the tools are already feature-gated out of the
        advertised set when disabled, but the handlers refuse too — a stale
        tool-call or an agent allowlist can't reach the filesystem.
        """
        ws = self.config.workspace
        if not ws.enabled or not ws.directory.strip():
            return None
        return ws.directory

    def _workspace_cwd(self) -> str | None:
        """Workspace root as a bash working directory, or None (#151).

        Unifies the two filesystem roots: when the harness is on, `bash`
        (git clone, git, gh) runs in the SAME tree the file tools resolve under,
        so a cloned repo is immediately visible to read/rg, editable with
        write/edit, and committable. Created on demand so a fresh, correctly
        configured workspace works without the owner pre-making the directory.
        """
        ws = self._workspace_dir()
        if ws is None:
            return None
        try:
            root = Path(ws).expanduser()
            root.mkdir(parents=True, exist_ok=True)
            return str(root.resolve())
        except OSError:
            return None  # unwritable path — fall back to the process cwd

    def _tool_read(self, params: dict) -> dict:
        workspace = self._workspace_dir()
        if workspace is None:
            return {"error": "The coding workspace is not enabled (workspace.enabled)."}
        path = str(params.get("path", "")).strip()
        if not path:
            return {"error": "Missing 'path'."}
        offset = _as_int(params.get("offset"), 0)
        limit = _as_int(params.get("limit"), 100)
        log.info("Tool call: read — %s:%d-%d", path, offset + 1, offset + limit)
        try:
            return coding.read_file(workspace, path, offset, limit)
        except coding.WorkspaceError as exc:
            return {"error": str(exc)}
        except OSError as exc:
            return {"error": f"Could not read file: {exc}"}

    def _tool_write(self, params: dict) -> dict:
        workspace = self._workspace_dir()
        if workspace is None:
            return {"error": "The coding workspace is not enabled (workspace.enabled)."}
        path = str(params.get("path", "")).strip()
        if not path:
            return {"error": "Missing 'path'."}
        log.info("Tool call: write — %s", path)
        try:
            return coding.write_file(workspace, path, str(params.get("content", "")))
        except coding.WorkspaceError as exc:
            return {"error": str(exc)}
        except OSError as exc:
            return {"error": f"Could not write file: {exc}"}

    def _tool_edit(self, params: dict) -> dict:
        workspace = self._workspace_dir()
        if workspace is None:
            return {"error": "The coding workspace is not enabled (workspace.enabled)."}
        path = str(params.get("path", "")).strip()
        if not path:
            return {"error": "Missing 'path'."}
        log.info("Tool call: edit — %s", path)
        edits = params.get("edits")
        try:
            return coding.edit_file(workspace, path, edits)
        except coding.WorkspaceError as exc:
            return {"error": str(exc)}
        except OSError as exc:
            return {"error": f"Could not edit file: {exc}"}

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
        agent_name = (request_state or {}).get("agent_name", "")
        log.info("Tool call: request_secret — %s (agent=%s)", sname, agent_name)
        token = await self.secret_store.create_request(
            sname, agent=agent_name, reason=reason, suggested_scope=scope
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

    async def _tool_create_calendar_event(
        self, params: dict, request_state: dict | None = None
    ) -> dict:
        """Create a calendar event via the CalDAV helper script."""
        calendar, err = self._resolve_calendar_write(self._accounts_identity(request_state), params)
        if err:
            return err
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

    async def _tool_search_contacts(self, params: dict, request_state: dict | None = None) -> dict:
        """Search/list contacts in the identity's bound CardDAV account (#110)."""
        account, err = self._resolve_contacts_access(
            self._accounts_identity(request_state), params, need_write=False
        )
        if err:
            return err
        query = str(params.get("query") or "").strip()
        log.info("Tool call: search_contacts — account=%s query=%r", account, query)
        cmd_parts = ["python3 /app/tools/contacts.py"]
        if query:
            cmd_parts += ["search", f"--query {_shell_quote(query)}"]
        else:
            cmd_parts.append("list")
        cmd_parts += [f"--provider {_shell_quote(account)}", "--output json"]
        return await self.executor.run_command(" ".join(cmd_parts))

    async def _tool_create_contact(self, params: dict, request_state: dict | None = None) -> dict:
        """Create a contact in the identity's writable CardDAV account (#110)."""
        account, err = self._resolve_contacts_access(
            self._accounts_identity(request_state), params, need_write=True
        )
        if err:
            return err
        name = str(params.get("name") or "").strip()
        if not name:
            return {"error": "A contact 'name' is required."}
        log.info("Tool call: create_contact — account=%s name=%s", account, name)
        cmd_parts = [
            "python3 /app/tools/contacts.py",
            "add",
            f"--provider {_shell_quote(account)}",
            f"--name {_shell_quote(name)}",
        ]
        if params.get("email"):
            cmd_parts.append(f"--email {_shell_quote(str(params['email']))}")
        if params.get("phone"):
            cmd_parts.append(f"--phone {_shell_quote(str(params['phone']))}")
        if params.get("organization"):
            cmd_parts.append(f"--org {_shell_quote(str(params['organization']))}")
        cmd_parts.append("--output json")
        return await self.executor.run_command(" ".join(cmd_parts))

    async def _tool_manage_jobs(self, params: dict, request_state: dict | None = None) -> dict:
        """Create, list, or cancel scheduled jobs via the JobStore.

        A created job captures its origin (the agent that scheduled it and the
        chat it was scheduled in) from ``request_state`` so the scheduler later
        runs it as that agent and delivers it back to the same chat — not the
        default identity in the owner's 1:1 DM (issue #71).
        """
        action = params.get("action", "")

        def _view(j: dict) -> dict:
            return {
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

        if action == "list":
            jobs = await self.job_store.list_jobs(
                status=params.get("filter_status"),
                include_done=bool(params.get("include_done")),
            )
            return {"ok": True, "jobs": [_view(j) for j in jobs]}

        if action == "get":
            job_id = params.get("job_id", "").strip()
            if not job_id:
                return {"error": "Missing job_id for get action."}
            job = await self.job_store.get_job(job_id)
            if not job:
                return {"error": f"Job not found: {job_id}"}
            return {"ok": True, "job": _view(job)}

        if action == "update":
            job_id = params.get("job_id", "").strip()
            if not job_id:
                return {"error": "Missing job_id for update action."}
            existing = await self.job_store.get_job(job_id)
            if not existing:
                return {"error": f"Job not found: {job_id}"}

            status = params.get("status")
            if status is not None and status not in ("active", "paused"):
                return {"error": "status must be 'active' or 'paused'."}

            cron_expr = params.get("cron")
            run_at_str = params.get("run_at")
            if cron_expr and run_at_str:
                return {"error": "Specify only one of 'cron' or 'run_at'."}

            schedule = existing["schedule"]
            new_cron = existing.get("cron")
            new_run_at = existing.get("run_at")
            if cron_expr:
                from core.scheduler import _parse_cron

                try:
                    _parse_cron(cron_expr)
                except ValueError as exc:
                    return {"error": str(exc)}
                schedule, new_cron, new_run_at = "cron", cron_expr, None
            elif run_at_str:
                try:
                    run_at = datetime.fromisoformat(run_at_str)
                except ValueError:
                    return {"error": f"Invalid datetime format: {run_at_str!r}. Use ISO format."}
                if run_at.tzinfo is None:
                    run_at = run_at.replace(tzinfo=ZoneInfo(self.config.agent.timezone))
                schedule, new_run_at, new_cron = "once", run_at.isoformat(), None

            task = params.get("task")
            channel = params.get("channel")
            description = params.get("description")
            # Omit agent/origin_* so upsert's COALESCE keeps the captured routing (#71).
            await self.job_store.upsert_job(
                job_id=job_id,
                type="agent",
                schedule=schedule,
                cron=new_cron,
                run_at=new_run_at,
                task=task if task is not None else existing["task"],
                channel=channel if channel is not None else existing["channel"],
                status=status if status is not None else existing["status"],
                description=(
                    description if description is not None else existing.get("description", "")
                ),
            )
            await self.scheduler.sync_job(job_id)
            updated = await self.job_store.get_job(job_id)
            return {"ok": True, "job": _view(updated)}

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

            # Capture the originating context (issue #71): the agent that
            # scheduled this and the chat it was scheduled in, so the scheduler
            # runs it as that agent and delivers back to the same chat. Deliver
            # from the same bot that received the request (an agent bot answers
            # as itself); non-telegram origins (e.g. the scheduler itself) keep
            # the explicit/default delivery channel.
            origin = (request_state or {}).get("origin") or {}
            origin_agent = (request_state or {}).get("agent_name") or ""
            origin_channel = origin.get("channel") or ""
            from_cli = origin_channel == "cli"
            if origin_channel == "telegram" or origin_channel.startswith("telegram:"):
                channel = origin_channel
            elif from_cli:
                # The CLI channel only exists in this transient process; the job
                # fires later in the server, where the scheduler falls back to
                # the owner's Telegram DM (#168).
                channel = "cli"
            origin_user_id = str(origin.get("user_id") or "")
            origin_chat_id = str(origin.get("chat_id") or "")
            cli_note = (
                "Saved to the job store. Output is delivered to your Telegram DM, not this "
                "terminal. It becomes active when the server next reloads jobs (on restart) — "
                "the running server isn't notified from this CLI process, so a time-sensitive "
                "one-shot may not fire. Prefer a cron job, or schedule it from Telegram."
                if from_cli
                else None
            )

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
                    created_by=origin_agent or "agent",
                    description=description,
                    agent=origin_agent,
                    origin_user_id=origin_user_id,
                    origin_chat_id=origin_chat_id,
                )
                await self.scheduler.sync_job(job_id)
                result = {
                    "ok": True,
                    "job_id": job_id,
                    "schedule": "cron",
                    "cron": cron_expr,
                    "task": task,
                    "channel": channel,
                }
                if cli_note:
                    result["note"] = cli_note
                return result

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
                    created_by=origin_agent or "agent",
                    description=description,
                    agent=origin_agent,
                    origin_user_id=origin_user_id,
                    origin_chat_id=origin_chat_id,
                )
                await self.scheduler.sync_job(job_id)
                result = {
                    "ok": True,
                    "job_id": job_id,
                    "schedule": "once",
                    "run_at": run_at.isoformat(),
                    "task": task,
                    "channel": channel,
                }
                if cli_note:
                    result["note"] = cli_note
                return result
            else:
                return {"error": "Must specify 'cron' for recurring or 'run_at' for one-time jobs."}

        return {
            "error": (
                f"Unknown action: {action!r}. Use 'create', 'list', 'get', 'update', or 'cancel'."
            )
        }

    async def _tool_remember(self, params: dict, request_state: dict | None = None) -> dict:
        """Save a long-term memory via the structured store (#13).

        Replaces hand-built sqlite3 INSERTs from the memory skill: parameterised, so
        arbitrary user text can't break quoting or inject SQL. Scope follows the active
        agent — same boundary recall and injection use.
        """
        content = str(params.get("content", "")).strip()
        if not content:
            return {"error": "Missing 'content'."}
        subject = str(params.get("subject", "")).strip()
        category = str(params.get("category", "") or "fact").strip()
        scope = (request_state or {}).get("agent_name") or ""
        try:
            stored = await self.memory.remember(
                content, subject=subject, category=category, scope=scope
            )
        except Exception:
            log.exception("remember failed for: %s", content[:80])
            return {"error": "Saving the memory failed."}
        if not stored:
            log.info("Tool call: remember — skipped transient/empty: %s", content[:80])
            return {"ok": True, "skipped": "transient confirmation; not stored as a durable memory"}
        log.info("Tool call: remember — %s/%s", category, subject or "-")
        return {"ok": True, "remembered": content}

    async def _tool_recall_memory(self, params: dict, request_state: dict | None = None) -> dict:
        """Deliberate semantic search over the full long-term memory store (#47).

        Scoped to the active agent (#42): ``agent_name`` on the per-turn
        request state is the agent's private memory scope (``""`` = the default
        identity's shared-only view), so recall never crosses into another
        agent's private memories — same boundary the injection readers enforce.
        """
        query = str(params.get("query", "")).strip()
        if not query:
            return {"error": "Missing 'query'."}
        limit = params.get("limit")
        limit = limit if isinstance(limit, int) and not isinstance(limit, bool) else None
        scope = (request_state or {}).get("agent_name") or ""
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
            agent_name=str(params.get("agent", "")).strip(),
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
        agent_name: str = "",
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

        # Resolve + narrow the agent. A name must exist; with no name the child
        # inherits the caller's identity and scope.
        if agent_name:
            requested = await self._load_agent(agent_name)
            if requested is None:
                try:
                    names = [p.name for p in await self.agents.list_agents() if p.enabled]
                except Exception:
                    names = []
                hint = f" Available: {', '.join(names)}." if names else ""
                return {
                    "error": (
                        f"Agent not found: {agent_name}.{hint} Omit 'agent' to run as yourself."
                    )
                }
            if not requested.enabled:  # kill-switch: can't spawn a disabled agent (#115 flw)
                return {"error": f"Agent '{agent_name}' is disabled and can't be spawned."}
        else:
            requested = parent_state.get("agent_obj")
        child_agent = self._narrow_agent(requested, parent_state) if requested else None

        run_id = f"sub_{uuid.uuid4().hex[:8]}"
        child_state = self._new_request_state(
            child_agent,
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
            agent=child_agent.name if child_agent else "",
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
                        "Wait for one to finish or cancel it via /subagents."
                    )
                }
            self.subagents.register(run)
            bg = asyncio.create_task(
                self._run_subagent_background(run, child_agent, child_state),
                name=f"subagent-{run_id}",
            )
            self.subagents.attach_task(run_id, bg)
            log.info("Spawned background subagent %s (agent=%s)", run_id, run.agent or "default")
            return {
                "ok": True,
                "run_id": run_id,
                "background": True,
                "status": "running",
                "agent": run.agent,
                "note": (
                    "Running in the background; its result is posted to this chat "
                    "automatically when done — you don't relay it. Each later turn "
                    "shows this run's status until it finishes."
                ),
            }

        # Synchronous: run to completion and return the result to the caller.
        self.subagents.register(run)
        log.info("Running subagent %s (agent=%s)", run_id, run.agent or "default")
        try:
            text = await self._run_subagent_loop(task, child_agent, child_state, run)
        except Exception as exc:
            log.exception("Subagent %s failed", run_id)
            self.subagents.finish(run_id, "error", error=str(exc))
            return {"error": f"Subagent failed: {exc}", "run_id": run_id}
        self.subagents.finish(run_id, "done", result=text)
        return {
            "ok": True,
            "run_id": run_id,
            "agent": run.agent,
            "summary": short_summary(text),
            "result": text,
        }

    def _narrow_agent(self, requested: Agent, parent_state: dict) -> Agent:
        """Build a child agent whose scopes are a subset of the caller's."""
        parent: Agent | None = parent_state.get("agent_obj")
        p_skills = parent.skills if parent else []
        p_tools = parent.tools if parent else []
        p_secrets = parent.secrets if parent else []
        # Account bindings are narrowed against the caller's *account* identity: the
        # parent agent, else the default-agent bindings when the top-level agent
        # ran unscoped. None (no agent and no default bindings) = unscoped owner,
        # so the child keeps its own — narrow_accounts distinguishes None from [].
        account_parent = parent if parent is not None else self._default_accounts
        p_email = account_parent.email_accounts if account_parent else None
        p_cal = account_parent.calendar_accounts if account_parent else None
        p_contacts = account_parent.contacts_accounts if account_parent else None
        return Agent(
            name=requested.name,
            agent_name=requested.agent_name,
            role=requested.role,
            voice=requested.voice,
            character=requested.character,
            skills=narrow_scope(p_skills, requested.skills),
            tools=narrow_scope(p_tools, requested.tools),
            secrets=narrow_scope(p_secrets, requested.secrets),
            # Tool identity travels verbatim with the agent (#93) — it is who the
            # child IS (its own gh token / browser profile), not a caller-subset
            # scope. Dropping it would silently fall back to the owner's token and
            # re-open the very identity bleed this feature prevents. The one scope
            # that IS narrowed is the GitHub repo allowlist (#111): a subagent can
            # never widen the repos its parent may touch.
            tool_config=_narrow_gh_repos(parent, requested.tool_config),
            # Account access is a grant, narrowed to the parent's — a subagent can
            # never reach an account (or a higher access level) its parent lacks (#110).
            email_accounts=narrow_accounts(p_email, requested.email_accounts),
            calendar_accounts=narrow_accounts(p_cal, requested.calendar_accounts),
            contacts_accounts=narrow_accounts(p_contacts, requested.contacts_accounts),
        )

    async def _run_subagent_loop(
        self, task: str, child_agent: Agent | None, child_state: dict, run: SubagentRun
    ) -> str:
        """Route this subagent's log records into its spawner's stream (#75).

        The label (agent slug, else run id) prefixes each line as
        ``[subagent:<label>]`` so it filters out of the shared stream; ``fallback``
        names the stream for a top-level scheduled run that inherited none.
        """
        # Suppress Inspect capture (#99): a subagent runs inside the spawner's
        # contextvar but is a different conversation — don't clobber the parent's
        # last-sent payload with the child's.
        cap_token = set_capture_context(None)
        # Attribute the subagent's token usage to the child agent (#199 flw) so its
        # (often heavy) generate() calls don't land in the parent's bucket or, for a
        # scheduled/background spawn that ran outside any turn, in "(unknown)". Only
        # override when there IS a named child: an anonymous subagent runs "as
        # yourself", so it should keep the spawner's inherited attribution.
        child_name = run.agent or (child_agent.name if child_agent else "")
        usage_token = set_usage_agent(child_name) if child_name else None
        try:
            with subagent_stream(run.agent or run.run_id, fallback=run.agent):
                return await self._run_subagent_loop_inner(task, child_agent, child_state, run)
        finally:
            if usage_token is not None:
                reset_usage_agent(usage_token)
            reset_capture_context(cap_token)

    async def _run_subagent_loop_inner(
        self, task: str, child_agent: Agent | None, child_state: dict, run: SubagentRun
    ) -> str:
        """The subagent's agentic loop — system semantics, budgeted and depth-capped.

        Mirrors the main injection loop but runs from a clean slate (no history),
        skips approval/decomposition/memory/reflection (channel='system'), and
        stops at this run's step/token budget (sized by the spawning agent).
        """
        cfg = self.config.subagents
        # Same gating as the main loop (incl. the #50 skill-discovery tools, which a
        # subagent needs in on-demand mode — its preamble carries the pointer too).
        tools = self._tools_for_turn(child_agent)
        # At the depth ceiling a subagent may not spawn further — don't even offer it.
        if child_state["depth"] >= cfg.recursion_depth:
            tools = [t for t in tools if t["name"] != "spawn_subagent"]
        # Subagents have no native-media delivery path (#55): a subagent returns
        # only text, so a generated image would be billed + saved + silently
        # dropped. Don't offer the tool at all.
        tools = [t for t in tools if t["name"] != "generate_image"]

        system = await self._build_system_prompt(agent=child_agent)
        system = f"{system}\n\n{RESULT_FOR_AGENT_INSTRUCTION}\n\n{FILE_HANDOFF_INSTRUCTION}"
        # Memory/reflections inject per-turn via the preamble (#41), scoped to the
        # child agent (#42); query=task keeps the injection relevant.
        preamble = await self._turn_preamble(None, query=task, scope=_agent_scope(child_agent))
        messages: list[dict] = [await self._build_user_message(task, None, preamble)]

        # The child runs on its own agent's LLM override when it has one (so a
        # "senior" subagent gets its bigger model); an explicit spawn effort
        # still wins over the inherited/overridden thinking level.
        llm, model, max_tokens = self._agent_llm(child_agent)
        if run.effort is not None:
            clone = copy.copy(llm)
            clone.thinking_level = run.effort
            llm = clone
        response = await llm.generate(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=cast(Any, tools),
        )
        steps = 0
        tokens = self._usage_total(response.usage)
        while response.tool_calls and steps < run.max_steps and tokens < run.token_budget:
            steps += 1
            run.progress = f"step {steps}: {', '.join(c.name for c in response.tool_calls)}"[:120]
            # A truncated round (issue #77) has half-built call args; skip
            # execution and feed back the notice. steps caps the retries here.
            if response.truncated:
                tool_results = _truncation_tool_results(response)
            else:
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
                model=model,
                max_tokens=max_tokens,
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
        self, run: SubagentRun, child_agent: Agent | None, child_state: dict
    ) -> None:
        """Run a subagent off-turn. When the chat's whole batch of background runs
        has finished, the spawning agent ingests their results and writes one reply
        — the user never sees raw subagent output; it works for the agent (#15)."""
        try:
            text = await self._run_subagent_loop(run.task, child_agent, child_state, run)
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
                r.agent or "",
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
        alternating for providers that require it (#15).

        Runs in the background subagent task, so it holds the same per-chat lock as
        ``process()``: this fold is another read-modify-write on the chat's history,
        and a concurrent foreground turn on the same chat would otherwise interleave
        with it. Background delivery never runs under a held lock for this key (the
        spawning turn returned long ago), so acquiring it here can't deadlock."""
        if not chat_id or channel == "system":
            return
        try:
            async with self._chat_lock(channel, user_id, chat_id):
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

    async def _record_inbound(
        self,
        channel: str,
        user_id: str,
        chat_id: str,
        message: str,
        attachments: list[Attachment] | None = None,
    ) -> None:
        """Record an inbound message as a user turn without generating a reply —
        the respond-gate's silent path for group rooms (#30).

        Folds into the trailing user turn (mirroring ``_record_subagent_context``)
        so a run of un-answered group messages stays a single turn and the
        replayed history keeps strict user/assistant alternation. ``message``
        already carries its ``[from <author>]`` speaker tag, so the bot sees who
        said what when it is later addressed. A refused fold (trailing turn is an
        assistant reply, a structured tool turn, or the cap below is hit) just
        starts a fresh user turn — ``_coalesce_user_messages`` merges the run back
        into one before the next LLM call, so alternation always holds.

        This fold is a read-modify-write; two silent records racing in one busy
        group would drop a line of ambient context. It runs inside ``process()``,
        which now holds a per-(channel,user,chat) lock, so records of one chat are
        serialized and the race is closed.
        """
        if channel == "system":
            return
        text = self._history_message_text(message, attachments)
        # Cap a single folded run so a high-traffic, never-addressed group can't
        # grow one turn without bound. A fresh turn then ages out via windowing in
        # injection mode; in session mode it persists until the next reply triggers
        # _maybe_compact, so a sustained never-addressed flood can bloat the session
        # — acceptable behind the opt-in group_chat flag. ponytail: add a per-chat
        # record budget only if a real room ever shows write abuse.
        cap = _SILENT_FOLD_MAX_CHARS
        try:
            if self.history_mode == "session":
                merged = await self.history.append_to_last_session_message(
                    channel,
                    user_id,
                    f"\n\n{text}",
                    chat_id,
                    role="user",
                    text_only=True,
                    max_len=cap,
                )
                if not merged:
                    await self.history.append_session_message(
                        channel, user_id, {"role": "user", "content": text}, chat_id
                    )
            else:
                merged = await self.history.append_to_last_turn(
                    channel, user_id, "user", f"\n\n{text}", chat_id, max_len=cap
                )
                if not merged:
                    await self.history.add_turn(channel, user_id, "user", text, chat_id)
        except Exception:
            log.exception("Failed to record silent inbound turn (chat=%s)", chat_id)

    @staticmethod
    def _usage_total(usage: dict | None) -> int:
        """Best-effort token count for budgeting (0 when the provider omits usage)."""
        if not usage:
            return 0
        return int(usage.get("input_tokens", 0) or 0) + int(usage.get("output_tokens", 0) or 0)

    async def _tool_web_search(self, params: dict) -> dict:
        """Search the web via the configured provider (Tavily or SearXNG)."""
        if not self.search_enabled:
            return {"error": "Web search is not configured."}

        query = params.get("query", "").strip()
        if not query:
            return {"error": "Empty search query."}

        max_results = self.config.search.max_results

        if self.config.search.provider == "searxng":
            return await self._searxng_search(query, max_results)

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

    async def _searxng_search(self, query: str, max_results: int) -> dict:
        """Search via a self-hosted SearXNG instance's JSON API. Returns the same
        {query, results:[{title,url,content}]} shape as the Tavily path."""
        import httpx

        base = self.config.search.searxng_url.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
                resp = await client.get(
                    f"{base}/search",
                    params={
                        "q": query,
                        "format": "json",
                        "engines": "google,bing,duckduckgo",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            log.exception("SearXNG search failed for query: %s", query)
            return {"error": f"Search failed: {exc}"}

        results = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            }
            for item in data.get("results", [])[:max_results]
        ]
        return {"query": query, "results": results}

    async def _tool_generate_image(self, params: dict, request_state: dict) -> dict:
        """Generate an image and queue it for native-media delivery (issue #55).

        The bytes ride back on the turn's ``AgentResponse.attachments`` (the
        channel sends them as a photo) — never as a path/base64 in the model's
        text. The budget guardrail is checked before spending, recorded after.
        """
        ig = self.config.tools.imagegen
        if not ig.enabled:
            return {"error": "Image generation is disabled. Enable it in the admin settings."}
        prompt = str(params.get("prompt", "")).strip()
        if not prompt:
            return {"error": "A prompt is required."}
        over = await self.image_budget.check(ig.daily_budget, ig.monthly_budget)
        if over:
            return {"error": over}
        try:
            data, mime = await imagegen.generate(
                self.config, prompt, str(params.get("size", "")).strip()
            )
        except Exception as exc:
            log.exception("Image generation failed")
            return {"error": f"Image generation failed: {exc}"}
        if mime not in IMAGE_MIME_TYPES:
            # A vector/other output (e.g. SVG from some OpenRouter models) can't be
            # sent as a photo. Fail before billing/saving so nothing is wasted.
            return {
                "error": (
                    f"The configured image model returned {mime}, which can't be sent "
                    "as a photo. Pick a model that outputs PNG/JPEG/GIF/WebP."
                )
            }
        await self.image_budget.record()
        path = imagegen.save(data, mime)
        request_state.setdefault("pending_attachments", []).append(
            Attachment(data=data, mime_type=mime, filename=Path(path).name)
        )
        log.info("Generated image (%d bytes, %s) → %s", len(data), mime, path)
        result = {
            "ok": True,
            "path": path,
            "note": (
                "Image generated and queued for delivery to the user as a photo. "
                "Do not include the path or base64 in your reply — just say briefly "
                "what you made."
            ),
        }
        # Issue #55 cost controls: warn the user when nearing a budget cap.
        warning = await self.image_budget.warning(ig.daily_budget, ig.monthly_budget)
        if warning:
            result["warning"] = warning
        return result

    async def _request_approval(
        self,
        tool_name: str,
        params: dict,
        channel: str,
        user_id: str,
        scope: str = "",
        chat_id: str = "",
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
            scope=scope,
            chat_id=chat_id,
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
        if channel == "system" or request_state.get("yolo"):
            return  # YOLO: writes fall through to _execute_tool's auto-approve
        scope = request_state.get("agent_name") or ""  # per-agent rules (#100)
        write_decisions = request_state.setdefault("write_decisions", {})
        pending: list[tuple[str, str]] = []  # (signature, description)
        seen: set[str] = set()
        for call in tool_calls:
            if not self.permissions.is_write_action(call.name, call.arguments):
                continue
            if self.permissions.check(call.name, call.arguments, scope=scope) != (
                PermissionLevel.ASK
            ):
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
        decision = await self._await_approval(
            description,
            channel,
            user_id,
            chat_id=str(request_state.get("origin", {}).get("chat_id") or ""),
        )
        for sig, _ in pending:
            write_decisions[sig] = decision

    def _approval_image(self, tool_name: str | None, params: dict | None) -> str | None:
        """Screenshot to attach to a browser `act` approval (mobile follow-along).

        The agent is told to screenshot the page before acting, which writes the
        per-profile preview; we surface it so the user sees the page next to the
        Approve/Deny buttons. Returns None for non-browser actions or no preview.
        """
        if tool_name != "bash" or not isinstance(params, dict):
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
        scope: str = "",
        chat_id: str = "",
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

        request_id, future = self.permissions.create_approval_request(tool_name, params, scope)

        # Send the approval prompt via the channel
        try:
            await ch.send_approval_request(
                user_id,
                request_id,
                description,
                image_path=self._approval_image(tool_name, params),
                chat_id=chat_id,
            )
        except AttributeError:
            # Channel doesn't support approval requests — auto-approve
            log.warning("Channel %r doesn't support approvals, auto-approving", channel)
            self.permissions.resolve_approval(request_id, True)
            return "approved"
        except Exception:
            # The prompt couldn't be delivered (commonly: too long for the
            # channel's message limit — a huge bash command). Retry once with a
            # clipped, image-less prompt so a legitimate long action stays
            # approvable. The request_id still maps to the real action, so
            # truncating the *display* text changes nothing that executes.
            log.warning("Approval send failed; retrying with truncated prompt", exc_info=True)
            try:
                await ch.send_approval_request(
                    user_id, request_id, _truncate_approval(description), chat_id=chat_id
                )
            except Exception:
                # Still undeliverable — fail CLOSED. A gate that cannot ask the
                # user must never silently approve (#79). Drop the pending
                # request and skip the action.
                log.exception("Approval request undeliverable; skipping action (fail-closed)")
                self.permissions._pending.pop(request_id, None)
                return "undeliverable"

        # Wait for the user's response (timeout after 2 minutes)
        try:
            return await asyncio.wait_for(future, timeout=120)
        except TimeoutError:
            log.info("Approval request %s timed out", request_id)
            self.permissions._pending.pop(request_id, None)
            return "skipped"

    async def _extract_memories(
        self, user_msg: str, agent_msg: str, agent: Agent | None = None
    ) -> None:
        """Run automatic memory extraction in the background.

        Uses a cheap/fast model to identify facts worth remembering
        from the conversation turn, then stores them in the memory DB.
        Exceptions are logged and swallowed — this must never crash the
        main agent loop.

        ``agent`` scopes what is written (#42): facts the extractor marks
        private land in that agent's scope, everything else stays shared.
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
                agent_scope=_agent_scope(agent),
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

    def _agent_llm(self, agent: Agent | None) -> tuple[LLMClient, str, int]:
        """The ``(client, model, max_tokens)`` a turn runs on.

        An agent's ``llm`` override wins key-by-key over the global LLM config
        (so a "senior" agent can run a bigger model while a "junior" one stays
        cheap); no override = the shared main client, unchanged. Credentials and
        base URLs are always the globally configured ones — the override only
        picks provider/model/thinking/max_tokens/temperature.
        """
        cfg = self.config.agent
        o = getattr(agent, "llm", None) if agent else None
        if not isinstance(o, dict) or not o:
            return self.llm, cfg.model, cfg.max_tokens
        provider = o.get("provider") or self.llm.provider
        thinking = o.get("thinking_level", cfg.thinking_level)
        # ponytail: a fresh (or cloned, same-provider) client per turn — same
        # cost profile as _background_llm's per-call construction; cache per
        # agent if profiling ever says otherwise.
        llm = self._background_llm(provider, thinking)
        llm.temperature = o.get("temperature", cfg.temperature)
        return llm, o.get("model") or cfg.model, o.get("max_tokens") or cfg.max_tokens

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

    def _is_group_chat(self, user_id: str, chat_id: str) -> bool:
        """Heuristic: a chat whose id differs from the user id is a group (#36).

        Telegram private chats use the user's own id as the chat id, and a
        WhatsApp DM falls back to the sender as chat_id — so ``chat_id == user_id``
        marks a 1:1 chat. Anything else (a negative Telegram group id, a
        ``"<chat>:<thread>"`` topic, a ``"...@g.us"`` WhatsApp jid) is shared.
        ponytail: a convention, not a protocol guarantee — if a channel ever
        sets chat_id == user_id for a real group, thread an explicit is_group
        flag through process() instead.
        """
        return bool(chat_id) and chat_id != user_id

    def _reserve_reply(self, channel: str, chat_id: str, cfg) -> float | None:
        """Reserve an auto-reply slot if under the per-chat cap (#36 backstop).

        Returns the reservation timestamp, or None if the rolling window is
        already full. Read-modify-write with no ``await`` in between, so it is
        atomic under the single-threaded event loop — concurrent messages in
        the same chat see each other's reservations and the cap holds even
        under a bursty bot-to-bot loop. Caller must ``_release_reply`` the slot
        if it ends up not replying (a SKIP), so quiet decisions don't burn the
        budget of a busy human group.
        """
        now = time.time()
        key = (channel, chat_id)
        recent = [t for t in self._reply_times.get(key, []) if now - t < cfg.window_seconds]
        if len(recent) >= cfg.max_replies_per_window:
            self._reply_times[key] = recent  # prune expired even when refusing
            return None
        recent.append(now)
        self._reply_times[key] = recent
        return now

    def _release_reply(self, channel: str, chat_id: str, reserved: float) -> None:
        """Give back a reserved slot when the gate decided not to reply (#36)."""
        slots = self._reply_times.get((channel, chat_id))
        if slots:
            try:
                slots.remove(reserved)
            except ValueError:
                pass  # already pruned by the window — nothing to release

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
        agent: Agent | None = None,
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
            agent=agent,
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
