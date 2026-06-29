"""Permission engine — glob-pattern rules with ALWAYS/ASK/NEVER levels.

Each rule maps a pattern like "run_command:himalaya*list*" to a permission level.
The engine checks tool calls against these patterns to decide whether to execute
immediately, ask the user for approval, or block entirely.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import TypedDict

log = logging.getLogger(__name__)


class PermissionLevel:
    ALWAYS = "ALWAYS"  # Pre-approved, execute without asking
    ASK = "ASK"  # Pause and ask the user for approval
    NEVER = "NEVER"  # Block entirely


# Programs that make a generalized "<prefix>*" rule unsafe — see _rule_pattern.
# Interpreters/shells: a trailing `*` becomes `-c <code>` = arbitrary execution.
_INTERPRETERS = {
    "python",
    "python2",
    "python3",
    "pypy",
    "pypy3",
    "perl",
    "ruby",
    "node",
    "deno",
    "bun",
    "bash",
    "sh",
    "zsh",
    "fish",
    "dash",
    "ksh",
    "php",
    "lua",
    "luajit",
    "rscript",
    "osascript",
    "awk",
    "gawk",
    "tclsh",
    "ed",
    "eval",
}
# Exec wrappers: run whatever follows, and their filler args can push an
# interpreter past _rule_pattern's token cap (`env A=1 python3 -c …`).
_EXEC_WRAPPERS = {
    "env",
    "sudo",
    "doas",
    "su",
    "xargs",
    "nohup",
    "nice",
    "ionice",
    "time",
    "timeout",
    "watch",
    "setsid",
    "stdbuf",
    "unbuffer",
    "flock",
    "chroot",
    "script",
}
# Net fetchers: a trailing `*` is any path/host (scheme-less, so the `://` break
# in _rule_pattern doesn't catch them).
_NET_FETCHERS = {"curl", "wget"}


# Default rules — read operations are ALWAYS, write operations ASK, destructive NEVER.
DEFAULT_RULES: dict[str, str] = {
    # Read operations — safe by default
    "run_command:himalaya*list*": "ALWAYS",
    "run_command:himalaya*read*": "ALWAYS",
    "run_command:himalaya*envelope*": "ALWAYS",
    "run_command:himalaya*folder*": "ALWAYS",
    "run_command:python3 /app/tools/contacts.py*": "ALWAYS",
    "run_command:python3 tools/contacts.py*": "ALWAYS",
    "run_command:python3 /app/tools/calendar_read.py*": "ALWAYS",
    "run_command:python3 tools/calendar_read.py*": "ALWAYS",
    # wacli read operations — all pre-approved
    "run_command:wacli*messages*": "ALWAYS",
    "run_command:wacli*contacts search*": "ALWAYS",
    "run_command:wacli*contacts show*": "ALWAYS",
    "run_command:wacli*chats*": "ALWAYS",
    "run_command:wacli*groups list*": "ALWAYS",
    "run_command:wacli*groups info*": "ALWAYS",
    "run_command:wacli*sync*": "ALWAYS",
    "run_command:wacli*search*": "ALWAYS",
    # wacli write operations — require approval
    "run_command:wacli*contacts refresh*": "ASK",
    "run_command:wacli*contacts alias*": "ASK",
    "run_command:wacli*contacts tags*": "ASK",
    "run_command:wacli*groups refresh*": "ASK",
    "run_command:wacli*groups rename*": "ASK",
    "run_command:wacli*groups participants*": "ASK",
    "run_command:wacli*groups invite*": "ASK",
    "run_command:wacli*groups join*": "ASK",
    "run_command:wacli*groups leave*": "ASK",
    "run_command:wacli*send*": "ASK",
    # Block direct access to wacli's internal SQLite databases
    "run_command:sqlite3*wacli*": "NEVER",
    "run_command:sqlite3*.wacli*": "NEVER",
    "run_command:sqlite3*/app/data/memory.db*SELECT*": "ALWAYS",
    "run_command:sqlite3*/app/data/memory.db*INSERT*": "ALWAYS",
    "run_command:sqlite3*/app/data/memory.db*UPDATE*": "ALWAYS",
    "run_command:sqlite3*/app/data/memory.db*DELETE*": "ALWAYS",
    "run_command:sqlite3*data/memory.db*SELECT*": "ALWAYS",
    "run_command:sqlite3*data/memory.db*INSERT*": "ALWAYS",
    "run_command:sqlite3*data/memory.db*UPDATE*": "ALWAYS",
    "run_command:sqlite3*data/memory.db*DELETE*": "ALWAYS",
    "run_command:python3 /app/tools/jobs.py list*": "ALWAYS",
    "run_command:python3 /app/tools/jobs.py show*": "ALWAYS",
    "run_command:python3 tools/jobs.py list*": "ALWAYS",
    "run_command:python3 tools/jobs.py show*": "ALWAYS",
    "run_command:python3 /app/tools/jobs.py create*": "ASK",
    "run_command:python3 /app/tools/jobs.py edit*": "ASK",
    "run_command:python3 /app/tools/jobs.py remove*": "ASK",
    "run_command:python3 /app/tools/jobs.py cancel*": "ASK",
    "run_command:python3 tools/jobs.py create*": "ASK",
    "run_command:python3 tools/jobs.py edit*": "ASK",
    "run_command:python3 tools/jobs.py remove*": "ASK",
    "run_command:python3 tools/jobs.py cancel*": "ASK",
    "run_command:python3 /app/tools/skills.py list*": "ALWAYS",
    "run_command:python3 /app/tools/skills.py show*": "ALWAYS",
    "run_command:python3 /app/tools/skills.py upsert*": "ASK",
    "run_command:python3 /app/tools/skills.py delete*": "ASK",
    "run_command:python3 tools/skills.py list*": "ALWAYS",
    "run_command:python3 tools/skills.py show*": "ALWAYS",
    "run_command:python3 tools/skills.py upsert*": "ASK",
    "run_command:python3 tools/skills.py delete*": "ASK",
    "run_command:jq*": "ALWAYS",
    "run_command:curl*wttr.in*": "ALWAYS",
    "run_command:w3m*": "ALWAYS",
    "run_command:pandoc*": "ALWAYS",
    "run_command:pdftotext*": "ALWAYS",
    "run_command:rg*": "ALWAYS",
    "run_command:yt-dlp*": "ALWAYS",
    "run_command:cal*": "ALWAYS",
    "run_command:git*log*": "ALWAYS",
    "run_command:git*status*": "ALWAYS",
    "run_command:git*diff*": "ALWAYS",
    "run_command:git*show*": "ALWAYS",
    "run_command:git*branch*": "ALWAYS",
    "run_command:gh*list*": "ALWAYS",
    "run_command:gh*view*": "ALWAYS",
    "run_command:gh*status*": "ALWAYS",
    "run_command:gh*api*": "ALWAYS",
    "run_command:gh*search*": "ALWAYS",
    "run_command:gh*issue create*": "ASK",
    "run_command:gh*pr create*": "ASK",
    "run_command:gh*release create*": "ASK",
    # Browser automation — reading is safe, acting (click/fill/submit) asks.
    # Per-domain rules work because every command carries `--url`, e.g. add
    # "run_command:*browser.py act*github.com*": "ALWAYS" via the admin UI.
    "run_command:*browser.py read*": "ALWAYS",
    "run_command:*browser.py screenshot*": "ALWAYS",
    "run_command:*browser.py profiles*": "ALWAYS",
    "run_command:*browser.py act*": "ASK",
    "run_command:git*push*": "ASK",
    "run_command:git*commit*": "ASK",
    "web_search": "ALWAYS",
    "recall_memory": "ALWAYS",
    # Write operations — ask first
    "send_email": "ASK",
    "reply_email": "ASK",
    "send_message": "ASK",
    "create_calendar_event": "ASK",
    "run_command:himalaya*send*": "ASK",
    "run_command:himalaya*delete*": "ASK",
    "run_command:himalaya*move*": "ASK",
    "schedule_task": "ASK",
    "manage_jobs": "ASK",
    # Delegating to a subagent is approved once per spawn; the subagent then runs
    # autonomously within its narrowed scope (system semantics), like a job.
    "spawn_subagent": "ASK",
    # Publishing inline content the agent authored is low-risk and reversible
    # (TTL cleanup) — no prompt, like web_search / load_skill. But copying an
    # on-disk file to a public URL can expose data the agent didn't author, so
    # that variant (source_path) asks first.
    "write_artifact": "ALWAYS",
    "write_artifact:publish_file": "ASK",
    # Dangerous — never allow
    "run_command:sqlite3*DROP*": "NEVER",
    "run_command:sqlite3*ALTER*": "NEVER",
    "load_skill": "ALWAYS",
}


class PermissionEngine:
    """Check tool actions against permission rules using glob patterns."""

    def __init__(self, db_path: str = "data/config.db") -> None:
        self.db_path = db_path
        self.rules: dict[str, str] = dict(DEFAULT_RULES)
        self._ready = False
        # Pending approval requests: request_id → PendingApproval
        self._pending: dict[str, PendingApproval] = {}
        # YOLO scopes (channels) with the approval prompt bypassed — see is_yolo.
        self._yolo: set[str] = set()
        self._load_persisted_rules()
        self._load_yolo()

    def _ensure_schema(self) -> None:
        if self._ready:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS permissions ("
                "pattern TEXT PRIMARY KEY, level TEXT NOT NULL, "
                "created_at DATETIME DEFAULT (datetime('now'))"
                ")"
            )
            db.execute("CREATE TABLE IF NOT EXISTS yolo (scope TEXT PRIMARY KEY)")
        self._ready = True

    def _load_persisted_rules(self) -> None:
        self._ensure_schema()
        with sqlite3.connect(self.db_path) as db:
            rows = db.execute("SELECT pattern, level FROM permissions").fetchall()
        for pattern, level in rows:
            if level in (PermissionLevel.ALWAYS, PermissionLevel.ASK, PermissionLevel.NEVER):
                self.rules[pattern] = level

    def _persist_rule(self, pattern: str, level: str) -> None:
        self._ensure_schema()
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "INSERT INTO permissions (pattern, level) VALUES (?, ?) "
                "ON CONFLICT(pattern) DO UPDATE SET level = excluded.level",
                (pattern, level),
            )
            db.commit()

    def _load_yolo(self) -> None:
        self._ensure_schema()
        with sqlite3.connect(self.db_path) as db:
            rows = db.execute("SELECT scope FROM yolo").fetchall()
        self._yolo = {scope for (scope,) in rows}

    def set_yolo(self, scope: str, on: bool) -> None:
        """Turn the approval-bypass (YOLO) on/off for a scope (a channel name).

        A scope in YOLO has ASK actions auto-approved without a prompt. NEVER
        rules still hold — this is "act without asking", not "self-destruct".
        Persisted so the choice survives a restart until explicitly turned off.
        """
        self._ensure_schema()
        with sqlite3.connect(self.db_path) as db:
            if on:
                db.execute("INSERT OR IGNORE INTO yolo (scope) VALUES (?)", (scope,))
                self._yolo.add(scope)
            else:
                db.execute("DELETE FROM yolo WHERE scope = ?", (scope,))
                self._yolo.discard(scope)
            db.commit()
        log.warning("YOLO mode %s for scope %r", "ON" if on else "OFF", scope)

    def is_yolo(self, scope: str) -> bool:
        """True if the scope (channel) currently bypasses approval prompts."""
        return scope in self._yolo

    def _build_match_key(self, tool_name: str, params: dict | None = None) -> str:
        if tool_name == "run_command" and params and "command" in params:
            return f"run_command:{params['command']}"
        # Publishing an on-disk file is gated separately from inline writes.
        if tool_name == "write_artifact" and params and params.get("source_path"):
            return "write_artifact:publish_file"
        return tool_name

    @staticmethod
    def _rule_pattern(match_key: str) -> str:
        """Generalize a concrete command into a reusable glob rule for "always".

        Keeps the leading program/script/subcommand tokens (e.g.
        `python3 /app/tools/browser.py explore`) and wildcards the arguments, so
        approving once covers every later call of the same command shape. Stops at
        the first flag/URL/quoted/redirect token and caps at 3 tokens. Two guards
        keep the wildcard from re-opening an arbitrary-code/target hole: at least 2
        kept tokens are required, and none of the kept tokens may be an interpreter
        (as the last token), an exec-wrapper (anywhere), or a net fetcher (as the
        program) — see the inline note. Such commands keep their exact form and
        re-ask on the next distinct call. Non run_command keys are returned
        unchanged.
        """
        prefix = "run_command:"
        if not match_key.startswith(prefix):
            return match_key
        kept: list[str] = []
        for tok in match_key[len(prefix) :].split():
            if tok.startswith("-") or "://" in tok or tok[:1] in "\"'" or tok in "|<>;&":
                break
            kept.append(tok)
            if len(kept) == 3:
                break
        # Need program + subcommand (≥2 tokens) before wildcarding the rest.
        # A single kept token means the very next token was a flag/URL/quoted arg
        # (`python3 -c …`, `curl https://…`, `sed -n …`, `echo "…"`), and
        # generalizing to `python3*` / `curl*` would auto-approve arbitrary code or
        # any URL — turning one "always" click into a blanket bypass of the engine.
        # Keep the exact command instead; the next distinct invocation re-asks.
        if len(kept) < 2:
            return match_key
        # Content check, not just token count: wildcarding is unsafe whenever the
        # trailing `*` could still expand into arbitrary code or an arbitrary
        # target. That happens when the LAST kept token is an interpreter/shell
        # (its `*` becomes `-c <code>`); when ANY kept token is an exec-wrapper
        # (env/sudo/xargs/… hide the real program AND their filler args can push an
        # interpreter past the 3-token cap, e.g. `env A=1 python3 …`); or when the
        # program itself fetches the network (`curl host*` = any path on that host,
        # scheme-less so the `://` break above never fired). Keep those exact.
        bases = [tok.rsplit("/", 1)[-1] for tok in kept]
        if (
            bases[-1] in _INTERPRETERS
            or any(b in _EXEC_WRAPPERS for b in bases)
            or bases[0] in _NET_FETCHERS
        ):
            return match_key
        return prefix + " ".join(kept) + "*"

    def match_key(self, tool_name: str, params: dict | None = None) -> str:
        """Public helper to build the match key for a tool call."""
        return self._build_match_key(tool_name, params)

    def is_write_action(self, tool_name: str, params: dict | None = None) -> bool:
        """Return True if a tool call is a write-like action.

        Write-like actions should prompt for permission each time. Read actions
        can be auto-approved after the first user confirmation.
        """
        if tool_name in {
            "send_email",
            "reply_email",
            "send_message",
            "create_calendar_event",
            "schedule_task",
            "manage_jobs",
            "spawn_subagent",
        }:
            return True

        # Inline artifact writes are not write-actions (frictionless); publishing
        # an on-disk file is (it exposes a file and needs per-call approval).
        if tool_name == "write_artifact":
            return bool(params and params.get("source_path"))

        match_key = self._build_match_key(tool_name, params)
        if match_key.startswith("run_command:"):
            command = match_key[len("run_command:") :].strip().lower()
            for pattern, level in self.rules.items():
                if level != PermissionLevel.ASK:
                    continue
                if not pattern.startswith("run_command:"):
                    continue
                if fnmatch.fnmatch(match_key, pattern):
                    return True
            if any(
                token in command
                for token in ("send", "delete", "move", "invite", "rename", "join", "leave")
            ):
                return True

        return False

    def check(self, tool_name: str, params: dict | None = None) -> str:
        """Return the permission level for a tool call.

        Builds a match key like "run_command:himalaya envelope list ..."
        and checks it against all rules. First match wins, with more
        specific (longer) patterns tried first.
        """
        match_key = self._build_match_key(tool_name, params)

        # Sort rules by pattern length descending so more specific rules match first
        for pattern in sorted(self.rules, key=len, reverse=True):
            if fnmatch.fnmatch(match_key, pattern):
                return self.rules[pattern]

        # Default: ASK for unknown actions (safe fallback)
        return PermissionLevel.ASK

    def add_rule(self, pattern: str, level: str) -> None:
        """Add or update a permission rule."""
        if level not in (PermissionLevel.ALWAYS, PermissionLevel.ASK, PermissionLevel.NEVER):
            raise ValueError(f"Invalid permission level: {level!r}")
        self.rules[pattern] = level
        self._persist_rule(pattern, level)
        log.info("Permission rule added: %s → %s", pattern, level)

    def remove_rule(self, pattern: str) -> bool:
        """Remove a permission rule if it exists."""
        existed = pattern in self.rules
        if existed:
            del self.rules[pattern]
            self._ensure_schema()
            with sqlite3.connect(self.db_path) as db:
                db.execute("DELETE FROM permissions WHERE pattern = ?", (pattern,))
                db.commit()
        return existed

    def create_approval_request(
        self, tool_name: str | None = None, params: dict | None = None
    ) -> tuple[str, asyncio.Future[str]]:
        """Create a pending approval request. Returns (request_id, future).

        The caller awaits the future. When the user approves/denies via
        a channel callback, resolve_approval() completes the future with
        one of ``"approved"``, ``"denied"``, or ``"skipped"``.
        """
        request_id = uuid.uuid4().hex[:12]
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        if tool_name is None:
            tool_name = "unknown"
        if params is None:
            params = {}
        self._pending[request_id] = {
            "future": future,
            "match_key": self._build_match_key(tool_name, params),
        }
        return request_id, future

    def format_approval_message(self, tool_name: str, params: dict) -> str:
        return format_approval_message(tool_name, params)

    def resolve_approval(
        self,
        request_id: str,
        approved: bool,
        always_allow: bool = False,
        *,
        skipped: bool = False,
    ) -> bool:
        """Resolve a pending approval request. Returns False if not found.

        The future is resolved with a string: ``"approved"``, ``"denied"``,
        or ``"skipped"`` so callers can distinguish all three outcomes.
        """
        entry = self._pending.pop(request_id, None)
        if not entry:
            return False
        future = entry["future"]
        if future.done():
            return False
        if always_allow:
            match_key = entry.get("match_key")
            if isinstance(match_key, str):
                # Persist a GENERALIZED pattern, not the exact command — otherwise
                # "always" only ever matches that one verbatim invocation and the
                # next (different --url/--task/args) prompts again. See _rule_pattern.
                pattern = self._rule_pattern(match_key)
                if pattern not in self.rules:
                    self.add_rule(pattern, PermissionLevel.ALWAYS)
        if skipped:
            future.set_result("skipped")
        elif approved:
            future.set_result("approved")
        else:
            future.set_result("denied")
        return True


class PendingApproval(TypedDict):
    future: asyncio.Future[str]
    match_key: str


def format_approval_message(tool_name: str, params: dict) -> str:
    """Format a human-readable approval prompt for a tool call."""
    if tool_name == "send_email":
        to = params.get("to", "?")
        subject = params.get("subject", "?")
        return f"Send email to {to}\nSubject: {subject}"
    if tool_name == "reply_email":
        account = params.get("account", "?")
        msg_id = params.get("message_id", "?")
        return f"Reply to message {msg_id} on {account}"
    if tool_name == "send_message":
        channel = params.get("channel", "?")
        to = params.get("to", "?")
        text = params.get("text", "")
        preview = text[:100] + ("…" if len(text) > 100 else "")
        return f"Send {channel} message to {to}\n{preview}"
    if tool_name == "create_calendar_event":
        summary = params.get("summary", "?")
        start = params.get("start", "?")
        return f"Create event: {summary}\nAt: {start}"
    if tool_name == "schedule_task":
        task = params.get("task", "?")
        run_at = params.get("run_at", "?")
        return f"Schedule task at {run_at}\n{task}"
    if tool_name == "manage_jobs":
        action = params.get("action", "?")
        if action == "create":
            task = params.get("task", "?")
            cron = params.get("cron")
            run_at = params.get("run_at")
            schedule = f"cron: {cron}" if cron else f"once at {run_at}" if run_at else "?"
            return f"Create scheduled job ({schedule})\n{task}"
        if action == "cancel":
            job_id = params.get("job_id", "?")
            return f"Cancel scheduled job: {job_id}"
        if action == "list":
            return "List all scheduled jobs"
        return f"Manage jobs: {action}"
    if tool_name == "run_command":
        cmd = params.get("command", "?")
        purpose = params.get("purpose", "")
        return f"Run command: {cmd}" + (f"\n({purpose})" if purpose else "")
    if tool_name == "write_artifact":
        return f"Publish a file as a public web artifact:\n{params.get('source_path', '?')}"
    return f"{tool_name}: {params}"
