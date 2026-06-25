"""Tool executor — runs CLI commands via subprocess with a prefix whitelist."""

from __future__ import annotations

import asyncio
import json
import os
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

    async def run_command(self, command: str, timeout: int = 30) -> dict:
        """Execute a shell command and return its output."""
        # Security: validate against whitelist
        if not any(command.startswith(p) for p in self.ALLOWED_PREFIXES):
            return {
                "error": f"Command not allowed. Must start with one of: {self.ALLOWED_PREFIXES}"
            }
        # `browser.py explore` runs an inner LLM loop (many page steps) and needs
        # minutes, not the 30s default — otherwise it's always killed mid-booking.
        if "browser.py explore" in command:
            timeout = max(timeout, 480)
        return await self._exec(self._resolve_command(command), timeout)

    async def run_command_trusted(self, command: str, timeout: int = 30) -> dict:
        """Execute a shell command without prefix validation.

        Only use this for commands constructed internally by the agent code,
        never for commands originating from LLM tool calls.
        """
        return await self._exec(self._resolve_command(command), timeout)

    async def _exec(self, command: str, timeout: int) -> dict:
        """Run a shell command and capture output."""
        env = None
        wants_wacli_label = "wacli" in command and "WACLI_DEVICE_LABEL" not in os.environ
        if "himalaya" in command or self.tool_env or wants_wacli_label:
            env = os.environ.copy()
            if "himalaya" in command:
                env.update(himalaya_env())
            # wacli: identify the linked device as MPA (matches the Docker ENV).
            if wants_wacli_label:
                env.setdefault("WACLI_DEVICE_LABEL", "MPA")
            # Tool auth (e.g. GH_TOKEN) — only set when a tool is enabled.
            env.update(self.tool_env)
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
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
