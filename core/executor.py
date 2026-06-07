"""Tool executor — runs CLI commands via subprocess with a prefix whitelist."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
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
    # Docker: /app/tools/wacli/dist/wacli
    docker_path = Path("/app/tools/wacli/dist/wacli")
    if docker_path.exists():
        return str(docker_path)
    # Local dev: <project_root>/tools/wacli/dist/wacli
    local_path = Path(__file__).resolve().parents[1] / "tools" / "wacli" / "dist" / "wacli"
    if local_path.exists():
        return str(local_path)
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
        if "himalaya" in command or self.tool_env:
            env = os.environ.copy()
            if "himalaya" in command:
                env.update(himalaya_env())
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
