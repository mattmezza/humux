from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def default_wacli_bin() -> str:
    env = os.getenv("WACLI_BIN")
    if env:
        return env
    from_path = shutil.which("wacli")
    if from_path:
        return from_path
    # Fallback for local dev: `go install` drops the binary in GOBIN / ~/go/bin.
    return str(Path.home() / "go" / "bin" / "wacli")


def default_wacli_store() -> str:
    return os.getenv("WACLI_STORE", str(Path.home() / ".wacli"))


def default_device_label() -> str:
    return os.getenv("WACLI_DEVICE_LABEL", "MPA")


# Lock-wait window for write commands: wait for the store lock instead of
# failing fast when another wacli process (e.g. sync --follow) holds it.
LOCK_WAIT = "30s"


@dataclass
class WacliManager:
    bin_path: str = field(default_factory=default_wacli_bin)
    store_dir: str = field(default_factory=default_wacli_store)
    device_label: str = field(default_factory=default_device_label)
    auth_proc: asyncio.subprocess.Process | None = None
    latest_qr: str = ""
    latest_qr_at: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def available(self) -> bool:
        return Path(self.bin_path).exists()

    def _env(self) -> dict[str, str]:
        """Subprocess env carrying the linked-device label (native since 0.2.0)."""
        return {**os.environ, "WACLI_DEVICE_LABEL": self.device_label}

    async def _run_json(
        self, args: list[str], *, timeout: float = 30, read_only: bool = False
    ) -> dict[str, Any]:
        if not self.available():
            return {"success": False, "error": "wacli not found"}
        flags = [
            "--store",
            self.store_dir,
            "--json",
            "--timeout",
            f"{int(timeout)}s",
            "--lock-wait",
            LOCK_WAIT,
        ]
        if read_only:
            # Skip the session-store lock entirely for pure reads.
            flags.append("--read-only")
        proc = await asyncio.create_subprocess_exec(
            self.bin_path,
            *flags,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env(),
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout + 5)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return {"success": False, "error": "wacli timed out"}
        output = (stdout or b"").decode().strip().split("\n")
        last = output[-1] if output else ""
        if proc.returncode != 0:
            err_text = (stderr or b"").decode().strip() or last
            try:
                err_json = json.loads(err_text)
                if isinstance(err_json, dict):
                    return err_json
            except json.JSONDecodeError, ValueError:
                pass
            return {"success": False, "error": err_text}
        try:
            return json.loads(last)
        except json.JSONDecodeError:
            return {"success": False, "error": last}

    async def auth_status(self) -> dict[str, Any]:
        res = await self._run_json(["auth", "status"], read_only=True)
        authed = bool(res.get("data", {}).get("authenticated")) if res.get("success") else False
        return {
            "authenticated": authed,
            "running": self.auth_proc is not None,
            "has_qr": bool(self.latest_qr),
            "latest_qr_at": self.latest_qr_at,
            "available": self.available(),
        }

    async def start_auth(self) -> None:
        async with self.lock:
            if self.auth_proc is not None:
                return
            if not self.available():
                return
            self.latest_qr = ""
            self.latest_qr_at = 0.0
            # wacli >=0.8 streams the QR as an NDJSON `qr_code` event on stderr
            # (with --events), not as JSON on stdout. Parse stderr accordingly.
            self.auth_proc = await asyncio.create_subprocess_exec(
                self.bin_path,
                "--store",
                self.store_dir,
                "--events",
                "--lock-wait",
                LOCK_WAIT,
                "auth",
                "--idle-exit",
                "30s",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                env=self._env(),
            )
            asyncio.create_task(self._consume_auth_output(self.auth_proc))

    async def _consume_auth_output(self, proc: asyncio.subprocess.Process) -> None:
        if proc.stderr is None:
            return
        async for raw in proc.stderr:
            line = raw.decode().strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            # NDJSON envelope: {"event": "qr_code", "data": {"code": "..."}}
            if parsed.get("event") == "qr_code":
                code = parsed.get("data", {}).get("code")
                if code:
                    self.latest_qr = code
                    self.latest_qr_at = time.time()
        await proc.wait()
        if self.auth_proc is proc:
            self.auth_proc = None

    async def stop_auth(self) -> None:
        async with self.lock:
            if self.auth_proc is None:
                return
            self.auth_proc.terminate()
            self.auth_proc = None

    async def fetch_latest_qr(self) -> None:
        # The QR is streamed by the long-lived `auth` process started in
        # start_auth() (see _consume_auth_output). Spawning a second `auth`
        # here would contend for the store lock, so this is a no-op: callers
        # read self.latest_qr, which the streaming consumer keeps current.
        return

    async def sync_once(self) -> dict[str, Any]:
        """Run a single sync pass (non-blocking, no long-lived process)."""
        return await self._run_json(["sync", "--once"])

    async def logout(self) -> None:
        await self.stop_auth()
        await self._run_json(["auth", "logout"])
        try:
            await asyncio.to_thread(lambda: shutil.rmtree(self.store_dir, ignore_errors=True))
        except Exception:
            pass

    async def send_text(self, to: str, text: str) -> dict[str, Any]:
        return await self._run_json(["send", "text", "--to", to, "--message", text])

    async def list_messages(self, limit: int = 100) -> list[dict[str, Any]]:
        res = await self._run_json(["messages", "list", "--limit", str(limit)], read_only=True)
        if res.get("success") is not True:
            return []
        return list(res.get("data", {}).get("messages") or [])

    @staticmethod
    def parse_timestamp(value: str) -> datetime | None:
        if not value:
            return None
        try:
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            return datetime.fromisoformat(value).astimezone(UTC)
        except ValueError:
            return None
