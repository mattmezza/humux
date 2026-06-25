"""Local REPL channel — talk to the agent from the terminal, no Telegram.

Run:  make repl   (or  uv run python -m core.repl)

Builds the agent from the same config store the server uses, registers itself
as the ``repl`` channel so permission approvals route to a y/n terminal prompt,
then loops on stdin. Ctrl-D or ``/exit`` quits.

While the agent works, a spinner shows it's busy and the chain of thought
(model reasoning + each tool call) streams live above it.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import sys
import time

from core.agent import AgentCore
from core.config_store import ConfigStore

try:  # POSIX-only: lets us watch for an ESC keypress mid-turn
    import termios
    import tty
except ImportError:  # pragma: no cover - non-POSIX
    termios = tty = None

try:  # readline auto-hooks input() for ↑/↓ history + line editing
    import readline
except ImportError:  # pragma: no cover - non-POSIX
    readline = None

log = logging.getLogger(__name__)

USER_ID = "repl"

# Loggers whose INFO output is the agent's "chain of thought" / activity trail.
_THOUGHT_LOGGERS = ("core.agent", "core.executor", "core.llm.reasoning")
_NOISY_LOGGERS = ("httpx", "httpcore", "apscheduler", "telegram")


_DIM = "\033[2m"  # thinking / reasoning — low contrast
_CYAN = "\033[36m"  # tool calls / agent activity — stands out
_RESET = "\033[0m"


class _SpinnerHandler(logging.Handler):
    """Prints log lines above the spinner, clearing its line first.

    Reasoning (``core.llm.reasoning``) renders dim; everything else
    (tool calls, agent activity) renders cyan so it stands out.
    """

    def __init__(self, spinner: Spinner):
        super().__init__()
        self.spinner = spinner

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage().startswith("Processing message"):
            return  # redundant in a REPL — you just typed it (and it shows "repl/repl/repl")
        color = _DIM if record.name == "core.llm.reasoning" else _CYAN
        line = f"  {color}· {record.getMessage()}{_RESET}"
        sys.stderr.write("\r\033[K" + line + "\n")
        sys.stderr.flush()
        self.spinner.redraw()


class Spinner:
    """Background \\r spinner on stderr. Start before a turn, stop after."""

    _frames = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._start = 0.0
        self._frame = "⠋"

    def redraw(self) -> None:
        if self._task is None:  # not running — startup/idle log records mustn't draw it
            return
        sys.stderr.write(f"\r\033[K\033[2m{self._frame} thinking… {self._elapsed():.0f}s\033[0m")
        sys.stderr.flush()

    def _elapsed(self) -> float:
        return time.monotonic() - self._start

    async def _run(self) -> None:
        while True:
            self._frame = next(self._frames)
            self.redraw()
            await asyncio.sleep(0.1)

    def start(self) -> None:
        self._start = time.monotonic()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()


class ReplChannel:
    """Minimal channel: prints approval prompts and reads a y/n from stdin."""

    def __init__(self, agent: AgentCore, spinner: Spinner):
        self.agent = agent
        self.spinner = spinner
        # Set per-turn by _run_turn: release/reclaim stdin from the ESC watcher
        # so the approval prompt can read it (else _on_key eats the keystrokes).
        self.pause_keys = None
        self.resume_keys = None

    async def send(self, chat_id, text: str) -> None:
        print(f"\n{text}\n")

    async def send_approval_request(self, user_id: str, request_id: str, description: str) -> None:
        await self.spinner.stop()  # don't fight the prompt for the line
        if self.pause_keys:
            self.pause_keys()
        hist_len = readline.get_current_history_length() if readline else 0
        try:
            ans = await asyncio.to_thread(
                input, f"\n[approval] {description}\nApprove? Always|Yes|[No] "
            )
        finally:
            if self.resume_keys:
                self.resume_keys()
        # Keep the approval reply out of ↑/↓ history — only real prompts belong there.
        if readline and readline.get_current_history_length() > hist_len:
            readline.remove_history_item(hist_len)
        c = ans.strip().lower()[:1]  # first char: a→always, y→yes, else (incl. Enter)→deny
        self.agent.permissions.resolve_approval(
            request_id, approved=c in ("a", "y"), always_allow=c == "a"
        )
        self.spinner.start()


def _setup_logging(spinner: Spinner) -> None:
    handler = _SpinnerHandler(spinner)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.WARNING)
    for name in _THOUGHT_LOGGERS:
        logging.getLogger(name).setLevel(logging.INFO)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _print_debug_config(config) -> None:
    a = config.agent
    th = a.thinking_level or "off"
    rows = [
        ("agent", f"{a.name} (owner {a.owner_name})"),
        ("inference", f"{a.llm_provider} / {a.model}  thinking={th}"),
        ("memory", f"{config.memory.extraction_provider}/{config.memory.extraction_model}"),
        ("history", config.history.mode),
        ("voice", "on" if config.voice.tts_enabled else "off"),
        ("timezone", a.timezone),
    ]
    print(f"\n{_CYAN}── REPL debug config ──{_RESET}")
    for k, v in rows:
        print(f"  {_DIM}{k:>10}{_RESET}  {v}")
    print("\nESC interrupts a turn · /clear resets context · Ctrl-D or /exit quits.\n")


async def _run_turn(agent: AgentCore, spinner: Spinner, text: str):
    """Run one turn, cancellable by pressing ESC. Returns None if interrupted."""
    proc = asyncio.create_task(
        agent.process(message=text, channel="repl", user_id=USER_ID, chat_id=USER_ID)
    )
    fd = sys.stdin.fileno()
    loop = asyncio.get_running_loop()
    watch = termios is not None and sys.stdin.isatty()
    old = termios.tcgetattr(fd) if watch else None

    def _on_key() -> None:
        # A lone ESC (b"\x1b") interrupts; escape sequences (arrows) read longer → ignore.
        try:
            if os.read(fd, 16) == b"\x1b":
                proc.cancel()
        except OSError:
            pass

    chan = agent.channels.get("repl")

    def _pause() -> None:  # hand stdin back to a blocking input() prompt
        loop.remove_reader(fd)
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _resume() -> None:
        tty.setcbreak(fd)
        loop.add_reader(fd, _on_key)

    if watch:
        tty.setcbreak(fd)
        loop.add_reader(fd, _on_key)
        if chan:
            chan.pause_keys, chan.resume_keys = _pause, _resume
    spinner.start()
    try:
        return await proc
    except asyncio.CancelledError:
        return None
    finally:
        if watch:
            loop.remove_reader(fd)
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            if chan:
                chan.pause_keys = chan.resume_keys = None
        await spinner.stop()


async def main() -> None:
    spinner = Spinner()
    _setup_logging(spinner)

    store = ConfigStore()
    await store.seed_if_empty()
    await store.ensure_admin_password()
    config = await store.export_to_config()

    agent = AgentCore(config)
    agent.channels["repl"] = ReplChannel(agent, spinner)

    _print_debug_config(config)

    while True:
        try:
            text = await asyncio.to_thread(input, "> ")
        except EOFError:
            break
        text = text.strip()
        if not text:
            continue
        if text in ("/exit", "/quit"):
            break
        if text == "/clear":
            await agent.history.clear("repl", USER_ID, USER_ID)
            print("[context cleared]\n")
            continue
        response = await _run_turn(agent, spinner, text)
        if response is None:
            print("\n[interrupted]\n")
            continue
        if response.text:
            print(f"\n{response.text}\n")
        if getattr(response, "system_notice", None):
            print(f"[system] {response.system_notice}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
