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
import sys
import time

from core.agent import AgentCore
from core.config_store import ConfigStore

log = logging.getLogger(__name__)

USER_ID = "repl"

# Loggers whose INFO output is the agent's "chain of thought" / activity trail.
_THOUGHT_LOGGERS = ("core.agent", "core.executor", "core.llm.reasoning")
_NOISY_LOGGERS = ("httpx", "httpcore", "apscheduler", "telegram")


class _SpinnerHandler(logging.Handler):
    """Prints log lines above the spinner, clearing its line first."""

    def __init__(self, spinner: Spinner):
        super().__init__()
        self.spinner = spinner
        self.setFormatter(logging.Formatter("  \033[2m· %(message)s\033[0m"))

    def emit(self, record: logging.LogRecord) -> None:
        sys.stderr.write("\r\033[K" + self.format(record) + "\n")
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

    async def send(self, chat_id, text: str) -> None:
        print(f"\n{text}\n")

    async def send_approval_request(self, user_id: str, request_id: str, description: str) -> None:
        await self.spinner.stop()  # don't fight the prompt for the line
        ans = await asyncio.to_thread(input, f"\n[approval] {description}\nallow? [y/N] ")
        self.agent.permissions.resolve_approval(request_id, ans.strip().lower() in ("y", "yes"))
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


async def main() -> None:
    spinner = Spinner()
    _setup_logging(spinner)

    store = ConfigStore()
    await store.seed_if_empty()
    await store.ensure_admin_password()
    config = await store.export_to_config()

    agent = AgentCore(config)
    agent.channels["repl"] = ReplChannel(agent, spinner)

    print(
        f"{config.agent.name} REPL — model={config.agent.model} "
        f"provider={config.agent.llm_provider}. Ctrl-D or /exit to quit.\n"
    )

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
        spinner.start()
        try:
            response = await agent.process(
                message=text, channel="repl", user_id=USER_ID, chat_id=USER_ID
            )
        finally:
            await spinner.stop()
        if response.text:
            print(f"\n{response.text}\n")
        if getattr(response, "system_notice", None):
            print(f"[system] {response.system_notice}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
