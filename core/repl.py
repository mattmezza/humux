"""Local REPL channel — talk to the agent from the terminal, no Telegram.

Run:  make repl   (or  uv run python -m core.repl)

Builds the agent from the same config store the server uses, registers itself
as the ``repl`` channel so permission approvals route to a y/n terminal prompt,
then loops on stdin. Ctrl-D or ``/exit`` quits.
"""

from __future__ import annotations

import asyncio
import logging

from core.agent import AgentCore
from core.config_store import ConfigStore

log = logging.getLogger(__name__)

USER_ID = "repl"


class ReplChannel:
    """Minimal channel: prints approval prompts and reads a y/n from stdin."""

    def __init__(self, agent: AgentCore):
        self.agent = agent

    async def send(self, chat_id, text: str) -> None:
        print(f"\n{text}\n")

    async def send_approval_request(self, user_id: str, request_id: str, description: str) -> None:
        ans = await asyncio.to_thread(input, f"\n[approval] {description}\nallow? [y/N] ")
        self.agent.permissions.resolve_approval(request_id, ans.strip().lower() in ("y", "yes"))


async def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s — %(message)s")

    store = ConfigStore()
    await store.seed_if_empty()
    await store.ensure_admin_password()
    config = await store.export_to_config()

    agent = AgentCore(config)
    agent.channels["repl"] = ReplChannel(agent)

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
        response = await agent.process(
            message=text, channel="repl", user_id=USER_ID, chat_id=USER_ID
        )
        if response.text:
            print(f"\n{response.text}\n")
        if getattr(response, "system_notice", None):
            print(f"[system] {response.system_notice}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
