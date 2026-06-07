"""WhatsApp channel — bridges a WhatsApp sidecar to the AgentCore."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.agent import AgentCore
    from core.config import WhatsAppConfig
    from core.wacli import WacliManager

log = logging.getLogger(__name__)


_APPROVE_ACTIONS = {"approve", "approved", "yes"}
_DENY_ACTIONS = {"deny", "denied", "no"}
_ALWAYS_ACTIONS = {"always", "allow"}
_SKIP_ACTIONS = {"skip", "skipped"}


def _normalize_number(value: str) -> str:
    if not value:
        return ""
    core = value.strip()
    if "@" in core:
        core = core.split("@", 1)[0]
    return "".join(ch for ch in core if ch.isdigit())


class WhatsAppChannel:
    def __init__(self, config: WhatsAppConfig, agent: AgentCore, wacli: WacliManager):
        self.config = config
        self.agent = agent
        self.wacli = wacli
        self.allowed_numbers = {_normalize_number(n) for n in (config.allowed_numbers or []) if n}

    async def send(self, to: str, text: str) -> None:
        res = await self.wacli.send_text(to, text)
        if res.get("success") is not True:
            error = res.get("error", "Unknown wacli error")
            raise RuntimeError(f"wacli send failed: {error}")

    async def send_approval_request(self, user_id: str, request_id: str, description: str) -> None:
        message = (
            f"Permission request:\n\n{description}\n\n"
            f"Reply with:\n"
            f"- approve {request_id}\n"
            f"- deny {request_id}\n"
            f"- always {request_id}"
        )
        await self.send(user_id, message)

    @staticmethod
    def _reply_context(payload: dict) -> str:
        context = payload.get("context") or {}
        quoted = None
        if isinstance(context, dict):
            quoted = context.get("quotedMessage") or context.get("quoted_message")
        if not quoted:
            return ""

        author = ""
        if isinstance(context, dict):
            author = (
                str(context.get("participant") or "")
                or str(context.get("quotedAuthor") or "")
                or str(context.get("quoted_author") or "")
            )
        author = author.strip() or "Unknown"

        text = ""
        if isinstance(quoted, dict):
            text = str(quoted.get("conversation") or quoted.get("text") or "")
            if not text:
                for key in (
                    "extendedTextMessage",
                    "imageMessage",
                    "videoMessage",
                    "documentMessage",
                ):
                    if key in quoted and isinstance(quoted[key], dict):
                        text = str(
                            quoted[key].get("text")
                            or quoted[key].get("caption")
                            or quoted[key].get("fileName")
                            or ""
                        )
                        if text:
                            break
        text = text.strip()
        if not text:
            text = "(non-text message)"

        return f"[reply_to]\n{author}: {text}\n[/reply_to]\n"

    async def handle_webhook(self, payload: dict) -> dict:
        sender = str(payload.get("from", "")).strip()
        text = str(payload.get("body", "")).strip()
        if not sender or not text:
            return {"ok": False, "error": "Missing sender or message body"}

        if not self._is_allowed(sender):
            log.warning("Ignoring WhatsApp message from unauthorized sender %s", sender)
            return {"ok": False, "error": "Sender not allowed"}

        if await self._maybe_handle_approval(sender, text):
            return {"ok": True, "handled": "approval"}

        # Use chatId from the payload when available (e.g. group chats),
        # otherwise fall back to the sender so that each conversation
        # (private or group) gets its own history.
        chat_id = str(payload.get("chatId", sender)).strip() or sender

        reply_context = self._reply_context(payload)
        content = f"{reply_context}{text}" if reply_context else text
        response = await self.agent.process(
            message=content,
            channel="whatsapp",
            user_id=sender,
            chat_id=chat_id,
        )
        await self.send(sender, response.text)
        if getattr(response, "system_notice", None):
            await self.send(sender, response.system_notice)
        return {"ok": True}

    def _is_allowed(self, sender: str) -> bool:
        if not self.allowed_numbers:
            return True
        normalized = _normalize_number(sender)
        return normalized in self.allowed_numbers

    async def _maybe_handle_approval(self, sender: str, text: str) -> bool:
        tokens = text.strip().lower().split()
        if not tokens:
            return False

        action = tokens[0]
        if action not in _APPROVE_ACTIONS | _DENY_ACTIONS | _ALWAYS_ACTIONS | _SKIP_ACTIONS:
            return False

        request_id = tokens[1] if len(tokens) > 1 else ""
        if not request_id:
            match = re.search(r"\b[a-f0-9]{12}\b", text.lower())
            request_id = match.group(0) if match else ""
        if not request_id:
            await self.send(sender, "Missing approval ID. Reply with: approve <id>.")
            return True

        skipped = action in _SKIP_ACTIONS
        always_allow = action in _ALWAYS_ACTIONS
        approved = action in _APPROVE_ACTIONS or always_allow
        resolved = self.agent.permissions.resolve_approval(
            request_id, approved, always_allow=always_allow, skipped=skipped
        )
        if resolved:
            if skipped:
                label = "Skipped"
            elif always_allow:
                label = "Always allowed"
            elif approved:
                label = "Approved"
            else:
                label = "Denied"
            await self.send(sender, f"{label}.")
        else:
            await self.send(sender, "No pending approval found for that ID.")
        return True
