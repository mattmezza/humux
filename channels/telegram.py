"""Telegram channel — wires python-telegram-bot to the AgentCore."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from telegram import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from channels.markdown_tg import to_telegram_html

if TYPE_CHECKING:
    from core.agent import AgentCore
    from core.config import TelegramConfig
    from core.models import Attachment
    from voice.pipeline import VoicePipeline

log = logging.getLogger(__name__)

# Callback data prefix for approval buttons
_APPROVE_PREFIX = "perm_approve:"
_DENY_PREFIX = "perm_deny:"
_ALWAYS_PREFIX = "perm_always:"
# Callback data prefix for subagent-run cancel buttons (issue #15)
_SUB_CANCEL_PREFIX = "sub_cancel:"
_HTML_TAG_RE = re.compile(
    r"</?(b|strong|i|em|u|ins|s|strike|del|code|pre|a|tg-spoiler)(\s+[^>]*)?>",
    re.IGNORECASE,
)


class TelegramChannel:
    def __init__(
        self,
        config: TelegramConfig,
        agent: AgentCore,
        voice: VoicePipeline | None = None,
        channel_name: str = "telegram",
    ):
        self.config = config
        self.agent = agent
        self.voice = voice
        # The channel string this bot reports to the agent. The default bot is
        # bare "telegram"; a per-persona bot is "telegram:<persona>" (#29), which
        # silos history and resolves straight to that persona.
        self.channel_name = channel_name
        # Last chat a user wrote from, used to route approval prompts. Holds a
        # folded "<chat>:<thread>" string when the message came from a topic.
        self._last_chat_for_user: dict[int, int | str] = {}
        self.app = Application.builder().token(config.bot_token).concurrent_updates(8).build()
        # Commands are checked before the text handler so "/jobs" doesn't reach the
        # agent as an ordinary message. (Plain text — incl. /new, /clear — still
        # falls through to _on_text, which handles those.)
        self.app.add_handler(CommandHandler("jobs", self._on_jobs_command))
        self.app.add_handler(MessageHandler(filters.TEXT, self._on_text))
        self.app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self._on_voice))
        self.app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, self._on_photo))
        # Topic→persona auto-bind only makes sense on the default bot: a persona
        # bot resolves straight to its own persona (rung 0), so a per-topic binding
        # would be ignored. Topic *folding* (history isolation) still applies below.
        if config.topics_enabled and channel_name == "telegram":
            self.app.add_handler(
                MessageHandler(
                    filters.StatusUpdate.FORUM_TOPIC_CREATED
                    | filters.StatusUpdate.FORUM_TOPIC_EDITED,
                    self._on_forum_topic,
                )
            )
        self.app.add_handler(CallbackQueryHandler(self._on_approval_callback))

    # -- Topic folding helpers -----------------------------------------------

    def _fold(self, chat, message) -> int | str | None:
        """Derive the context id for a message.

        With ``topics_enabled``, a forum-topic message folds its
        ``message_thread_id`` into the chat id as ``"<chat>:<thread>"`` so each
        topic is a separate context. The forum's General topic carries no thread
        id, so it maps to the bare chat (the default context). Returns ``None``
        when there is no chat (caller falls back to the user id).
        """
        if not chat:
            return None
        if not self.config.topics_enabled:
            return chat.id
        # message_thread_id is also set on reply-chains in non-forum groups and on
        # linked-discussion comments (there it is just the root message id), so it
        # alone would fragment an ordinary chat. is_topic_message marks a genuine
        # forum topic and is False for the General topic — gate on it.
        thread = getattr(message, "message_thread_id", None)
        if thread and getattr(message, "is_topic_message", False):
            return f"{chat.id}:{thread}"
        return chat.id

    @staticmethod
    def _route(chat_id: int | str) -> tuple[int | str, dict]:
        """Split a folded ``"<chat>:<thread>"`` id for the Bot API.

        Returns ``(chat_id, kwargs)`` where kwargs carries ``message_thread_id``
        when a topic is encoded, and is empty otherwise — so non-topic calls are
        unchanged.
        """
        base, sep, thread = str(chat_id).partition(":")
        if sep and thread.isdigit() and base.lstrip("-").isdigit():
            return int(base), {"message_thread_id": int(thread)}
        return chat_id, {}

    # -- Incoming handlers ---------------------------------------------------

    def _reply_context(self, message) -> str:
        reply = getattr(message, "reply_to_message", None)
        if not reply:
            return ""
        author = ""
        from_user = getattr(reply, "from_user", None)
        if from_user:
            author = from_user.full_name or from_user.username or str(from_user.id)
        sender_chat = getattr(reply, "sender_chat", None)
        if not author and sender_chat:
            author = sender_chat.title or sender_chat.username or str(sender_chat.id)
        if not author:
            author = "Unknown"

        text = getattr(reply, "text", None) or getattr(reply, "caption", None) or ""
        if not text:
            if getattr(reply, "photo", None):
                text = "(photo)"
            elif getattr(reply, "document", None):
                filename = getattr(reply.document, "file_name", None)
                text = f"(document: {filename})" if filename else "(document)"
            elif getattr(reply, "voice", None):
                text = "(voice message)"
            elif getattr(reply, "audio", None):
                text = "(audio message)"
            else:
                text = "(non-text message)"

        return f"[reply_to]\n{author}: {text}\n[/reply_to]\n"

    async def _on_text(self, update: Update, context) -> None:
        user = update.effective_user
        message = update.message
        chat = update.effective_chat
        if not user or not message:
            return
        user_id = user.id
        folded = self._fold(chat, message)
        if folded is not None:
            self._last_chat_for_user[user_id] = folded
        if not self._is_allowed(user_id):
            return

        chat_id = folded if folded is not None else user_id
        reply_context = self._reply_context(message)
        text = (message.text or "").strip()
        payload = f"{reply_context}{text}" if reply_context else text
        asyncio.create_task(
            self._handle_text(payload, user_id, chat_id),
            name=f"tg-text-{user_id}",
        )

    async def _on_voice(self, update: Update, context) -> None:
        """Handle incoming voice messages: download, transcribe, process, reply."""
        user = update.effective_user
        message = update.message
        chat = update.effective_chat
        if not user or not message:
            return
        user_id = user.id
        folded = self._fold(chat, message)
        if folded is not None:
            self._last_chat_for_user[user_id] = folded
        if not self._is_allowed(user_id):
            return

        chat_id = folded if folded is not None else user_id
        if not self.voice:
            await self.send(
                chat_id, "Voice messages are not supported (voice pipeline not configured)."
            )
            return

        # Download the voice/audio file
        voice_msg = message.voice or message.audio
        if not voice_msg:
            return
        reply_context = self._reply_context(message)
        asyncio.create_task(
            self._handle_voice(voice_msg.file_id, user_id, chat_id, reply_context),
            name=f"tg-voice-{user_id}",
        )

    async def _on_photo(self, update: Update, context) -> None:
        """Handle incoming photos and image documents."""
        user = update.effective_user
        message = update.message
        chat = update.effective_chat
        if not user or not message:
            return
        user_id = user.id
        folded = self._fold(chat, message)
        if folded is not None:
            self._last_chat_for_user[user_id] = folded
        if not self._is_allowed(user_id):
            return

        chat_id = folded if folded is not None else user_id
        caption = message.caption or ""

        # Collect file IDs to download.
        # Photos: Telegram sends multiple sizes; pick the largest (last).
        # Document: a single file with a known mime type.
        file_ids: list[tuple[str, str | None]] = []  # (file_id, mime_type | None)
        if message.photo:
            largest = message.photo[-1]
            file_ids.append((largest.file_id, None))  # Telegram photos are always JPEG
        if message.document and message.document.mime_type:
            file_ids.append((message.document.file_id, message.document.mime_type))

        if not file_ids:
            return

        reply_context = self._reply_context(message)
        asyncio.create_task(
            self._handle_photo(file_ids, caption, reply_context, user_id, chat_id),
            name=f"tg-photo-{user_id}",
        )

    async def _on_jobs_command(self, update: Update, context) -> None:
        """/jobs — list active subagent runs with inline cancel buttons (issue #15)."""
        user = update.effective_user
        message = update.message
        if not user or not message:
            return
        if not self._is_allowed(user.id):
            return
        runs = self.agent.subagents.list_runs(active_only=True)
        if not runs:
            await message.reply_text("No active subagent runs.")
            return
        for r in runs:
            text = (
                f"🤖 <b>{r.persona or 'default'}</b> · {r.status} · {r.elapsed_str}\n"
                f"{(r.progress or '—')}\n"
                f"<i>{r.task[:160]}</i>"
            )
            keyboard = InlineKeyboardMarkup(
                [[InlineKeyboardButton("Cancel", callback_data=f"{_SUB_CANCEL_PREFIX}{r.run_id}")]]
            )
            await message.reply_text(text, parse_mode="HTML", reply_markup=keyboard)

    async def _on_approval_callback(self, update: Update, context) -> None:
        """Handle inline keyboard button presses for permission approvals."""
        query = update.callback_query
        user = update.effective_user
        chat = update.effective_chat
        if not query or not user:
            return
        await query.answer()  # Acknowledge the button press

        user_id = user.id
        folded = self._fold(chat, getattr(query, "message", None))
        if folded is not None:
            self._last_chat_for_user[user_id] = folded
        if not self._is_allowed(user_id):
            return

        data = query.data or ""

        if data.startswith(_APPROVE_PREFIX):
            request_id = data[len(_APPROVE_PREFIX) :]
            resolved = self.agent.permissions.resolve_approval(request_id, True)
            await self._finalize_approval_response(query, resolved, "Approved")

        elif data.startswith(_DENY_PREFIX):
            request_id = data[len(_DENY_PREFIX) :]
            resolved = self.agent.permissions.resolve_approval(request_id, False)
            await self._finalize_approval_response(query, resolved, "Denied")

        elif data.startswith(_ALWAYS_PREFIX):
            request_id = data[len(_ALWAYS_PREFIX) :]
            resolved = self.agent.permissions.resolve_approval(request_id, True, always_allow=True)
            await self._finalize_approval_response(query, resolved, "Always allowed")

        elif data.startswith(_SUB_CANCEL_PREFIX):
            run_id = data[len(_SUB_CANCEL_PREFIX) :]
            ok = self.agent.subagents.cancel(run_id)
            label = "Cancelled" if ok else "Already finished / not found"
            await self._finalize_approval_response(query, True, label)

    async def _on_forum_topic(self, update: Update, context) -> None:
        """Auto-bind a freshly created/renamed forum topic to a matching persona.

        The topic name is only carried on these service messages (not on ordinary
        messages), so this is the one place a topic→persona name match can happen
        without a web round-trip. Binding is skipped when the topic is already
        bound, so a manual rebind is never clobbered.
        """
        message = update.message
        chat = update.effective_chat
        user = update.effective_user
        if not message or not chat:
            return
        created = getattr(message, "forum_topic_created", None)
        edited = getattr(message, "forum_topic_edited", None)
        name = getattr(created or edited, "name", None)
        thread = getattr(message, "message_thread_id", None)
        if not name or not thread:
            return
        user_id = user.id if user else None
        if user_id is None or not self._is_allowed(user_id):
            return
        chat_id = f"{chat.id}:{thread}"
        bound = await self.agent.bind_chat_persona_by_label(
            self.channel_name, str(user_id), chat_id, name
        )
        if bound:
            await self.send(chat_id, f"Bound this topic to {bound}.")

    # -- Outgoing ------------------------------------------------------------

    async def send(self, chat_id: int | str, text: str) -> None:
        """Send a message to a specific chat (used by scheduler, send_message tool, etc.)."""
        # Agent output is Markdown; render to Telegram HTML unless it already carries HTML tags.
        if _HTML_TAG_RE.search(text):
            html = text
        else:
            html = to_telegram_html(text)
        cid, kw = self._route(chat_id)
        try:
            await self.app.bot.send_message(cid, html, parse_mode="HTML", **kw)
        except BadRequest as exc:
            if "parse entities" in str(exc).lower():
                log.warning("Telegram HTML parse failed; sending as plain text: %s", exc)
                await self.app.bot.send_message(cid, text, **kw)
                return
            raise

    async def send_approval_request(
        self, user_id: str, request_id: str, description: str, image_path: str | None = None
    ) -> None:
        """Send a permission approval prompt with Approve/Deny inline buttons.

        When ``image_path`` is given (e.g. a browser screenshot), send it as a
        photo with the buttons so the user can watch and approve from their phone.
        """
        target_id = int(user_id)
        chat_id = self._last_chat_for_user.get(target_id, target_id)
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Approve", callback_data=f"{_APPROVE_PREFIX}{request_id}"),
                    InlineKeyboardButton("Deny", callback_data=f"{_DENY_PREFIX}{request_id}"),
                    InlineKeyboardButton(
                        "Always allow", callback_data=f"{_ALWAYS_PREFIX}{request_id}"
                    ),
                ]
            ]
        )
        text = f"Permission request:\n\n{description}"
        cid, kw = self._route(chat_id)
        if image_path:
            try:
                with open(image_path, "rb") as photo:
                    # Telegram caption hard limit is 1024 chars.
                    await self.app.bot.send_photo(
                        cid, photo, caption=text[:1024], reply_markup=keyboard, **kw
                    )
                return
            except Exception:
                log.exception("Failed to send approval screenshot; falling back to text")
        await self.app.bot.send_message(cid, text, reply_markup=keyboard, **kw)

    # -- Helpers -------------------------------------------------------------

    @asynccontextmanager
    async def _typing(self, chat_id: int | str):
        """Send 'typing' chat action continuously until the wrapped block completes.

        Telegram's typing indicator expires after ~5 seconds, so we resend it
        every 4 seconds to keep it visible for the duration of agent processing.
        """
        cid, kw = self._route(chat_id)

        async def _send_typing():
            try:
                while True:
                    await self.app.bot.send_chat_action(cid, action=ChatAction.TYPING, **kw)
                    await asyncio.sleep(4)
            except asyncio.CancelledError:
                pass

        task = asyncio.create_task(_send_typing(), name=f"tg-typing-{chat_id}")
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @asynccontextmanager
    async def _progress(self, chat_id: int | str):
        """Mirror an in-flight `browser.py explore` run into ONE edited message.

        explore writes a per-step line to data/browser/last/explore.status; we
        poll it and edit a single Telegram message in place (the chat equivalent
        of the REPL's self-updating spinner line). No-op when nothing is running.
        """
        # ponytail: the explore status file is a single global singleton, so only
        # the default bot mirrors it — otherwise a run triggered via one persona-bot
        # would bubble into every other bot's chat (#29). Per-run scoping (a status
        # path keyed by channel/profile) belongs in the browser tool — follow-up.
        if self.channel_name != "telegram":
            yield
            return
        status = Path("/app/data" if Path("/app/data").exists() else "data")
        status = status / "browser" / "last" / "explore.status"
        cid, kw = self._route(chat_id)  # split a folded "<chat>:<thread>" topic id
        message_id: int | None = None
        last = None

        async def _poll():
            nonlocal message_id, last
            while True:
                await asyncio.sleep(3)
                try:
                    if time.time() - status.stat().st_mtime > 10:
                        continue  # stale → no run active
                    text = "🌐 " + status.read_text().strip()[:120]
                except OSError:
                    continue
                if text == last:
                    continue  # Telegram rejects no-op edits
                last = text
                try:
                    if message_id is None:
                        msg = await self.app.bot.send_message(cid, text, **kw)
                        message_id = msg.message_id
                    else:
                        await self.app.bot.edit_message_text(
                            text, chat_id=cid, message_id=message_id
                        )
                except Exception:
                    pass  # transient edit/rate-limit error — keep polling

        task = asyncio.create_task(_poll(), name=f"tg-progress-{chat_id}")
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            if message_id is not None:  # tidy the progress message away
                try:
                    await self.app.bot.delete_message(cid, message_id)
                except Exception:
                    pass

    def _is_allowed(self, user_id: int) -> bool:
        if self.config.allowed_user_ids and user_id not in self.config.allowed_user_ids:
            log.warning("Ignoring message from unauthorized user %s", user_id)
            return False
        return True

    async def _handle_text(self, text: str, user_id: int, chat_id: int | str) -> None:
        async with self._typing(chat_id), self._progress(chat_id):
            response = await self.agent.process(
                message=text,
                channel=self.channel_name,
                user_id=str(user_id),
                chat_id=str(chat_id),
            )
        await self._send_response(chat_id, response)

    async def _handle_voice(
        self, file_id: str, user_id: int, chat_id: int | str, reply_context: str
    ) -> None:
        async with self._typing(chat_id), self._progress(chat_id):
            file = await self.app.bot.get_file(file_id)
            audio_bytes = await file.download_as_bytearray()

            # Transcribe via Whisper
            log.info(
                "Transcribing voice message from user %s (%d bytes)", user_id, len(audio_bytes)
            )
            voice = self.voice
            if not voice:
                await self.send(
                    chat_id, "Voice messages are not supported (voice pipeline not configured)."
                )
                return
            transcript = await voice.transcribe(bytes(audio_bytes))

            if not transcript.strip():
                await self.send(chat_id, "(could not transcribe voice)")
                return

            log.info("Transcript: %s", transcript[:200])

            content = f"[voice] {transcript}"
            if reply_context:
                content = f"{reply_context}{content}"
            response = await self.agent.process(
                message=content,
                channel=self.channel_name,
                user_id=str(user_id),
                chat_id=str(chat_id),
            )
        await self._send_response(chat_id, response)

    async def _handle_photo(
        self,
        file_ids: list[tuple[str, str | None]],
        caption: str,
        reply_context: str,
        user_id: int,
        chat_id: int | str,
    ) -> None:
        from core.models import IMAGE_MIME_TYPES, Attachment

        async with self._typing(chat_id), self._progress(chat_id):
            attachments: list[Attachment] = []
            for file_id, mime_type in file_ids:
                file = await self.app.bot.get_file(file_id)
                data = bytes(await file.download_as_bytearray())
                # Telegram photos are always JPEG; documents carry their own mime.
                resolved_mime = mime_type or "image/jpeg"
                if resolved_mime not in IMAGE_MIME_TYPES:
                    log.info("Skipping non-image attachment: %s", resolved_mime)
                    continue
                attachments.append(Attachment(data=data, mime_type=resolved_mime))
                log.info(
                    "Downloaded image from user %s (%d bytes, %s)",
                    user_id,
                    len(data),
                    resolved_mime,
                )

            if not attachments:
                await self.send(
                    chat_id, "Sorry, I can only process image files (JPEG, PNG, GIF, WebP) for now."
                )
                return

            content = caption.strip()
            if reply_context:
                content = f"{reply_context}{content}" if content else reply_context
            response = await self.agent.process(
                message=content,
                channel=self.channel_name,
                user_id=str(user_id),
                attachments=attachments,
                chat_id=str(chat_id),
            )
        await self._send_response(chat_id, response)

    async def _send_response(self, chat_id: int | str, response) -> None:
        """Send an AgentResponse back — voice if present, otherwise text."""
        if response.voice:
            cid, kw = self._route(chat_id)
            await self.app.bot.send_voice(cid, response.voice, **kw)
        elif response.text:
            await self.send(chat_id, response.text)
        else:
            log.warning("Skipping empty response for chat_id=%s", chat_id)
        # Out-of-band system notice (e.g. context compaction), sent separately.
        if getattr(response, "system_notice", None):
            await self.send(chat_id, response.system_notice)

    async def _finalize_approval_response(
        self, query: CallbackQuery, resolved: bool, label: str
    ) -> None:
        if not resolved:
            await query.edit_message_text("(approval request expired or already handled)")
            return
        message = query.message
        try:
            text = getattr(message, "text", None) if message else None
            if text:
                await query.edit_message_text(text + f"\n\n--- {label}")
            else:
                await self.app.bot.send_message(query.from_user.id, f"{label}.")
        except Exception:
            log.exception("Failed to update approval message")
