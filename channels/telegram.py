"""Telegram channel — wires python-telegram-bot to the AgentCore."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReactionTypeEmoji,
    Update,
)
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

# Plain-text commands the agent handles in process() (not Telegram CommandHandlers).
# _on_text sends these bare (no speaker tag / reply-quote) and honours an explicit
# "@bot" target so a command never acts on the wrong bot in a multi-agent room.
_AGENT_COMMANDS = frozenset({"/new", "/clear", "/yolo-on", "/yolo-off"})

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

# Telegram rejects messages over 4096 chars with "Message is too long" (#80).
TELEGRAM_LIMIT = 4096

# Descriptive reaction name → Telegram emoji (#70). The model picks a name from
# the set_reaction tool's enum; only the emojis in Telegram's *default* reaction
# set are accepted by setMessageReaction, so the proposed check/cross/star/rocket/
# muscle/target (which have no native reaction) map to the closest valid emoji
# rather than silently failing.
REACTION_EMOJI: dict[str, str] = {
    "thumbsup": "👍",
    "thumbsdown": "👎",
    "heart": "❤",
    "fire": "🔥",
    "party": "🎉",
    "laugh": "😁",
    "cry": "😢",
    "surprise": "😱",
    "pray": "🙏",
    "100": "💯",
    "thinking": "🤔",
    "eyes": "👀",
    "check": "👌",  # no native ✅ reaction — 👌 reads as "acknowledged/approved"
    "cross": "👎",  # no native ❌ reaction — 👎 reads as "denied"
    "star": "🤩",  # no native ⭐ reaction — 🤩 "star-struck"
    "rocket": "⚡",  # no native 🚀 reaction — ⚡ "fast/launch"
    "clap": "👏",
    "muscle": "🆒",  # no native 💪 reaction
    "handshake": "🤝",
    "target": "🏆",  # no native 🎯 reaction — 🏆 "on target / achieved"
}


def _chunk(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Split ``text`` into pieces no longer than ``limit``, breaking on newlines.

    Splitting the Markdown *source* (before HTML conversion) is safe: the
    fence/bold/italic regexes in markdown_tg require matching delimiters, so a
    chunk boundary inside one leaves the markers as literal text rather than an
    unbalanced tag. A single line longer than ``limit`` is hard-split.
    """
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if not current:
            current = line
        elif len(current) + 1 + len(line) <= limit:
            current += "\n" + line
        else:
            chunks.append(current)
            current = line
    if current:
        chunks.append(current)
    return chunks


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
        # bare "telegram"; a per-agent bot is "telegram:<agent>" (#29), which
        # silos history and resolves straight to that agent.
        self.channel_name = channel_name
        # Last chat a user wrote from, used to route approval prompts. Holds a
        # folded "<chat>:<thread>" string when the message came from a topic.
        self._last_chat_for_user: dict[int, int | str] = {}
        # This bot's own identity, cached lazily from the first update (the Bot
        # API has it only once polling is initialised). Used to detect @mentions
        # and replies aimed at this bot in group rooms (#30).
        self._bot_id: int | None = None
        self._bot_username: str | None = None
        self.app = Application.builder().token(config.bot_token).concurrent_updates(8).build()
        # Commands are checked before the text handler so "/jobs" doesn't reach the
        # agent as an ordinary message. (Plain text — incl. /new, /clear — still
        # falls through to _on_text, which handles those.)
        self.app.add_handler(CommandHandler("jobs", self._on_jobs_command))
        self.app.add_handler(MessageHandler(filters.TEXT, self._on_text))
        self.app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self._on_voice))
        self.app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, self._on_photo))
        # Topic→agent auto-bind only makes sense on the default bot: an agent
        # bot resolves straight to its own agent (rung 0), so a per-topic binding
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

    # -- Group multi-agent rooms (#30) ---------------------------------------

    def _is_group(self, chat, message=None) -> bool:
        """True for a Telegram group/supergroup with group-room behaviour on.

        A genuine forum topic under ``topics_enabled`` is exempt: a topic is its
        own agent-bound 1:1-style context (#14), so group turn-taking would
        wrongly silence its bound agent. Group rooms (#30) apply to the rest of
        the group/supergroup.
        """
        if not (
            chat
            and getattr(chat, "type", "") in ("group", "supergroup")
            and self.config.group_chat.enabled
        ):
            return False
        if (
            self.config.topics_enabled
            and message is not None
            and getattr(message, "is_topic_message", False)
        ):
            return False
        return True

    def _convo_user_id(self, chat, sender_id: int, message=None) -> str:
        """The ``user_id`` key the agent stores history/bindings under.

        In a group room every participant shares one conversation per bot, so the
        group itself is the key — that is what lets a bot see other people's (and
        other bots') messages as inbound turns. In a 1:1 DM (or an exempt forum
        topic) it stays the sender (where ``chat_id == user_id`` anyway), so the
        plain flow is unchanged.
        """
        if self._is_group(chat, message):
            return str(chat.id)
        return str(sender_id)

    def _ensure_bot_identity(self, context) -> None:
        """Cache this bot's id + username from the first update that needs them."""
        if self._bot_id is not None:
            return
        bot = getattr(context, "bot", None) or self.app.bot
        try:
            self._bot_id = bot.id
            uname = bot.username
            self._bot_username = uname.lower() if uname else None
        except RuntimeError, AttributeError:
            # Bot info not populated yet — leave unset; addressing checks return
            # False this once and the next update retries once it is available.
            self._bot_id = None
            self._bot_username = None

    @staticmethod
    def _entity_text(message, text: str, ent) -> str:
        """The substring an entity spans, UTF-16-safe.

        Telegram entity offsets/lengths are UTF-16 code units, so slicing the
        Python str by code point misaligns once a non-BMP char (emoji) precedes
        the entity. Prefer python-telegram-bot's own parser and fall back to a
        code-point slice only for plain test doubles. A photo/document caption
        has no ``.text``, and ``parse_entity`` raises there — use
        ``parse_caption_entity`` for the entity that lives in the caption.
        """
        parse = (
            getattr(message, "parse_caption_entity", None)
            if not getattr(message, "text", None)
            else getattr(message, "parse_entity", None)
        )
        if callable(parse):
            try:
                return parse(ent)
            except Exception:
                pass
        return text[ent.offset : ent.offset + ent.length]

    @staticmethod
    def _speaker_name(user) -> str:
        if user is None:
            return "Unknown"
        name = (
            getattr(user, "full_name", None)
            or getattr(user, "username", None)
            or str(getattr(user, "id", ""))
        )
        return name or "Unknown"

    def _addressed_to_me(self, message) -> bool:
        """True when this message is addressed to THIS bot: a reply to one of the
        bot's own messages, an @mention, a text-mention, or a ``/cmd@bot`` command.

        Addressing is decided by Telegram's own entities (exact handle match, not
        a substring) so a sibling bot whose @username merely contains or extends
        this one's — ``@coach`` vs ``@coachbot``, an email ``x@coachbot.com`` —
        never makes the wrong bot reply, which would re-open the N×-reply storm
        the respond-gate exists to close. A loose substring match is used only as
        a last resort when Telegram supplied no entities at all.
        """
        reply = getattr(message, "reply_to_message", None)
        if reply is not None:
            rf = getattr(reply, "from_user", None)
            if rf is not None and self._bot_id is not None and rf.id == self._bot_id:
                return True
        text = getattr(message, "text", None) or getattr(message, "caption", None) or ""
        entities = list(getattr(message, "entities", None) or []) + list(
            getattr(message, "caption_entities", None) or []
        )
        handle = f"@{self._bot_username}" if self._bot_username else ""
        for ent in entities:
            etype = getattr(ent, "type", "")
            if etype in ("mention", "bot_command") and handle:
                seg = self._entity_text(message, text, ent).lower()
                # "@coachbot" (mention) is an exact match; "/jobs@coachbot"
                # (bot_command) ends with the handle.
                if seg == handle or seg.endswith(handle):
                    return True
            elif etype == "text_mention":
                u = getattr(ent, "user", None)
                if u is not None and self._bot_id is not None and u.id == self._bot_id:
                    return True
        # Entity-less last resort (forwards / odd clients): when Telegram gave us
        # entities we trust them — an absent match means not addressed. Require a
        # boundary so "@coach" can't match inside "@coachbot".
        if entities or not handle:
            return False
        return re.search(rf"(?<![\w@]){re.escape(handle)}(?![\w])", text.lower()) is not None

    def _turn_routing(self, update: Update, message, context) -> dict:
        """Decide how to handle an inbound message in a group room (#30).

        Returns ``user_id`` (the shared conversation key — the group, or the
        sender for a DM), the ``speaker_tag`` to prepend so the agent knows who
        spoke, ``respond`` (reply now, or just record for context), and
        ``addressed`` (was the message explicitly directed at THIS bot). Outside a
        group room every message gets ``respond=True``/``addressed=True`` and no
        tag, so 1:1 DMs are untouched.
        """
        user = update.effective_user
        chat = update.effective_chat
        sender_id = user.id if user else 0
        if not self._is_group(chat, message):
            return {
                "user_id": str(sender_id),
                "speaker_tag": "",
                "respond": True,
                "addressed": True,
            }

        self._ensure_bot_identity(context)
        gc = self.config.group_chat
        is_bot = bool(getattr(user, "is_bot", False))
        marker = " (bot)" if is_bot else ""
        speaker_tag = f"[from {self._speaker_name(user)}{marker}]\n"
        addressed = self._addressed_to_me(message)
        if is_bot and gc.ignore_bots:
            respond = False  # loop guard — record only, never reply to another bot
        elif gc.reply_when_addressed_only and not addressed:
            respond = False  # respond-gate — not addressed, stay silent but record
        else:
            respond = True
        return {
            "user_id": str(chat.id),
            "speaker_tag": speaker_tag,
            "respond": respond,
            "addressed": addressed,
        }

    def _remember_chat(self, convo_user: str, folded) -> None:
        """Note the chat to route an approval prompt back to (keyed by the
        conversational id, so a group's approval lands in the group)."""
        if folded is None:
            return
        try:
            self._last_chat_for_user[int(convo_user)] = folded
        except TypeError, ValueError:
            pass

    async def _on_text(self, update: Update, context) -> None:
        user = update.effective_user
        message = update.message
        chat = update.effective_chat
        if not user or not message:
            return
        sender_id = user.id
        folded = self._fold(chat, message)
        routing = self._turn_routing(update, message, context)
        convo_user = routing["user_id"]
        self._remember_chat(convo_user, folded)
        # The whitelist gates who can make the bot *reply/act*; a record-only turn
        # (respond=False) writes to history but runs no inference and takes no
        # action, so it bypasses the gate — that is what lets the loop guard
        # record other bots and the shared history tag every speaker (#30).
        if routing["respond"] and not self._is_allowed(sender_id):
            return

        chat_id = folded if folded is not None else sender_id
        text = (message.text or "").strip()
        respond = routing["respond"]
        first = text.split(maxsplit=1)[0] if text else ""
        base, _, target = first.partition("@") if first.startswith("/") else ("", "", "")
        if base.lower() in _AGENT_COMMANDS:
            # A recognised command is explicit and self-contained: send it bare, with
            # no speaker tag or reply-quote (either would stop the agent-side match).
            # But "/cmd@otherbot" is aimed at a DIFFERENT bot — even when this message
            # is a reply to us — so this bot must not act on it (which would toggle or
            # clear the wrong agent); record it for context instead.
            payload = text
            if target and self._bot_username and target.lower() != self._bot_username:
                respond = False
        else:
            payload = f"{routing['speaker_tag']}{self._reply_context(message)}{text}"
        asyncio.create_task(
            self._handle_text(
                payload,
                convo_user,
                str(chat_id),
                respond,
                routing["addressed"],
                getattr(message, "message_id", None),
            ),
            name=f"tg-text-{convo_user}",
        )

    async def _on_voice(self, update: Update, context) -> None:
        """Handle incoming voice messages: download, transcribe, process, reply."""
        user = update.effective_user
        message = update.message
        chat = update.effective_chat
        if not user or not message:
            return
        sender_id = user.id
        folded = self._fold(chat, message)
        routing = self._turn_routing(update, message, context)
        convo_user = routing["user_id"]
        self._remember_chat(convo_user, folded)
        # The whitelist gates who can make the bot *reply/act*; a record-only turn
        # (respond=False) writes to history but runs no inference and takes no
        # action, so it bypasses the gate — that is what lets the loop guard
        # record other bots and the shared history tag every speaker (#30).
        if routing["respond"] and not self._is_allowed(sender_id):
            return

        chat_id = folded if folded is not None else sender_id
        reply_context = self._reply_context(message)
        prefix = f"{routing['speaker_tag']}{reply_context}"
        # Respond-gate: a voice message we're staying silent on is recorded as a
        # cheap placeholder — no point downloading/transcribing audio we won't
        # answer (#30).
        if not routing["respond"]:
            await self.agent.process(
                message=f"{prefix}(voice message)",
                channel=self.channel_name,
                user_id=str(convo_user),
                chat_id=str(chat_id),
                respond=False,
            )
            return

        if not self.voice:
            await self.send(
                chat_id, "Voice messages are not supported (voice pipeline not configured)."
            )
            return

        # Download the voice/audio file
        voice_msg = message.voice or message.audio
        if not voice_msg:
            return
        asyncio.create_task(
            self._handle_voice(
                voice_msg.file_id,
                convo_user,
                str(chat_id),
                prefix,
                getattr(message, "message_id", None),
            ),
            name=f"tg-voice-{convo_user}",
        )

    async def _on_photo(self, update: Update, context) -> None:
        """Handle incoming photos and image documents."""
        user = update.effective_user
        message = update.message
        chat = update.effective_chat
        if not user or not message:
            return
        sender_id = user.id
        folded = self._fold(chat, message)
        routing = self._turn_routing(update, message, context)
        convo_user = routing["user_id"]
        self._remember_chat(convo_user, folded)
        # The whitelist gates who can make the bot *reply/act*; a record-only turn
        # (respond=False) writes to history but runs no inference and takes no
        # action, so it bypasses the gate — that is what lets the loop guard
        # record other bots and the shared history tag every speaker (#30).
        if routing["respond"] and not self._is_allowed(sender_id):
            return

        chat_id = folded if folded is not None else sender_id
        caption = message.caption or ""
        reply_context = self._reply_context(message)
        prefix = f"{routing['speaker_tag']}{reply_context}"
        # Respond-gate: record a placeholder for an image we're staying silent
        # on instead of downloading it (#30).
        if not routing["respond"]:
            label = f"{caption} (image)" if caption else "(image)"
            await self.agent.process(
                message=f"{prefix}{label}",
                channel=self.channel_name,
                user_id=str(convo_user),
                chat_id=str(chat_id),
                respond=False,
            )
            return

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

        asyncio.create_task(
            self._handle_photo(
                file_ids,
                caption,
                prefix,
                convo_user,
                str(chat_id),
                getattr(message, "message_id", None),
            ),
            name=f"tg-photo-{convo_user}",
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
                f"🤖 <b>{r.agent or 'default'}</b> · {r.status} · {r.elapsed_str}\n"
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
            await self._finalize_approval_response(query, resolved, "✅ Approved")

        elif data.startswith(_DENY_PREFIX):
            request_id = data[len(_DENY_PREFIX) :]
            resolved = self.agent.permissions.resolve_approval(request_id, False)
            await self._finalize_approval_response(query, resolved, "❌ Denied")

        elif data.startswith(_ALWAYS_PREFIX):
            request_id = data[len(_ALWAYS_PREFIX) :]
            resolved = self.agent.permissions.resolve_approval(request_id, True, always_allow=True)
            await self._finalize_approval_response(query, resolved, "♾️ Always allowed")

        elif data.startswith(_SUB_CANCEL_PREFIX):
            run_id = data[len(_SUB_CANCEL_PREFIX) :]
            ok = self.agent.subagents.cancel(run_id)
            label = "Cancelled" if ok else "Already finished / not found"
            await self._finalize_approval_response(query, True, label)

    async def _on_forum_topic(self, update: Update, context) -> None:
        """Auto-bind a freshly created/renamed forum topic to a matching agent.

        The topic name is only carried on these service messages (not on ordinary
        messages), so this is the one place a topic→agent name match can happen
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
        # Bind under the same conversational id messages resolve with. A forum
        # topic under topics_enabled is exempt from group rooms (#30), so this
        # resolves to the sender id — matching how messages in the topic key,
        # keeping the per-topic agent binding intact.
        convo_user = self._convo_user_id(chat, user_id, message)
        bound = await self.agent.bind_chat_agent_by_label(
            self.channel_name, convo_user, chat_id, name
        )
        if bound:
            await self.send(chat_id, f"Bound this topic to {bound}.")

    # -- Outgoing ------------------------------------------------------------

    async def send(self, chat_id: int | str, text: str) -> None:
        """Send a message to a specific chat (used by scheduler, send_message tool, etc.).

        Long output is chunked to Telegram's 4096-char limit (#80); each chunk
        is a separate message rather than crashing the turn.
        """
        # Agent output is Markdown; render to Telegram HTML unless it already carries HTML tags.
        html_input = bool(_HTML_TAG_RE.search(text))
        cid, kw = self._route(chat_id)
        for chunk in _chunk(text):
            payload = chunk if html_input else to_telegram_html(chunk)
            try:
                await self.app.bot.send_message(cid, payload, parse_mode="HTML", **kw)
            except BadRequest as exc:
                if "parse entities" in str(exc).lower():
                    log.warning("Telegram HTML parse failed; sending as plain text: %s", exc)
                    await self.app.bot.send_message(cid, chunk, **kw)
                    continue
                raise

    async def react(self, chat_id: int | str, message_id: int, emoji: str) -> None:
        """Set an emoji reaction on a message (#70).

        ``emoji`` is a descriptive name from REACTION_EMOJI (validated by the
        caller). A folded "<chat>:<thread>" id is split to its base chat — a
        reaction targets a message, not a thread, so message_thread_id is dropped.
        Telegram only allows reactions on recent messages and rejects an unknown
        emoji; both surface as BadRequest, which is swallowed (a cosmetic ack must
        never fail a turn — issue #70 edge cases).
        """
        char = REACTION_EMOJI.get(emoji)
        if char is None:
            raise ValueError(f"Unsupported reaction: {emoji!r}")
        cid, _ = self._route(chat_id)
        try:
            await self.app.bot.set_message_reaction(
                chat_id=cid, message_id=int(message_id), reaction=[ReactionTypeEmoji(emoji=char)]
            )
        except BadRequest as exc:
            # Expired (>24h), deleted, or an emoji this chat disallows — nothing to
            # recover, and a failed reaction must not break the reply.
            log.info("Reaction skipped on %s/%s: %s", cid, message_id, exc)

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
        # Chunk so a long prompt sends (#80); the keyboard rides the final piece.
        *head, last = _chunk(text)
        for piece in head:
            await self.app.bot.send_message(cid, piece, **kw)
        await self.app.bot.send_message(cid, last, reply_markup=keyboard, **kw)

    # -- Helpers -------------------------------------------------------------

    # Seconds the agent must keep working before the "Thinking…" placeholder is
    # posted — short enough to reassure on a slow turn, long enough that quick
    # replies (already covered by native typing dots) never flash a throwaway
    # bubble. Overridable in tests.
    _PLACEHOLDER_DELAY = 0.1

    @asynccontextmanager
    async def _typing(self, chat_id: int | str):
        """Keep a 'working' signal visible for the whole turn, on every client.

        Telegram Web (K) does not render the ``sendChatAction`` typing indicator
        that mobile and desktop show (#57), but it does render normal messages.
        So two signals run together:

        * the chat action, resent every 4s (it expires after ~5s) — native typing
          dots on clients that honour it;
        * a real, silent placeholder message ("🤔 Thinking…"), which every client
          renders and which doubles as a "the agent is thinking" (CoT) signal. It
          is posted only once the turn is slow and removed before the answer.
        """
        cid, kw = self._route(chat_id)

        async def _send_typing():
            try:
                while True:
                    await self.app.bot.send_chat_action(cid, action=ChatAction.TYPING, **kw)
                    await asyncio.sleep(4)
            except asyncio.CancelledError:
                pass

        turn_over = asyncio.Event()

        async def _placeholder():
            # Wait out the delay; a turn that finishes first never posts (no flash).
            try:
                await asyncio.wait_for(turn_over.wait(), timeout=self._PLACEHOLDER_DELAY)
                return
            except TimeoutError:
                pass
            # disable_notification: deleting a message does NOT retract its push, so
            # a notifying placeholder would ping the user every turn. Keep it silent.
            # ponytail: static "Thinking…". Streaming the real per-step CoT would
            # need a progress callback threaded through agent.process() — add that
            # if the generic signal proves not enough.
            msg = await self.app.bot.send_message(
                cid, "🤔 Thinking…", disable_notification=True, **kw
            )
            await turn_over.wait()  # leave it up for the rest of the turn
            try:
                await self.app.bot.delete_message(cid, msg.message_id)
            except Exception:
                pass

        typing_task = asyncio.create_task(_send_typing(), name=f"tg-typing-{chat_id}")
        # The placeholder owns its full post→delete lifecycle and is signalled via
        # turn_over, never cancelled mid-send, so a turn that ends during the send
        # round-trip can't orphan the bubble (the bot would lose the id to delete).
        placeholder_task = asyncio.ensure_future(_placeholder())
        try:
            yield
        finally:
            turn_over.set()
            typing_task.cancel()
            for t in (typing_task, placeholder_task):
                try:
                    await t
                except asyncio.CancelledError, Exception:
                    pass

    @asynccontextmanager
    async def _progress(self, chat_id: int | str):
        """Mirror an in-flight `browser.py explore` run into ONE edited message.

        explore writes a per-step line to data/browser/last/explore.status; we
        poll it and edit a single Telegram message in place (the chat equivalent
        of the REPL's self-updating spinner line). No-op when nothing is running.
        """
        # ponytail: the explore status file is a single global singleton, so only
        # the default bot mirrors it — otherwise a run triggered via one agent-bot
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

    async def _handle_text(
        self,
        text: str,
        user_id: str,
        chat_id: int | str,
        respond: bool = True,
        addressed: bool = True,
        message_id: int | None = None,
    ) -> None:
        # Respond-gate silent path (#30): record the turn for context, no typing
        # indicator, no reply.
        if not respond:
            await self.agent.process(
                message=text,
                channel=self.channel_name,
                user_id=str(user_id),
                chat_id=str(chat_id),
                respond=False,
            )
            return
        async with self._typing(chat_id), self._progress(chat_id):
            response = await self.agent.process(
                message=text,
                channel=self.channel_name,
                user_id=str(user_id),
                chat_id=str(chat_id),
                addressed=addressed,
                message_id=message_id,
            )
        await self._send_response(chat_id, response)

    async def _handle_voice(
        self,
        file_id: str,
        user_id: str,
        chat_id: int | str,
        prefix: str,
        message_id: int | None = None,
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
            if prefix:
                content = f"{prefix}{content}"
            response = await self.agent.process(
                message=content,
                channel=self.channel_name,
                user_id=str(user_id),
                chat_id=str(chat_id),
                message_id=message_id,
            )
        await self._send_response(chat_id, response)

    async def _handle_photo(
        self,
        file_ids: list[tuple[str, str | None]],
        caption: str,
        prefix: str,
        user_id: str,
        chat_id: int | str,
        message_id: int | None = None,
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
            if prefix:
                content = f"{prefix}{content}" if content else prefix
            response = await self.agent.process(
                message=content,
                channel=self.channel_name,
                user_id=str(user_id),
                attachments=attachments,
                chat_id=str(chat_id),
                message_id=message_id,
            )
        await self._send_response(chat_id, response)

    async def _send_response(self, chat_id: int | str, response) -> None:
        """Send an AgentResponse back — voice or text, then any generated images.

        Text goes through ``send()`` so Markdown is rendered (HTML), and photos are
        sent bare afterwards — simpler and correct, vs. a raw-Markdown caption.
        """
        cid, kw = self._route(chat_id)
        images = [a for a in (getattr(response, "attachments", None) or []) if a.is_image]
        if response.voice:
            await self.app.bot.send_voice(cid, response.voice, **kw)
        elif response.text:
            await self.send(chat_id, response.text)
        elif not images:
            log.warning("Skipping empty response for chat_id=%s", chat_id)
        for att in images:
            await self.app.bot.send_photo(cid, att.data, **kw)
        # Out-of-band system notice (e.g. context compaction), sent separately.
        if getattr(response, "system_notice", None):
            await self.send(chat_id, response.system_notice)

    async def _finalize_approval_response(
        self, query: CallbackQuery, resolved: bool, label: str
    ) -> None:
        if not resolved:
            await query.edit_message_text("(approval request expired or already handled)")
            return
        try:
            await query.edit_message_text(f"{label}.")
            asyncio.create_task(self._delete_after_delay(query))
        except Exception:
            log.exception("Failed to update approval message")

    async def _delete_after_delay(self, query: CallbackQuery, delay: int = 30) -> None:
        """Delete message after a delay to keep chat clean (issue #62)."""
        await asyncio.sleep(delay)
        try:
            await query.message.delete()
        except Exception:
            pass
