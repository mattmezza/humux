"""Group multi-agent rooms — turn-taking, loop guard, speaker tags (#30).

Covers the three required behaviours from the issue:
- respond-gate (reply only when addressed; otherwise record the turn silently),
- loop guard (never reply to another bot, but still record it),
- speaker tagging (each inbound turn carries a [from <author>] tag),
plus the alternation-safety net (consecutive user turns are coalesced before the
LLM call) and the /new@bot command normalisation.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.agent import AgentCore, _strip_command_suffix
from core.config import GroupChatConfig, TelegramConfig
from core.history import ConversationHistory
from core.llm import _coalesce_user_messages
from core.permissions import PermissionEngine

# ---------------------------------------------------------------------------
# _coalesce_user_messages — the alternation safety net
# ---------------------------------------------------------------------------


def test_coalesce_merges_consecutive_user_strings() -> None:
    out = _coalesce_user_messages(
        [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    )
    assert out == [{"role": "user", "content": "a\n\nb"}]


def test_coalesce_leaves_alternating_untouched() -> None:
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "x"},
        {"role": "user", "content": "b"},
    ]
    assert _coalesce_user_messages(msgs) == msgs


def test_coalesce_does_not_merge_assistant_or_tool() -> None:
    msgs = [
        {"role": "assistant", "content": "x"},
        {"role": "assistant", "content": "y"},  # would never happen, must stay split
        {"role": "user", "content": "a"},
        {"role": "tool", "content": "t"},
    ]
    assert _coalesce_user_messages(msgs) == msgs


def test_coalesce_runs_of_three() -> None:
    out = _coalesce_user_messages(
        [
            {"role": "user", "content": "1"},
            {"role": "user", "content": "2"},
            {"role": "user", "content": "3"},
        ]
    )
    assert out == [{"role": "user", "content": "1\n\n2\n\n3"}]


def test_coalesce_mixed_string_and_blocks_falls_back_to_blocks() -> None:
    out = _coalesce_user_messages(
        [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": [{"type": "text", "text": "world"}, {"type": "image"}]},
        ]
    )
    assert out == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"},
                {"type": "image"},
            ],
        }
    ]


def test_coalesce_does_not_mutate_input() -> None:
    msgs = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    _coalesce_user_messages(msgs)
    assert msgs == [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]


def test_coalesce_empty() -> None:
    assert _coalesce_user_messages([]) == []


# ---------------------------------------------------------------------------
# _strip_command_suffix — /new@bot normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("/new", "/new"),
        ("/new@coachbot", "/new"),
        ("  /Clear  ", "/clear"),
        ("/CLEAR@Bot", "/clear"),
        ("hello there", "hello there"),
        ("email bob@example.com", "email bob@example.com"),  # not a command, no @-split
    ],
)
def test_strip_command_suffix(raw: str, expected: str) -> None:
    assert _strip_command_suffix(raw) == expected


# ---------------------------------------------------------------------------
# AgentCore.process(respond=False) — the silent record path
# ---------------------------------------------------------------------------


def _bare_agent(tmp_path, mode: str) -> AgentCore:
    config = MagicMock()
    config.history.db_path = str(tmp_path / "agent.db")
    config.history.max_turns = 10
    config.history.mode = mode
    agent = object.__new__(AgentCore)
    agent.config = config
    agent.history = ConversationHistory(db_path=config.history.db_path, max_turns=10)
    agent.history_mode = mode
    return agent


async def _silent(agent: AgentCore, message: str):
    return await agent.process(
        message, channel="telegram", user_id="g1", chat_id="g1", respond=False
    )


@pytest.mark.asyncio
async def test_silent_record_injection_returns_empty_and_stores(tmp_path) -> None:
    agent = _bare_agent(tmp_path, "injection")
    resp = await _silent(agent, "[from Bob]\nhi all")
    assert resp.text == ""
    msgs = await agent.history.get_messages("telegram", "g1", "g1")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "[from Bob]\nhi all"


@pytest.mark.asyncio
async def test_silent_records_fold_into_one_user_turn(tmp_path) -> None:
    """Several un-answered group messages stay a single user turn (alternation)."""
    agent = _bare_agent(tmp_path, "injection")
    await _silent(agent, "[from Bob]\none")
    await _silent(agent, "[from Cy]\ntwo")
    msgs = await agent.history.get_messages("telegram", "g1", "g1")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "[from Bob]\none\n\n[from Cy]\ntwo"


@pytest.mark.asyncio
async def test_silent_record_session_mode(tmp_path) -> None:
    agent = _bare_agent(tmp_path, "session")
    await _silent(agent, "[from Bob]\none")
    await _silent(agent, "[from Cy]\ntwo")
    session = await agent.history.get_session("telegram", "g1", "g1")
    assert len(session) == 1
    assert session[0]["role"] == "user"
    assert session[0]["content"] == "[from Bob]\none\n\n[from Cy]\ntwo"


@pytest.mark.asyncio
async def test_silent_record_does_not_clear_on_new(tmp_path) -> None:
    """A silent /new (unaddressed) is recorded, never honoured as a clear."""
    agent = _bare_agent(tmp_path, "injection")
    await agent.history.add_turn("telegram", "g1", "user", "hello", "g1")
    await agent.history.add_turn("telegram", "g1", "assistant", "hi", "g1")
    resp = await _silent(agent, "/new")
    assert resp.text == ""
    msgs = await agent.history.get_messages("telegram", "g1", "g1")
    # The prior exchange survives (a clear would have wiped it); /new is a turn.
    assert [m["content"] for m in msgs] == ["hello", "hi", "/new"]


@pytest.mark.asyncio
async def test_addressed_new_with_bot_suffix_clears(tmp_path) -> None:
    """An addressed "/new@bot" still clears, despite the @-suffix."""
    agent = _bare_agent(tmp_path, "injection")
    await agent.history.add_turn("telegram", "g1", "user", "hello", "g1")
    await agent.history.add_turn("telegram", "g1", "assistant", "hi", "g1")
    resp = await agent.process("/new@coachbot", channel="telegram", user_id="g1", chat_id="g1")
    assert resp.text == "Conversation cleared."
    assert await agent.history.get_messages("telegram", "g1", "g1") == []


@pytest.mark.asyncio
async def test_yolo_toggle_requires_addressing_and_scopes_per_chat(tmp_path) -> None:
    """`/yolo-on` only toggles the addressed bot, and only for the chat it ran in."""
    agent = _bare_agent(tmp_path, "injection")
    agent.permissions = PermissionEngine(db_path=str(tmp_path / "config.db"))
    scope = agent._yolo_scope("telegram", "g1")

    # Unaddressed (a bare "/yolo-on" reaching a gate-off bot): must NOT toggle, so
    # one bare command can't flip every bot in a multi-agent room.
    resp = await agent.process(
        "/yolo-on", channel="telegram", user_id="g1", chat_id="g1", addressed=False
    )
    assert resp.text == ""
    assert agent.permissions.is_yolo(scope) is False

    # Addressed "/yolo-on@coachbot": this agent+chat enters YOLO...
    resp = await agent.process("/yolo-on@coachbot", channel="telegram", user_id="g1", chat_id="g1")
    assert "YOLO mode ON" in resp.text
    assert agent.permissions.is_yolo(scope) is True
    # ...but only this chat — the same bot in another chat is unaffected.
    assert agent.permissions.is_yolo(agent._yolo_scope("telegram", "g2")) is False

    resp = await agent.process("/yolo-off@coachbot", channel="telegram", user_id="g1", chat_id="g1")
    assert "YOLO mode OFF" in resp.text
    assert agent.permissions.is_yolo(scope) is False


# ---------------------------------------------------------------------------
# Telegram channel routing — respond-gate, loop guard, speaker tags
# ---------------------------------------------------------------------------


def _channel(
    group_chat: GroupChatConfig | None = None,
    channel_name: str = "telegram",
    *,
    allowed_user_ids=None,
):
    # group_chat defaults OFF (opt-in); the tests want it ON to exercise the path.
    cfg = TelegramConfig(
        enabled=True,
        bot_token="123456:TESTTOKEN",
        allowed_user_ids=allowed_user_ids or [],
        group_chat=group_chat or GroupChatConfig(enabled=True),
    )
    from channels.telegram import TelegramChannel

    ch = TelegramChannel(cfg, MagicMock(), channel_name=channel_name)
    ch._bot_id = 999
    ch._bot_username = "coachbot"
    # Per-chat trigger/DM gate (#129) is async and defaults permissive here; the
    # tests that exercise it set their own return value.
    ch.agent.may_act_in_chat = AsyncMock(return_value=True)
    return ch


def _ent(type_, text="", *, user=None, offset=0):
    """A MessageEntity-ish double. ``_seg`` is what PTB's UTF-16-safe parser
    returns; ``offset/length`` drive the raw code-point slice fallback."""
    return SimpleNamespace(type=type_, offset=offset, length=len(text), user=user, _seg=text)


def _msg(
    text="",
    *,
    entities=None,
    reply_to=None,
    caption=None,
    caption_entities=None,
    is_topic_message=False,
):
    # Mirror PTB: parse_entity raises on a caption-only message (no .text); the
    # caption entity must be read via parse_caption_entity instead.
    def _parse_entity(ent):
        if not text:
            raise RuntimeError("This Message has no 'text'.")
        return getattr(ent, "_seg", "")

    return SimpleNamespace(
        text=text,
        caption=caption,
        entities=entities or [],
        caption_entities=caption_entities or [],
        reply_to_message=reply_to,
        is_topic_message=is_topic_message,
        parse_entity=_parse_entity,
        parse_caption_entity=lambda ent: getattr(ent, "_seg", ""),
    )


def _user(uid=1, name="Alice", is_bot=False, username=None):
    return SimpleNamespace(id=uid, full_name=name, username=username, is_bot=is_bot)


GROUP = SimpleNamespace(id=-100, type="supergroup")
PRIVATE = SimpleNamespace(id=1, type="private")


def test_is_group_and_convo_user_id() -> None:
    ch = _channel()
    assert ch._is_group(GROUP) is True
    assert ch._is_group(PRIVATE) is False
    assert ch._convo_user_id(GROUP, 1) == "-100"  # shared across senders
    assert ch._convo_user_id(PRIVATE, 7) == "7"


def test_is_group_off_when_disabled() -> None:
    ch = _channel(GroupChatConfig(enabled=False))
    assert ch._is_group(GROUP) is False
    assert ch._convo_user_id(GROUP, 1) == "1"  # falls back to per-sender


@pytest.mark.asyncio
async def test_callback_respects_per_chat_gate() -> None:
    # A /jobs Cancel (and any approve/deny) button is an action in the chat, so a
    # user the resolved agent bars there (#129) must not drive it via the callback.
    ch = _channel()
    ch._finalize_approval_response = AsyncMock()
    ch.agent.subagents = SimpleNamespace(cancel=MagicMock(return_value=True))
    query = SimpleNamespace(data="sub_cancel:run1", message=None, answer=AsyncMock())
    upd = SimpleNamespace(callback_query=query, effective_user=_user(uid=7), effective_chat=GROUP)
    ctx = SimpleNamespace(bot=SimpleNamespace(id=999, username="coachbot"))

    ch.agent.may_act_in_chat = AsyncMock(return_value=False)  # blocked
    await ch._on_approval_callback(upd, ctx)
    ch.agent.subagents.cancel.assert_not_called()
    # Gate was consulted with the turn's own (convo_user, chat_id) keys.
    ch.agent.may_act_in_chat.assert_awaited_once_with("telegram", "-100", "-100", 7)

    ch.agent.may_act_in_chat = AsyncMock(return_value=True)  # allowed
    await ch._on_approval_callback(upd, ctx)
    ch.agent.subagents.cancel.assert_called_once_with("run1")


def test_speaker_name_fallbacks() -> None:
    ch = _channel()
    assert ch._speaker_name(_user(name="Alice")) == "Alice"
    assert ch._speaker_name(SimpleNamespace(id=5, full_name="", username="al")) == "al"
    assert ch._speaker_name(SimpleNamespace(id=5, full_name="", username="")) == "5"
    assert ch._speaker_name(None) == "Unknown"


def test_addressed_via_mention_entity() -> None:
    ch = _channel()
    msg = _msg("@coachbot help me", entities=[_ent("mention", "@coachbot")])
    assert ch._addressed_to_me(msg) is True


def test_addressed_via_command_entity() -> None:
    ch = _channel()
    msg = _msg("/jobs@coachbot", entities=[_ent("bot_command", "/jobs@coachbot")])
    assert ch._addressed_to_me(msg) is True


def test_addressed_via_photo_caption_mention() -> None:
    """A photo caption @mention must be read via parse_caption_entity (a caption
    message has no .text, where parse_entity raises) — and stay UTF-16-safe so an
    emoji before the mention doesn't misalign the slice."""
    ch = _channel()
    # Emoji prefix: a raw code-point slice at the UTF-16 offset would be wrong;
    # only parse_caption_entity yields the correct "@coachbot".
    ent = _ent("mention", "@coachbot", offset=3)
    msg = _msg(text="", caption="😀 @coachbot hi", caption_entities=[ent])
    assert ch._addressed_to_me(msg) is True


def test_addressed_via_reply_to_bot() -> None:
    ch = _channel()
    reply = SimpleNamespace(from_user=SimpleNamespace(id=999))
    assert ch._addressed_to_me(_msg("thanks", reply_to=reply)) is True


def test_addressed_via_text_mention() -> None:
    ch = _channel()
    ent = _ent("text_mention", user=SimpleNamespace(id=999))
    assert ch._addressed_to_me(_msg("hey you", entities=[ent])) is True


def test_not_addressed_to_another_bot() -> None:
    ch = _channel()
    assert ch._addressed_to_me(_msg("@chefbot hi", entities=[_ent("mention", "@chefbot")])) is False
    reply = SimpleNamespace(from_user=SimpleNamespace(id=12345))  # someone else's msg
    assert ch._addressed_to_me(_msg("ok", reply_to=reply)) is False


def test_not_addressed_when_handle_is_prefix_of_another_bot() -> None:
    """@coachbot must NOT consider itself addressed by a mention of @coachbotpro."""
    ch = _channel()
    msg = _msg("@coachbotpro do X", entities=[_ent("mention", "@coachbotpro")])
    assert ch._addressed_to_me(msg) is False


def test_not_addressed_when_shorter_handle_is_prefix() -> None:
    """A bot named @coach must NOT match a mention of the longer @coachbot."""
    ch = _channel()
    ch._bot_username = "coach"
    msg = _msg("@coachbot hi", entities=[_ent("mention", "@coachbot")])
    assert ch._addressed_to_me(msg) is False


def test_not_addressed_by_email_substring() -> None:
    """An email containing the handle (its own entity) must not address the bot."""
    ch = _channel()
    msg = _msg("mail me at admin@coachbot.com", entities=[_ent("email", "admin@coachbot.com")])
    assert ch._addressed_to_me(msg) is False


def test_entity_less_fallback_matches_with_boundary() -> None:
    ch = _channel()
    # No entities at all (forwarded / odd client): a bare handle still addresses.
    assert ch._addressed_to_me(_msg("@coachbot help")) is True
    # ...but a longer handle as a bare substring does not.
    assert ch._addressed_to_me(_msg("@coachbotpro help")) is False


def _route(ch, *, chat=GROUP, user=None, message=None):
    user = user or _user()
    message = message if message is not None else _msg("hello")
    upd = SimpleNamespace(effective_user=user, effective_chat=chat, message=message)
    ctx = SimpleNamespace(bot=SimpleNamespace(id=999, username="coachbot"))
    return ch._turn_routing(upd, message, ctx)


def test_routing_private_chat_is_untouched() -> None:
    ch = _channel()
    r = _route(ch, chat=PRIVATE, user=_user(uid=7))
    assert r == {"user_id": "7", "speaker_tag": "", "respond": True, "addressed": True}


def test_routing_group_unaddressed_human_records_silently() -> None:
    ch = _channel()
    r = _route(ch, user=_user(name="Bob"), message=_msg("just chatting"))
    assert r["respond"] is False
    assert r["user_id"] == "-100"
    assert r["speaker_tag"] == "[from Bob]\n"


def test_routing_group_addressed_human_replies() -> None:
    ch = _channel()
    msg = _msg("@coachbot hi", entities=[_ent("mention", "@coachbot")])
    r = _route(ch, user=_user(name="Alice"), message=msg)
    assert r["respond"] is True
    assert r["addressed"] is True
    assert r["speaker_tag"] == "[from Alice]\n"


def test_routing_loop_guard_ignores_bots_even_if_addressed() -> None:
    ch = _channel()
    msg = _msg("@coachbot hello fellow bot", entities=[_ent("mention", "@coachbot")])
    r = _route(ch, user=_user(name="Chef", is_bot=True), message=msg)
    assert r["respond"] is False  # never reply to another bot
    assert r["speaker_tag"] == "[from Chef (bot)]\n"  # but tagged + recorded


def test_routing_reply_to_all_humans_when_gate_disabled() -> None:
    ch = _channel(GroupChatConfig(enabled=True, reply_when_addressed_only=False))
    r = _route(ch, user=_user(name="Bob"), message=_msg("anything"))
    assert r["respond"] is True
    # ...but the message was NOT addressed to this bot. `addressed` must stay
    # independent of `respond` so a bare "/yolo-on" here can't flip every bot.
    assert r["addressed"] is False
    # bots still ignored even with the address-gate off
    r2 = _route(ch, user=_user(name="Chef", is_bot=True), message=_msg("loop?"))
    assert r2["respond"] is False


def test_routing_keep_bots_when_ignore_disabled() -> None:
    ch = _channel(GroupChatConfig(enabled=True, ignore_bots=False))
    chef = _user(name="Chef", is_bot=True)
    msg = _msg("@coachbot hi", entities=[_ent("mention", "@coachbot")])
    r = _route(ch, user=chef, message=msg)
    assert r["respond"] is True  # addressed bot reply allowed when ignore_bots off


# ---------------------------------------------------------------------------
# _handle_text wiring — silent path skips typing + passes respond through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_text_silent_calls_process_respond_false() -> None:
    ch = _channel()
    ch.agent.process = AsyncMock()
    await ch._handle_text("[from Bob]\nhi", "-100", "-100", respond=False)
    ch.agent.process.assert_awaited_once()
    kwargs = ch.agent.process.await_args.kwargs
    assert kwargs["respond"] is False
    assert kwargs["user_id"] == "-100"
    assert kwargs["message"] == "[from Bob]\nhi"


# ---------------------------------------------------------------------------
# _on_text gate — whitelist applies to replies, not to record-only turns
# ---------------------------------------------------------------------------


async def _drive_on_text(ch, user, message, chat=GROUP):
    upd = SimpleNamespace(effective_user=user, effective_chat=chat, message=message)
    ctx = SimpleNamespace(bot=SimpleNamespace(id=999, username="coachbot"))
    before = set(asyncio.all_tasks())
    await ch._on_text(upd, ctx)
    spawned = [t for t in asyncio.all_tasks() if t not in before]
    if spawned:
        await asyncio.gather(*spawned)


@pytest.mark.asyncio
async def test_whitelist_still_records_other_bots_for_context() -> None:
    """A whitelist gates replies, but a bot's message (respond=False) is still
    recorded — otherwise the loop guard + cross-bot speaker tags would be dead."""
    ch = _channel(allowed_user_ids=[1])  # only human id 1 may drive the bot
    ch.agent.process = AsyncMock()
    chef = _user(uid=555, name="Chef", is_bot=True)
    await _drive_on_text(ch, chef, _msg("hello fellow bot"))
    ch.agent.process.assert_awaited_once()
    kwargs = ch.agent.process.await_args.kwargs
    assert kwargs["respond"] is False
    assert kwargs["message"] == "[from Chef (bot)]\nhello fellow bot"


@pytest.mark.asyncio
async def test_whitelist_blocks_non_allowed_human_reply() -> None:
    """A non-whitelisted human addressing the bot is still blocked from replies."""
    ch = _channel(allowed_user_ids=[1])
    ch.agent.process = AsyncMock()
    intruder = _user(uid=2, name="Mallory")
    msg = _msg("@coachbot do X", entities=[_ent("mention", "@coachbot")])
    await _drive_on_text(ch, intruder, msg)
    ch.agent.process.assert_not_awaited()


@pytest.mark.asyncio
async def test_allowed_human_reply_still_works(tmp_path) -> None:
    ch = _channel(allowed_user_ids=[1])
    ch.agent.process = AsyncMock(
        return_value=SimpleNamespace(voice=None, text="", system_notice=None)
    )
    msg = _msg("@coachbot hi", entities=[_ent("mention", "@coachbot")])
    await _drive_on_text(ch, _user(uid=1, name="Owner"), msg)
    ch.agent.process.assert_awaited_once()
    # The reply path omits the respond kwarg (defaults True).
    assert ch.agent.process.await_args.kwargs.get("respond", True) is True


def _reply_to_me():
    """A message that is a reply to one of THIS bot's (id 999) own messages."""
    return SimpleNamespace(from_user=SimpleNamespace(id=999))


@pytest.mark.asyncio
async def test_command_via_reply_runs_bare_without_quote_prefix() -> None:
    """A "/yolo-on" sent as a reply to this bot still matches: no reply-quote prefix
    is prepended (that would have broken the command match)."""
    ch = _channel()
    ch.agent.process = AsyncMock(
        return_value=SimpleNamespace(voice=None, text="", system_notice=None)
    )
    msg = _msg("/yolo-on@coachbot", reply_to=_reply_to_me())
    await _drive_on_text(ch, _user(uid=1, name="Owner"), msg)
    kwargs = ch.agent.process.await_args.kwargs
    # The reply/respond path omits the respond kwarg (defaults True).
    assert kwargs.get("respond", True) is True
    assert kwargs["addressed"] is True
    assert kwargs["message"] == "/yolo-on@coachbot"  # bare, no [reply_to] wrapper


@pytest.mark.asyncio
async def test_command_targeting_other_bot_is_recorded_not_run() -> None:
    """ "/yolo-on@financebot" sent as a reply to coachbot must NOT toggle coachbot —
    the explicit @handle targets another bot, so it is recorded, not executed."""
    ch = _channel()
    ch.agent.process = AsyncMock()
    msg = _msg("/yolo-on@financebot", reply_to=_reply_to_me())
    await _drive_on_text(ch, _user(uid=1, name="Owner"), msg)
    ch.agent.process.assert_awaited_once()
    assert ch.agent.process.await_args.kwargs["respond"] is False


# ---------------------------------------------------------------------------
# History safety — session tool_result fold guard + fold cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_fold_refuses_tool_result_message(tmp_path) -> None:
    """A silent user record must not be spliced into an in-flight tool_result
    (a role=user message whose content is a list)."""
    hist = ConversationHistory(db_path=str(tmp_path / "h.db"))
    tool_result = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1"}]}
    await hist.append_session_message("telegram", "g1", tool_result, "g1")
    merged = await hist.append_to_last_session_message(
        "telegram", "g1", "\n\n[from Bob]\nchatter", "g1", role="user", text_only=True
    )
    assert merged is False
    session = await hist.get_session("telegram", "g1", "g1")
    # The tool_result block list is untouched (no stray text block appended).
    assert session[0]["content"] == [{"type": "tool_result", "tool_use_id": "tu1"}]


@pytest.mark.asyncio
async def test_fold_cap_refuses_when_oversized(tmp_path) -> None:
    hist = ConversationHistory(db_path=str(tmp_path / "h.db"))
    await hist.add_turn("telegram", "g1", "user", "x" * 100, "g1")
    refused = await hist.append_to_last_turn("telegram", "g1", "user", "y" * 50, "g1", max_len=120)
    assert refused is False  # 100 + 50 > 120
    ok = await hist.append_to_last_turn("telegram", "g1", "user", "z" * 5, "g1", max_len=120)
    assert ok is True


# ---------------------------------------------------------------------------
# Integration: silent records then a reply stay alternation-safe via coalesce
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_silent_then_reply_array_is_alternation_safe(tmp_path) -> None:
    agent = _bare_agent(tmp_path, "injection")
    await _silent(agent, "[from Bob]\none")
    await _silent(agent, "[from Cy]\ntwo")
    # Build the message array the way _process_injection does for the reply turn:
    history = await agent.history.get_messages("telegram", "g1", "g1")
    messages = [{"role": t["role"], "content": t["content"]} for t in history]
    messages.append({"role": "user", "content": "preamble\n\n[from Alice]\n@coachbot help"})
    # The raw array has a consecutive user/user pair (the silent run + the reply)...
    raw = [m["role"] for m in messages]
    assert any(raw[i] == raw[i + 1] == "user" for i in range(len(raw) - 1))
    # ...which the coalescer (called inside llm.generate) collapses before the API.
    collapsed = [m["role"] for m in _coalesce_user_messages(messages)]
    assert all(not (collapsed[i] == collapsed[i + 1] == "user") for i in range(len(collapsed) - 1))
