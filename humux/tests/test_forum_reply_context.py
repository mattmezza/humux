from types import SimpleNamespace

from channels.telegram import TelegramChannel

_rc = TelegramChannel._reply_context


def test_forum_topic_root_ignored():
    # First msg in a forum topic: reply_to is the topic-creation service message.
    root = SimpleNamespace(forum_topic_created=object(), message_id=40, text=None)
    msg = SimpleNamespace(reply_to_message=root, message_thread_id=40)
    assert _rc(None, msg) == ""


def test_real_reply_still_rendered():
    replied = SimpleNamespace(
        forum_topic_created=None,
        message_id=99,
        text="hi",
        from_user=SimpleNamespace(full_name="Bob", username=None, id=1),
    )
    msg = SimpleNamespace(reply_to_message=replied, message_thread_id=40)
    assert "Bob: hi" in _rc(None, msg)
