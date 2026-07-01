"""WhatsApp is a tool, not a channel (issue #97).

The agent reads/sends WhatsApp by running `wacli` via run_command; it is registered
as an optional external tool (advertised + env-gated) instead of an inbound channel.
"""

from __future__ import annotations

from core.config import Config
from core.tools import active_tool_prompts, registry, tool_env


def test_whatsapp_registered_but_off_by_default() -> None:
    cfg = Config()
    assert any(spec.key == "whatsapp" for spec in registry())
    # Disabled by default: not advertised, no env injected.
    assert tool_env(cfg) == {}
    assert active_tool_prompts(cfg) == []


def test_whatsapp_advert_and_env_when_enabled() -> None:
    cfg = Config()
    cfg.tools.whatsapp.enabled = True
    cfg.tools.whatsapp.store = "/data/wacli"
    cfg.tools.whatsapp.device_label = "Agent1"
    assert tool_env(cfg) == {
        "WACLI_STORE": "/data/wacli",
        "WACLI_DEVICE_LABEL": "Agent1",
    }
    blocks = active_tool_prompts(cfg)
    assert any('name="whatsapp"' in b and "send text" in b for b in blocks)


def test_whatsapp_enabled_without_identity_overrides_has_empty_env() -> None:
    cfg = Config()
    cfg.tools.whatsapp.enabled = True  # no store/label → wacli's own defaults
    assert tool_env(cfg) == {}
    assert active_tool_prompts(cfg)  # still advertised so the agent knows it exists


def test_send_message_tool_is_telegram_only() -> None:
    # Migration guard: send_message no longer routes WhatsApp (that channel is gone);
    # WhatsApp send goes through the wacli CLI via run_command.
    from core.agent import TOOLS

    send = next(t for t in TOOLS if t["name"] == "send_message")
    assert send["input_schema"]["properties"]["channel"]["enum"] == ["telegram"]
