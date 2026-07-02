"""Tests for the optional browser tool: gating, permissions, CLI helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import Config
from core.permissions import PermissionEngine, PermissionLevel
from core.tools import active_tool_prompts, tool_env

# ---------------------------------------------------------------------------
# Registry gating — invisible when disabled, advertised + wired when enabled
# ---------------------------------------------------------------------------


def test_browser_inactive_by_default() -> None:
    cfg = Config()
    assert "browser" not in " ".join(active_tool_prompts(cfg))
    assert "BROWSER_HEADLESS" not in tool_env(cfg)


def test_browser_advertised_and_env_when_enabled() -> None:
    cfg = Config()
    cfg.tools.browser.enabled = True
    blocks = "\n".join(active_tool_prompts(cfg))
    assert "browser.py" in blocks
    env = tool_env(cfg)
    assert env["BROWSER_HEADLESS"] == "1"
    assert "BROWSER_CDP_URL" not in env  # only present when a sidecar is configured


def test_browser_headless_and_cdp_env() -> None:
    cfg = Config()
    cfg.tools.browser.enabled = True
    cfg.tools.browser.headless = False
    cfg.tools.browser.cdp_url = "ws://sidecar:9222"
    env = tool_env(cfg)
    assert env["BROWSER_HEADLESS"] == "0"
    assert env["BROWSER_CDP_URL"] == "ws://sidecar:9222"


# ---------------------------------------------------------------------------
# Permissions — read is pre-approved, act asks, per-domain overrides win
# ---------------------------------------------------------------------------


def _level(engine: PermissionEngine, command: str) -> str:
    return engine.check("run_command", {"command": command})


def test_browser_read_is_always_act_asks(tmp_path) -> None:
    eng = PermissionEngine(db_path=str(tmp_path / "c.db"))
    assert (
        _level(eng, "python3 tools/browser.py read --url https://x.com") == PermissionLevel.ALWAYS
    )
    assert (
        _level(eng, "python3 tools/browser.py screenshot --url https://x.com")
        == PermissionLevel.ALWAYS
    )
    assert (
        _level(eng, "python3 tools/browser.py act --url https://x.com --steps []")
        == PermissionLevel.ASK
    )


def test_browser_act_is_write_action(tmp_path) -> None:
    eng = PermissionEngine(db_path=str(tmp_path / "c.db"))
    # act must re-ask every time (write), read must not.
    assert eng.is_write_action(
        "run_command", {"command": "python3 tools/browser.py act --url https://x.com --steps []"}
    )
    assert not eng.is_write_action(
        "run_command", {"command": "python3 tools/browser.py read --url https://x.com"}
    )


def test_per_domain_rule_overrides_default(tmp_path) -> None:
    eng = PermissionEngine(db_path=str(tmp_path / "c.db"))
    eng.add_rule("run_command:*browser.py act*github.com*", PermissionLevel.ALWAYS)
    # github.com is now pre-approved; other domains still ask.
    assert (
        _level(eng, "python3 tools/browser.py act --url https://github.com/x --steps []")
        == PermissionLevel.ALWAYS
    )
    assert (
        _level(eng, "python3 tools/browser.py act --url https://evil.com/x --steps []")
        == PermissionLevel.ASK
    )


# ---------------------------------------------------------------------------
# CLI helpers (no browser needed)
# ---------------------------------------------------------------------------


def test_validate_profile() -> None:
    from tools.browser import _validate_profile

    assert _validate_profile("Acme") == "acme"
    for bad in ["", "a/b", "../x", "a b"]:
        with pytest.raises(ValueError):
            _validate_profile(bad)


def test_parse_steps() -> None:
    from tools.browser import _parse_steps

    steps = _parse_steps('[{"fill":["#u","a"]},{"click":"#go"}]')
    assert steps == [{"fill": ["#u", "a"]}, {"click": "#go"}]
    for bad in ["{}", "[]", "not json", '[{"a":1,"b":2}]']:
        with pytest.raises(ValueError):
            _parse_steps(bad)


# ---------------------------------------------------------------------------
# Approval screenshot — browser `act` surfaces the per-profile preview
# ---------------------------------------------------------------------------


def test_approval_image_for_browser_act(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    from core.agent import AgentCore

    agent = AgentCore(Config())
    preview = tmp_path / "data" / "browser" / "last" / "acme.png"
    preview.parent.mkdir(parents=True, exist_ok=True)
    preview.write_bytes(b"\x89PNG")

    act_cmd = {"command": "python3 tools/browser.py act --url https://x --profile acme --steps []"}
    got = agent._approval_image("run_command", act_cmd)
    assert got is not None and Path(got).resolve() == preview.resolve()
    # No preview file -> None; read command -> None; non-run_command -> None.
    other = {"command": "python3 tools/browser.py act --url https://x --profile ghost --steps []"}
    assert agent._approval_image("run_command", other) is None
    read_cmd = {"command": "python3 tools/browser.py read --url https://x"}
    assert agent._approval_image("run_command", read_cmd) is None
    assert agent._approval_image("send_email", {"to": "a"}) is None
