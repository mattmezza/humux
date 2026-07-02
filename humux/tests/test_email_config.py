"""Tests for Himalaya config materialization from structured providers."""

from __future__ import annotations

import json

import pytest

from core import email_config
from core.email_config import providers_to_toml


class _Store:
    """Minimal config store stub."""

    def __init__(self, providers: list[dict] | None) -> None:
        self._raw = json.dumps(providers) if providers is not None else None

    async def get(self, key: str) -> str | None:
        if key == email_config.EMAIL_PROVIDERS_KEY:
            return self._raw
        return None


# ---------------------------------------------------------------------------
# providers_to_toml unit tests
# ---------------------------------------------------------------------------


def test_empty_providers_returns_empty_string() -> None:
    assert providers_to_toml([]) == ""


def test_single_provider_generates_valid_toml() -> None:
    toml = providers_to_toml(
        [
            {
                "name": "personal",
                "email": "me@example.com",
                "display_name": "Me",
                "imap_host": "imap.example.com",
                "imap_port": 993,
                "smtp_host": "smtp.example.com",
                "smtp_port": 465,
                "login": "me@example.com",
                "password": "secret",
            }
        ]
    )
    assert "[accounts.personal]" in toml
    assert "default = true" in toml
    assert 'email = "me@example.com"' in toml
    assert 'display-name = "Me"' in toml
    assert 'backend.type = "imap"' in toml
    assert 'backend.host = "imap.example.com"' in toml
    assert "backend.port = 993" in toml
    assert 'backend.login = "me@example.com"' in toml
    assert 'backend.auth.command = "echo -n secret"' in toml
    assert 'message.send.backend.type = "smtp"' in toml
    assert 'message.send.backend.host = "smtp.example.com"' in toml
    assert "message.send.backend.port = 465" in toml


def test_multiple_providers_only_first_is_default() -> None:
    toml = providers_to_toml(
        [
            {
                "name": "personal",
                "email": "me@example.com",
                "imap_host": "imap.example.com",
                "smtp_host": "smtp.example.com",
            },
            {
                "name": "work",
                "email": "me@work.com",
                "imap_host": "imap.work.com",
                "smtp_host": "smtp.work.com",
            },
        ]
    )
    assert "[accounts.personal]" in toml
    assert "[accounts.work]" in toml

    # Only the first account should be default
    personal_section = toml.split("[accounts.work]")[0]
    work_section = toml.split("[accounts.work]")[1]
    assert "default = true" in personal_section
    assert "default = true" not in work_section


def test_login_defaults_to_email() -> None:
    toml = providers_to_toml(
        [
            {
                "name": "test",
                "email": "user@example.com",
                "imap_host": "imap.example.com",
                "smtp_host": "smtp.example.com",
                "login": "",  # blank login should fall back to email
            }
        ]
    )
    assert 'backend.login = "user@example.com"' in toml
    assert 'message.send.backend.login = "user@example.com"' in toml


def test_password_embedded_in_auth_command() -> None:
    toml = providers_to_toml(
        [
            {
                "name": "my-work",
                "email": "a@b.com",
                "imap_host": "imap.b.com",
                "smtp_host": "smtp.b.com",
                "password": "s3cret",
            }
        ]
    )
    assert 'backend.auth.command = "echo -n s3cret"' in toml
    assert 'message.send.backend.auth.command = "echo -n s3cret"' in toml


def test_provider_without_display_name_omits_field() -> None:
    toml = providers_to_toml(
        [
            {
                "name": "bare",
                "email": "a@b.com",
                "imap_host": "imap.b.com",
                "smtp_host": "smtp.b.com",
            }
        ]
    )
    assert "display-name" not in toml


def test_custom_ports() -> None:
    toml = providers_to_toml(
        [
            {
                "name": "custom",
                "email": "a@b.com",
                "imap_host": "imap.b.com",
                "imap_port": 143,
                "smtp_host": "smtp.b.com",
                "smtp_port": 587,
            }
        ]
    )
    assert "backend.port = 143" in toml
    assert "message.send.backend.port = 587" in toml


# ---------------------------------------------------------------------------
# materialize_himalaya_config integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_materialize_writes_tmp_files(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "himalaya.toml"
    xdg_dir = tmp_path / "xdg"
    xdg_config_path = xdg_dir / "himalaya" / "config.toml"

    monkeypatch.setattr(email_config, "HIMALAYA_CONFIG_PATH", config_path)
    monkeypatch.setattr(email_config, "HIMALAYA_XDG_DIR", xdg_dir)
    monkeypatch.setattr(email_config, "HIMALAYA_XDG_CONFIG_PATH", xdg_config_path)

    store = _Store(
        [
            {
                "name": "personal",
                "email": "you@example.com",
                "imap_host": "imap.example.com",
                "smtp_host": "smtp.example.com",
            }
        ]
    )
    changed = await email_config.materialize_himalaya_config(store)

    assert changed is True
    content = config_path.read_text()
    assert "[accounts.personal]" in content
    assert 'email = "you@example.com"' in content
    assert xdg_config_path.read_text() == content


@pytest.mark.asyncio
async def test_materialize_removes_tmp_files(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "himalaya.toml"
    xdg_dir = tmp_path / "xdg"
    xdg_config_path = xdg_dir / "himalaya" / "config.toml"
    config_path.write_text("[accounts.personal]\n")
    xdg_config_path.parent.mkdir(parents=True, exist_ok=True)
    xdg_config_path.write_text("[accounts.personal]\n")

    monkeypatch.setattr(email_config, "HIMALAYA_CONFIG_PATH", config_path)
    monkeypatch.setattr(email_config, "HIMALAYA_XDG_DIR", xdg_dir)
    monkeypatch.setattr(email_config, "HIMALAYA_XDG_CONFIG_PATH", xdg_config_path)

    store = _Store([])
    changed = await email_config.materialize_himalaya_config(store)

    assert changed is True
    assert not config_path.exists()
    assert not xdg_config_path.exists()


@pytest.mark.asyncio
async def test_materialize_empty_store_no_files(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "himalaya.toml"
    xdg_dir = tmp_path / "xdg"
    xdg_config_path = xdg_dir / "himalaya" / "config.toml"

    monkeypatch.setattr(email_config, "HIMALAYA_CONFIG_PATH", config_path)
    monkeypatch.setattr(email_config, "HIMALAYA_XDG_DIR", xdg_dir)
    monkeypatch.setattr(email_config, "HIMALAYA_XDG_CONFIG_PATH", xdg_config_path)

    store = _Store(None)
    changed = await email_config.materialize_himalaya_config(store)

    assert changed is False
    assert not config_path.exists()


@pytest.mark.asyncio
async def test_materialize_skips_incomplete_providers(tmp_path, monkeypatch) -> None:
    """Providers missing required fields (email, imap_host, smtp_host) are filtered out."""
    config_path = tmp_path / "himalaya.toml"
    xdg_dir = tmp_path / "xdg"
    xdg_config_path = xdg_dir / "himalaya" / "config.toml"

    monkeypatch.setattr(email_config, "HIMALAYA_CONFIG_PATH", config_path)
    monkeypatch.setattr(email_config, "HIMALAYA_XDG_DIR", xdg_dir)
    monkeypatch.setattr(email_config, "HIMALAYA_XDG_CONFIG_PATH", xdg_config_path)

    store = _Store(
        [
            {"name": "incomplete", "email": "a@b.com"},  # missing imap/smtp hosts
        ]
    )
    changed = await email_config.materialize_himalaya_config(store)

    # No valid providers, so no file written
    assert changed is False
    assert not config_path.exists()


@pytest.mark.asyncio
async def test_materialize_invalid_json(tmp_path, monkeypatch) -> None:
    """Invalid JSON in config should not crash."""
    config_path = tmp_path / "himalaya.toml"
    xdg_dir = tmp_path / "xdg"
    xdg_config_path = xdg_dir / "himalaya" / "config.toml"

    monkeypatch.setattr(email_config, "HIMALAYA_CONFIG_PATH", config_path)
    monkeypatch.setattr(email_config, "HIMALAYA_XDG_DIR", xdg_dir)
    monkeypatch.setattr(email_config, "HIMALAYA_XDG_CONFIG_PATH", xdg_config_path)

    # Simulate invalid JSON by using a raw store
    class _RawStore:
        async def get(self, key: str) -> str | None:
            if key == email_config.EMAIL_PROVIDERS_KEY:
                return "not valid json{{"
            return None

    changed = await email_config.materialize_himalaya_config(_RawStore())
    assert changed is False
    assert not config_path.exists()
