"""Materialize Himalaya config from the config store.

Structured email provider data is stored as JSON in the ``email.providers``
config key.  This module generates a valid Himalaya TOML configuration from
that structured data and writes it to well-known temporary paths so the CLI
can pick it up at runtime.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

HIMALAYA_CONFIG_PATH = Path("/tmp/humux-himalaya-config.toml")
HIMALAYA_XDG_DIR = Path("/tmp/humux-himalaya-xdg")
HIMALAYA_XDG_CONFIG_PATH = HIMALAYA_XDG_DIR / "himalaya" / "config.toml"

# Config key used to store the structured provider list.
EMAIL_PROVIDERS_KEY = "email.providers"


def himalaya_env() -> dict[str, str]:
    return {
        "HIMALAYA_CONFIG": str(HIMALAYA_CONFIG_PATH),
        "XDG_CONFIG_HOME": str(HIMALAYA_XDG_DIR),
    }


def _quote(value: str) -> str:
    """TOML-quote a string value."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _provider_to_toml(provider: dict, *, is_default: bool = False, resolve=None) -> str:
    """Convert a single provider dict to a Himalaya TOML account section.

    Expected provider keys:
        name         – account name slug (e.g. "personal", "work")
        email        – email address
        display_name – sender display name (optional)
        imap_host    – IMAP server hostname
        imap_port    – IMAP server port (default 993)
        smtp_host    – SMTP server hostname
        smtp_port    – SMTP server port (default 465)
        login        – login username (defaults to email)
        password     – app password / secret; may be a ``${vault:NAME}`` reference

    ``resolve`` (name -> str | None), when given, expands ``${vault:NAME}``
    references from the infra vault so the account password never sits in the
    stored config as plaintext (#110). The resolved secret is written only into
    the Himalaya config file (0600, /tmp) that the CLI reads — it never returns
    to the model's context or the logs.
    """
    if resolve is not None:
        from core.config import resolve_vault_vars

        provider = resolve_vault_vars(provider, resolve)
    name = provider.get("name", "default").strip()
    email = provider.get("email", "").strip()
    display_name = provider.get("display_name", "").strip()
    imap_host = provider.get("imap_host", "").strip()
    imap_port = int(provider.get("imap_port", 993) or 993)
    smtp_host = provider.get("smtp_host", "").strip()
    smtp_port = int(provider.get("smtp_port", 465) or 465)
    login = provider.get("login", "").strip() or email
    password = provider.get("password", "").strip()

    # Sanitise account name – only allow alphanumeric, dash, underscore
    safe_name = "".join(c if c.isalnum() or c in "-_" else "-" for c in name) or "default"

    lines: list[str] = []
    lines.append(f"[accounts.{safe_name}]")
    if is_default:
        lines.append("default = true")
    lines.append(f"email = {_quote(email)}")
    if display_name:
        lines.append(f"display-name = {_quote(display_name)}")

    # IMAP backend
    lines.append('backend.type = "imap"')
    lines.append(f"backend.host = {_quote(imap_host)}")
    lines.append(f"backend.port = {imap_port}")
    lines.append(f"backend.login = {_quote(login)}")
    lines.append('backend.encryption.type = "tls"')
    lines.append('backend.auth.type = "password"')
    lines.append(f"backend.auth.command = {_quote(f'echo -n {password}')}")

    # SMTP send backend
    lines.append('message.send.backend.type = "smtp"')
    lines.append(f"message.send.backend.host = {_quote(smtp_host)}")
    lines.append(f"message.send.backend.port = {smtp_port}")
    lines.append(f"message.send.backend.login = {_quote(login)}")
    lines.append('message.send.backend.encryption.type = "tls"')
    lines.append('message.send.backend.auth.type = "password"')
    lines.append(f"message.send.backend.auth.command = {_quote(f'echo -n {password}')}")

    return "\n".join(lines)


def providers_to_toml(providers: list[dict], resolve=None) -> str:
    """Generate a complete Himalaya TOML config from a list of providers.

    ``resolve`` is threaded to :func:`_provider_to_toml` to expand
    ``${vault:NAME}`` credential references (#110)."""
    if not providers:
        return ""
    sections: list[str] = []
    for idx, provider in enumerate(providers):
        sections.append(_provider_to_toml(provider, is_default=(idx == 0), resolve=resolve))
    return "\n\n".join(sections) + "\n"


async def materialize_himalaya_config(config_store) -> bool:
    """Write Himalaya TOML config from config DB.

    Reads structured provider data from ``email.providers`` (JSON list),
    generates Himalaya TOML, and writes it to the well-known paths.

    Returns True if a file was written or removed.
    """
    raw = await config_store.get(EMAIL_PROVIDERS_KEY)
    providers: list[dict] = []
    if raw:
        try:
            providers = json.loads(raw)
        except json.JSONDecodeError, TypeError:
            log.warning("Invalid JSON in %s, ignoring", EMAIL_PROVIDERS_KEY)

    # Filter out providers missing required fields
    providers = [
        p for p in providers if p.get("email") and p.get("imap_host") and p.get("smtp_host")
    ]

    if not providers:
        removed = False
        if HIMALAYA_CONFIG_PATH.exists():
            HIMALAYA_CONFIG_PATH.unlink()
            removed = True
        if HIMALAYA_XDG_CONFIG_PATH.exists():
            HIMALAYA_XDG_CONFIG_PATH.unlink()
            removed = True
        return removed

    # Expand ${vault:NAME} credential references, if a resolver was attached to
    # the config store (main.py wires config_store.vault_resolve after the infra
    # vault cache loads). Absent = literal values, preserving prior behaviour (#110).
    resolve = getattr(config_store, "vault_resolve", None)
    content = providers_to_toml(providers, resolve)

    # 0600: the file holds resolved account passwords, so keep it owner-only (#110).
    HIMALAYA_CONFIG_PATH.write_text(content)
    HIMALAYA_CONFIG_PATH.chmod(0o600)
    HIMALAYA_XDG_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    HIMALAYA_XDG_CONFIG_PATH.write_text(content)
    HIMALAYA_XDG_CONFIG_PATH.chmod(0o600)
    log.info(
        "Materialized Himalaya config (%d accounts) to %s", len(providers), HIMALAYA_CONFIG_PATH
    )
    return True


if __name__ == "__main__":
    # ponytail: one runnable check — TOML build + ${vault:} password resolution (#110).
    prov = {
        "name": "fitness",
        "email": "fitness@agent.example.com",
        "imap_host": "imap.example.com",
        "smtp_host": "smtp.example.com",
        "password": "${vault:IMAP_PW_FITNESS}",
    }
    plain = _provider_to_toml(prov)
    assert "${vault:IMAP_PW_FITNESS}" in plain  # unresolved without a resolver
    resolved = _provider_to_toml(
        prov, resolve=lambda n: "s3cret" if n == "IMAP_PW_FITNESS" else None
    )
    assert "${vault:" not in resolved and "echo -n s3cret" in resolved, resolved
    # A vault miss leaves the reference literal rather than blanking the field.
    miss = _provider_to_toml(prov, resolve=lambda n: None)
    assert "${vault:IMAP_PW_FITNESS}" in miss
    print("email_config.py self-check OK")
