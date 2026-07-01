"""Admin API — FastAPI app for health checks, config management, permissions,
memory inspection, log streaming, and agent lifecycle control.

Uses Jinja2 templates with HTMX for the UI. All endpoints (except /health,
/setup/*, /login, and /static/*) require Bearer token auth matching the
stored admin password hash.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import logging
import os
import re
import secrets
import sys
import urllib.parse
from base64 import urlsafe_b64encode
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests as http_requests
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field

from core.artifacts import ARTIFACT_CSP, NOT_FOUND_HTML, resolve, serving_base, valid_id
from core.config_store import ConfigStore
from core.goal_decomposition import classify_complexity, decompose_goal
from core.llm import LLMClient, get_sent_payload
from core.log_streams import current_stream, current_subagent
from core.prompt_builder import (
    DEFAULT_HISTORY_HANDLING_BLOCK,
    DEFAULT_TOOL_USAGE_BLOCK,
    build_prompt_sections,
)
from core.tools import gh_token_secret_name, tool_env
from core.tools import registry as tool_registry
from core.wacli import WacliManager

if TYPE_CHECKING:
    from core.agent import AgentCore
    from core.secret_store import SecretStore
    from core.skills import SkillsStore

log = logging.getLogger(__name__)

# A persona slug becomes part of a channel name (``telegram:<slug>``) and a URL
# path (``/admin/personae/<slug>``), so it must avoid ':' , '/' and whitespace.
# Capped at 64 chars so it can't bloat every channel string it is embedded in.
_VALID_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_SLUG_ERROR = "Slug must be 1-64 chars: letters, digits, '-' and '_' only (no spaces or ':')."

# ---------------------------------------------------------------------------
# Google OAuth 2.0 constants (for CalDAV calendar access)
# ---------------------------------------------------------------------------

_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar"]
_GOOGLE_CONTACTS_SCOPES = ["https://www.googleapis.com/auth/contacts.readonly"]

# In-memory PKCE state (short-lived, per auth attempt)
_oauth_pending: dict[str, dict] = {}  # state -> {code_verifier, client_id, client_secret}


def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


# ---------------------------------------------------------------------------
# Jinja2 template environment
# ---------------------------------------------------------------------------

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_jinja_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=True,
    auto_reload=True,
)
_jinja_env.globals["step_ctx"] = {}  # default empty dict for wizard templates


def _render(template_name: str, **ctx: object) -> HTMLResponse:
    """Render a Jinja2 template and return an HTMLResponse."""
    tmpl = _jinja_env.get_template(template_name)
    return HTMLResponse(tmpl.render(**ctx))


def _render_partial(template_name: str, **ctx: object) -> HTMLResponse:
    """Render a partial template (no base layout) for HTMX swaps."""
    tmpl = _jinja_env.get_template(template_name)
    return HTMLResponse(tmpl.render(**ctx))


def _humanize_ts(ts_utc: str, now: datetime, tz: str = "UTC") -> str:
    """Relative "last active" label from a SQLite ``datetime('now')`` UTC string.

    Recent times read as just now / Nm / Nh / Nd ago; older than a week falls
    back to a date in the agent's timezone. "" in, "" out (a never-messaged chat).
    """
    if not ts_utc:
        return ""
    try:
        dt = datetime.strptime(ts_utc, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        return ts_utc
    delta = max(0.0, (now - dt).total_seconds())
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    if delta < 7 * 86400:
        return f"{int(delta // 86400)}d ago"
    try:
        return dt.astimezone(ZoneInfo(tz)).strftime("%Y-%m-%d")
    except ZoneInfoNotFoundError:
        return dt.strftime("%Y-%m-%d")


def _elide_image_data(value: object) -> object:
    """Return a copy of ``value`` with long base64 ``data`` fields replaced by a
    short placeholder, so a captured image payload doesn't bloat the Inspect view.
    Recurses through the message/block dict+list structure; leaves all else as-is.
    """
    if isinstance(value, dict):
        out: dict = {}
        for k, v in value.items():
            if k == "data" and isinstance(v, str) and len(v) > 256:
                out[k] = f"<{len(v)} base64 chars elided>"
            else:
                out[k] = _elide_image_data(v)
        return out
    if isinstance(value, list):
        return [_elide_image_data(v) for v in value]
    return value


def _is_vault_ref(value: str | None) -> bool:
    """True if a stored config value points at the infra vault (issue #35).

    Vault-managed credentials must never be echoed back into a form field — the
    tab/wizard shows a read-only note instead — so callers filter these out.
    """
    return bool(value) and str(value).startswith("${vault:")


async def _wizard_step_context(step: str, config_store: ConfigStore) -> dict[str, str]:
    """Fetch previously-saved config values relevant to a wizard step.

    Returns a flat dict of template variable names to values so that
    navigating *back* in the wizard pre-populates form fields.
    """
    ctx: dict[str, str] = {}
    if step == "llm":
        for key, var in (
            ("agent.llm_provider", "provider"),
            ("agent.anthropic_api_key", "anthropic_api_key"),
            ("agent.openai_api_key", "openai_api_key"),
            ("agent.openai_base_url", "openai_base_url"),
            ("agent.google_api_key", "google_api_key"),
            ("agent.google_base_url", "google_base_url"),
            ("agent.grok_api_key", "grok_api_key"),
            ("agent.grok_base_url", "grok_base_url"),
            ("agent.deepseek_api_key", "deepseek_api_key"),
            ("agent.deepseek_base_url", "deepseek_base_url"),
            ("agent.model", "model"),
        ):
            val = await config_store.get(key)
            # Don't pre-fill a form field with a ${vault:} reference (issue #35).
            if val and not _is_vault_ref(val):
                ctx[var] = val
    elif step == "identity":
        for key, var in (
            ("agent.name", "agent_name"),
            ("agent.owner_name", "owner_name"),
            ("agent.timezone", "timezone"),
        ):
            val = await config_store.get(key)
            if val:
                ctx[var] = val
    elif step == "persona":
        store = await _persona_store_from_config(config_store)
        # Plain dicts so the wizard template (Jinja) can read emoji/role/name.
        ctx["personae"] = [  # type: ignore[assignment]
            {"name": p.name, "role": p.role, "emoji": p.emoji} for p in await store.list_personae()
        ]
        ctx["active"] = (await config_store.get("agent.active_persona") or "").strip()
    elif step == "email":
        raw = await config_store.get("email.providers")
        if raw:
            try:
                providers = json.loads(raw)
                if providers and isinstance(providers, list) and providers:
                    p = providers[0]
                    ctx["email_name"] = p.get("name", "")
                    ctx["email_addr"] = p.get("email", "")
                    ctx["email_display_name"] = p.get("display_name", "")
                    ctx["email_imap_host"] = p.get("imap_host", "")
                    ctx["email_imap_port"] = p.get("imap_port", "993")
                    ctx["email_smtp_host"] = p.get("smtp_host", "")
                    ctx["email_smtp_port"] = p.get("smtp_port", "465")
                    ctx["email_login"] = p.get("login", "")
                    ctx["email_password"] = p.get("password", "")
            except json.JSONDecodeError, TypeError:
                pass
    elif step == "telegram":
        for key, var in (
            ("channels.telegram.bot_token", "bot_token"),
            ("channels.telegram.allowed_user_ids", "user_ids"),
        ):
            val = await config_store.get(key)
            if val and not _is_vault_ref(val):
                ctx[var] = val
    elif step == "browser":
        enabled = await config_store.get("tools.browser.enabled")
        if enabled is not None:
            ctx["browser_enabled"] = enabled
    elif step == "imagegen":
        enabled = await config_store.get("tools.imagegen.enabled")
        if enabled is not None:
            ctx["imagegen_enabled"] = enabled
        provider = await config_store.get("tools.imagegen.provider")
        if provider:
            ctx["imagegen_provider"] = provider
        # Auto-detect a reusable LLM key (OpenRouter/OpenAI) → zero-step setup (#55).
        openai_key = await config_store.get("agent.openai_api_key")
        openai_base = (await config_store.get("agent.openai_base_url") or "").lower()
        if openai_key:
            ctx["imagegen_reuse"] = "openrouter" if "openrouter" in openai_base else "openai"
    elif step == "calendar":
        raw = await config_store.get("calendar.providers")
        if raw:
            try:
                providers = json.loads(raw)
                if providers and isinstance(providers, list):
                    p = providers[0]
                    ctx["cal_name"] = p.get("name", "")
                    ctx["cal_url"] = p.get("url", "")
                    ctx["cal_username"] = p.get("username", "")
                    ctx["cal_password"] = p.get("password", "")
            except json.JSONDecodeError, IndexError:
                pass
    elif step == "search":
        val = await config_store.get("search.api_key")
        if val and not _is_vault_ref(val):
            ctx["tavily_key"] = val
    elif step == "admin":
        val = await config_store.get("admin.password_hash")
        if val:
            ctx["admin_key"] = ""
    elif step == "secrets":
        import os as _os

        from core.secret_store import INFRA_VAULT_KEYS
        from core.vault import load_machine_key

        ctx["machine_key_present"] = bool(load_machine_key())  # type: ignore[assignment]
        ctx["env_seed_present"] = bool(  # type: ignore[assignment]
            _os.environ.get("ADMIN_PASSWORD") or _os.environ.get("ADMIN_API_KEY")
        )
        detected = []
        for cfg_key in INFRA_VAULT_KEYS:
            val = await config_store.get(cfg_key)
            if val and not val.startswith("${"):
                detected.append(cfg_key)
        ctx["detected_secrets"] = detected  # type: ignore[assignment]
    return ctx


async def _channel_list_context(
    config_store: ConfigStore,
    wacli: WacliManager | None = None,
) -> dict[str, list[dict[str, object]]]:
    channels: list[dict[str, object]] = []

    tg_enabled_raw = await config_store.get("channels.telegram.enabled")
    tg_enabled = str(tg_enabled_raw).lower() == "true"
    tg_token = await config_store.get("channels.telegram.bot_token")
    tg_users = await config_store.get("channels.telegram.allowed_user_ids")
    tg_configured = bool(tg_token or tg_users or tg_enabled)
    if tg_configured:
        tg_detail = "Not configured"
        if tg_token:
            tg_detail = "Bot token set"
            if tg_users:
                tg_detail = f"Bot token set · Users: {tg_users}"
        channels.append(
            {
                "key": "telegram",
                "label": "Telegram",
                "enabled": tg_enabled,
                "detail": tg_detail,
            }
        )

    # WhatsApp is a tool now (#97), not a channel — its enable toggle and wacli
    # linking live on the Tools tab.

    return {"channels": channels}


async def _channel_wizard_context(
    config_store: ConfigStore,
    channel: str,
) -> dict[str, str]:
    ctx: dict[str, str] = {}
    if channel == "telegram":
        bot_token = await config_store.get("channels.telegram.bot_token")
        user_ids = await config_store.get("channels.telegram.allowed_user_ids")
        # Vault-managed token (issue #35): mark it so the editor shows a read-only
        # note instead of the input, and never ship the ref to the browser.
        bot_token_vaulted = _is_vault_ref(bot_token)
        ctx["bot_token_vaulted"] = bot_token_vaulted  # type: ignore[assignment]
        if bot_token and not bot_token_vaulted:
            ctx["bot_token"] = bot_token
        if user_ids:
            ctx["user_ids"] = user_ids
        ctx["topics_enabled"] = (
            str(await config_store.get("channels.telegram.topics_enabled")).lower() == "true"
        )
        # Group multi-agent rooms (#30). Absent keys fall back to the model
        # defaults: enabled off, the two sub-options on.
        g_enabled = await config_store.get("channels.telegram.group_chat.enabled")
        g_addressed = await config_store.get(
            "channels.telegram.group_chat.reply_when_addressed_only"
        )
        g_ignore = await config_store.get("channels.telegram.group_chat.ignore_bots")
        ctx["group_chat_enabled"] = str(g_enabled).lower() == "true"  # default off
        ctx["group_reply_addressed_only"] = (
            g_addressed is None or str(g_addressed).lower() == "true"
        )
        ctx["group_ignore_bots"] = g_ignore is None or str(g_ignore).lower() == "true"
    return ctx


async def _calendar_providers_context(config_store: ConfigStore) -> list[dict[str, str]]:
    raw = await config_store.get("calendar.providers")
    if not raw:
        return []
    try:
        providers = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(providers, list):
        return []
    cleaned: list[dict[str, str]] = []
    for p in providers:
        if not isinstance(p, dict):
            continue
        cleaned.append(
            {
                "name": str(p.get("name", "")),
                "url": str(p.get("url", "")),
                "username": str(p.get("username", "")),
                "password": str(p.get("password", "")),
            }
        )
    return cleaned


async def _email_account_names(config_store: ConfigStore) -> list[str]:
    """Names of the configured email accounts, for persona-binding UIs (#110)."""
    raw = await config_store.get("email.providers")
    if not raw:
        return []
    try:
        providers = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(providers, list):
        return []
    return [
        str(p.get("name", "")).strip()
        for p in providers
        if isinstance(p, dict) and str(p.get("name", "")).strip()
    ]


async def _contact_providers_context(config_store: ConfigStore) -> list[dict[str, str]]:
    raw = await config_store.get("contacts.providers")
    if not raw:
        return []
    try:
        providers = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(providers, list):
        return []
    cleaned: list[dict[str, str]] = []
    for p in providers:
        if not isinstance(p, dict):
            continue
        cleaned.append(
            {
                "name": str(p.get("name", "")),
                "type": str(p.get("type", "carddav")),
                "url": str(p.get("url", "")),
                "username": str(p.get("username", "")),
                "password": str(p.get("password", "")),
                "client_id": str(p.get("client_id", "")),
                "client_secret": str(p.get("client_secret", "")),
            }
        )
    return cleaned


def _render_wizard_step(
    step: str,
    steps: list[str],
    ctx: dict[str, str] | None = None,
) -> HTMLResponse:
    """Render a wizard step partial with OOB progress dots update."""
    step_html = _jinja_env.get_template(f"wizard/{step}.html").render(
        step_ctx=ctx or {},
    )
    progress_html = _jinja_env.get_template("wizard/progress.html").render(
        steps=steps, current_step=step
    )
    return HTMLResponse(step_html + progress_html)


# ---------------------------------------------------------------------------
# Shared mutable agent state
# ---------------------------------------------------------------------------


class AgentState:
    """Mutable container for the currently running agent."""

    def __init__(self, agent: AgentCore | None = None, status: str = "STOPPED"):
        self.agent: AgentCore | None = agent
        self.status: str = status


# ---------------------------------------------------------------------------
# In-memory ring buffer for recent log lines
# ---------------------------------------------------------------------------

# Structured entries (not formatted strings) so the Logs tab can filter by
# stream / level / time / text server-side (#75). 5000 lines ≈ a couple of MB and
# gives the stream/time filters enough history to be useful (the view still caps
# at the last 300 matches).
_LOG_BUFFER: collections.deque[dict] = collections.deque(maxlen=5000)

_LOG_INCLUDE_PREFIXES = ("core.", "channels.", "voice.", "tools.")
_LOG_INCLUDE_NAMES = {"core", "channels", "voice", "tools"}

# Model chain-of-thought logger (see core/llm.py). The admin log viewer styles
# lines from this logger distinctly; the template detects them by this name.
_REASONING_LOGGER = "core.llm.reasoning"

# Severity levels offered in the Logs tab filter, low → high.
_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]


def _should_capture_log_record(record: logging.LogRecord) -> bool:
    """Return True when a record should be visible in the admin log viewer."""
    name = record.name
    if name in _LOG_INCLUDE_NAMES:
        return True
    return name.startswith(_LOG_INCLUDE_PREFIXES)


def _stream_hue(stream: str) -> int:
    """A stable hue (0-359) for a stream name, so each agent's lines read in a
    consistent colour. Deterministic across restarts (no salted hash())."""
    return (sum(stream.encode()) * 47) % 360


class _BufferHandler(logging.Handler):
    """Logging handler that appends structured records to an in-memory deque."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if not _should_capture_log_record(record):
                return
            stream = current_stream()
            sub = current_subagent()
            message = record.getMessage()
            if sub:
                message = f"[subagent:{sub}] {message}"
            _LOG_BUFFER.append(
                {
                    "ts": record.created,
                    "time": datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
                    "level": record.levelname,
                    "levelno": record.levelno,
                    "name": record.name,
                    "stream": stream,
                    "hue": _stream_hue(stream),
                    "message": message,
                    "is_reasoning": record.name == _REASONING_LOGGER,
                }
            )
        except Exception:
            pass


def _filter_log_entries(
    entries: list[dict],
    *,
    stream: str = "",
    level: str = "",
    q: str = "",
    since: str = "",
    until: str = "",
) -> list[dict]:
    """Apply the Logs tab filters. ``stream`` is a regex over the stream name;
    ``level`` is a minimum severity; ``q`` is a case-insensitive substring over
    the message; ``since``/``until`` are ``datetime-local`` bounds. Each empty
    filter is a no-op, so the unfiltered case returns everything."""
    try:
        rx = re.compile(stream, re.IGNORECASE) if stream else None
    except re.error:
        rx = None  # a half-typed regex shouldn't blank the viewer
    minlevel = logging.getLevelName(level) if level in _LOG_LEVELS else 0
    ql = q.strip().lower()
    lo = _parse_local_dt(since)
    hi = _parse_local_dt(until)
    out = []
    for e in entries:
        if rx and not rx.search(e["stream"]):
            continue
        if minlevel and e["levelno"] < minlevel:
            continue
        if ql and ql not in e["message"].lower() and ql not in e["name"].lower():
            continue
        if lo is not None and e["ts"] < lo:
            continue
        if hi is not None and e["ts"] > hi:
            continue
        out.append(e)
    return out


def _parse_local_dt(value: str) -> float | None:
    """A ``datetime-local`` field (``2026-06-30T14:30``) → epoch seconds, or None."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def install_log_buffer() -> None:
    """Attach the ring-buffer handler to the root logger.

    Also surface model chain-of-thought in the viewer: the reasoning logger is
    silent (WARNING) by default so it never spams server stdout, but the admin
    UI wants it. Bump it to INFO so its records reach the buffer, and filter it
    off the pre-existing console handlers so server logs stay clean.
    """
    root = logging.getLogger()
    logging.getLogger(_REASONING_LOGGER).setLevel(logging.INFO)

    def _drop_reasoning(record: logging.LogRecord) -> bool:
        return record.name != _REASONING_LOGGER

    for existing in root.handlers:  # console handler(s) from basicConfig
        existing.addFilter(_drop_reasoning)

    handler = _BufferHandler()  # added after, so it keeps the reasoning records
    root.addHandler(handler)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class PermissionRuleIn(BaseModel):
    pattern: str
    level: str  # ALWAYS | ASK | NEVER


class ConfigPatchIn(BaseModel):
    values: dict[str, str]


class SetupStepIn(BaseModel):
    step: str
    values: dict[str, str] = {}


class SkillUpsertIn(BaseModel):
    name: str
    content: str


def _persona_public(persona) -> dict:
    """Persona as JSON for read-only APIs, with the bot token redacted (#29).

    Mirrors how the global Telegram token is redacted in config reads — the token
    is a secret and must not leave the server in cleartext (exports, list views).
    """
    from dataclasses import asdict, replace

    from core.config_store import _redact
    from core.personae import to_markdown

    safe = replace(persona, bot_token=_redact(persona.bot_token))
    return {**asdict(safe), "markdown": to_markdown(safe)}


class PersonaUpsertIn(BaseModel):
    name: str
    agent_name: str = ""
    role: str = ""
    emoji: str = ""
    voice: str = ""
    character: str = ""
    skills: list[str] = []
    tools: list[str] = []
    secrets: list[str] = []
    bot_token: str = ""  # per-persona Telegram bot (#29); empty = no own bot
    allowed_user_ids: str = ""  # comma/newline-separated; empty = inherit global
    tool_config: dict = {}  # per-persona external-tool config (gh/browser) — #93
    email_accounts: list[dict] = []  # [{account, access_level, is_sender_identity}] — #110
    calendar_accounts: list[dict] = []  # [{account, access_level}] — #110
    gh_token: str = ""  # this persona's GitHub PAT → infra vault; empty = leave unchanged
    raw: str = ""  # when set, the markdown doc is parsed instead of the fields above


class PasswordChangeIn(BaseModel):
    current_password: str
    new_password: str


class CalendarProvidersIn(BaseModel):
    providers: list[dict[str, str]]


class GoogleOAuthClientIn(BaseModel):
    client_id: str
    client_secret: str


class WhatsAppTestIn(BaseModel):
    pass


class ContactProvidersIn(BaseModel):
    providers: list[dict[str, str]]


class EmailProvidersIn(BaseModel):
    providers: list[dict[str, str]]


class EmailTestIn(BaseModel):
    """One email account to probe for IMAP + SMTP reachability (#110)."""

    imap_host: str = ""
    imap_port: str = "993"
    smtp_host: str = ""
    smtp_port: str = "465"
    login: str = ""
    email: str = ""
    password: str = ""


class CalendarTestIn(BaseModel):
    """One CalDAV account to probe for reachability (#110)."""

    url: str = ""
    username: str = ""
    password: str = ""


class PromptPreviewIn(BaseModel):
    message: str = ""
    include_memories: bool = True
    include_reflections: bool = True


class VoicePreviewIn(BaseModel):
    # Bounded so an authenticated admin can't request synthesis of megabytes of
    # text (RAM / worker-thread exhaustion). A preview is a short sample.
    voice: str = Field("", max_length=64)
    text: str = Field("Hi! This is a quick preview of this voice.", max_length=600)
    lang: str = Field("", max_length=16)  # Kokoro pronunciation override; "" = derive from voice


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

_bearer = HTTPBearer(auto_error=False)


def _make_auth_dependency(config_store: ConfigStore, secret_store: SecretStore | None = None):
    """Return a FastAPI dependency that validates the admin API key.

    On a successful auth, if the persona secrets vault is still locked, the
    bearer password is used to unseal it (issue #19) — so the first admin page
    view after a boot caches the DEK for the agent runtime.
    """

    async def _check_auth(
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ) -> None:
        setup_complete = await config_store.is_setup_complete()
        if not setup_complete:
            return

        password_hash = await config_store.get("admin.password_hash")
        password_salt = await config_store.get("admin.password_salt")
        if not password_hash or not password_salt:
            return

        if not credentials:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        if not await config_store.verify_admin_password(credentials.credentials):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        # Auth passed — unseal the persona vault if it isn't already.
        if secret_store is not None and not secret_store.persona_unsealed():
            try:
                await secret_store.unseal_persona(credentials.credentials)
            except Exception:
                log.exception("Failed to unseal persona vault on auth")

    return _check_auth


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_admin_app(
    agent_state: AgentState,
    config_store: ConfigStore,
    lifespan=None,
    secret_store: SecretStore | None = None,
) -> tuple[FastAPI, object]:
    wacli = WacliManager()

    if lifespan is None:

        @asynccontextmanager
        async def _lifespan(app: FastAPI):  # noqa: ANN001
            yield
            await wacli.stop_auth()

        lifespan = _lifespan

    app = FastAPI(
        title="Personal Agent Admin",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.wacli = wacli

    # Mount static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ── Agent web artifacts (public files under {workspace}/artifacts/, issue #82) ──
    @app.get("/artifacts/{artifact_id}", response_model=None)
    async def artifact_root(artifact_id: str) -> Response:
        # Redirect to the trailing-slash form so relative links inside the
        # artifact (href="style.css", "img/logo.png") resolve under /artifacts/<slug>/.
        # Validate first so a malformed slug never reaches the Location header.
        if not valid_id(artifact_id):
            return HTMLResponse(NOT_FOUND_HTML, status_code=404)
        return RedirectResponse(f"/artifacts/{artifact_id}/", status_code=307)

    @app.get("/artifacts/{artifact_id}/{file_path:path}", response_model=None)
    async def serve_artifact(artifact_id: str, file_path: str = "") -> Response:
        """Serve a file from ``{workspace}/artifacts/<slug>/``. No auth — the slug
        is the only handle (artifacts are public shareables, issue #82).

        ``resolve`` blocks traversal/dotfiles/symlinks/hardlinks and anything
        outside the artifacts dir, so a sibling source file or a planted link to
        ``../.env`` can't leak. The CSP sandbox keeps artifact JS off the admin
        origin's localStorage; nosniff stops MIME-sniffing.
        """
        base = await serving_base(config_store)
        target = resolve(base, artifact_id, file_path) if base else None
        if target is None:
            return HTMLResponse(NOT_FOUND_HTML, status_code=404)
        return FileResponse(
            target,
            headers={
                "Content-Security-Policy": ARTIFACT_CSP,
                "X-Content-Type-Options": "nosniff",
            },
        )

    auth = _make_auth_dependency(config_store, secret_store)

    async def _resolved_config():
        """Export the Config with ``${vault:NAME}`` references resolved.

        The infra vault owns migrated credentials (issue #35), so any code that
        reconstructs a live Config from the store — rebuilding the LLM/search
        clients, testing embeddings — must resolve vault refs or it would hand a
        literal ``${vault:…}`` string to a client. ``secret_store`` is absent in
        some unit tests; fall back to the unresolved export there.
        """
        if secret_store is not None:
            return await config_store.export_to_config(vault_resolve=secret_store.infra_resolve)
        return await config_store.export_to_config()

    async def _preserve_vault_refs(values: dict) -> dict:
        """Drop blank/echoed writes to vault-managed secret keys.

        Once a credential is migrated its tab shows a read-only "managed in
        vault" field that submits empty (or echoes the ref). Without this guard
        that save would overwrite the ``${vault:NAME}`` reference and orphan the
        secret. A real, freshly-typed value (non-empty, not a ``${…}`` ref) is
        always allowed through, so re-entering a key still works.
        """
        from core.secret_store import INFRA_VAULT_KEYS

        out = dict(values)
        for key in INFRA_VAULT_KEYS:
            if key not in out:
                continue
            incoming = str(out[key])
            if incoming and not incoming.startswith("${"):
                continue
            if _is_vault_ref(await config_store.get(key)):
                del out[key]
        return out

    # Keys managed by dedicated tabs — excluded from the generic Config tab.
    _IDENTITY_KEYS = {
        "agent.character",
        "agent.name",
        "agent.owner_name",
        "agent.timezone",
    }
    _LLM_KEYS = {
        "agent.llm_provider",
        "agent.anthropic_api_key",
        "agent.openai_api_key",
        "agent.openai_base_url",
        "agent.google_api_key",
        "agent.google_base_url",
        "agent.grok_api_key",
        "agent.grok_base_url",
        "agent.deepseek_api_key",
        "agent.deepseek_base_url",
        "agent.model",
    }
    _SEARCH_PREFIX = "search."
    _MEMORY_PREFIX = "memory."
    _CHANNEL_PREFIX = "channels."
    _SCHEDULER_PREFIX = "scheduler."
    _CALENDAR_PREFIX = "calendar."
    _CONTACTS_PREFIX = "contacts."
    _YOU_PREFIX = "you."
    _VOICE_PREFIX = "voice."
    _HISTORY_PREFIX = "history."
    _EMAIL_PREFIX = "email."
    _PROMPT_PREFIX = "prompt."
    _TOOLS_PREFIX = "tools."
    _WORKSPACE_PREFIX = "workspace."
    _COMPACTION_PREFIX = "compaction."

    def _is_managed_key(key: str) -> bool:
        """Return True if this key is managed by a dedicated tab (not Config)."""
        if key in _IDENTITY_KEYS or key in _LLM_KEYS:
            return True
        for prefix in (
            _SEARCH_PREFIX,
            _MEMORY_PREFIX,
            _CHANNEL_PREFIX,
            _SCHEDULER_PREFIX,
            _CALENDAR_PREFIX,
            _CONTACTS_PREFIX,
            _YOU_PREFIX,
            _VOICE_PREFIX,
            _HISTORY_PREFIX,
            _EMAIL_PREFIX,
            _PROMPT_PREFIX,
            _TOOLS_PREFIX,
            _WORKSPACE_PREFIX,
            _COMPACTION_PREFIX,
        ):
            if key.startswith(prefix):
                return True
        return False

    # ── Health ──────────────────────────────────────────────────────────

    @app.get("/health")
    async def health() -> dict:
        agent = agent_state.agent
        setup_complete = await config_store.is_setup_complete()
        return {
            "status": "ok",
            "setup_complete": setup_complete,
            "agent_running": agent is not None and bool(agent.channels),
        }

    # ── Page routes (full HTML pages) ──────────────────────────────────

    @app.get("/login", response_class=HTMLResponse)
    async def login_page() -> HTMLResponse:
        """Login page for the admin dashboard."""
        return _render("login.html")

    @app.get("/setup", response_model=None)
    async def setup_page() -> Response:
        """Setup wizard page."""
        from core.config_store import SETUP_STEPS

        complete = await config_store.is_setup_complete()
        if complete:
            return RedirectResponse("/admin", status_code=302)

        step = await config_store.get_setup_step()
        step_ctx = await _wizard_step_context(step, config_store)
        return _render("setup.html", steps=SETUP_STEPS, current_step=step, step_ctx=step_ctx)

    @app.get("/admin", response_model=None)
    async def admin_page() -> Response:
        """Admin dashboard page."""
        setup_complete = await config_store.is_setup_complete()
        if not setup_complete:
            return RedirectResponse("/setup", status_code=302)
        owner_name = await config_store.get("agent.owner_name") or ""
        agent_name = await config_store.get("agent.name") or ""
        return _render(
            "dashboard.html",
            owner_name=owner_name,
            agent_name=agent_name,
        )

    @app.get("/admin/skills/new", response_model=None)
    async def admin_skill_new() -> Response:
        """New skill editor page."""
        setup_complete = await config_store.is_setup_complete()
        if not setup_complete:
            return RedirectResponse("/setup", status_code=302)
        return _render(
            "skill_editor.html",
            skill_name="",
            skill_content="",
            is_new=True,
        )

    @app.get("/admin/skills/{name}", response_model=None)
    async def admin_skill_editor(name: str) -> Response:
        """Skill editor page."""
        setup_complete = await config_store.is_setup_complete()
        if not setup_complete:
            return RedirectResponse("/setup", status_code=302)
        store = await _skills_store_from_config(config_store)
        skill = await store.get_skill(name)
        if not skill:
            raise HTTPException(404, f"Skill not found: {name}")
        return _render(
            "skill_editor.html",
            skill_name=skill.get("name", ""),
            skill_content=skill.get("content", ""),
            is_new=False,
        )

    async def _persona_editor_ctx() -> dict:
        """Shared context for the persona editor: all skills + gateable tools.

        A globally-disabled feature (e.g. artifacts) is dropped so its tool is
        hidden from the persona scope UI.
        """
        from voice.pipeline import KOKORO_LANGUAGES, KOKORO_VOICES

        store = await _skills_store_from_config(config_store)
        all_skills = [s["name"] for s in await store.list_skills()]
        sub_enabled = await config_store.get("subagents.enabled")
        sub_on = sub_enabled is None or sub_enabled == "true"
        ig_on = (await config_store.get("tools.imagegen.enabled")) == "true"
        ws_on = (await config_store.get("workspace.enabled")) == "true" and bool(
            (await config_store.get("workspace.directory") or "").strip()
        )
        return {
            "all_skills": all_skills,
            "all_tools": gateable_tools_for(
                subagents_enabled=sub_on,
                imagegen_enabled=ig_on,
                workspace_enabled=ws_on,
            ),
            "kokoro_voices": KOKORO_VOICES,
            "kokoro_languages": KOKORO_LANGUAGES,
            # External CLI tools that support a per-persona identity (#93). Only the
            # system-wide enabled ones are offered, so the registry stays the
            # source of truth for what exists.
            "tool_specs": [
                {"key": s.key, "label": s.label, "summary": s.summary}
                for s in tool_registry()
                if (await config_store.get(f"tools.{s.key}.enabled")) == "true"
            ],
            "infra_available": bool(secret_store and secret_store.infra.available),
            # Existing infra-vault secret names a persona can reuse as its gh token
            # instead of storing its own copy (#93). Infra vault only (boot-unsealed,
            # so it resolves headless like the per-persona token does).
            "infra_names": (
                [r["name"] for r in await secret_store.list_infra_names()]
                if secret_store and secret_store.infra.available
                else []
            ),
            # Email/calendar accounts a persona can be bound to (#110). Names only —
            # the account registry (Email/Calendar tabs) is the source of truth.
            "available_email_accounts": await _email_account_names(config_store),
            "available_calendar_accounts": [
                p["name"] for p in await _calendar_providers_context(config_store) if p["name"]
            ],
        }

    async def _persona_gh_token_set(name: str) -> bool:
        """Whether this persona already has a GitHub token in the infra vault (#93)."""
        if not name or secret_store is None or not secret_store.infra.available:
            return False
        try:
            return await secret_store.get_infra_secret(gh_token_secret_name(name)) is not None
        except Exception:
            return False

    @app.get("/admin/personae/new", response_model=None)
    async def admin_persona_new() -> Response:
        """New persona editor page."""
        if not await config_store.is_setup_complete():
            return RedirectResponse("/setup", status_code=302)
        from core.personae import Persona, to_markdown

        ctx = await _persona_editor_ctx()
        return _render(
            "persona_editor.html",
            is_new=True,
            persona=Persona(name=""),
            raw=to_markdown(Persona(name="")),
            gh_token_set=False,
            **ctx,
        )

    @app.get("/admin/personae/{name}", response_model=None)
    async def admin_persona_editor(name: str) -> Response:
        """Persona editor page."""
        if not await config_store.is_setup_complete():
            return RedirectResponse("/setup", status_code=302)
        from core.personae import to_markdown

        store = await _persona_store_from_config(config_store)
        persona = await store.get(name)
        if not persona:
            raise HTTPException(404, f"Persona not found: {name}")
        ctx = await _persona_editor_ctx()
        return _render(
            "persona_editor.html",
            is_new=False,
            persona=persona,
            raw=to_markdown(persona),
            gh_token_set=await _persona_gh_token_set(name),
            **ctx,
        )

    @app.get("/", response_model=None)
    async def root_redirect() -> RedirectResponse:
        """Redirect root to setup or admin based on state."""
        setup_complete = await config_store.is_setup_complete()
        if setup_complete:
            return RedirectResponse("/admin", status_code=302)
        return RedirectResponse("/setup", status_code=302)

    # ── HTMX partial routes ────────────────────────────────────────────

    @app.get("/partials/status", dependencies=[Depends(auth)])
    async def partial_status() -> HTMLResponse:
        """Status bar partial for the dashboard."""
        agent = agent_state.agent
        if agent:
            running = True
            channels = list(agent.channels.keys())
            scheduler_jobs = len(agent.scheduler.scheduler.get_jobs())
        else:
            running = False
            channels = []
            scheduler_jobs = 0
        status = agent_state.status
        if running and status not in ("STARTING", "RESTARTING", "STOPPING"):
            status = "RUNNING"
        return _render_partial(
            "partials/status.html",
            running=running,
            status=status,
            channels=channels,
            scheduler_jobs=scheduler_jobs,
        )

    @app.get("/partials/identity", dependencies=[Depends(auth)])
    async def partial_identity() -> HTMLResponse:
        """Agent identity tab partial."""
        from voice.pipeline import KOKORO_LANGUAGES, KOKORO_VOICES

        character = await config_store.get("agent.character") or ""
        agent_name = await config_store.get("agent.name") or ""
        stt_model = await config_store.get("voice.stt_model") or "base"
        tts_voice = await config_store.get("voice.tts_voice") or "en-US-AvaNeural"
        tts_enabled = await config_store.get("voice.tts_enabled") or "true"
        backend = await config_store.get("voice.backend") or "edge-tts"
        kokoro_voice = await config_store.get("voice.kokoro.default_voice") or "af_bella"
        return _render_partial(
            "partials/identity.html",
            character=character,
            agent_name=agent_name,
            stt_model=stt_model,
            tts_voice=tts_voice,
            tts_enabled=tts_enabled,
            backend=backend,
            kokoro_voice=kokoro_voice,
            kokoro_voices=KOKORO_VOICES,
            kokoro_languages=KOKORO_LANGUAGES,
        )

    @app.get("/partials/you", dependencies=[Depends(auth)])
    async def partial_you() -> HTMLResponse:
        """You tab partial — info about the user the assistant serves."""
        owner_name = await config_store.get("agent.owner_name") or ""
        timezone = await config_store.get("agent.timezone") or ""
        you_personalia = await config_store.get("you.personalia") or ""
        return _render_partial(
            "partials/you.html",
            owner_name=owner_name,
            timezone=timezone,
            you_personalia=you_personalia,
        )

    async def _render_permissions(scope: str = "") -> HTMLResponse:
        """Render the permissions tab for one scope (#100): "" = global default,
        else a persona slug whose own overrides are shown."""
        agent = agent_state.agent
        store = await _persona_store_from_config(config_store)
        personae = [p.name for p in await store.list_personae()]
        if agent:
            rules = agent.permissions.rules_for_scope(scope)
        elif scope:
            rules = {}
        else:
            from core.permissions import DEFAULT_RULES

            rules = dict(DEFAULT_RULES)
        return _render_partial(
            "partials/permissions.html", rules=rules, scope=scope, personae=personae
        )

    @app.get("/partials/permissions", dependencies=[Depends(auth)])
    async def partial_permissions(scope: str = "") -> HTMLResponse:
        """Permissions tab partial."""
        return await _render_permissions(scope)

    @app.get("/partials/skills", dependencies=[Depends(auth)])
    async def partial_skills() -> HTMLResponse:
        """Skills tab partial."""
        store = await _skills_store_from_config(config_store)
        skills = await store.list_skills()
        return _render_partial("partials/skills.html", skills=skills)

    @app.get("/partials/personae", dependencies=[Depends(auth)])
    async def partial_personae() -> HTMLResponse:
        """Personae tab partial — cards + active-persona selector."""
        store = await _persona_store_from_config(config_store)
        personae = await store.list_personae()
        active = (await config_store.get("agent.active_persona") or "").strip()
        return _render_partial("partials/personae.html", personae=personae, active=active)

    @app.get("/partials/channels", dependencies=[Depends(auth)])
    async def partial_channels() -> HTMLResponse:
        """Channels tab partial."""
        channel_data = await _channel_list_context(config_store, wacli)
        return _render_partial("partials/channels.html", **channel_data)

    @app.get("/partials/calendars", dependencies=[Depends(auth)])
    async def partial_calendars() -> HTMLResponse:
        """Calendars tab partial."""
        providers = await _calendar_providers_context(config_store)
        token_raw = await config_store.get("calendar.google_oauth_token")
        google_connected = bool(token_raw)
        client_id = await config_store.get("calendar.google_oauth_client_id") or ""
        client_secret = await config_store.get("calendar.google_oauth_client_secret") or ""
        has_oauth_creds = bool(client_id and client_secret)
        return _render_partial(
            "partials/calendars.html",
            providers=providers,
            google_connected=google_connected,
            google_client_id=client_id,
            has_oauth_creds=has_oauth_creds,
        )

    @app.get("/partials/contacts", dependencies=[Depends(auth)])
    async def partial_contacts() -> HTMLResponse:
        """Contacts tab partial."""
        providers = await _contact_providers_context(config_store)
        return _render_partial("partials/contacts.html", providers=providers)

    @app.get("/partials/email", dependencies=[Depends(auth)])
    async def partial_email() -> HTMLResponse:
        """Email tab partial."""
        raw = await config_store.get("email.providers") or "[]"
        try:
            providers = json.loads(raw)
        except json.JSONDecodeError, TypeError:
            providers = []
        return _render_partial("partials/email.html", providers=providers)

    @app.get("/partials/admin", dependencies=[Depends(auth)])
    async def partial_admin() -> HTMLResponse:
        """Admin tab partial."""
        return _render_partial("partials/admin.html")

    @app.get("/partials/llm", dependencies=[Depends(auth)])
    async def partial_llm() -> HTMLResponse:
        """LLM tab partial."""
        provider = await config_store.get("agent.llm_provider") or "anthropic"
        anthropic_api_key = await config_store.get("agent.anthropic_api_key") or ""
        openai_api_key = await config_store.get("agent.openai_api_key") or ""
        openai_base_url = await config_store.get("agent.openai_base_url") or ""
        google_api_key = await config_store.get("agent.google_api_key") or ""
        google_base_url = await config_store.get("agent.google_base_url") or ""
        grok_api_key = await config_store.get("agent.grok_api_key") or ""
        grok_base_url = await config_store.get("agent.grok_base_url") or ""
        deepseek_api_key = await config_store.get("agent.deepseek_api_key") or ""
        deepseek_base_url = await config_store.get("agent.deepseek_base_url") or ""
        # Vault-managed keys (issue #35): show a read-only "in vault" note instead
        # of the input, and don't ship the ref to the browser.
        anthropic_vaulted = _is_vault_ref(anthropic_api_key)
        openai_vaulted = _is_vault_ref(openai_api_key)
        google_vaulted = _is_vault_ref(google_api_key)
        grok_vaulted = _is_vault_ref(grok_api_key)
        deepseek_vaulted = _is_vault_ref(deepseek_api_key)
        model = await config_store.get("agent.model") or "claude-4-6-sonnet"
        max_tokens = await config_store.get("agent.max_tokens") or "8192"
        thinking_level = await config_store.get("agent.thinking_level") or ""
        temperature = await config_store.get("agent.temperature") or "0.5"
        extraction_provider = await config_store.get("memory.extraction_provider") or "deepseek"
        extraction_model = await config_store.get("memory.extraction_model") or "deepseek-v4-flash"
        consolidation_provider = (
            await config_store.get("memory.consolidation_provider") or "deepseek"
        )
        consolidation_model = (
            await config_store.get("memory.consolidation_model") or "deepseek-v4-flash"
        )
        extraction_thinking_level = await config_store.get("memory.extraction_thinking_level") or ""
        consolidation_thinking_level = (
            await config_store.get("memory.consolidation_thinking_level") or ""
        )
        gd_enabled = await config_store.get("goal_decomposition.enabled")
        gd_enabled = gd_enabled if gd_enabled is not None else "true"
        gd_provider = await config_store.get("goal_decomposition.provider") or "deepseek"
        gd_model = await config_store.get("goal_decomposition.model") or "deepseek-v4-flash"
        tr_enabled = await config_store.get("task_reflection.enabled")
        tr_enabled = tr_enabled if tr_enabled is not None else "true"
        tr_provider = await config_store.get("task_reflection.provider") or "deepseek"
        tr_model = await config_store.get("task_reflection.model") or "deepseek-v4-flash"
        tr_max_reflections = await config_store.get("task_reflection.max_reflections") or "12"
        gd_thinking_level = await config_store.get("goal_decomposition.thinking_level") or ""
        tr_thinking_level = await config_store.get("task_reflection.thinking_level") or ""
        rd_enabled = await config_store.get("reply_decision.enabled")
        rd_enabled = rd_enabled if rd_enabled is not None else "false"
        rd_provider = await config_store.get("reply_decision.provider") or "deepseek"
        rd_model = await config_store.get("reply_decision.model") or "deepseek-v4-flash"
        rd_thinking_level = await config_store.get("reply_decision.thinking_level") or ""
        rd_group_only = await config_store.get("reply_decision.group_only")
        rd_group_only = rd_group_only if rd_group_only is not None else "true"
        rd_max_replies = await config_store.get("reply_decision.max_replies_per_window") or "6"
        rd_window_seconds = await config_store.get("reply_decision.window_seconds") or "120"
        compaction_provider = await config_store.get("compaction.provider") or "deepseek"
        compaction_model = await config_store.get("compaction.model") or "deepseek-v4-flash"
        compaction_thinking_level = await config_store.get("compaction.thinking_level") or ""
        vision_enabled = await config_store.get("vision.enabled")
        vision_enabled = vision_enabled if vision_enabled is not None else "false"
        vision_provider = await config_store.get("vision.provider") or "anthropic"
        vision_model = await config_store.get("vision.model") or "claude-haiku-4-5"
        prompt_tool_usage_override = await config_store.get("prompt.tool_usage_override") or ""
        prompt_history_override = await config_store.get("prompt.history_handling_override") or ""
        prompt_capture_enabled = await config_store.get("admin.capture_prompts")
        prompt_capture_enabled = False
        if prompt_capture_enabled is not None:
            prompt_capture_enabled = str(prompt_capture_enabled).lower() == "true"
        return _render_partial(
            "partials/llm.html",
            provider=provider,
            anthropic_api_key="" if anthropic_vaulted else anthropic_api_key,
            anthropic_vaulted=anthropic_vaulted,
            openai_api_key="" if openai_vaulted else openai_api_key,
            openai_vaulted=openai_vaulted,
            openai_base_url=openai_base_url,
            google_api_key="" if google_vaulted else google_api_key,
            google_vaulted=google_vaulted,
            google_base_url=google_base_url,
            grok_api_key="" if grok_vaulted else grok_api_key,
            grok_vaulted=grok_vaulted,
            grok_base_url=grok_base_url,
            deepseek_api_key="" if deepseek_vaulted else deepseek_api_key,
            deepseek_vaulted=deepseek_vaulted,
            deepseek_base_url=deepseek_base_url,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_level=thinking_level,
            extraction_provider=extraction_provider,
            extraction_model=extraction_model,
            extraction_thinking_level=extraction_thinking_level,
            consolidation_provider=consolidation_provider,
            consolidation_model=consolidation_model,
            consolidation_thinking_level=consolidation_thinking_level,
            gd_enabled=gd_enabled,
            gd_provider=gd_provider,
            gd_model=gd_model,
            gd_thinking_level=gd_thinking_level,
            tr_enabled=tr_enabled,
            tr_provider=tr_provider,
            tr_model=tr_model,
            tr_max_reflections=tr_max_reflections,
            tr_thinking_level=tr_thinking_level,
            rd_enabled=rd_enabled,
            rd_provider=rd_provider,
            rd_model=rd_model,
            rd_thinking_level=rd_thinking_level,
            rd_group_only=rd_group_only,
            rd_max_replies=rd_max_replies,
            rd_window_seconds=rd_window_seconds,
            compaction_provider=compaction_provider,
            compaction_model=compaction_model,
            compaction_thinking_level=compaction_thinking_level,
            vision_enabled=vision_enabled,
            vision_provider=vision_provider,
            vision_model=vision_model,
            prompt_tool_usage_override=prompt_tool_usage_override,
            prompt_history_override=prompt_history_override,
            default_tool_usage=DEFAULT_TOOL_USAGE_BLOCK,
            default_history_handling=DEFAULT_HISTORY_HANDLING_BLOCK,
            prompt_capture_enabled=prompt_capture_enabled,
        )

    def _browser_rules() -> list[dict]:
        """Per-domain browser `act` rules (excludes the generic default rule)."""
        agent = agent_state.agent
        if not agent:
            return []
        marker = "browser.py act*"
        out: list[dict] = []
        for pattern, level in agent.permissions.rules.items():
            if marker not in pattern:
                continue
            domain = pattern.split(marker, 1)[1].rstrip("*")
            if not domain:
                continue  # the generic "ask for all acts" default, not a per-domain rule
            out.append({"domain": domain, "pattern": pattern, "level": level})
        return out

    @app.get("/partials/tools", dependencies=[Depends(auth)])
    async def partial_tools() -> HTMLResponse:
        """Tools tab partial — manage optional external CLI tools (gh, browser)."""
        gh_enabled = await config_store.get("tools.gh.enabled")
        gh_enabled = gh_enabled if gh_enabled is not None else "false"
        gh_token = await config_store.get("tools.gh.token") or ""
        gh_token_vaulted = _is_vault_ref(gh_token)

        browser_enabled = await config_store.get("tools.browser.enabled")
        browser_enabled = browser_enabled if browser_enabled is not None else "false"
        browser_headless = await config_store.get("tools.browser.headless")
        browser_headless = browser_headless if browser_headless is not None else "true"
        browser_cdp = await config_store.get("tools.browser.cdp_url") or ""
        browser_ua = await config_store.get("tools.browser.user_agent") or ""
        try:
            from tools.browser import cmd_profiles

            browser_profiles = cmd_profiles(None).get("profiles", [])
        except Exception:
            browser_profiles = []

        # Web artifacts (issue #82) — only the public-serving toggle remains; the
        # files live under the workspace. Key may be absent on an old store.
        artifacts_enabled = await config_store.get("artifacts.enabled")
        artifacts_enabled = "false" if artifacts_enabled == "false" else "true"

        # Subagents (issue #15) — keys may be absent on a store seeded before the
        # feature existed, so fall back to the SubagentsConfig defaults.
        sub_enabled = await config_store.get("subagents.enabled")
        sub_enabled = sub_enabled if sub_enabled is not None else "true"
        sub_recursion = await config_store.get("subagents.recursion_depth") or "3"
        sub_steps = await config_store.get("subagents.max_steps") or "12"
        sub_tokens = await config_store.get("subagents.token_budget") or "100000"
        sub_concurrent = await config_store.get("subagents.max_concurrent") or "3"
        # Result-summary inference (notification + context digest) for finished
        # background batches — fast/cheap model by default.
        ss_enabled = await config_store.get("subagent_summary.enabled")
        ss_enabled = ss_enabled if ss_enabled is not None else "true"
        ss_provider = await config_store.get("subagent_summary.provider") or "deepseek"
        ss_model = await config_store.get("subagent_summary.model") or "deepseek-v4-flash"
        ss_thinking = await config_store.get("subagent_summary.thinking_level") or ""

        # Image generation (issue #55).
        ig_enabled = await config_store.get("tools.imagegen.enabled")
        ig_enabled = ig_enabled if ig_enabled is not None else "false"
        ig_provider = await config_store.get("tools.imagegen.provider") or "openrouter"
        ig_model = await config_store.get("tools.imagegen.model") or ""
        ig_key = await config_store.get("tools.imagegen.api_key") or ""
        ig_key_vaulted = _is_vault_ref(ig_key)
        ig_daily = await config_store.get("tools.imagegen.daily_budget") or "0"
        ig_monthly = await config_store.get("tools.imagegen.monthly_budget") or "0"

        # WhatsApp tool (#97) — enable flag + wacli link status for the badge.
        wa_enabled = await config_store.get("tools.whatsapp.enabled")
        wa_enabled = wa_enabled if wa_enabled is not None else "false"
        wa_device_label = await config_store.get("tools.whatsapp.device_label") or ""
        try:
            wa_status = await wacli.auth_status()
        except Exception:
            wa_status = {"available": False, "authenticated": False}

        return _render_partial(
            "partials/tools.html",
            tools=tool_registry(),
            whatsapp_enabled=wa_enabled,
            whatsapp_device_label=wa_device_label,
            whatsapp_available=wa_status.get("available") is True,
            whatsapp_authenticated=wa_status.get("authenticated") is True,
            gh_enabled=gh_enabled,
            gh_token="" if gh_token_vaulted else gh_token,
            gh_token_vaulted=gh_token_vaulted,
            browser_enabled=browser_enabled,
            browser_headless=browser_headless,
            browser_cdp=browser_cdp,
            browser_ua=browser_ua,
            browser_profiles=browser_profiles,
            browser_rules=_browser_rules(),
            artifacts_enabled=artifacts_enabled,
            subagents_enabled=sub_enabled,
            subagents_recursion_depth=sub_recursion,
            subagents_max_steps=sub_steps,
            subagents_token_budget=sub_tokens,
            subagents_max_concurrent=sub_concurrent,
            summary_enabled=ss_enabled,
            summary_provider=ss_provider,
            summary_model=ss_model,
            summary_thinking_level=ss_thinking,
            imagegen_enabled=ig_enabled,
            imagegen_provider=ig_provider,
            imagegen_model=ig_model,
            imagegen_api_key="" if ig_key_vaulted else ig_key,
            imagegen_key_vaulted=ig_key_vaulted,
            imagegen_daily_budget=ig_daily,
            imagegen_monthly_budget=ig_monthly,
        )

    @app.get("/partials/workspace", dependencies=[Depends(auth)])
    async def partial_workspace() -> HTMLResponse:
        """Workspace tab partial — coding harness file tools (issue #76)."""
        ws_enabled = await config_store.get("workspace.enabled")
        ws_enabled = ws_enabled if ws_enabled is not None else "false"
        ws_directory = await config_store.get("workspace.directory") or ""
        return _render_partial(
            "partials/workspace.html",
            workspace_enabled=ws_enabled,
            workspace_directory=ws_directory,
        )

    @app.post("/tools/gh/test", dependencies=[Depends(auth)])
    async def test_gh_tool(request: Request) -> dict:
        """Verify a GitHub token by calling the GitHub API as that token."""
        body = await request.json()
        token = str(body.get("token", "")).strip()
        if not token:
            return {"ok": False, "error": "Token is required."}
        try:
            resp = await asyncio.to_thread(
                http_requests.get,
                "https://api.github.com/user",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                timeout=10,
            )
        except Exception as exc:  # noqa: BLE001 — surface any network error to the UI
            return {"ok": False, "error": str(exc)}
        if resp.status_code == 200:
            login = resp.json().get("login", "")
            return {"ok": True, "login": login}
        if resp.status_code in (401, 403):
            return {
                "ok": False,
                "error": "Token rejected by GitHub (invalid or insufficient scope).",
            }
        return {"ok": False, "error": f"GitHub returned HTTP {resp.status_code}."}

    @app.post("/tools/imagegen/test", dependencies=[Depends(auth)])
    async def test_imagegen_tool(request: Request) -> dict:
        """Generate a sample image so the user confirms the integration works (#55)."""
        import base64 as _b64

        from core import imagegen as _ig

        body = await request.json()
        provider = str(body.get("provider", "openrouter")).strip().lower()
        model = str(body.get("model", "")).strip()
        api_key = str(body.get("api_key", "")).strip()
        if not api_key:
            # Reuse a stored/vaulted image key or the matching LLM key.
            cfg = await _resolved_config()
            api_key = cfg.tools.imagegen.api_key or _ig.llm_fallback_key(cfg, provider)
        if not api_key:
            return {
                "ok": False,
                "error": "An API key is required (or configure an LLM key to reuse).",
            }
        prompt = (
            str(body.get("prompt", "")).strip()
            or "a friendly robot waving hello, simple flat vector illustration"
        )
        try:
            data, mime = await _ig.generate_bytes(provider, model, api_key, prompt)
        except Exception as exc:  # noqa: BLE001 — surface any error to the UI
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "image_b64": _b64.b64encode(data).decode("ascii"), "mime": mime}

    @app.post("/tools/browser/test", dependencies=[Depends(auth)])
    async def test_browser_tool(request: Request) -> dict:
        """Load a page via the browser CLI to confirm Chromium works (and a profile)."""
        body = await request.json()
        url = str(body.get("url", "")).strip() or "https://example.com"
        profile = str(body.get("profile", "")).strip() or "default"
        root = Path(__file__).resolve().parent.parent
        # Subprocess, not in-process: the sync Playwright API can't run in this event loop.
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "tools/browser.py",
                "read",
                "--url",
                url,
                "--profile",
                profile,
                cwd=str(root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "BROWSER_HEADLESS": "1"},
            )
            out, err = await asyncio.wait_for(proc.communicate(), timeout=60)
        except TimeoutError:
            return {"ok": False, "error": "Browser test timed out (60s)."}
        except Exception as exc:  # noqa: BLE001 — surface any error to the UI
            return {"ok": False, "error": str(exc)}
        try:
            data = json.loads(out.decode() or "{}")
        except json.JSONDecodeError:
            return {"ok": False, "error": (err.decode() or "no output")[:300]}
        if data.get("error"):
            return {"ok": False, "error": data["error"]}
        return {"ok": True, "title": data.get("title", ""), "url": data.get("url", "")}

    @app.post("/tools/browser/rules", dependencies=[Depends(auth)])
    async def add_browser_rule(request: Request) -> dict:
        """Pre-approve/ask/block browser actions on a domain (a permission rule)."""
        agent = agent_state.agent
        if not agent:
            raise HTTPException(503, "Agent not running")
        body = await request.json()
        domain = str(body.get("domain", "")).strip()
        level = str(body.get("level", "ALWAYS")).strip().upper()
        if not domain:
            return {"ok": False, "error": "Domain is required."}
        if level not in ("ALWAYS", "ASK", "NEVER"):
            level = "ASK"
        # Global default scope on purpose: a per-domain browser trust toggle applies
        # to every persona, not just one (#100 scoping is opt-in via the perms tab).
        agent.permissions.add_rule(f"run_command:*browser.py act*{domain}*", level)
        return {"ok": True, "rules": _browser_rules()}

    @app.post("/tools/browser/rules/delete", dependencies=[Depends(auth)])
    async def delete_browser_rule(request: Request) -> dict:
        agent = agent_state.agent
        if not agent:
            raise HTTPException(503, "Agent not running")
        body = await request.json()
        pattern = str(body.get("pattern", ""))
        if pattern:
            agent.permissions.remove_rule(pattern)
        return {"ok": True, "rules": _browser_rules()}

    @app.get("/partials/search", dependencies=[Depends(auth)])
    async def partial_search() -> HTMLResponse:
        """Search tab partial."""
        enabled = await config_store.get("search.enabled") or "false"
        provider = await config_store.get("search.provider") or "tavily"
        api_key = await config_store.get("search.api_key") or ""
        api_key_vaulted = _is_vault_ref(api_key)
        max_results = await config_store.get("search.max_results") or "5"
        return _render_partial(
            "partials/search.html",
            enabled=enabled,
            provider=provider,
            api_key="" if api_key_vaulted else api_key,
            api_key_vaulted=api_key_vaulted,
            max_results=max_results,
        )

    async def _render_memory_partial() -> HTMLResponse:
        """Build the Memory tab partial (config + stored memories).

        Shared by the tab load and the post-delete refresh so both render the
        full embedding/lifecycle config, not just the memory tables.
        """
        import aiosqlite

        async def _cfg(key: str, default: str) -> str:
            val = await config_store.get(key)
            return default if val is None or val == "" else str(val)

        async def _bool(key: str, default: str) -> str:
            val = await config_store.get(key)
            return default if val is None else str(val).lower()

        ctx: dict[str, object] = {
            "memory_long_term_limit": await _cfg("memory.long_term_limit", "50"),
            "emb_enabled": await _bool("memory.embedding.enabled", "true"),
            "emb_provider": await _cfg("memory.embedding.provider", "local"),
            "emb_model": await _cfg("memory.embedding.model", "BAAI/bge-small-en-v1.5"),
            "emb_base_url": await _cfg("memory.embedding.base_url", ""),
            "emb_top_k": await _cfg("memory.embedding.injection_top_k", "12"),
            "emb_recall_top_k": await _cfg("memory.embedding.recall_top_k", "10"),
            "hygiene_enabled": await _bool("memory.hygiene_enabled", "true"),
            "default_importance": await _cfg("memory.default_importance", "5.0"),
            "archive_after_days": await _cfg("memory.archive_after_days", "90"),
            "archive_max_importance": await _cfg("memory.archive_max_importance", "4.0"),
            "archive_min_idle_days": await _cfg("memory.archive_min_idle_days", "45"),
            "hygiene_threshold": await _cfg("memory.hygiene_similarity_threshold", "0.45"),
        }

        # Memory data — read directly from DB (works even when agent is stopped)
        memory_db = await config_store.get("memory.db_path") or "data/memory.db"
        long_term: list[dict] = []
        short_term: list[dict] = []
        if Path(memory_db).exists():
            # Idempotent migrate-on-read so a legacy DB has the scope column (#42)
            # even when no agent is running to have migrated it on startup.
            from core.memory import MemoryStore

            await MemoryStore(db_path=memory_db)._ensure_schema()
            cols = (
                "id, category, subject, content, source, confidence, created_at, updated_at, scope"
            )
            async with aiosqlite.connect(memory_db) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(f"SELECT {cols} FROM long_term ORDER BY updated_at DESC")
                long_term = [dict(row) for row in await cursor.fetchall()]
                cursor = await db.execute(
                    "SELECT id, content, context, expires_at, created_at, scope "
                    "FROM short_term WHERE expires_at > datetime('now') "
                    "ORDER BY created_at DESC"
                )
                short_term = [dict(row) for row in await cursor.fetchall()]

        return _render_partial(
            "partials/memory.html", long_term=long_term, short_term=short_term, **ctx
        )

    @app.get("/partials/memory", dependencies=[Depends(auth)])
    async def partial_memory() -> HTMLResponse:
        """Memory tab partial."""
        return await _render_memory_partial()

    @app.get("/partials/history", dependencies=[Depends(auth)])
    async def partial_history() -> HTMLResponse:
        """History tab partial."""
        mode = await config_store.get("history.mode") or "injection"
        max_turns = await config_store.get("history.max_turns") or "10"
        c_enabled = await config_store.get("compaction.enabled")
        c_enabled = c_enabled if c_enabled is not None else "true"
        c_threshold_type = await config_store.get("compaction.threshold_type") or "percent"
        c_threshold_percent = await config_store.get("compaction.threshold_percent") or "80"
        c_threshold_tokens = await config_store.get("compaction.threshold_tokens") or "150000"
        c_context_window = await config_store.get("compaction.context_window") or "200000"
        c_keep_recent_turns = await config_store.get("compaction.keep_recent_turns") or "4"
        return _render_partial(
            "partials/history.html",
            mode=mode,
            max_turns=max_turns,
            compaction_enabled=c_enabled,
            compaction_threshold_type=c_threshold_type,
            compaction_threshold_percent=c_threshold_percent,
            compaction_threshold_tokens=c_threshold_tokens,
            compaction_context_window=c_context_window,
            compaction_keep_recent_turns=c_keep_recent_turns,
        )

    @app.get("/partials/logs", dependencies=[Depends(auth)])
    async def partial_logs() -> HTMLResponse:
        """Logs tab partial (filter controls + auto-refresh)."""
        return _render_partial("partials/logs.html", levels=_LOG_LEVELS)

    @app.get("/partials/logs-content", dependencies=[Depends(auth)])
    async def partial_logs_content(
        stream: str = "",
        level: str = "",
        q: str = "",
        since: str = "",
        until: str = "",
    ) -> HTMLResponse:
        """Filtered log lines for HTMX swap (#75). Filters: stream (regex), level
        (min severity), q (text), since/until (time range)."""
        snapshot = list(_LOG_BUFFER)
        entries = _filter_log_entries(
            snapshot, stream=stream, level=level, q=q, since=since, until=until
        )[-300:]
        # All stream names ever seen (not just the filtered slice) for the picker.
        streams = sorted({e["stream"] for e in snapshot})
        return _render_partial("partials/logs_content.html", entries=entries, streams=streams)

    # ── Jobs partial + API ─────────────────────────────────────────────

    def _get_job_store():
        """Get the JobStore, either from the running agent or create a standalone one."""
        agent = agent_state.agent
        if agent and agent.job_store:
            return agent.job_store
        from core.job_store import JobStore

        return JobStore(db_path="data/jobs.db")

    def _get_jobs_list(include_done: bool = False) -> list[dict]:
        """Build a list of job dicts from the JobStore + APScheduler next_run times.

        By default only live jobs (active/paused) are returned; pass
        ``include_done=True`` to also include completed/cancelled jobs.
        """
        store = _get_job_store()
        db_jobs = store.list_jobs_sync(include_done=include_done)
        agent = agent_state.agent

        # Build a map of APScheduler next_run times
        next_runs: dict[str, str] = {}
        if agent:
            for ap_job in agent.scheduler.scheduler.get_jobs():
                if ap_job.next_run_time:
                    next_runs[ap_job.id] = ap_job.next_run_time.strftime("%Y-%m-%d %H:%M")

        jobs = []
        for j in db_jobs:
            schedule = j.get("schedule", "cron")
            if schedule == "once":
                cron_display = f"once @ {j.get('run_at', '?')}"
            else:
                cron_display = j.get("cron", "")

            jobs.append(
                {
                    "id": j["id"],
                    "cron": cron_display,
                    "type": j["type"],
                    "task": j.get("task", ""),
                    "channel": j.get("channel", "telegram"),
                    "next_run": next_runs.get(j["id"], "(not scheduled)" if not agent else ""),
                    "status": j.get("status", "active"),
                    "schedule": schedule,
                    "raw_cron": j.get("cron", ""),
                    "run_at": j.get("run_at", ""),
                    "description": j.get("description", ""),
                    "created_by": j.get("created_by", ""),
                    "persona": j.get("persona", ""),
                }
            )
        return jobs

    @app.get("/partials/jobs", dependencies=[Depends(auth)])
    async def partial_jobs(show_completed: bool = False) -> HTMLResponse:
        """Jobs tab partial. ``show_completed`` reveals done/cancelled jobs."""
        jobs = _get_jobs_list(include_done=show_completed)
        agent_running = agent_state.agent is not None
        return _render_partial(
            "partials/jobs.html",
            jobs=jobs,
            agent_running=agent_running,
            show_completed=show_completed,
        )

    @app.post("/jobs", dependencies=[Depends(auth)])
    async def upsert_job(request: Request) -> HTMLResponse:
        """Add or update a scheduled job. Returns refreshed jobs partial."""
        content_type = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" in content_type:
            body = await request.form()
        else:
            body = await request.json()

        job_id = str(body.get("job_id", "")).strip()
        schedule = str(body.get("schedule", "cron")).strip()
        cron = str(body.get("cron", "")).strip()
        run_at = str(body.get("run_at", "")).strip()
        job_type = str(body.get("type", "agent")).strip()
        task = str(body.get("task", "")).strip()
        channel = str(body.get("channel", "telegram")).strip()
        description = str(body.get("description", "")).strip()
        persona = str(body.get("persona", "")).strip()

        if not job_id:
            raise HTTPException(400, "Job ID is required")
        if schedule not in ("cron", "once"):
            raise HTTPException(400, f"Invalid schedule type: {schedule}")
        if schedule == "cron":
            if not cron:
                raise HTTPException(400, "Cron schedule is required")
        else:
            if not run_at:
                raise HTTPException(400, "Run-at datetime is required for one-time jobs")
        if job_type not in ("agent", "agent_silent", "system", "memory_consolidation", "subagent"):
            raise HTTPException(400, f"Invalid job type: {job_type}")
        if job_type != "memory_consolidation" and not task:
            raise HTTPException(400, "Task is required for agent/system/subagent jobs")
        # Persona scopes who an agent job runs as (#101); meaningless for a raw
        # system command or memory consolidation, so don't persist a stray value.
        if job_type in ("system", "memory_consolidation"):
            persona = ""

        if schedule == "cron":
            from core.scheduler import _parse_cron

            try:
                _parse_cron(cron)
            except ValueError as exc:
                raise HTTPException(400, str(exc))
        else:
            # Validate the run_at datetime
            try:
                datetime.fromisoformat(run_at)
            except ValueError:
                raise HTTPException(400, f"Invalid datetime format: {run_at!r}. Use ISO format.")

        # Write to JobStore
        store = _get_job_store()
        await store.upsert_job(
            job_id=job_id,
            type=job_type,
            schedule=schedule,
            cron=cron if schedule == "cron" else None,
            run_at=run_at if schedule == "once" else None,
            task=task,
            channel=channel,
            status="active",
            created_by="admin",
            description=description,
            persona=persona,
        )

        # Sync with APScheduler if the agent is running
        agent = agent_state.agent
        if agent:
            await agent.scheduler.sync_job(job_id)

        log.info("Job %r upserted via admin: %s (%s)", job_id, cron, job_type)

        show_completed = str(body.get("show_completed", "")).strip().lower() in ("true", "on", "1")
        jobs = _get_jobs_list(include_done=show_completed)
        agent_running = agent is not None
        resp = _render_partial(
            "partials/jobs.html",
            jobs=jobs,
            agent_running=agent_running,
            show_completed=show_completed,
        )
        resp.headers["HX-Trigger"] = json.dumps({"showToast": f'Job "{job_id}" saved'})
        return resp

    @app.post("/jobs/delete", dependencies=[Depends(auth)])
    async def delete_job(request: Request) -> HTMLResponse:
        """Delete a scheduled job. Returns refreshed jobs partial."""
        content_type = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" in content_type:
            body = await request.form()
        else:
            body = await request.json()

        job_id = str(body.get("job_id", "")).strip()
        if not job_id:
            raise HTTPException(400, "Missing 'job_id' in request body")

        store = _get_job_store()
        deleted = await store.delete_job(job_id)
        if not deleted:
            raise HTTPException(404, f"Job not found: {job_id}")

        # Remove from APScheduler if running
        agent = agent_state.agent
        if agent:
            await agent.scheduler.sync_job(job_id)

        log.info("Job %r deleted via admin", job_id)

        show_completed = str(body.get("show_completed", "")).strip().lower() in ("true", "on", "1")
        jobs = _get_jobs_list(include_done=show_completed)
        agent_running = agent is not None
        resp = _render_partial(
            "partials/jobs.html",
            jobs=jobs,
            agent_running=agent_running,
            show_completed=show_completed,
        )
        resp.headers["HX-Trigger"] = json.dumps({"showToast": f'Job "{job_id}" deleted'})
        return resp

    @app.post("/jobs/run", dependencies=[Depends(auth)])
    async def run_job_now(request: Request) -> HTMLResponse:
        """Trigger a job to run immediately. Returns a flash message."""
        agent = agent_state.agent
        if not agent:
            raise HTTPException(503, "Agent not running")
        content_type = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" in content_type:
            body = await request.form()
        else:
            body = await request.json()

        job_id = str(body.get("job_id", "")).strip()
        if not job_id:
            raise HTTPException(400, "Missing 'job_id' in request body")

        # Look up the job in JobStore
        store = _get_job_store()
        job = await store.get_job(job_id)
        if not job:
            raise HTTPException(404, f"Job not found: {job_id}")

        import asyncio

        from core.scheduler import (
            run_agent_task,
            run_memory_consolidation,
            run_subagent_task,
            run_system_command,
        )

        job_type = job.get("type", "agent")
        task = job.get("task", "")
        channel_name = job.get("channel", "telegram")

        # Restore the originating persona + chat (issue #71) so a manual run
        # behaves exactly like the scheduled one (empty for pre-#71/admin jobs).
        persona = job.get("persona", "")
        origin_user_id = job.get("origin_user_id", "")
        origin_chat_id = job.get("origin_chat_id", "")

        if job_type in ("agent", "agent_silent"):
            silent = job_type == "agent_silent"
            asyncio.create_task(
                run_agent_task(
                    task=task,
                    channel=channel_name,
                    job_id=job_id,
                    silent=silent,
                    persona=persona,
                    origin_user_id=origin_user_id,
                    origin_chat_id=origin_chat_id,
                )
            )
        elif job_type == "system":
            asyncio.create_task(run_system_command(command=task))
        elif job_type == "memory_consolidation":
            asyncio.create_task(run_memory_consolidation())
        elif job_type == "subagent":
            asyncio.create_task(
                run_subagent_task(
                    persona=persona,
                    task=task,
                    channel=channel_name,
                    job_id=job_id,
                    origin_user_id=origin_user_id,
                    origin_chat_id=origin_chat_id,
                )
            )
        else:
            return HTMLResponse('<span class="alert-error">Unknown job type</span>')

        log.info("Job %r triggered manually via admin", job_id)
        return HTMLResponse(
            f'<span class="alert-success">Job &quot;{job_id}&quot; triggered — check logs '
            "for output</span>"
        )

    # ── Subagent runs (issue #15) ──────────────────────────────────────

    def _subagent_runs() -> list:
        """Live subagent runs from the running agent (newest first)."""
        agent = agent_state.agent
        if not agent:
            return []
        return agent.subagents.list_runs()

    @app.get("/partials/subagent-runs", dependencies=[Depends(auth)])
    async def partial_subagent_runs() -> HTMLResponse:
        """Subagent runs card grid — polled by the Jobs tab for live status."""
        return _render_partial(
            "partials/subagent_runs.html",
            runs=_subagent_runs(),
            agent_running=agent_state.agent is not None,
        )

    @app.post("/subagents/cancel", dependencies=[Depends(auth)])
    async def cancel_subagent(request: Request) -> HTMLResponse:
        """Cancel a running subagent. Returns the refreshed runs partial."""
        agent = agent_state.agent
        if not agent:
            raise HTTPException(503, "Agent not running")
        content_type = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" in content_type:
            body = await request.form()
        else:
            body = await request.json()
        run_id = str(body.get("run_id", "")).strip()
        if not run_id:
            raise HTTPException(400, "Missing 'run_id' in request body")
        agent.subagents.cancel(run_id)
        log.info("Subagent %r cancelled via admin", run_id)
        return _render_partial(
            "partials/subagent_runs.html", runs=_subagent_runs(), agent_running=True
        )

    # ── Config API ─────────────────────────────────────────────────────

    @app.get("/config", dependencies=[Depends(auth)])
    async def get_config() -> dict:
        data = await config_store.get_all_redacted()
        return {k: v for k, v in data.items() if not _is_managed_key(k)}

    @app.get("/config/character", dependencies=[Depends(auth)])
    async def get_character() -> dict:
        value = await config_store.get("agent.character") or ""
        return {"content": value}

    @app.post("/config/character", dependencies=[Depends(auth)])
    async def put_character(request: Request) -> dict:
        body = await request.json()
        content = body.get("content", "")
        await config_store.set("agent.character", content)
        return {"updated": "agent.character"}

    @app.get("/config/you-personalia", dependencies=[Depends(auth)])
    async def get_you_personalia() -> dict:
        value = await config_store.get("you.personalia") or ""
        return {"content": value}

    @app.post("/config/you-personalia", dependencies=[Depends(auth)])
    async def put_you_personalia(request: Request) -> dict:
        body = await request.json()
        content = body.get("content", "")
        await config_store.set("you.personalia", content)
        return {"updated": "you.personalia"}

    @app.get("/config/{section}", dependencies=[Depends(auth)])
    async def get_config_section(section: str) -> dict:
        return await config_store.get_section_redacted(section)

    @app.patch("/config", dependencies=[Depends(auth)])
    async def patch_config(body: ConfigPatchIn) -> dict:
        # Which keys actually changed — so restart_required only fires when a
        # restart-bound value really moved. A form may re-send unchanged keys
        # (e.g. History saves both mode (hot-applied) and max_turns every time).
        values = await _preserve_vault_refs(body.values)
        changed = {k: v for k, v in values.items() if str(await config_store.get(k)) != str(v)}
        await config_store.set_many(values)
        agent = agent_state.agent
        if agent:
            try:
                new_config = await _resolved_config()
                agent.config = new_config
                agent.llm = LLMClient.from_agent_config(new_config.agent)
                agent.llm.temperature = new_config.agent.temperature  # #12: live temp update
                agent.executor.tool_env = tool_env(new_config)
                agent.history_mode = new_config.history.mode
                mem_cfg = new_config.memory
                agent.memory.long_term_limit = mem_cfg.long_term_limit
                # Rebuild the embedder (lazy — no model load here) and refresh the
                # Tier 3/4 lifecycle knobs so memory config changes apply live.
                agent.memory.embedder = agent._build_embedder()
                agent.memory.injection_top_k = mem_cfg.embedding.injection_top_k
                agent.memory.recall_top_k = mem_cfg.embedding.recall_top_k
                agent.memory.default_importance = mem_cfg.default_importance
                agent.memory.archive_after_days = mem_cfg.archive_after_days
                agent.memory.archive_max_importance = mem_cfg.archive_max_importance
                agent.memory.archive_min_idle_days = mem_cfg.archive_min_idle_days
                agent.memory.hygiene_enabled = mem_cfg.hygiene_enabled
                agent.memory.hygiene_similarity_threshold = mem_cfg.hygiene_similarity_threshold
                agent.reflections.max_reflections = new_config.task_reflection.max_reflections
                if new_config.search.enabled and new_config.search.api_key:
                    from tavily import TavilyClient

                    agent.search_client = TavilyClient(api_key=new_config.search.api_key)
                else:
                    agent.search_client = None
            except Exception:
                log.exception("Failed to apply updated config to running agent")
        return {
            "updated": list(values.keys()),
            "restart_required": _config_requires_restart(changed),
        }

    @app.post("/voice/preview", dependencies=[Depends(auth)])
    async def voice_preview(body: VoicePreviewIn) -> Response:
        """Synthesize a short sample of the given voice and return the audio so
        the admin UI can play it (voice selection/testing, #84).  Edge-tts
        voices preview anytime; Kokoro voices need the Kokoro backend loaded."""
        agent = agent_state.agent
        pipeline = getattr(agent, "voice", None) if agent else None
        if pipeline is None:
            raise HTTPException(
                503, "Voice pipeline not loaded — enable TTS and restart the agent."
            )
        text = (body.text or "").strip() or "Hi! This is a quick preview of this voice."
        try:
            audio, mime = await pipeline.preview(text, body.voice.strip(), body.lang.strip())
        except Exception as exc:
            raise HTTPException(503, f"Preview failed: {exc}") from exc
        return Response(content=audio, media_type=mime)

    @app.post("/debug/system-prompt/preview", dependencies=[Depends(auth)])
    async def system_prompt_preview(body: PromptPreviewIn) -> dict:
        message = body.message.strip()
        config = await _resolved_config()

        skills_store = await _skills_store_from_config(config_store)
        await skills_store.ensure_seeded()
        skills = await skills_store.list_skills()
        skill_lines = []
        for skill in skills:
            summary = str(skill.get("summary", "")).strip()
            name = str(skill.get("name", "")).strip()
            if not name:
                continue
            if summary:
                skill_lines.append(f"- {name}: {summary}")
            else:
                skill_lines.append(f"- {name}")
        skills_index = "\n".join(skill_lines)

        memories = ""
        if body.include_memories:
            query = message or None
            if agent_state.agent:
                memories = await agent_state.agent.memory.format_for_prompt(query=query)
            else:
                from core.memory import MemoryStore

                memories = await MemoryStore(
                    db_path=config.memory.db_path,
                    long_term_limit=config.memory.long_term_limit,
                ).format_for_prompt(query=query)

        reflections = ""
        if body.include_reflections and config.task_reflection.enabled:
            if agent_state.agent:
                reflections = await agent_state.agent.reflections.format_for_prompt()
            else:
                from core.task_reflection import ReflectionStore

                reflections = await ReflectionStore(
                    db_path=config.task_reflection.db_path,
                    max_reflections=config.task_reflection.max_reflections,
                ).format_for_prompt()

        decomposed_goal = None
        if message and config.goal_decomposition.enabled:
            try:
                if agent_state.agent and hasattr(agent_state.agent, "_background_llm"):
                    llm = agent_state.agent._background_llm(config.goal_decomposition.provider)
                else:
                    provider = config.goal_decomposition.provider
                    llm = LLMClient(
                        provider=provider,
                        api_key=getattr(config.agent, f"{provider}_api_key", ""),
                        base_url=getattr(
                            config.agent,
                            f"{provider}_base_url",
                            None,
                        ),
                    )
                gd_model = config.goal_decomposition.model
                is_complex = await classify_complexity(llm, gd_model, message)
                if is_complex:
                    decomposed_goal = await decompose_goal(llm, gd_model, message)
            except Exception:
                log.exception("Prompt preview decomposition failed")

        history_mode = config.history.mode
        if agent_state.agent and hasattr(agent_state.agent, "history_mode"):
            history_mode = cast(str, agent_state.agent.history_mode)

        sections = build_prompt_sections(
            config=config,
            history_mode=history_mode,
            skills_index=skills_index,
            memories=memories,
            reflections=reflections,
            decomposed_goal=decomposed_goal,
            secrets_available=secret_store is not None,
            include_memories=body.include_memories,
            include_reflections=body.include_reflections,
            skills_on_demand=config.agent.skills_index_mode == "on_demand",
        )
        full_prompt = sections.full_prompt
        section_map = sections.as_dict()
        token_estimate = max(1, len(full_prompt) // 4) if full_prompt else 0
        return {
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "history_mode": history_mode,
            "token_estimate": token_estimate,
            "full_prompt": full_prompt,
            "sections": section_map,
            "lengths": {k: len(v or "") for k, v in section_map.items()},
            "flags": {
                "include_memories": body.include_memories,
                "include_reflections": body.include_reflections,
                "decomposition_applied": decomposed_goal is not None,
            },
        }

    @app.get("/debug/system-prompt/recent", dependencies=[Depends(auth)])
    async def system_prompt_recent() -> dict:
        agent = agent_state.agent
        if not agent:
            return {"enabled": False, "items": []}
        agent_cfg = getattr(agent, "config", None)
        admin_cfg = getattr(agent_cfg, "admin", None)
        capture_cfg = bool(getattr(admin_cfg, "capture_prompts", False))
        enabled = capture_cfg and hasattr(agent, "get_recent_system_prompts")
        items = agent.get_recent_system_prompts() if enabled else []
        return {"enabled": enabled, "items": items}

    @app.post("/admin/password", dependencies=[Depends(auth)])
    async def change_admin_password(body: PasswordChangeIn) -> dict:
        current = body.current_password.strip()
        new_password = body.new_password.strip()
        if not current or not new_password:
            return {"ok": False, "error": "Both current and new passwords are required."}
        if not await config_store.verify_admin_password(current):
            return {"ok": False, "error": "Current password is incorrect."}
        # Re-wrap the persona vault DEK FIRST; only advance the admin password if
        # it succeeds, so a rewrap failure can never orphan the vault (new auth
        # password but DEK still wrapped under the old one).
        if secret_store is not None:
            try:
                await secret_store.rotate_password(current, new_password)
            except Exception:
                log.exception("Vault rewrap failed; aborting password change")
                return {
                    "ok": False,
                    "error": "Could not re-encrypt the secrets vault; password unchanged.",
                }
        await config_store.set_admin_password(new_password)
        return {"ok": True, "token": new_password}

    @app.post("/config/delete", dependencies=[Depends(auth)])
    async def delete_config(request: Request) -> HTMLResponse:
        """Delete a config key. Returns refreshed config partial for HTMX."""
        content_type = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" in content_type:
            body = await request.form()
        else:
            body = await request.json()
        key = str(body.get("key", ""))
        if not key:
            raise HTTPException(400, "Missing 'key' in request body")
        deleted = await config_store.delete(key)
        if not deleted:
            raise HTTPException(404, f"Config key not found: {key}")
        # Return refreshed config partial
        data = await config_store.get_all_redacted()
        filtered = {k: v for k, v in data.items() if not _is_managed_key(k)}
        config_items = sorted(filtered.items())
        return _render_partial("partials/config.html", config_items=config_items)

    @app.post("/calendar/providers", dependencies=[Depends(auth)])
    async def save_calendar_providers(body: CalendarProvidersIn) -> dict:
        providers = []
        for p in body.providers:
            name = str(p.get("name", "")).strip()
            url = str(p.get("url", "")).strip()
            username = str(p.get("username", "")).strip()
            password = str(p.get("password", "")).strip()
            if not any([name, url, username, password]):
                continue
            providers.append(
                {
                    "name": name,
                    "url": url,
                    "username": username,
                    "password": password,
                }
            )
        await config_store.set("calendar.providers", json.dumps(providers))
        return {"ok": True}

    @app.post("/contacts/providers", dependencies=[Depends(auth)])
    async def save_contact_providers(body: ContactProvidersIn) -> dict:
        providers = []
        for p in body.providers:
            name = str(p.get("name", "")).strip()
            provider_type = str(p.get("type", "carddav")).strip() or "carddav"
            url = str(p.get("url", "")).strip()
            username = str(p.get("username", "")).strip()
            password = str(p.get("password", "")).strip()
            client_id = str(p.get("client_id", "")).strip()
            client_secret = str(p.get("client_secret", "")).strip()
            if not any([name, url, username, password, client_id, client_secret]):
                continue
            providers.append(
                {
                    "name": name,
                    "type": provider_type,
                    "url": url,
                    "username": username,
                    "password": password,
                    "client_id": client_id,
                    "client_secret": client_secret,
                }
            )
        await config_store.set("contacts.providers", json.dumps(providers))
        return {"ok": True}

    @app.post("/email/providers", dependencies=[Depends(auth)])
    async def save_email_providers(body: EmailProvidersIn) -> dict:
        """Save structured email (IMAP/SMTP) providers to the config store."""
        providers = []
        for p in body.providers:
            name = str(p.get("name", "")).strip()
            email = str(p.get("email", "")).strip()
            display_name = str(p.get("display_name", "")).strip()
            imap_host = str(p.get("imap_host", "")).strip()
            imap_port = str(p.get("imap_port", "993")).strip() or "993"
            smtp_host = str(p.get("smtp_host", "")).strip()
            smtp_port = str(p.get("smtp_port", "465")).strip() or "465"
            login = str(p.get("login", "")).strip()
            password = str(p.get("password", "")).strip()
            if not any([name, email, imap_host, smtp_host]):
                continue
            providers.append(
                {
                    "name": name,
                    "email": email,
                    "display_name": display_name,
                    "imap_host": imap_host,
                    "imap_port": imap_port,
                    "smtp_host": smtp_host,
                    "smtp_port": smtp_port,
                    "login": login,
                    "password": password,
                }
            )
        await config_store.set("email.providers", json.dumps(providers))
        return {"ok": True}

    def _resolve_secret_ref(value: str) -> str:
        """Expand a ${vault:NAME} reference for a connection test, so a vault-backed
        password can be verified without ever revealing it to the client (#110)."""
        if secret_store is None or "${vault:" not in value:
            return value
        from core.config import resolve_vault_vars

        return resolve_vault_vars(value, secret_store.infra_resolve)

    @app.post("/email/test", dependencies=[Depends(auth)])
    async def test_email_account(body: EmailTestIn) -> dict:
        """Probe IMAP + SMTP login for one account (#110). Returns per-protocol
        status; the password is resolved from the vault if it is a reference and
        is never echoed back."""
        import asyncio

        password = _resolve_secret_ref(body.password.strip())
        login = body.login.strip() or body.email.strip()

        def _probe() -> dict:
            import imaplib
            import smtplib

            out: dict = {"imap": None, "smtp": None}
            if body.imap_host.strip():
                try:
                    port = int(body.imap_port or 993)
                    with imaplib.IMAP4_SSL(body.imap_host.strip(), port, timeout=10) as m:
                        m.login(login, password)
                    out["imap"] = "ok"
                except Exception as exc:  # noqa: BLE001 — surface the reason to the admin
                    out["imap"] = f"error: {exc}"
            if body.smtp_host.strip():
                try:
                    port = int(body.smtp_port or 465)
                    with smtplib.SMTP_SSL(body.smtp_host.strip(), port, timeout=10) as s:
                        s.login(login, password)
                    out["smtp"] = "ok"
                except Exception as exc:  # noqa: BLE001
                    out["smtp"] = f"error: {exc}"
            return out

        result = await asyncio.to_thread(_probe)
        result["ok"] = all(v == "ok" for v in (result["imap"], result["smtp"]) if v is not None)
        return result

    @app.post("/calendar/test", dependencies=[Depends(auth)])
    async def test_calendar_account(body: CalendarTestIn) -> dict:
        """Probe CalDAV reachability for one account (#110): connect and count the
        calendars found. Vault-backed passwords resolve server-side."""
        import asyncio

        password = _resolve_secret_ref(body.password.strip())

        def _probe() -> dict:
            import caldav

            try:
                client = caldav.DAVClient(
                    url=body.url.strip(), username=body.username.strip(), password=password
                )
                cals = client.principal().calendars()
                return {"ok": True, "calendars": len(cals)}
            except Exception as exc:  # noqa: BLE001 — surface the reason to the admin
                return {"ok": False, "error": str(exc)}

        return await asyncio.to_thread(_probe)

    # ── Google Calendar OAuth 2.0 flow ─────────────────────────────────

    @app.post("/calendar/google/oauth/save-credentials", dependencies=[Depends(auth)])
    async def save_google_oauth_credentials(body: GoogleOAuthClientIn) -> dict:
        """Save Google OAuth client_id and client_secret to the config store."""
        await config_store.set("calendar.google_oauth_client_id", body.client_id.strip())
        await config_store.set("calendar.google_oauth_client_secret", body.client_secret.strip())
        return {"ok": True}

    async def _start_google_oauth(request: Request, *, scope: list[str], kind: str) -> dict:
        client_id = await config_store.get("calendar.google_oauth_client_id")
        client_secret = await config_store.get("calendar.google_oauth_client_secret")
        if not client_id or not client_secret:
            raise HTTPException(
                400,
                "Google OAuth client_id and client_secret must be configured first.",
            )

        code_verifier, code_challenge = _generate_pkce()
        state = secrets.token_urlsafe(32)

        callback_url = str(request.base_url).rstrip("/") + f"/google/oauth/callback?kind={kind}"

        _oauth_pending[state] = {
            "code_verifier": code_verifier,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": callback_url,
            "kind": kind,
        }

        params = {
            "client_id": client_id,
            "redirect_uri": callback_url,
            "response_type": "code",
            "scope": " ".join(scope),
            "access_type": "offline",
            "prompt": "consent",
            "code_challenge_method": "S256",
            "code_challenge": code_challenge,
            "state": state,
        }
        auth_url = f"{_GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"
        return {"auth_url": auth_url}

    @app.get("/calendar/google/oauth/start", dependencies=[Depends(auth)])
    async def google_oauth_start(request: Request) -> dict:
        """Initiate the Google OAuth 2.0 flow for calendars."""
        return await _start_google_oauth(request, scope=_GOOGLE_CALENDAR_SCOPES, kind="calendar")

    @app.get("/contacts/google/oauth/start", dependencies=[Depends(auth)])
    async def google_contacts_oauth_start(request: Request) -> dict:
        """Initiate the Google OAuth 2.0 flow for contacts."""
        return await _start_google_oauth(request, scope=_GOOGLE_CONTACTS_SCOPES, kind="contacts")

    @app.get("/google/oauth/callback")
    async def google_oauth_callback(
        kind: str = "calendar", code: str = "", state: str = "", error: str = ""
    ):
        """Handle the OAuth 2.0 callback from Google.

        This is opened in a popup window — no auth header needed since
        Google redirects the browser here directly.
        """
        if error:
            return HTMLResponse(
                f"<html><body><h2>Authorization failed: {error}</h2>"
                f"<script>window.opener && window.opener.postMessage("
                f"{{type:'google-oauth-error',error:'{error}'}},'*');"
                f"setTimeout(()=>window.close(),2000)</script></body></html>"
            )

        pending = _oauth_pending.pop(state, None)
        if not pending:
            return HTMLResponse(
                "<html><body><h2>Invalid or expired OAuth state.</h2>"
                "<p>Please try connecting again.</p></body></html>",
                status_code=400,
            )

        # Exchange the authorization code for tokens
        try:
            resp = http_requests.post(
                _GOOGLE_TOKEN_URL,
                data={
                    "code": code,
                    "client_id": pending["client_id"],
                    "client_secret": pending["client_secret"],
                    "redirect_uri": pending["redirect_uri"],
                    "grant_type": "authorization_code",
                    "code_verifier": pending["code_verifier"],
                },
                timeout=30,
            )
            resp.raise_for_status()
            token_data = resp.json()
        except Exception as exc:
            log.exception("Google OAuth token exchange failed")
            return HTMLResponse(
                f"<html><body><h2>Token exchange failed</h2><p>{exc}</p></body></html>",
                status_code=500,
            )

        if "refresh_token" not in token_data:
            return HTMLResponse(
                "<html><body><h2>No refresh token received</h2>"
                "<p>You may need to revoke access in your Google account and try again.</p>"
                "</body></html>",
                status_code=400,
            )

        # Save token to ConfigStore
        token_out = {
            "client_id": pending["client_id"],
            "client_secret": pending["client_secret"],
            "refresh_token": token_data["refresh_token"],
            "token_type": token_data.get("token_type", "Bearer"),
        }
        if token_data.get("access_token"):
            token_out["access_token"] = token_data["access_token"]
        if token_data.get("expires_in"):
            token_out["expires_in"] = token_data["expires_in"]
        if token_data.get("scope"):
            token_out["scope"] = token_data["scope"]
        if pending.get("kind") == "contacts" or kind == "contacts":
            await config_store.set("contacts.google_oauth_token", json.dumps(token_out))
            log.info("Google Contacts OAuth token saved to config store")
        else:
            await config_store.set("calendar.google_oauth_token", json.dumps(token_out))
            log.info("Google Calendar OAuth token saved to config store")

        title = (
            "Google Contacts connected!"
            if (pending.get("kind") == "contacts")
            else "Google Calendar connected!"
        )
        return HTMLResponse(
            f"<html><body><h2>{title}</h2>"
            "<p>You can close this window.</p>"
            "<script>window.opener && window.opener.postMessage("
            "{type:'google-oauth-success'},'*');"
            "setTimeout(()=>window.close(),2000)</script></body></html>"
        )

    @app.get("/calendar/google/oauth/status", dependencies=[Depends(auth)])
    async def google_oauth_status() -> dict:
        """Check if a Google OAuth token is stored."""
        token_raw = await config_store.get("calendar.google_oauth_token")
        has_token = bool(token_raw)
        client_id = await config_store.get("calendar.google_oauth_client_id") or ""
        return {"connected": has_token, "client_id": client_id}

    @app.get("/contacts/google/oauth/status", dependencies=[Depends(auth)])
    async def google_contacts_oauth_status() -> dict:
        """Check if a Google Contacts OAuth token is stored."""
        token_raw = await config_store.get("contacts.google_oauth_token")
        has_token = bool(token_raw)
        client_id = await config_store.get("calendar.google_oauth_client_id") or ""
        return {"connected": has_token, "client_id": client_id}

    # ── Permissions API ────────────────────────────────────────────────

    @app.get("/permissions", dependencies=[Depends(auth)])
    async def list_permissions(scope: str = "") -> dict:
        agent = agent_state.agent
        if not agent:
            raise HTTPException(503, "Agent not running")
        return {"scope": scope, "rules": agent.permissions.rules_for_scope(scope)}

    @app.post("/permissions", dependencies=[Depends(auth)])
    async def upsert_permission(request: Request) -> HTMLResponse:
        """Add/update a permission rule. Returns refreshed partial for HTMX."""
        agent = agent_state.agent
        if not agent:
            raise HTTPException(503, "Agent not running")
        body = await request.form()
        pattern = body.get("pattern", "")
        level = body.get("level", "ASK")
        scope = str(body.get("scope", ""))
        if pattern:
            agent.permissions.add_rule(str(pattern), str(level), scope)
        return await _render_permissions(scope)

    @app.post("/permissions/delete", dependencies=[Depends(auth)])
    async def delete_permission(request: Request) -> HTMLResponse:
        """Delete a permission rule. Returns refreshed partial for HTMX."""
        agent = agent_state.agent
        if not agent:
            raise HTTPException(503, "Agent not running")
        content_type = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" in content_type:
            body = await request.form()
        else:
            body = await request.json()
        pattern = str(body.get("pattern", ""))
        scope = str(body.get("scope", ""))
        if pattern:
            agent.permissions.remove_rule(pattern, scope)
        return await _render_permissions(scope)

    # ── Skills API ────────────────────────────────────────────────────────

    @app.get("/skills", dependencies=[Depends(auth)])
    async def list_skills() -> dict:
        store = await _skills_store_from_config(config_store)
        skills = await store.list_skills()
        return {"count": len(skills), "skills": skills}

    @app.get("/skills/{name}", dependencies=[Depends(auth)])
    async def get_skill(name: str) -> dict:
        store = await _skills_store_from_config(config_store)
        skill = await store.get_skill(name)
        if not skill:
            raise HTTPException(404, f"Skill not found: {name}")
        return skill

    # ── Channels API ──────────────────────────────────────────────────────

    @app.get("/channels/wizard", dependencies=[Depends(auth)])
    async def channel_wizard(channel: str = "telegram") -> HTMLResponse:
        key = channel.strip().lower()
        if key == "telegram":
            ctx = await _channel_wizard_context(config_store, "telegram")
            return _render_partial("partials/channel_wizard_telegram.html", **ctx)
        raise HTTPException(400, f"Unknown channel: {channel}")

    @app.post("/channels/telegram", dependencies=[Depends(auth)])
    async def save_channel_telegram(request: Request) -> HTMLResponse:
        body = await request.json()
        bot_token = str(body.get("bot_token", "")).strip()
        user_ids = str(body.get("user_ids", "")).strip()
        enabled = str(body.get("enabled", "true")).lower() == "true"
        topics_enabled = bool(body.get("topics_enabled", False))
        # Group multi-agent rooms (#30). Sub-options default on; whole feature off.
        group_enabled = bool(body.get("group_chat_enabled", False))
        group_addressed = bool(body.get("group_reply_addressed_only", True))
        group_ignore_bots = bool(body.get("group_ignore_bots", True))
        # When the token lives in the vault the editor submits an empty field —
        # treat the existing ${vault:} ref as "present" and leave it untouched.
        token_is_vaulted = _is_vault_ref(await config_store.get("channels.telegram.bot_token"))
        if not bot_token and not token_is_vaulted:
            raise HTTPException(400, "Bot token is required")
        values = {
            "channels.telegram.enabled": str(enabled).lower(),
            "channels.telegram.allowed_user_ids": user_ids,
            "channels.telegram.topics_enabled": str(topics_enabled).lower(),
            "channels.telegram.group_chat.enabled": str(group_enabled).lower(),
            "channels.telegram.group_chat.reply_when_addressed_only": str(group_addressed).lower(),
            "channels.telegram.group_chat.ignore_bots": str(group_ignore_bots).lower(),
        }
        if bot_token:  # only overwrite when a real new token was typed
            values["channels.telegram.bot_token"] = bot_token
        await config_store.set_many(values)
        channel_data = await _channel_list_context(config_store, wacli)
        return _render_partial("partials/channels.html", **channel_data)

    # WhatsApp wacli linking (#97): the tool's enable flag is `tools.whatsapp.enabled`
    # (saved via /config like the other tools); the routes below drive device auth and
    # sync from the Tools tab. WhatsApp is a tool now, not a channel.
    @app.post("/tools/whatsapp/test", dependencies=[Depends(auth)])
    async def test_channel_whatsapp(body: WhatsAppTestIn) -> dict:
        status = await wacli.auth_status()
        available = status.get("available") is True
        result: dict = {"ok": available, "response": status}
        if not available:
            result["error"] = (
                "wacli binary not found. "
                "Run 'make dev-wa' to install it, or set WACLI_BIN to its path."
            )
        return result

    @app.get("/tools/whatsapp/auth/status", dependencies=[Depends(auth)])
    async def whatsapp_auth_status() -> dict:
        status = await wacli.auth_status()
        return {"ok": True, **status}

    @app.post("/tools/whatsapp/auth/start", dependencies=[Depends(auth)])
    async def whatsapp_auth_start() -> dict:
        if not wacli.available():
            return {
                "ok": False,
                "available": False,
                "error": "wacli binary not found. Run 'make dev-wa' to install it, "
                "or set WACLI_BIN to its path.",
            }
        await wacli.start_auth()
        return {"ok": True, "available": True}

    @app.post("/tools/whatsapp/auth/stop", dependencies=[Depends(auth)])
    async def whatsapp_auth_stop() -> dict:
        await wacli.stop_auth()
        return {"ok": True}

    @app.get("/tools/whatsapp/auth/qr", dependencies=[Depends(auth)])
    async def whatsapp_auth_qr() -> dict:
        if not wacli.latest_qr:
            await wacli.fetch_latest_qr()
        if not wacli.latest_qr:
            raise HTTPException(404, "No QR available")
        return {"ok": True, "qr": wacli.latest_qr, "latest_qr_at": wacli.latest_qr_at}

    @app.post("/tools/whatsapp/auth/logout", dependencies=[Depends(auth)])
    async def whatsapp_auth_logout() -> dict:
        await wacli.logout()
        return {"ok": True}

    @app.post("/tools/whatsapp/sync", dependencies=[Depends(auth)])
    async def whatsapp_sync() -> dict:
        res = await wacli.sync_once()
        return {"ok": res.get("success") is True, "response": res}

    @app.delete("/channels/{channel}", dependencies=[Depends(auth)])
    async def delete_channel(channel: str) -> HTMLResponse:
        key = channel.strip().lower()
        values: dict[str, str] = {}
        if key == "telegram":
            values = {
                "channels.telegram.enabled": "false",
                "channels.telegram.bot_token": "",
                "channels.telegram.allowed_user_ids": "",
            }
        else:
            raise HTTPException(400, f"Unknown channel: {channel}")

        # Disabling the channel must not blank a vault-managed token — that would
        # orphan the secret (issue #35). Lifecycle of vaulted secrets lives on the
        # Secrets tab; here we just disable + clear the non-secret fields.
        await config_store.set_many(await _preserve_vault_refs(values))
        channel_data = await _channel_list_context(config_store, wacli)
        return _render_partial("partials/channels.html", **channel_data)

    @app.post("/skills", dependencies=[Depends(auth)])
    async def upsert_skill(body: SkillUpsertIn) -> HTMLResponse:
        store = await _skills_store_from_config(config_store)
        name = body.name.strip()
        content = body.content.strip()
        if not name:
            raise HTTPException(400, "Skill name is required")
        if not content:
            raise HTTPException(400, "Skill content is required")
        await store.upsert_skill(name, content)
        skills = await store.list_skills()
        return _render_partial("partials/skills.html", skills=skills)

    @app.post("/skills/delete", dependencies=[Depends(auth)])
    async def delete_skill(request: Request) -> HTMLResponse:
        content_type = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" in content_type:
            body = await request.form()
        else:
            body = await request.json()
        name = str(body.get("name", "")).strip()
        if not name:
            raise HTTPException(400, "Missing 'name' in request body")
        store = await _skills_store_from_config(config_store)
        deleted = await store.delete_skill(name)
        if not deleted:
            raise HTTPException(404, f"Skill not found: {name}")
        skills = await store.list_skills()
        return _render_partial("partials/skills.html", skills=skills)

    # ── Personae API ───────────────────────────────────────────────────

    async def _personae_partial() -> HTMLResponse:
        store = await _persona_store_from_config(config_store)
        personae = await store.list_personae()
        active = (await config_store.get("agent.active_persona") or "").strip()
        return _render_partial("partials/personae.html", personae=personae, active=active)

    @app.get("/personae", dependencies=[Depends(auth)])
    async def list_personae() -> dict:
        store = await _persona_store_from_config(config_store)
        personae = await store.list_personae()
        active = (await config_store.get("agent.active_persona") or "").strip()
        return {
            "count": len(personae),
            "active": active,
            "personae": [_persona_public(p) for p in personae],
        }

    @app.get("/personae/{name}", dependencies=[Depends(auth)])
    async def get_persona(name: str) -> dict:
        store = await _persona_store_from_config(config_store)
        persona = await store.get(name)
        if not persona:
            raise HTTPException(404, f"Persona not found: {name}")
        return _persona_public(persona)

    @app.post("/personae", dependencies=[Depends(auth)])
    async def upsert_persona(body: PersonaUpsertIn) -> HTMLResponse:
        from core.personae import (
            Persona,
            _as_account_list,
            _as_int_list,
            _as_tool_config,
            parse_markdown,
        )

        name = body.name.strip()
        if not name:
            raise HTTPException(400, "Persona name is required")
        if not _VALID_SLUG.match(name):
            raise HTTPException(400, _SLUG_ERROR)
        # Raw markdown (power-user mode) wins over the structured fields.
        if body.raw.strip():
            persona = parse_markdown(body.raw, name=name)
        else:
            persona = Persona(
                name=name,
                agent_name=body.agent_name.strip(),
                role=body.role.strip(),
                emoji=body.emoji.strip(),
                voice=body.voice.strip(),
                character=body.character,
                skills=[s.strip() for s in body.skills if s.strip()],
                tools=[t.strip() for t in body.tools if t.strip()],
                secrets=[s.strip() for s in body.secrets if s.strip()],
                bot_token=body.bot_token.strip(),
                allowed_user_ids=_as_int_list(body.allowed_user_ids),
                tool_config=_as_tool_config(body.tool_config),
                email_accounts=_as_account_list(body.email_accounts, sender=True),
                calendar_accounts=_as_account_list(body.calendar_accounts),
            )
        # A per-persona GitHub token goes into the infra vault (machine-key,
        # boot-unsealed so it works headless), namespaced per persona (#93). Empty
        # = leave any existing token untouched.
        if body.gh_token.strip():
            if secret_store is None or not secret_store.infra.available:
                raise HTTPException(
                    400,
                    "Set a master key in the Secrets tab before storing per-persona tokens.",
                )
            await secret_store.set_infra_secret(
                gh_token_secret_name(name),
                body.gh_token.strip(),
                f"GitHub token for persona {name}",
            )
        store = await _persona_store_from_config(config_store)
        await store.upsert(persona)
        return await _personae_partial()

    @app.post("/personae/delete", dependencies=[Depends(auth)])
    async def delete_persona(request: Request) -> HTMLResponse:
        content_type = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" in content_type:
            body = await request.form()
        else:
            body = await request.json()
        name = str(body.get("name", "")).strip()
        if not name:
            raise HTTPException(400, "Missing 'name' in request body")
        store = await _persona_store_from_config(config_store)
        if not await store.delete(name):
            raise HTTPException(404, f"Persona not found: {name}")
        # If the deleted persona was active, fall back to the default identity.
        if (await config_store.get("agent.active_persona") or "").strip() == name:
            await _set_active_persona("")
        return await _personae_partial()

    async def _set_active_persona(name: str) -> None:
        """Persist the active persona and hot-reload it into the running agent."""
        await config_store.set("agent.active_persona", name)
        agent = agent_state.agent
        if agent:
            agent.config.agent.active_persona = name

    @app.post("/personae/rename", dependencies=[Depends(auth)])
    async def rename_persona(request: Request) -> HTMLResponse:
        """Change a persona's slug, cascading it to every store that keys off it.

        The slug is a foreign key without a DB constraint: per-chat bindings,
        private memory scope, scheduled jobs, the active-persona selection and the
        ``telegram:<slug>`` bot channel all reference it by value, so each is
        repointed here (#69). A persona with its own bot needs an agent restart for
        the bot to re-register under the new slug.
        """
        content_type = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" in content_type:
            body = await request.form()
        else:
            body = await request.json()
        old = str(body.get("old", "")).strip()
        new = str(body.get("new", "")).strip()
        if not old or not new:
            raise HTTPException(400, "Missing 'old' or 'new' in request body")
        if old == new:
            return await _personae_partial()  # no-op
        if not _VALID_SLUG.match(new):
            raise HTTPException(400, _SLUG_ERROR)
        store = await _persona_store_from_config(config_store)
        if await store.get(old) is None:
            raise HTTPException(404, f"Persona not found: {old}")
        if await store.get(new) is not None:
            raise HTTPException(409, f"A persona named '{new}' already exists")
        # ponytail: best-effort cascade across the separate SQLite DBs — no cross-DB
        # transaction. The references are repointed first and the personae PK row is
        # renamed LAST, so if any step fails the old slug still fully resolves and the
        # whole rename is safe to retry (the ref updates are idempotent no-ops once
        # moved). A missed ref only ever falls back to the default identity, never
        # corrupts data.
        history = await _history_from_config(config_store)
        await history.rename_persona(old, new)
        from core.memory import MemoryStore

        memory_db = await config_store.get("memory.db_path") or "data/memory.db"
        await MemoryStore(db_path=memory_db).rename_scope(old, new)
        await _get_job_store().rename_persona(old, new)
        await store.rename(old, new)
        if (await config_store.get("agent.active_persona") or "").strip() == old:
            await _set_active_persona(new)
        # Re-register live scheduler jobs so a renamed persona's cron/once jobs fire
        # under the new slug + channel immediately, not only after a restart. Bot
        # channels still need a restart (they are created at startup).
        agent = agent_state.agent
        if agent is not None:
            await agent.scheduler.load_jobs()
        return await _personae_partial()

    @app.post("/personae/activate", dependencies=[Depends(auth)])
    async def activate_persona(request: Request) -> HTMLResponse:
        content_type = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" in content_type:
            body = await request.form()
        else:
            body = await request.json()
        name = str(body.get("name", "")).strip()  # "" = default identity
        if name:
            store = await _persona_store_from_config(config_store)
            if not await store.get(name):
                raise HTTPException(404, f"Persona not found: {name}")
        await _set_active_persona(name)
        return await _personae_partial()

    # ── Inspect API (active contexts + last-sent LLM payload) ──────────────

    async def _inspect_partial() -> HTMLResponse:
        history = await _history_from_config(config_store)
        chats = await history.list_chats()
        tz = await config_store.get("agent.timezone") or "UTC"
        now = datetime.now(UTC)
        for c in chats:
            c["last_active_h"] = _humanize_ts(c.get("last_active", ""), now, tz)
        return _render_partial("partials/inspect.html", chats=chats)

    @app.get("/partials/inspect", dependencies=[Depends(auth)])
    async def partial_inspect() -> HTMLResponse:
        """Inspect tab partial — active contexts (most-recently-active first) and
        a master/detail view of the exact payload last sent to the LLM (#99)."""
        return await _inspect_partial()

    @app.get("/inspect/payload", dependencies=[Depends(auth)])
    async def inspect_payload(
        channel: str = "", user_id: str = "", chat_id: str = ""
    ) -> HTMLResponse:
        """Render the last-sent inference payload for one context (#99).

        Reads the in-memory capture keyed by (channel, user_id, chat_id) — the
        exact system/messages/tools/model that generate() last sent. Empty until
        the context has had a turn since the agent started."""
        payload = get_sent_payload((channel, user_id, chat_id))
        meta: dict[str, object] | None = None
        system = ""
        messages: list = []
        tools: list = []
        pretty: str | None = None
        if payload:
            ts = payload.get("captured_at")
            captured = ""
            if ts:
                tz = await config_store.get("agent.timezone") or "UTC"
                try:
                    captured = datetime.fromtimestamp(float(ts), ZoneInfo(tz)).strftime(
                        "%Y-%m-%d %H:%M:%S %Z"
                    )
                except ZoneInfoNotFoundError:
                    captured = datetime.fromtimestamp(float(ts), UTC).isoformat()
            system = payload.get("system") or ""
            # Elide base64 image data so the detail pane never ships megabytes of
            # base64 (a sent photo) — the transcript shows "[image]" and the raw
            # view shows a placeholder, while every other field stays verbatim.
            messages = _elide_image_data(payload.get("messages") or [])
            tools = payload.get("tools") or []
            meta = {
                "provider": payload.get("provider", ""),
                "model": payload.get("model", ""),
                "max_tokens": payload.get("max_tokens", ""),
                "n_messages": len(messages),
                "n_tools": len(tools),
                "captured_at": captured,
            }
            pretty = json.dumps(
                {
                    "model": payload.get("model"),
                    "max_tokens": payload.get("max_tokens"),
                    "system": system,
                    "tools": tools,
                    "messages": messages,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{channel}-{chat_id or 'default'}").strip("_")
        download_name = f"inspect-{safe or 'payload'}.json"
        return _render_partial(
            "partials/inspect_payload.html",
            meta=meta,
            system=system,
            messages=messages,
            tools=tools,
            pretty=pretty,
            download_name=download_name,
        )

    # ── Memory API ─────────────────────────────────────────────────────

    @app.get("/memory/long-term", dependencies=[Depends(auth)])
    async def list_long_term(
        subject: str | None = None,
        category: str | None = None,
    ) -> dict:
        agent = agent_state.agent
        if not agent:
            raise HTTPException(503, "Agent not running")
        import aiosqlite

        await agent.memory._ensure_schema()
        cols = "id, category, subject, content, source, confidence, created_at, updated_at, scope"
        query = f"SELECT {cols} FROM long_term"
        conditions = []
        params: list[str] = []
        if subject:
            conditions.append("subject = ?")
            params.append(subject)
        if category:
            conditions.append("category = ?")
            params.append(category)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY updated_at DESC"

        async with aiosqlite.connect(agent.memory.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = [dict(row) for row in await cursor.fetchall()]
        return {"count": len(rows), "memories": rows}

    @app.get("/memory/short-term", dependencies=[Depends(auth)])
    async def list_short_term() -> dict:
        agent = agent_state.agent
        if not agent:
            raise HTTPException(503, "Agent not running")
        import aiosqlite

        await agent.memory._ensure_schema()
        async with aiosqlite.connect(agent.memory.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id, content, context, expires_at, created_at, scope "
                "FROM short_term WHERE expires_at > datetime('now') "
                "ORDER BY created_at DESC"
            )
            rows = [dict(row) for row in await cursor.fetchall()]
        return {"count": len(rows), "memories": rows}

    @app.post("/memory/delete", dependencies=[Depends(auth)])
    async def delete_memory(request: Request) -> HTMLResponse:
        """Delete a memory entry. Returns refreshed memory partial for HTMX."""
        agent = agent_state.agent
        if not agent:
            raise HTTPException(503, "Agent not running")
        content_type = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" in content_type:
            body = await request.form()
        else:
            body = await request.json()
        tier = str(body.get("tier", ""))
        memory_id_raw = body.get("memory_id")
        if tier not in ("long-term", "short-term"):
            raise HTTPException(400, "Tier must be 'long-term' or 'short-term'")
        if memory_id_raw is None:
            raise HTTPException(400, "Missing 'memory_id' in request body")
        # Coerce to int (form-encoded values arrive as strings)
        try:
            memory_id = int(cast(str, memory_id_raw))
        except TypeError, ValueError:
            raise HTTPException(400, "memory_id must be an integer")

        import aiosqlite

        table = "long_term" if tier == "long-term" else "short_term"
        await agent.memory._ensure_schema()
        async with aiosqlite.connect(agent.memory.db_path) as db:
            cursor = await db.execute(f"DELETE FROM {table} WHERE id = ?", (memory_id,))
            await db.commit()
            if cursor.rowcount == 0:
                raise HTTPException(404, f"Memory {memory_id} not found in {tier}")

        # Return refreshed memory partial (full config + tables)
        return await _render_memory_partial()

    @app.get("/memory/embedding/status", dependencies=[Depends(auth)])
    async def embedding_status() -> dict:
        """Report embedding config + whether a local model is already on disk."""
        from core.embeddings import LOCAL_PROVIDERS

        config = await _resolved_config()
        emb = config.memory.embedding
        is_local = emb.provider in LOCAL_PROVIDERS
        model_ready: bool | None = None
        if is_local:
            cache = Path(emb.cache_dir)
            model_ready = cache.exists() and any(cache.rglob("*.onnx"))
        return {
            "enabled": emb.enabled,
            "provider": emb.provider,
            "model": emb.model,
            "local": is_local,
            "model_ready": model_ready,
            "cache_dir": emb.cache_dir,
        }

    @app.post("/memory/embedding/prefetch", dependencies=[Depends(auth)])
    async def embedding_prefetch() -> dict:
        """Download the local embedding model now (also done at Docker build)."""
        from core.embeddings import LOCAL_PROVIDERS, prefetch_local_model

        config = await _resolved_config()
        emb = config.memory.embedding
        if emb.provider not in LOCAL_PROVIDERS:
            raise HTTPException(400, "Prefetch only applies to the local embedding provider")
        try:
            dim = await asyncio.to_thread(prefetch_local_model, emb.model, emb.cache_dir)
        except Exception as exc:
            log.exception("Embedding model prefetch failed")
            raise HTTPException(500, f"Prefetch failed: {exc}") from exc
        return {"ok": True, "model": emb.model, "dimensions": dim, "cache_dir": emb.cache_dir}

    @app.post("/memory/embedding/test", dependencies=[Depends(auth)])
    async def embedding_test() -> dict:
        """Embed a few probe sentences and report dimension + a sanity cosine."""
        from core.embeddings import (
            LOCAL_PROVIDERS,
            EmbeddingClient,
            LocalEmbeddingClient,
            cosine_similarity,
        )

        config = await _resolved_config()
        emb = config.memory.embedding
        try:
            if emb.provider in LOCAL_PROVIDERS:
                client: object = LocalEmbeddingClient(model=emb.model, cache_dir=emb.cache_dir)
            else:
                cfg = config.agent
                api_key = emb.api_key or getattr(cfg, f"{emb.provider}_api_key", "")
                base_url = emb.base_url or getattr(cfg, f"{emb.provider}_base_url", "") or None
                if not api_key:
                    raise HTTPException(400, f"No API key configured for provider {emb.provider}")
                client = EmbeddingClient(
                    provider=emb.provider,
                    api_key=api_key,
                    model=emb.model,
                    base_url=base_url,
                    dimensions=emb.dimensions,
                )
            probes = ["allergic to shellfish", "cannot eat prawns", "the weather is sunny today"]
            vecs = await client.embed(probes)  # type: ignore[attr-defined]
            if len(vecs) < 3 or not vecs[0]:
                raise HTTPException(500, "Embedding returned no vectors")
            return {
                "ok": True,
                "model": emb.model,
                "dimensions": len(vecs[0]),
                "similar_pair": round(cosine_similarity(vecs[0], vecs[1]), 3),
                "unrelated_pair": round(cosine_similarity(vecs[0], vecs[2]), 3),
            }
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("Embedding test failed")
            raise HTTPException(500, f"Test failed: {exc}") from exc

    @app.post("/memory/consolidate", dependencies=[Depends(auth)])
    async def trigger_consolidation() -> HTMLResponse:
        agent = agent_state.agent
        if not agent:
            raise HTTPException(503, "Agent not running")
        result = await agent.memory.consolidate_and_cleanup(
            llm=agent._memory_llm(agent.config.memory.consolidation_provider),
            model=agent.config.memory.consolidation_model,
        )
        promoted = result.get("promoted_to_long_term", 0)
        expired = result.get("expired_deleted", 0)
        return HTMLResponse(
            f'<span class="alert-success">{promoted} promoted, {expired} expired deleted</span>'
        )

    # ── Logs API ───────────────────────────────────────────────────────

    @app.get("/logs", dependencies=[Depends(auth)])
    async def get_logs(lines: int = 100) -> dict:
        recent = list(_LOG_BUFFER)
        if lines < len(recent):
            recent = recent[-lines:]
        return {"count": len(recent), "lines": recent}

    # ── Agent status (JSON API) ──────────────────────────────────────

    @app.get("/agent/status", dependencies=[Depends(auth)])
    async def agent_status() -> dict:
        agent = agent_state.agent
        if not agent:
            return {
                "running": False,
                "status": agent_state.status,
                "channels": [],
                "scheduler_jobs": 0,
            }
        return {
            "running": True,
            "status": agent_state.status,
            "channels": list(agent.channels.keys()),
            "scheduler_jobs": len(agent.scheduler.scheduler.get_jobs()),
        }

    # NOTE: lifecycle POST endpoints (/agent/start, /agent/stop,
    # /agent/restart) are attached in core/main.py via
    # _attach_lifecycle_routes() so they can access _start_agent /
    # _stop_agent without circular imports.

    # ── Setup wizard ────────────────────────────────────────────────────

    @app.get("/setup/status")
    async def setup_status() -> dict:
        from core.config_store import SETUP_STEPS

        complete = await config_store.is_setup_complete()
        step = await config_store.get_setup_step()
        return {
            "complete": complete,
            "current_step": step,
            "steps": SETUP_STEPS,
        }

    @app.post("/setup/step")
    async def setup_save_step(request: Request) -> HTMLResponse:
        """Save config values for a setup step and advance. Returns the next step partial."""
        from core.config_store import SETUP_STEPS

        content_type = request.headers.get("content-type", "")

        if "application/x-www-form-urlencoded" in content_type:
            form_data = await request.form()
            step = str(form_data.get("step", ""))
            values = {}
            for key, val in form_data.items():
                if key.startswith("values[") and key.endswith("]"):
                    config_key = key[7:-1]
                    values[config_key] = str(val)
        else:
            body = await request.json()
            step = body.get("step", "")
            values = body.get("values", {})

        if values:
            # Same guard as the tabs: a wizard step re-submitted after migration
            # carries empty secret fields — don't let them clobber a ${vault:} ref.
            values = await _preserve_vault_refs(values)
            await config_store.set_many(values)
            log.info("Setup step %r: saved %d values", step, len(values))
            # Initialise the persona vault DEK from the admin password set in the
            # wizard (the plaintext is available here, before it is hashed at boot).
            if secret_store is not None and values.get("admin.api_key"):
                try:
                    await secret_store.ensure_wrapped_dek(values["admin.api_key"])
                except Exception:
                    log.exception("Failed to initialise persona vault DEK from wizard")

        if step not in SETUP_STEPS:
            raise HTTPException(400, f"Unknown step: {step}")
        await config_store.set_setup_step(step)

        ctx = await _wizard_step_context(step, config_store)
        return _render_wizard_step(step, SETUP_STEPS, ctx)

    @app.post("/setup/step/identity")
    async def setup_save_identity(request: Request) -> HTMLResponse:
        """Handle identity step form submission with default character seeding."""
        from core.config_store import SETUP_STEPS

        form_data = await request.form()
        agent_name = str(form_data.get("agent_name", "")).strip() or "Clio"
        owner_name = str(form_data.get("owner_name", "")).strip() or "User"
        timezone = str(form_data.get("timezone", "")).strip() or "UTC"

        values = {
            "agent.name": agent_name,
            "agent.owner_name": owner_name,
            "agent.timezone": timezone,
        }

        # Seed default character — identity + tone (personalia merged in, #98)
        today = datetime.now().strftime("%Y-%m-%d")
        values["agent.character"] = "\n".join(
            [
                "# Character",
                "",
                "## Tone",
                "",
                "- Be concise. Messages should be short and direct — no filler.",
                '- Be warm but not sycophantic. No "Great question!" or "Of course!". Just answer.',
                (
                    f"- When acting on {owner_name}'s behalf (emails, messages), match their "
                    "communication style."
                ),
                (
                    f"- When messaging {owner_name}'s contacts, always identify yourself unless "
                    "told otherwise."
                ),
                "",
                "## Decision-making",
                "",
                "- When unsure about an action, ask. When confident and pre-approved, just do it.",
                "- If multiple contacts match a name, present the options — never guess.",
                "- If a command fails, read the error and try to fix it before reporting back.",
                (
                    "- Prefer structured data (JSON output flags) over free-text parsing when "
                    "available."
                ),
                "",
                "## Language",
                "",
                "- Default to English.",
                "",
                "## Proactive behaviors",
                "",
                "When running scheduled tasks (morning briefing, email checks):",
                "- Be brief and scannable.",
                "- Only flag truly important items.",
                "- Group related information together.",
                "",
                "## Identity",
                "",
                f"- Name: {agent_name}",
                f"- Owner: {owner_name}",
                "- Role: Personal AI assistant",
                "",
                "## Capabilities",
                "",
                "- Conversational interaction via Telegram (text)",
                "- Can execute CLI commands via a whitelisted tool executor",
                "",
                "## Limitations",
                "",
                "- Cannot make phone calls",
                "- Cannot access websites or browse the internet",
                f"- Cannot access files on {owner_name}'s personal devices",
                (
                    f"- Always needs permission before sending messages or emails on {owner_name}'s"
                    " behalf"
                ),
                "",
                "## History",
                "",
                f"- {today}: Initial setup",
            ]
        )

        await config_store.set_many(values)
        log.info("Setup identity: saved %d values", len(values))

        next_step = "persona"
        await config_store.set_setup_step(next_step)
        ctx = await _wizard_step_context(next_step, config_store)
        return _render_wizard_step(next_step, SETUP_STEPS, ctx)

    @app.post("/setup/step/email")
    async def setup_save_email(request: Request) -> HTMLResponse:
        """Handle email step form submission."""
        from core.config_store import SETUP_STEPS

        form_data = await request.form()
        name = str(form_data.get("name", "")).strip()
        email_addr = str(form_data.get("email", "")).strip()
        display_name = str(form_data.get("display_name", "")).strip()
        imap_host = str(form_data.get("imap_host", "")).strip()
        imap_port = str(form_data.get("imap_port", "993")).strip() or "993"
        smtp_host = str(form_data.get("smtp_host", "")).strip()
        smtp_port = str(form_data.get("smtp_port", "465")).strip() or "465"
        login = str(form_data.get("login", "")).strip()
        password = str(form_data.get("password", "")).strip()

        values: dict[str, str] = {}
        if name and email_addr and imap_host and smtp_host:
            values["email.providers"] = json.dumps(
                [
                    {
                        "name": name,
                        "email": email_addr,
                        "display_name": display_name,
                        "imap_host": imap_host,
                        "imap_port": imap_port,
                        "smtp_host": smtp_host,
                        "smtp_port": smtp_port,
                        "login": login,
                        "password": password,
                    }
                ]
            )

        if values:
            await config_store.set_many(values)
            log.info("Setup email: saved %d values", len(values))

        next_step = "calendar"
        await config_store.set_setup_step(next_step)
        ctx = await _wizard_step_context(next_step, config_store)
        return _render_wizard_step(next_step, SETUP_STEPS, ctx)

    @app.post("/setup/step/calendar")
    async def setup_save_calendar(request: Request) -> HTMLResponse:
        """Handle calendar step form submission."""
        from core.config_store import SETUP_STEPS

        form_data = await request.form()
        cal_name = str(form_data.get("cal_name", "")).strip()
        cal_url = str(form_data.get("cal_url", "")).strip()
        cal_user = str(form_data.get("cal_username", "")).strip()
        cal_pass = str(form_data.get("cal_password", "")).strip()

        values = {}
        if cal_name and cal_url:
            values["calendar.providers"] = json.dumps(
                [{"name": cal_name, "url": cal_url, "username": cal_user, "password": cal_pass}]
            )

        if values:
            await config_store.set_many(values)
            log.info("Setup calendar: saved %d values", len(values))

        next_step = "search"
        await config_store.set_setup_step(next_step)
        ctx = await _wizard_step_context(next_step, config_store)
        return _render_wizard_step(next_step, SETUP_STEPS, ctx)

    @app.post("/setup/test-connection")
    async def test_connection(request: Request) -> dict:
        payload = await request.json()
        service = payload.get("service", "")

        if service == "anthropic":
            return await _test_anthropic(payload.get("api_key", ""))
        if service == "openai":
            return await _test_openai(
                payload.get("api_key", ""),
                payload.get("base_url"),
            )
        if service == "google":
            return await _test_openai(
                payload.get("api_key", ""),
                payload.get("base_url"),
                model="gemini-flash-latest",
            )
        if service == "grok":
            return await _test_openai(
                payload.get("api_key", ""),
                payload.get("base_url"),
                model="grok-2-latest",
            )
        if service == "deepseek":
            return await _test_openai(
                payload.get("api_key", ""),
                payload.get("base_url"),
                model="deepseek-chat",
            )
        if service == "telegram":
            return await _test_telegram(payload.get("bot_token", ""))
        if service == "tavily":
            return await _test_tavily(payload.get("api_key", ""))
        return {"ok": False, "error": f"Unknown service: {service}"}

    @app.post("/setup/list-models")
    async def list_models(request: Request) -> dict:
        payload = await request.json()
        service = payload.get("service", "")
        api_key = payload.get("api_key", "")
        base_url = payload.get("base_url")

        if service == "anthropic":
            return await _list_models_anthropic(api_key)
        if service == "openai":
            return await _list_models_openai(api_key, base_url)
        if service == "google":
            return await _list_models_openai(api_key, base_url, strip_prefix="models/")
        if service in ("grok", "deepseek"):
            return await _list_models_openai(api_key, base_url)
        return {"ok": False, "error": f"Unknown service: {service}"}

    @app.post("/setup/thinking-levels")
    async def thinking_levels(request: Request) -> dict:
        """Autodiscover supported reasoning-effort levels for a model.

        Only Anthropic exposes this via the Models API; other providers must be
        configured by typing the effort value (see the docs link in the UI).
        """
        payload = await request.json()
        service = payload.get("service", "")
        api_key = payload.get("api_key", "")
        model = payload.get("model", "")
        if service == "anthropic":
            return await _thinking_levels_anthropic(api_key, model)
        return {
            "ok": False,
            "error": "Autodiscovery is only available for Anthropic — "
            "enter the effort value manually for this provider.",
        }

    # ── Secrets vault (issue #19, #35) ─────────────────────────────────
    # INFRA_VAULT_KEYS (core.secret_store) maps the single-value credential keys
    # to their vault names; migrate_config_to_infra_vault moves the plaintext.
    # Used by the wizard "import" step and the post-setup Secrets tab button
    # (provider blobs — email/calendar/contacts — are not auto-moved here).

    def _duration_fields(duration: str, until: str) -> tuple[str | None, int | None]:
        """Map a UI duration choice to (expires_at, max_uses)."""
        if duration == "once":
            return None, 1
        if duration == "until" and until.strip():
            return f"{until.strip()}T23:59:59+00:00", None
        return None, None

    async def _grant_personae(names: list[str], secret_name: str, grant: bool = True) -> None:
        persona_store = await _persona_store_from_config(config_store)
        for pname in names:
            pname = pname.strip()
            if not pname:
                continue
            p = await persona_store.get(pname)
            if not p:
                continue
            s = set(p.secrets)
            s.add(secret_name) if grant else s.discard(secret_name)
            p.secrets = sorted(s)
            await persona_store.upsert(p)

    async def _secrets_ctx() -> dict:
        from core.secret_store import INFRA_VAULT_KEYS

        persona_store = await _persona_store_from_config(config_store)
        personae = await persona_store.list_personae()
        meta = await secret_store.list_secret_meta()
        for m in meta:
            holders = [p.name for p in personae if m["name"] in p.secrets]
            m["holders"] = holders
            m["personae"] = "all" if m["shared"] else (", ".join(holders) if holders else "—")
        # Credential keys still holding a plaintext value (migratable to the vault).
        migratable = []
        for cfg_key in INFRA_VAULT_KEYS:
            val = await config_store.get(cfg_key)
            if val and not val.startswith("${"):
                migratable.append(cfg_key)
        return {
            "configured": True,
            "unsealed": secret_store.persona_unsealed(),
            "infra_available": secret_store.infra.available,
            "secrets": meta,
            "infra": await secret_store.list_infra_names(),
            "personae": [p.name for p in personae],
            "migratable": migratable,
        }

    def _no_vault_partial() -> HTMLResponse:
        return _render_partial("partials/secrets.html", configured=False)

    @app.get("/partials/secrets", response_class=HTMLResponse, dependencies=[Depends(auth)])
    async def partial_secrets() -> HTMLResponse:
        if secret_store is None:
            return _no_vault_partial()
        return _render_partial("partials/secrets.html", **(await _secrets_ctx()))

    @app.post("/admin/secrets/migrate", response_class=HTMLResponse, dependencies=[Depends(auth)])
    async def migrate_secrets() -> HTMLResponse:
        """Move any still-plaintext credentials onto the infra vault (issue #35).

        The non-destructive, post-setup counterpart of the wizard's import step:
        existing installs can collapse their scattered LLM/Telegram/Tavily/gh
        credentials into the vault without re-running setup. Idempotent.
        """
        if secret_store is None:
            return _no_vault_partial()
        from core.secret_store import migrate_config_to_infra_vault

        ctx = await _secrets_ctx()
        if not secret_store.infra.available:
            ctx["error"] = "No machine key configured (set MPA_MASTER_KEY or generate one)."
            return _render_partial("partials/secrets.html", **ctx)
        migrated = await migrate_config_to_infra_vault(config_store, secret_store)
        ctx = await _secrets_ctx()
        ctx["flash"] = (
            f"Migrated {len(migrated)} credential(s) into the vault."
            if migrated
            else "Nothing to migrate — all credentials already reference the vault."
        )
        return _render_partial("partials/secrets.html", **ctx)

    @app.post("/admin/secrets", response_class=HTMLResponse, dependencies=[Depends(auth)])
    async def add_secret(request: Request) -> HTMLResponse:
        if secret_store is None:
            return _no_vault_partial()
        form = await request.form()
        name = str(form.get("name", "")).strip()
        value = str(form.get("value", ""))
        from core.secret_store import valid_name

        if not valid_name(name):
            ctx = await _secrets_ctx()
            ctx["error"] = "Invalid name — use letters, digits, _ - : only."
            return _render_partial("partials/secrets.html", **ctx)
        if not secret_store.persona_unsealed():
            ctx = await _secrets_ctx()
            ctx["error"] = "Vault is locked — reload the page to unlock it with your password."
            return _render_partial("partials/secrets.html", **ctx)
        scope = str(form.get("scope", "this"))
        personae = str(form.get("personae", "")).replace("\n", ",").split(",")
        expires_at, max_uses = _duration_fields(
            str(form.get("duration", "forever")), str(form.get("until", ""))
        )
        await secret_store.set_secret(
            name,
            value,
            shared=(scope == "all"),
            description=str(form.get("description", "")),
            expires_at=expires_at,
            max_uses=max_uses,
        )
        if scope == "personae":
            await _grant_personae(personae, name)
        return _render_partial("partials/secrets.html", **(await _secrets_ctx()))

    @app.post("/admin/secrets/delete", response_class=HTMLResponse, dependencies=[Depends(auth)])
    async def delete_secret(request: Request) -> HTMLResponse:
        if secret_store is None:
            return _no_vault_partial()
        form = await request.form()
        name = str(form.get("name", "")).strip()
        if await secret_store.delete_secret(name):
            log.info("Persona secret %r deleted via admin", name)
        return _render_partial("partials/secrets.html", **(await _secrets_ctx()))

    @app.post("/admin/secrets/grant", response_class=HTMLResponse, dependencies=[Depends(auth)])
    async def grant_secret(request: Request) -> HTMLResponse:
        if secret_store is None:
            return _no_vault_partial()
        form = await request.form()
        name = str(form.get("name", "")).strip()
        persona = str(form.get("persona", "")).strip()
        grant = str(form.get("grant", "true")) == "true"
        await _grant_personae([persona], name, grant=grant)
        return _render_partial("partials/secrets.html", **(await _secrets_ctx()))

    @app.post("/admin/secrets/infra", response_class=HTMLResponse, dependencies=[Depends(auth)])
    async def add_infra_secret(request: Request) -> HTMLResponse:
        if secret_store is None:
            return _no_vault_partial()
        form = await request.form()
        name = str(form.get("name", "")).strip()
        from core.secret_store import valid_name

        if not secret_store.infra.available:
            ctx = await _secrets_ctx()
            ctx["error"] = "No machine key configured (set MPA_MASTER_KEY or generate one)."
            return _render_partial("partials/secrets.html", **ctx)
        if not valid_name(name):
            ctx = await _secrets_ctx()
            ctx["error"] = "Invalid infra secret name."
            return _render_partial("partials/secrets.html", **ctx)
        await secret_store.set_infra_secret(
            name, str(form.get("value", "")), str(form.get("description", ""))
        )
        return _render_partial("partials/secrets.html", **(await _secrets_ctx()))

    @app.post(
        "/admin/secrets/infra/delete", response_class=HTMLResponse, dependencies=[Depends(auth)]
    )
    async def delete_infra_secret(request: Request) -> HTMLResponse:
        if secret_store is None:
            return _no_vault_partial()
        form = await request.form()
        name = str(form.get("name", "")).strip()
        if await secret_store.delete_infra_secret(name):
            log.info("Infra secret %r deleted via admin", name)
        return _render_partial("partials/secrets.html", **(await _secrets_ctx()))

    # -- Secure-link credential fill flow --
    @app.get("/vault/request/{token}", dependencies=[Depends(auth)])
    async def vault_request_detail(token: str) -> dict:
        if secret_store is None:
            raise HTTPException(404, "Vault not configured")
        req = await secret_store.get_request(token)
        if not req:
            raise HTTPException(404, "Request not found or expired")
        return {
            "name": req["name"],
            "reason": req["reason"],
            "persona": req["persona"],
            "suggested_scope": req["suggested_scope"],
        }

    @app.get("/vault/fill/{token}", response_class=HTMLResponse)
    async def vault_fill_page(token: str) -> HTMLResponse:
        # The page shell is public; details + submit require admin auth (the
        # page fetches them with the stored bearer, like the rest of the UI).
        return _render("vault_fill.html", token=token)

    @app.post("/vault/fill/{token}", dependencies=[Depends(auth)])
    async def vault_fill_submit(token: str, request: Request) -> dict:
        if secret_store is None:
            raise HTTPException(404, "Vault not configured")
        req = await secret_store.get_request(token)
        if not req:
            raise HTTPException(404, "Request not found or expired")
        if not secret_store.persona_unsealed():
            return {"ok": False, "error": "Vault is locked."}
        form = await request.form()
        name = req["name"]
        # Structured if any login field beyond a bare value/password is present.
        fields = {
            k: str(form.get(k, ""))
            for k in ("username", "password", "url", "totp")
            if str(form.get(k, "")).strip()
        }
        bare = str(form.get("value", ""))
        value: str | dict
        if fields and not (len(fields) == 1 and "password" in fields and not bare):
            value = fields
        else:
            value = bare or fields.get("password", "")
        scope = str(form.get("scope", "this"))
        personae = str(form.get("personae", "")).replace("\n", ",").split(",")
        expires_at, max_uses = _duration_fields(
            str(form.get("duration", "forever")), str(form.get("until", ""))
        )
        # "This persona only" with no requesting persona (base agent) is incoherent —
        # it would orphan the secret (no owner, no grant, not shared → never resolvable).
        # Treat it as global so the agent that asked can actually use it.
        shared = scope == "all" or (scope == "this" and not req["persona"])
        await secret_store.set_secret(
            name,
            value,
            shared=shared,
            owner=f"persona:{req['persona']}" if req["persona"] else "",
            description=req["reason"][:200],
            expires_at=expires_at,
            max_uses=max_uses,
        )
        if scope == "personae":
            await _grant_personae(personae, name)
        elif scope == "this" and req["persona"]:
            await _grant_personae([req["persona"]], name)
        await secret_store.resolve_request(token)
        return {"ok": True, "name": name}

    # -- Bitwarden import --
    @app.post(
        "/admin/secrets/import/parse", response_class=HTMLResponse, dependencies=[Depends(auth)]
    )
    async def import_parse(request: Request) -> HTMLResponse:
        if secret_store is None:
            return _no_vault_partial()
        from core.secret_store import parse_bitwarden_export

        form = await request.form()
        upload = form.get("file")
        try:
            raw = await upload.read() if hasattr(upload, "read") else str(upload).encode()
            data = json.loads(raw)
            items = parse_bitwarden_export(data)
        except Exception as exc:
            ctx = await _secrets_ctx()
            ctx["error"] = f"Could not parse export: {exc}"
            return _render_partial("partials/secrets.html", **ctx)
        persona_store = await _persona_store_from_config(config_store)
        return _render_partial(
            "partials/secrets_import.html",
            items=items,
            personae=[p.name for p in await persona_store.list_personae()],
        )

    @app.post(
        "/admin/secrets/import/commit", response_class=HTMLResponse, dependencies=[Depends(auth)]
    )
    async def import_commit(request: Request) -> HTMLResponse:
        if secret_store is None:
            return _no_vault_partial()
        if not secret_store.persona_unsealed():
            ctx = await _secrets_ctx()
            ctx["error"] = "Vault is locked."
            return _render_partial("partials/secrets.html", **ctx)
        form = await request.form()
        selected = form.getlist("selected")
        scope = str(form.get("scope", "this"))
        personae = str(form.get("personae", "")).replace("\n", ",").split(",")
        count = 0
        for name in selected:
            value = {
                "username": str(form.get(f"username__{name}", "")),
                "password": str(form.get(f"password__{name}", "")),
                "url": str(form.get(f"url__{name}", "")),
                "totp": str(form.get(f"totp__{name}", "")),
            }
            value = {k: v for k, v in value.items() if v}
            if not value:
                continue
            store_val = value if len(value) > 1 else next(iter(value.values()))
            await secret_store.set_secret(name, store_val, shared=(scope == "all"))
            if scope == "personae":
                await _grant_personae(personae, name)
            count += 1
        ctx = await _secrets_ctx()
        ctx["flash"] = f"Imported {count} secret(s)."
        return _render_partial("partials/secrets.html", **ctx)

    # -- Wizard: master key + import-from-.env --
    @app.post("/setup/step/secrets")
    async def setup_save_secrets(request: Request) -> HTMLResponse:
        from core.config_store import SETUP_STEPS
        from core.vault import generate_and_save_machine_key

        # Wizard-only endpoint: refuse once setup is complete so it can't be used
        # post-setup to (re)generate a machine key or migrate config.
        if await config_store.is_setup_complete():
            raise HTTPException(403, "Setup already complete")

        form = await request.form()
        action = str(form.get("action", ""))
        if action == "generate" and secret_store is not None:
            key = generate_and_save_machine_key()
            secret_store.infra = type(secret_store.infra)(key)  # reload infra vault
        elif action == "import" and secret_store is not None:
            from core.secret_store import migrate_config_to_infra_vault

            await migrate_config_to_infra_vault(config_store, secret_store)
        ctx = await _wizard_step_context("secrets", config_store)
        return _render_wizard_step("secrets", SETUP_STEPS, ctx)

    return app, auth


async def _skills_store_from_config(config_store: ConfigStore) -> SkillsStore:
    from core.skills import SkillsStore

    skills_db_path = await config_store.get("agent.skills_db_path") or "data/skills.db"
    skills_dir = await config_store.get("agent.skills_dir") or "skills/"
    return SkillsStore(db_path=skills_db_path, seed_dir=skills_dir)


async def _persona_store_from_config(config_store: ConfigStore):
    from core.personae import PersonaStore

    db_path = await config_store.get("agent.personae_db_path") or "data/personae.db"
    seed_dir = await config_store.get("agent.personae_dir") or "personae/"
    return PersonaStore(db_path=db_path, seed_dir=seed_dir)


async def _history_from_config(config_store: ConfigStore):
    from core.history import ConversationHistory

    db_path = await config_store.get("history.db_path") or "data/history.db"
    return ConversationHistory(db_path=db_path)


# Config keys the running agent only reads at startup, so a change to them via
# PATCH /config takes effect only after an agent restart. Everything else is
# hot-applied in patch_config (agent.config swap + llm/memory/search rebuild).
# Channels are saved through their own routes (always restart-bound) and handled
# client-side, so they are not listed here.
def _config_requires_restart(values: dict) -> bool:
    """True if any saved key is consumed only at agent startup (voice pipeline,
    history window) and so needs a restart to take effect."""
    return any(key == "history.max_turns" or key.startswith("voice.") for key in values)


# Function-tools that a persona may scope. ``load_skill`` is intentionally
# excluded — it is always available (the core mechanic personae use to read
# their allowlisted skills); so are ``search_skills``/``list_skills`` (its
# on-demand discovery counterparts — #50), the vault tools, and ``recall_memory``
# (memory is injected for every persona, scope-filtered, so its on-demand
# counterpart is always available too). Kept here (not imported from core.agent)
# to avoid pulling the agent's heavy import graph into the admin app.
GATEABLE_TOOLS = [
    "run_command",
    "send_email",
    "reply_email",
    "send_message",
    "set_reaction",
    "create_calendar_event",
    "web_search",
    "manage_jobs",
    "spawn_subagent",
    "generate_image",
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "grep",
    "run_command_in_dir",
]

_WORKSPACE_TOOLS = (
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "grep",
    "run_command_in_dir",
)


def gateable_tools_for(
    subagents_enabled: bool = True,
    imagegen_enabled: bool = True,
    workspace_enabled: bool = True,
) -> list[str]:
    """GATEABLE_TOOLS minus tools whose feature is globally disabled."""
    out = list(GATEABLE_TOOLS)
    if not subagents_enabled:
        out = [t for t in out if t != "spawn_subagent"]
    if not imagegen_enabled:
        out = [t for t in out if t != "generate_image"]
    if not workspace_enabled:
        out = [t for t in out if t not in _WORKSPACE_TOOLS]
    return out


# ---------------------------------------------------------------------------
# Connection test helpers
# ---------------------------------------------------------------------------


async def _test_anthropic(api_key: str) -> dict:
    if not api_key:
        return {"ok": False, "error": "API key is empty"}
    try:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=16,
            messages=[{"role": "user", "content": "Say 'ok'"}],
        )
        text = ""
        for block in response.content:
            text = getattr(block, "text", "")
            if text:
                break
        return {"ok": True, "response": text}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _test_openai(api_key: str, base_url: str | None, model: str = "gpt-4o-mini") -> dict:
    if not api_key:
        return {"ok": False, "error": "API key is empty"}
    try:
        import importlib

        module = importlib.import_module("openai")
        client_class = cast(Any, getattr(module, "AsyncOpenAI"))
        client = cast(Any, client_class)(api_key=api_key, base_url=base_url or None)
        response = await client.chat.completions.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": "Say 'ok'"}],
        )
        text = response.choices[0].message.content or ""
        return {"ok": True, "response": text}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _thinking_levels_anthropic(api_key: str, model: str) -> dict:
    if not api_key:
        return {"ok": False, "error": "API key is empty"}
    if not model:
        return {"ok": False, "error": "Enter a model id first"}
    try:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=api_key)
        m = await client.models.retrieve(model)
        caps = getattr(m, "capabilities", None)
        if caps is not None and not isinstance(caps, dict):
            caps = getattr(caps, "model_dump", lambda: {})() or {}
        caps = caps or {}
        effort = caps.get("effort") or {}
        levels = [
            lvl
            for lvl in ("low", "medium", "high", "xhigh", "max")
            if isinstance(effort.get(lvl), dict) and effort[lvl].get("supported")
        ]
        thinking = caps.get("thinking") or {}
        supported = bool(thinking.get("supported")) or bool(levels)
        return {"ok": True, "supported": supported, "levels": levels}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _list_models_anthropic(api_key: str) -> dict:
    if not api_key:
        return {"ok": False, "error": "API key is empty"}
    try:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=api_key)
        ids: list[str] = []
        async for model in client.models.list(limit=1000):
            mid = getattr(model, "id", "")
            if mid:
                ids.append(mid)
        return {"ok": True, "models": ids}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _list_models_openai(
    api_key: str, base_url: str | None, strip_prefix: str | None = None
) -> dict:
    if not api_key:
        return {"ok": False, "error": "API key is empty"}
    try:
        import importlib

        module = importlib.import_module("openai")
        client_class = cast(Any, getattr(module, "AsyncOpenAI"))
        client = cast(Any, client_class)(api_key=api_key, base_url=base_url or None)
        resp = await client.models.list()
        ids = []
        for model in resp.data:
            mid = getattr(model, "id", "")
            if strip_prefix and mid.startswith(strip_prefix):
                mid = mid[len(strip_prefix) :]
            if mid:
                ids.append(mid)
        ids.sort()
        return {"ok": True, "models": ids}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _test_telegram(bot_token: str) -> dict:
    if not bot_token:
        return {"ok": False, "error": "Bot token is empty"}
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"https://api.telegram.org/bot{bot_token}/getMe")
            data = resp.json()
            if data.get("ok"):
                bot_info = data["result"]
                return {
                    "ok": True,
                    "bot_name": bot_info.get("first_name", ""),
                    "bot_username": bot_info.get("username", ""),
                }
            return {"ok": False, "error": data.get("description", "Unknown error")}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _test_tavily(api_key: str) -> dict:
    if not api_key:
        return {"ok": False, "error": "API key is empty"}
    try:
        import asyncio

        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        result = await asyncio.to_thread(client.search, query="test", max_results=1)
        return {"ok": True, "results": len(result.get("results", []))}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
