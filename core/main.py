"""Entrypoint — boots the agent and runs Telegram + admin API concurrently.

Supports two boot modes:
  1. **Setup mode** — config store is empty or setup not complete.
     Only the admin API runs (serves the setup wizard).  No agent, no
     Telegram, no scheduler.
  2. **Normal mode** — setup complete.  Full agent with all channels,
     scheduler, and admin API.

Usage:
  Production:  ``python -m core.main``
  Development: ``uvicorn core.main:app --reload``
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from os import environ

import uvicorn
from fastapi import Depends, Request
from fastapi.responses import HTMLResponse

from api.admin import AgentState, create_admin_app, install_log_buffer
from core.config_store import ConfigStore
from core.email_config import materialize_himalaya_config
from core.secret_store import SecretStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    force=True,
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Install the in-memory log buffer handler before anything else
install_log_buffer()


async def _start_agent(config_store: ConfigStore):
    """Build and start the full agent (channels, scheduler, voice)."""
    from channels.telegram import TelegramChannel
    from core.agent import AgentCore
    from core.config import TelegramConfig
    from voice.pipeline import VoicePipeline

    # Decrypt infra secrets into memory so ${vault:NAME} resolves at config load
    # (with .env fallback). Then build the config and hand the shared secret store
    # to the agent so the executor can resolve {{secret:NAME}} at runtime.
    await _secret_store.load_infra_cache()
    config = await config_store.export_to_config(vault_resolve=_secret_store.infra_resolve)

    agent = AgentCore(config, secret_store=_secret_store)

    # Ensure scheduler jobs can resolve the current agent instance
    from core.scheduler import set_agent_context

    set_agent_context(agent)

    # -- Migrate jobs from config store to jobs.db (one-time) --
    await agent.job_store.migrate_from_config_store(config_store)

    # -- Seed jobs from config.yml if jobs.db is empty --
    if config.scheduler.jobs:
        seed_data = [j.model_dump() for j in config.scheduler.jobs]
        await agent.job_store.seed_from_config(seed_data)

    # -- One-time #110 account-binding migration --
    # Before #110 any agent could use any configured email/calendar account. Now
    # an empty binding means no access, so on first start we grant every agent
    # (that has none yet) full access to all existing accounts, preserving prior
    # behaviour. Agents created afterwards start empty (safe default). Runs once.
    if not await config_store.get("accounts.agent_binding_migrated"):
        from core.agents import bind_existing_accounts

        def _account_names(raw: str | None) -> list[str]:
            try:
                items = json.loads(raw) if raw else []
            except ValueError, TypeError:
                return []
            return [
                str(i.get("name", "")).strip()
                for i in items
                if isinstance(i, dict) and str(i.get("name", "")).strip()
            ]

        n = await bind_existing_accounts(
            agent.agents,
            _account_names(await config_store.get("email.providers")),
            _account_names(await config_store.get("calendar.providers")),
        )
        await config_store.set("accounts.agent_binding_migrated", "true")
        if n:
            log.info("Bound existing email/calendar accounts to %d agent(s) (#110)", n)

    # -- Voice pipeline --
    voice: VoicePipeline | None = None
    if config.voice.tts_enabled:
        log.info(
            "Initializing voice pipeline (model=%s, voice=%s, backend=%s)…",
            config.voice.stt_model,
            config.voice.tts_voice,
            config.voice.backend,
        )
        voice = VoicePipeline(
            stt_model=config.voice.stt_model,
            tts_voice=config.voice.tts_voice,
            tts_enabled=config.voice.tts_enabled,
            backend=config.voice.backend,
            kokoro_model_path=config.voice.kokoro.model_path,
            kokoro_voices_path=config.voice.kokoro.voices_path,
            kokoro_default_voice=config.voice.kokoro.default_voice,
        )
        agent.voice = voice

    # -- Telegram: the default bot plus one bot per agent that carries a token (#29).
    # A single bad/revoked token must never abort the others, WhatsApp, or the
    # scheduler — each bot is brought up independently and failures are isolated.
    async def _start_tg(conf, name: str, channel_name: str = "telegram") -> None:
        try:
            tg = TelegramChannel(conf, agent, voice=voice, channel_name=channel_name)
            await tg.app.initialize()
            await tg.app.start()
            if tg.app.updater is not None:
                await tg.app.updater.start_polling()
            agent.channels[name] = tg  # registered only once it is actually polling
            log.info("Telegram bot started (%s)", name)
        except Exception:
            log.exception("Failed to start Telegram bot %s — skipping", name)

    try:
        tg_global = config.channels.telegram
        seen_tokens: set[str] = set()
        if tg_global.enabled and tg_global.bot_token:
            seen_tokens.add(tg_global.bot_token)
            await _start_tg(tg_global, "telegram")

        for ag in await agent.agents.list_agents():
            token = (ag.bot_token or "").strip()
            if not token:
                continue  # no own bot — reachable only via the default bot
            if token in seen_tokens:
                log.warning(
                    "Agent %s shares a bot token with another bot — skipping its bot "
                    "(one token can only be polled once)",
                    ag.name,
                )
                continue
            seen_tokens.add(token)
            pconf = TelegramConfig(
                enabled=True,
                bot_token=token,
                allowed_user_ids=ag.allowed_user_ids or tg_global.allowed_user_ids,
                topics_enabled=tg_global.topics_enabled,
                group_chat=tg_global.group_chat,  # inherit group-room behaviour (#30)
            )
            await _start_tg(pconf, f"telegram:{ag.name}", f"telegram:{ag.name}")

        # WhatsApp is a tool now (#97), not a channel: the agent reads/sends via
        # the `wacli` CLI through run_command. Linking/sync live on the admin app's
        # WacliManager (api/admin.py); no inbound channel to start here.

        # -- Scheduler --
        await agent.scheduler.load_jobs()
        agent.scheduler.start()
        log.info("Scheduler started with %d jobs", len(agent.scheduler.scheduler.get_jobs()))
    except Exception:
        # Bring-up failed after some bots were already polling — stop them so we
        # don't leak orphaned pollers (which would 409 on the next start).
        await _stop_telegram_bots(agent)
        raise

    return agent


async def _stop_telegram_bots(agent) -> None:
    """Stop and deregister the default bot and every per-agent bot (#29).

    Each bot is torn down independently: one that fails to stop must not strand
    the rest still polling (which would 409 on the next start).
    """
    for name, ch in list(agent.channels.items()):
        if name != "telegram" and not name.startswith("telegram:"):
            continue
        try:
            if ch.app.updater is not None:
                await ch.app.updater.stop()
            await ch.app.stop()
            await ch.app.shutdown()
        except Exception:
            log.exception("Error stopping Telegram bot %s", name)
        agent.channels.pop(name, None)


async def _stop_agent(agent) -> None:
    """Gracefully shut down the agent."""
    agent.scheduler.shutdown()

    from core.scheduler import set_agent_context

    set_agent_context(None)

    await _stop_telegram_bots(agent)


# ---------------------------------------------------------------------------
# Shared state — populated once during lifespan, used by lifecycle routes.
# ---------------------------------------------------------------------------

_config_store = ConfigStore()
_secret_store = SecretStore()
_agent_state = AgentState()


@asynccontextmanager
async def _lifespan(application):  # noqa: ANN001
    """FastAPI lifespan: seed config, start agent, yield, then tear down."""
    # -- startup --
    # Load .env so MPA_MASTER_KEY / ADMIN_PASSWORD etc. are in the environment even
    # for existing installs where the config store is already seeded (Docker injects
    # env_file directly; this covers `make run` / bare-process deployments).
    from dotenv import load_dotenv

    load_dotenv()
    await _config_store.seed_if_empty()
    await _config_store.ensure_admin_password()
    # Initialise the agent vault's wrapped DEK when an admin password is set via
    # the environment (the only point at boot where plaintext is available). When
    # the password is set through the wizard/UI instead, the admin routes mint it.
    seed_pw = environ.get("ADMIN_PASSWORD") or environ.get("ADMIN_API_KEY")
    if seed_pw:
        await _secret_store.ensure_wrapped_dek(seed_pw)
    # Load the infra-vault cache and attach its resolver to the config store so the
    # Himalaya materialisation below can expand ${vault:NAME} email passwords (#110).
    # Safe with no master key: the cache is empty and infra_resolve falls back to env.
    await _secret_store.load_infra_cache()
    _config_store.vault_resolve = _secret_store.infra_resolve
    await materialize_himalaya_config(_config_store)

    setup_complete = await _config_store.is_setup_complete()

    if setup_complete:
        log.info("Setup complete — starting agent")
        _agent_state.status = "STARTING"
        try:
            _agent_state.agent = await _start_agent(_config_store)
            _agent_state.status = "RUNNING"
        except Exception:
            log.exception("Failed to start agent — falling back to setup mode")
            _agent_state.status = "STOPPED"
    else:
        log.info("Setup not complete — running in setup-only mode (admin API + wizard)")
        _agent_state.status = "STOPPED"

    yield

    # -- shutdown --
    log.info("Shutting down…")
    if _agent_state.agent:
        _agent_state.status = "STOPPING"
        await _stop_agent(_agent_state.agent)
        _agent_state.agent = None
        _agent_state.status = "STOPPED"


# ---------------------------------------------------------------------------
# Build the FastAPI app at module level so ``uvicorn core.main:app`` works.
# ---------------------------------------------------------------------------

app, _auth = create_admin_app(
    _agent_state, _config_store, lifespan=_lifespan, secret_store=_secret_store
)


def _attach_lifecycle_routes(
    application, config_store: ConfigStore, agent_state: AgentState, auth
) -> None:
    """Add /agent/start, /agent/stop, and /agent/restart endpoints.

    These share the same ``AgentState`` object used by ``create_admin_app``
    so all endpoints see agent changes immediately.

    Content-negotiation: returns HTML snippets for HTMX requests (the
    dashboard buttons) and JSON for programmatic callers (the setup
    wizard's fetch() call).
    """

    def _is_htmx(request: Request) -> bool:
        return request.headers.get("HX-Request") == "true"

    @application.post("/agent/start", dependencies=[Depends(auth)])
    async def start_agent(request: Request):
        if agent_state.agent is not None:
            result = {
                "status": "already_running",
                "channels": list(agent_state.agent.channels.keys()),
            }
        else:
            try:
                agent_state.status = "STARTING"
                agent_state.agent = await _start_agent(config_store)
                log.info("Agent started via API")
                agent_state.status = "RUNNING"
                result = {
                    "status": "started",
                    "channels": list(agent_state.agent.channels.keys()),
                }
            except Exception as exc:
                log.exception("Failed to start agent via API")
                agent_state.status = "STOPPED"
                result = {"status": "error", "error": str(exc)}

        if _is_htmx(request):
            css = (
                "alert-success"
                if result["status"] in ("started", "already_running")
                else "alert-error"
            )
            label = result["status"].replace("_", " ").title()
            resp = HTMLResponse(f'<span class="{css}">{label}</span>')
            resp.headers["HX-Trigger"] = "refresh-status"
            return resp
        return result

    @application.post("/agent/stop", dependencies=[Depends(auth)])
    async def stop_agent(request: Request):
        if agent_state.agent is None:
            result = {"status": "not_running"}
        else:
            try:
                agent_state.status = "STOPPING"
                await _stop_agent(agent_state.agent)
                agent_state.agent = None
                log.info("Agent stopped via API")
                agent_state.status = "STOPPED"
                result = {"status": "stopped"}
            except Exception as exc:
                log.exception("Failed to stop agent via API")
                agent_state.status = "RUNNING"
                result = {"status": "error", "error": str(exc)}

        if _is_htmx(request):
            css = "alert-success" if result["status"] == "stopped" else "alert-error"
            label = result["status"].replace("_", " ").title()
            resp = HTMLResponse(f'<span class="{css}">{label}</span>')
            resp.headers["HX-Trigger"] = "refresh-status"
            return resp
        return result

    @application.post("/agent/restart", dependencies=[Depends(auth)])
    async def restart_agent(request: Request):
        # Stop
        if agent_state.agent is not None:
            try:
                agent_state.status = "STOPPING"
                await _stop_agent(agent_state.agent)
            except Exception:
                log.exception("Error during agent stop (restart)")
            agent_state.agent = None
            agent_state.status = "STOPPED"

        # Start
        try:
            agent_state.status = "RESTARTING"
            agent_state.agent = await _start_agent(config_store)
            log.info("Agent restarted via API")
            agent_state.status = "RUNNING"
            result = {
                "status": "restarted",
                "channels": list(agent_state.agent.channels.keys()),
            }
        except Exception as exc:
            log.exception("Failed to restart agent via API")
            agent_state.status = "STOPPED"
            result = {"status": "error", "error": str(exc)}

        if _is_htmx(request):
            css = "alert-success" if result["status"] == "restarted" else "alert-error"
            label = result["status"].replace("_", " ").title()
            resp = HTMLResponse(f'<span class="{css}">{label}</span>')
            resp.headers["HX-Trigger"] = "refresh-status"
            return resp
        return result


_attach_lifecycle_routes(app, _config_store, _agent_state, _auth)


# ---------------------------------------------------------------------------
# Production entrypoint: ``python -m core.main``
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=environ.get("HOST", "0.0.0.0"),
        port=int(environ.get("PORT", "8000")),
        log_level="info",
        log_config=None,
    )
