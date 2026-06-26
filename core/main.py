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

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
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
    from channels.whatsapp import WhatsAppChannel
    from core.agent import AgentCore
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

    # -- Voice pipeline --
    voice: VoicePipeline | None = None
    if config.voice.tts_enabled:
        log.info(
            "Initializing voice pipeline (model=%s, voice=%s)…",
            config.voice.stt_model,
            config.voice.tts_voice,
        )
        voice = VoicePipeline(
            stt_model=config.voice.stt_model,
            tts_voice=config.voice.tts_voice,
            tts_enabled=config.voice.tts_enabled,
        )
        agent.voice = voice

    # -- Telegram --
    if config.channels.telegram.enabled and config.channels.telegram.bot_token:
        tg = TelegramChannel(config.channels.telegram, agent, voice=voice)
        agent.channels["telegram"] = tg
        log.info("Starting Telegram bot…")

        await tg.app.initialize()
        await tg.app.start()
        updater = tg.app.updater
        if updater is not None:
            await updater.start_polling()

    # -- WhatsApp --
    if config.channels.whatsapp.enabled:
        from core.wacli import WacliManager

        wacli = WacliManager()
        wa = WhatsAppChannel(config.channels.whatsapp, agent, wacli=wacli)
        agent.channels["whatsapp"] = wa
        log.info("WhatsApp channel enabled (wacli)")

    # -- Scheduler --
    await agent.scheduler.load_jobs()
    agent.scheduler.start()
    log.info("Scheduler started with %d jobs", len(agent.scheduler.scheduler.get_jobs()))

    return agent


async def _stop_agent(agent) -> None:
    """Gracefully shut down the agent."""
    agent.scheduler.shutdown()

    from core.scheduler import set_agent_context

    set_agent_context(None)

    if "telegram" in agent.channels:
        tg = agent.channels["telegram"]
        await tg.app.updater.stop()
        await tg.app.stop()
        await tg.app.shutdown()


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
    # Initialise the persona vault's wrapped DEK when an admin password is set via
    # the environment (the only point at boot where plaintext is available). When
    # the password is set through the wizard/UI instead, the admin routes mint it.
    seed_pw = environ.get("ADMIN_PASSWORD") or environ.get("ADMIN_API_KEY")
    if seed_pw:
        await _secret_store.ensure_wrapped_dek(seed_pw)
    await materialize_himalaya_config(_config_store)

    # Background sweep of expired web artifacts (runs regardless of setup state).
    from core.artifacts import cleanup_loop

    application.state.artifacts_task = asyncio.create_task(cleanup_loop(_config_store))

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
    artifacts_task = getattr(application.state, "artifacts_task", None)
    if artifacts_task:
        artifacts_task.cancel()
        with suppress(asyncio.CancelledError):
            await artifacts_task
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
