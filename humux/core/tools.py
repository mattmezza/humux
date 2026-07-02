"""Optional external CLI tools the agent can use (e.g. the GitHub `gh` CLI).

Tools are configured under ``config.tools.*``.  When a tool is *enabled*, two
things happen:

1. Its authentication is wired into the executor environment (via :func:`tool_env`),
   so the underlying CLI is authenticated when the agent runs it.
2. It is advertised to the LLM in the system prompt (via :func:`active_tool_prompts`),
   so the model knows the capability exists and how to use it.

A tool that is *not* enabled is neither authenticated nor advertised, keeping the
prompt lean and the capability hidden.

Adding a new tool means: add a config sub-model in ``core/config.py``, then add an
entry to ``_REGISTRY`` below describing its env + prompt advertisement.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.config import Config

if TYPE_CHECKING:
    from core.agents import Agent

# Advertisement injected into the system prompt when `gh` is active.
_GH_PROMPT = """<tool name="gh">
The GitHub CLI `gh` is installed and authenticated. Run it with the `run_command`
tool for GitHub operations. Read operations run without asking; creating issues,
PRs or releases ask for confirmation first.
Examples:
  gh issue list --repo owner/name
  gh pr view 123 --repo owner/name
  gh pr list --repo owner/name --state open
  gh api repos/owner/name/commits
  gh search issues "is:open label:bug" --repo owner/name
Always pass `--repo owner/name` unless the working directory is a checkout of the
target repository. Use `-o json` / `gh api` and parse JSON when you need fields.
</tool>"""


# Advertisement injected into the system prompt when `browser` is active.
_BROWSER_PROMPT = """<tool name="browser">
A headless browser (`/app/tools/browser.py`, Playwright) is available via `run_command`
for JS-heavy pages and acting on the user's behalf. Prefer an existing API/CLI
over the browser whenever one exists — it is a last resort.
Verbs (always pass `--url`; add `--profile NAME` to reuse a logged-in session):
  python3 /app/tools/browser.py read --url URL                  # readable page text
  python3 /app/tools/browser.py screenshot --url URL            # save a PNG (path in result)
  python3 /app/tools/browser.py act --url URL --profile P --steps JSON
  python3 /app/tools/browser.py explore --url URL --task "..."  # self-driving loop
  python3 /app/tools/browser.py profiles                        # list saved sessions
PREFER `explore` for ANYTHING interactive — clicking buttons, opening modals/
widgets, multi-step forms, bookings, checkouts, anything inside an iframe. One
browser stays open and an inner loop sees every frame and clicks/types on its
own until done, then returns an answer. It is the ONLY verb that can drive
embedded widgets and payment iframes; `read`/`act` only see the top page and
will fail on them.
CRITICAL: every browser command starts a BRAND-NEW browser that reloads `--url`
from scratch — there is NO shared session or page state between calls. So you
CANNOT do a flow step-by-step across several commands; each call would restart
from the beginning and lose all progress. A whole interactive flow MUST be ONE
explore call that carries the entire task.
CAUTION: `explore` self-drives to completion under a SINGLE approval — there is NO
per-step gate once it starts. Do NOT use it to spend money or submit irreversible
actions on its own: for purchases/payments confirm the exact details with the owner
first, and prefer guided `act` steps (each fill/submit is approved separately).
How to use it well: put the ENTIRE flow in one `--task` as numbered steps with every
value it needs (product, dates, name, email, phone, card details) — it cannot ask
mid-run. It runs for a few minutes and returns ONE JSON `answer`: do NOT treat the wait
as a hang, split the task, retry, or fall back to `read`/`act` (those only see the top
page, never the widget/iframe). Quote the `answer` (and screenshot) back; if it reports
pending/awaiting-approval don't upgrade to "confirmed"; if it returns `done:false` with a
`reason`, report what blocked it and don't blindly re-run.
`read`/`screenshot` run without asking; `act` asks approval each call (shows a screenshot
on chat) and takes `--steps`, a JSON array of single-key objects, e.g.
  [{"fill":["#user","alice"]},{"click":"#login"}]   (fill/click/select/press/wait/goto).
For the full reference — steps syntax, guided first-time login + 2FA, profiles — run
`load_skill browser`.
</tool>"""


# Advertisement injected into the system prompt when `whatsapp` is active (#97).
_WHATSAPP_PROMPT = """<tool name="whatsapp">
WhatsApp is available through the `wacli` CLI — run it with the `run_command` tool.
Read operations (sync, messages, contacts, chats, groups) run without asking;
sending a message asks for confirmation first.
Send a message:
  wacli --json send text --to <jid> --message "..."
Read (sync first whenever checking for new/recent messages):
  wacli --json sync --once --idle-exit 5s
  wacli --json messages list --limit 20
  wacli --json messages search "invoice" --chat <jid>
  wacli --json contacts search "Marco"
JIDs: users are `<phone>@s.whatsapp.net` (digits only, no `+`); groups `<id>@g.us`.
Run `load_skill wacli-whatsapp` for the full command reference.
</tool>"""


@dataclass(frozen=True)
class ToolSpec:
    """Describes an optional external tool the agent can use."""

    key: str  # config sub-key under `tools.` (e.g. "gh")
    label: str  # human-friendly name for the admin UI
    summary: str  # one-line description for the admin UI
    # Returns env vars to inject when the tool is enabled (auth, etc.).
    env: Callable[[Config], dict[str, str]]
    # Returns the system-prompt advertisement block for the tool.
    prompt: Callable[[Config], str]


def _gh_prefers_app(gh) -> bool:
    """Whether the system default should use the GitHub App (#111).

    Explicit ``auth`` wins: ``"app"`` → App, ``"pat"`` → never App (even if App
    fields linger in config). ``""`` infers from whether the App fields are set.
    """
    ready = bool(gh.app_id and gh.installation_id and gh.private_key)
    if not ready:
        return False
    return (gh.auth or "").strip().lower() != "pat"


def _gh_env(config: Config) -> dict[str, str]:
    gh = config.tools.gh
    if not gh.enabled:
        return {}
    # GitHub App takes precedence over the PAT (#111) unless auth is forced to
    # "pat": mint a short-lived installation token so `gh` acts as the bot.
    if _gh_prefers_app(gh):
        from core import github_app

        token = github_app.installation_token(gh.app_id, gh.installation_id, gh.private_key)
        if token:
            return {"GH_TOKEN": token}
        # App selected but the mint failed → fall back to the PAT if present.
    if gh.token:
        # `gh` reads GH_TOKEN (preferred) / GITHUB_TOKEN for non-interactive auth.
        return {"GH_TOKEN": gh.token}
    return {}


def _gh_app_configured(config: Config) -> bool:
    """Whether the system App can be a shared bot for agents (respects auth)."""
    gh = config.tools.gh
    return bool(gh.enabled) and _gh_prefers_app(gh)


def _agent_has_own_app(gh: dict) -> bool:
    """Whether the agent carries its OWN GitHub App creds (#111)."""
    return bool(
        str(gh.get("app_id") or "").strip()
        and str(gh.get("installation_id") or "").strip()
        and str(gh.get("private_key_secret") or "").strip()
    )


def _agent_app_token(gh: dict, resolve_secret: Callable[[str], str | None]) -> str | None:
    """Mint an agent's OWN GitHub App installation token, or ``None`` (#111).

    ``app_id`` + ``installation_id`` are non-secret; the PEM is referenced by
    ``private_key_secret`` (an infra-vault name) so it is never stored in the
    agent doc — same posture as ``token_secret`` for PATs.
    """
    if not _agent_has_own_app(gh):
        return None
    app_id = str(gh.get("app_id")).strip()
    installation_id = str(gh.get("installation_id")).strip()
    key_secret = str(gh.get("private_key_secret")).strip()
    pem = resolve_secret(key_secret)
    if not pem:
        return None
    from core import github_app

    return github_app.installation_token(app_id, installation_id, pem)


def _agent_gh_token(
    gh: dict,
    agent_name: str,
    config: Config,
    resolve_secret: Callable[[str], str | None],
) -> str | None:
    """Resolve an agent's ``GH_TOKEN`` honoring its explicit auth mode (#111).

    * ``auth="pat"`` → its own PAT (``token_secret`` or ``GH_TOKEN_<slug>``); it
      never borrows the App.
    * ``auth="app"`` → its own GitHub App if configured, else the shared *system*
      App bot.
    * ``auth=""`` (legacy/inferred) → PAT if it has one, else own App, else the
      system App bot.

    An agent NEVER falls back to the owner's PAT (#93 no-borrow): only its own
    credentials or the App bot (which is not the owner) are ever used.
    """
    auth = (gh.get("auth") or "").strip().lower()

    def own_pat() -> str | None:
        name = (gh.get("token_secret") or "").strip() or gh_token_secret_name(agent_name)
        return resolve_secret(name)

    def system_app() -> str | None:
        if not _gh_app_configured(config):
            return None
        from core import github_app

        c = config.tools.gh
        return github_app.installation_token(c.app_id, c.installation_id, c.private_key)

    if auth == "pat":
        return own_pat()
    if auth == "app":
        # Own app → own app or NOTHING: never silently cross over to the system
        # bot on a transient mint failure (that could be a different, broader
        # identity). Only an agent that configured no own app uses the shared bot.
        return _agent_app_token(gh, resolve_secret) if _agent_has_own_app(gh) else system_app()
    # Legacy / inferred: PAT first, then own app (own-app-or-nothing — same
    # no-crossover rule as auth="app"), else the shared system bot.
    token = own_pat()
    if token:
        return token
    if _agent_has_own_app(gh):
        return _agent_app_token(gh, resolve_secret)
    return system_app()


def _whatsapp_env(config: Config) -> dict[str, str]:
    wa = config.tools.whatsapp
    if not wa.enabled:
        return {}
    # Identity knobs for the wacli store. Per-agent overrides ride #93's
    # per-agent tool_env on top of these defaults.
    env: dict[str, str] = {}
    if wa.store:
        env["WACLI_STORE"] = wa.store
    if wa.device_label:
        env["WACLI_DEVICE_LABEL"] = wa.device_label
    return env


def _browser_env(config: Config) -> dict[str, str]:
    browser = config.tools.browser
    if not browser.enabled:
        return {}
    env = {"BROWSER_HEADLESS": "1" if browser.headless else "0"}
    if browser.cdp_url:
        env["BROWSER_CDP_URL"] = browser.cdp_url
    if browser.user_agent:
        env["BROWSER_USER_AGENT"] = browser.user_agent
    return env


_REGISTRY: tuple[ToolSpec, ...] = (
    ToolSpec(
        key="gh",
        label="GitHub CLI (gh)",
        summary="Let the agent query and act on GitHub (issues, PRs, repos, API).",
        env=_gh_env,
        prompt=lambda _cfg: _GH_PROMPT,
    ),
    ToolSpec(
        key="browser",
        label="Browser automation",
        summary="Let the agent read JS-heavy pages and act on sites via a headless browser.",
        env=_browser_env,
        prompt=lambda _cfg: _BROWSER_PROMPT,
    ),
    ToolSpec(
        key="whatsapp",
        label="WhatsApp (wacli)",
        summary="Let the agent read and send WhatsApp messages via the local wacli CLI.",
        env=_whatsapp_env,
        prompt=lambda _cfg: _WHATSAPP_PROMPT,
    ),
)


# Env vars the tool registry manages. When a per-agent env override is applied
# (#93), any managed key absent from the override is stripped from the inherited
# process environment too — so an agent that switched `gh` off can never inherit
# a token that leaked in via `.env`/Docker ENV and act as the owner. `gh` reads
# GH_TOKEN, then GITHUB_TOKEN, then the enterprise variants (in that precedence),
# so ALL of them must be stripped, not just GH_TOKEN.
MANAGED_TOOL_ENV_KEYS: frozenset[str] = frozenset(
    {
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "GH_ENTERPRISE_TOKEN",
        "GITHUB_ENTERPRISE_TOKEN",
        "BROWSER_HEADLESS",
        "BROWSER_CDP_URL",
        "BROWSER_USER_AGENT",
        "BROWSER_PROFILE",
    }
)


def registry() -> tuple[ToolSpec, ...]:
    """Return all known optional tools."""
    return _REGISTRY


def _is_enabled(config: Config, key: str) -> bool:
    sub = getattr(config.tools, key, None)
    return bool(getattr(sub, "enabled", False))


def active_tool_prompts(config: Config, agent: Agent | None = None) -> list[str]:
    """Return system-prompt advertisement blocks for every *enabled* tool.

    When ``agent`` is active and has per-tool config (#93), a tool it opted out
    of is dropped, and a note about its own identity (own ``gh`` token, own browser
    profile) is injected into the tool block so the model authenticates as itself.
    """
    blocks: list[str] = []
    for spec in _REGISTRY:
        if not _is_enabled(config, spec.key):
            continue
        setting = agent.tool_setting(spec.key) if agent else None
        if setting is not None and not setting.get("enabled"):
            continue  # this agent has the tool switched off
        block = spec.prompt(config).strip()
        if not block:
            continue
        note = _agent_tool_note(spec.key, setting, agent)
        if note:
            block = block.replace("</tool>", f"{note}\n</tool>", 1)
        blocks.append(block)
    return blocks


def tool_env(config: Config) -> dict[str, str]:
    """Return the merged environment for every *enabled* tool (auth tokens, etc.)."""
    env: dict[str, str] = {}
    for spec in _REGISTRY:
        if _is_enabled(config, spec.key):
            env.update(spec.env(config))
    return env


# -- Per-agent tool identity (#93) ----------------------------------------


def gh_token_secret_name(agent_name: str) -> str:
    """Infra-vault name holding an agent's own GitHub token.

    Namespaced per agent so each agent authenticates as a distinct GitHub user.
    Stored in the *infra* vault (machine-key, boot-unsealed) so it works headless
    and in scheduled jobs — same on-disk posture as the system-wide ``GH_TOKEN``.

    The slug is kept verbatim (case preserved; infra names are case-sensitive) so
    two agents whose slugs differ only by case don't collide on one token.
    """
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", (agent_name or "").strip())
    return f"GH_TOKEN_{slug}"


# `gh` invoked as a word (so `grep -R foo/bar` etc. aren't repo-gated).
_GH_INVOKED_RE = re.compile(r"\bgh\b")
# The `--repo`/`--repository`/`-R` target flag, tolerant of `=`, quotes, and the
# glued short form (`-Rowner/name`) — all forms `gh` itself accepts.
_GH_REPO_FLAG_RE = re.compile(
    r"""(?:--repo(?:sitory)?[=\s]+|-R\s*)["']?([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)"""
)


def github_repo_violation(agent: Agent | None, command: str) -> str | None:
    """First GitHub repo the agent is NOT allowed to touch, or ``None`` (#111).

    Best-effort per-agent repo allowlist from ``tool_config["gh"]["repos"]``.
    Absent ``repos`` key = unrestricted; a *present* list restricts to it, and a
    present-but-empty list (e.g. a subagent whose scope was narrowed to a set
    disjoint from its parent's) allows **nothing** — it blocks every ``--repo``
    target. Only ``gh`` invocations are gated, and only the explicit
    ``--repo``/``-R`` flag is inspected (its various quoted/glued forms).
    ponytail: the HARD boundary is the GitHub App installation's own repo
    selection (server-enforced); this is defense-in-depth, so it deliberately
    does NOT parse cwd checkouts, `gh api` paths, or git remotes — a parser for
    those is a bug farm and would give false confidence. Tighten the App
    installation to narrow access for real.
    """
    if agent is None or not command:
        return None
    gh = agent.tool_setting("gh") or {}
    repos = gh.get("repos")
    if repos is None:
        return None  # no allowlist → unrestricted
    allowed = {str(r).strip().lower() for r in repos if r and str(r).strip()}
    if not _GH_INVOKED_RE.search(command):
        return None  # not a gh command → nothing to gate here
    for repo in _GH_REPO_FLAG_RE.findall(command):
        if repo.lower() not in allowed:
            return repo
    return None


def _agent_tool_note(key: str, setting: dict | None, agent: Agent | None) -> str:
    """A one-line identity note injected into a tool's prompt block for an agent."""
    if agent is None or setting is None:
        return ""
    if key == "gh":
        repos = [r for r in (setting.get("repos") or []) if str(r).strip()]
        repo_note = (
            f" You may only target these repos with `--repo`: {', '.join(repos)}." if repos else ""
        )
        if (setting.get("token_secret") or "").strip():
            return (
                f'Running as agent "{agent.name}": `gh` is authenticated with the '
                "GitHub token configured for this agent. Every gh/git action appears "
                "as that token's GitHub user." + repo_note
            )
        return (
            f'Running as agent "{agent.name}": `gh` is authenticated with this '
            "agent's OWN GitHub identity (a distinct identity from the owner). Every "
            "gh/git action appears as this agent's GitHub user or the configured "
            "GitHub App bot." + repo_note
        )
    if key == "browser":
        profile = (setting.get("profile") or agent.name).strip()
        if profile:
            return (
                f'Running as agent "{agent.name}": your browser profile is '
                f'"{profile}" — it is used by default, so your logged-in sessions are '
                "isolated from other agents. Pass `--profile " + profile + "` "
                "explicitly when a command needs it."
            )
    return ""


def effective_tool_env(
    config: Config,
    agent: Agent | None,
    resolve_secret: Callable[[str], str | None],
) -> dict[str, str]:
    """The tool environment for a turn, adjusted for the active agent (#93).

    Starts from the system-wide :func:`tool_env` and, for an agent that has its
    own tool config, swaps in its own identity:

    * ``gh`` — replace ``GH_TOKEN`` with the agent's own token (resolved from the
      infra vault). If the agent switched ``gh`` off, ``GH_TOKEN`` is *removed*
      so it can never act as the owner; if it has no own token, it also falls back
      to no token rather than silently borrowing the owner's.
    * ``browser`` — set ``BROWSER_PROFILE`` to the agent's isolated profile.

    An agent with no entry for a tool inherits the system config unchanged, so
    existing setups keep working (migration §6).
    """
    env = tool_env(config)
    if agent is None:
        return env

    gh = agent.tool_setting("gh")
    if gh is not None:
        # Agent has an explicit gh policy → never inherit the owner's token.
        env.pop("GH_TOKEN", None)
        if gh.get("enabled") and config.tools.gh.enabled:
            token = _agent_gh_token(gh, agent.name, config, resolve_secret)
            if token:
                env["GH_TOKEN"] = token

    browser = agent.tool_setting("browser")
    if browser is not None and browser.get("enabled") and config.tools.browser.enabled:
        profile = (browser.get("profile") or agent.name).strip()
        if profile:
            env["BROWSER_PROFILE"] = profile

    return env


if __name__ == "__main__":
    # ponytail: one runnable check covering per-agent env swap + prompt notes.
    from core.agents import Agent

    cfg = Config()
    cfg.tools.gh.enabled = True
    cfg.tools.gh.token = "owner-token"
    cfg.tools.browser.enabled = True

    assert gh_token_secret_name("coding-helper") == "GH_TOKEN_coding-helper"
    assert gh_token_secret_name("Hopper") != gh_token_secret_name("hopper")  # case-distinct

    vault = {"GH_TOKEN_hopper": "hopper-token", "SHARED_PAT": "shared-token"}
    resolve = vault.get  # Callable[[str], str | None]

    # No agent → system token, no profile.
    base = effective_tool_env(cfg, None, resolve)
    assert base["GH_TOKEN"] == "owner-token" and "BROWSER_PROFILE" not in base

    # Agent with no tool_config → inherits system config unchanged (migration).
    plain = Agent(name="plain")
    assert effective_tool_env(cfg, plain, resolve)["GH_TOKEN"] == "owner-token"

    # Agent with its own gh token + browser profile → own identity.
    hopper = Agent(
        name="hopper",
        tool_config={"gh": {"enabled": True}, "browser": {"enabled": True, "profile": "hop"}},
    )
    env = effective_tool_env(cfg, hopper, resolve)
    assert env["GH_TOKEN"] == "hopper-token", env.get("GH_TOKEN")
    assert env["BROWSER_PROFILE"] == "hop"

    # gh enabled but no own token stored → no token (never borrows the owner's).
    atlas = Agent(name="atlas", tool_config={"gh": {"enabled": True}})
    assert "GH_TOKEN" not in effective_tool_env(cfg, atlas, resolve)

    # token_secret reuses an existing vault secret instead of the namespaced one.
    ref = Agent(name="atlas", tool_config={"gh": {"enabled": True, "token_secret": "SHARED_PAT"}})
    assert effective_tool_env(cfg, ref, resolve)["GH_TOKEN"] == "shared-token"

    # gh explicitly disabled → GH_TOKEN removed.
    lingua = Agent(name="lingua", tool_config={"gh": {"enabled": False}})
    assert "GH_TOKEN" not in effective_tool_env(cfg, lingua, resolve)

    # GitHub App configured + agent has no own PAT → uses the shared bot token (#111).
    app_cfg = Config()
    app_cfg.tools.gh.enabled = True
    app_cfg.tools.gh.app_id = "42"
    app_cfg.tools.gh.installation_id = "7"
    app_cfg.tools.gh.private_key = "PEM"
    import core.github_app as _ga

    # Tag the minted token with the app_id so we can tell own-App from system-App.
    _real_it = _ga.installation_token
    _ga.installation_token = lambda app_id, *_a: f"app:{app_id}"
    try:
        # Legacy (no auth): no own PAT → shared system App bot (app 42).
        assert effective_tool_env(app_cfg, atlas, resolve)["GH_TOKEN"] == "app:42"

        # Explicit auth="pat" → own PAT only, NEVER the App (even if App fields set).
        pat_only = Agent(
            name="hopper", tool_config={"gh": {"enabled": True, "auth": "pat", "app_id": "99"}}
        )
        assert effective_tool_env(app_cfg, pat_only, resolve)["GH_TOKEN"] == "hopper-token"
        pat_none = Agent(name="none", tool_config={"gh": {"enabled": True, "auth": "pat"}})
        assert "GH_TOKEN" not in effective_tool_env(app_cfg, pat_none, resolve)

        # Agent's OWN GitHub App (multiple apps) → its own bot (app 500), not 42.
        own_app = Agent(
            name="coder",
            tool_config={
                "gh": {
                    "enabled": True,
                    "auth": "app",
                    "app_id": "500",
                    "installation_id": "9",
                    "private_key_secret": "CODER_APP_KEY",
                }
            },
        )
        vault["CODER_APP_KEY"] = "coder-pem"
        assert effective_tool_env(app_cfg, own_app, resolve)["GH_TOKEN"] == "app:500"

        # auth="app" but no own App creds → falls back to the shared system App bot.
        app_shared = Agent(name="w", tool_config={"gh": {"enabled": True, "auth": "app"}})
        assert effective_tool_env(app_cfg, app_shared, resolve)["GH_TOKEN"] == "app:42"

        # System auth forced to "pat" → the App bot is NOT offered to agents.
        pat_sys = Config()
        pat_sys.tools.gh.enabled = True
        pat_sys.tools.gh.auth = "pat"
        pat_sys.tools.gh.app_id = "42"
        pat_sys.tools.gh.installation_id = "7"
        pat_sys.tools.gh.private_key = "PEM"
        assert "GH_TOKEN" not in effective_tool_env(pat_sys, atlas, resolve)
    finally:
        _ga.installation_token = _real_it

    # Per-agent repo allowlist (#111): only --repo targets outside the list are blocked.
    scoped = Agent(name="coder", tool_config={"gh": {"enabled": True, "repos": ["me/humux"]}})
    assert github_repo_violation(scoped, "gh issue list --repo me/humux") is None
    assert github_repo_violation(scoped, "gh pr view 1 -R me/other") == "me/other"
    assert github_repo_violation(scoped, "gh api user") is None  # no --repo → can't tell → allow
    assert github_repo_violation(plain, "gh pr view 1 --repo any/thing") is None  # no allowlist
    # Quote/glue-tolerant + gh-scoped (no false-block on grep -R).
    assert github_repo_violation(scoped, 'gh pr view 1 --repo "me/evil"') == "me/evil"
    assert github_repo_violation(scoped, "gh pr view 1 -Rme/evil") == "me/evil"
    assert github_repo_violation(scoped, "gh pr view 1 --repo=me/humux") is None
    assert github_repo_violation(scoped, "grep -R foo/bar .") is None  # not a gh command
    # Present-but-empty allowlist = block every --repo (disjoint-narrow result).
    blocked = Agent(name="sub", tool_config={"gh": {"enabled": True, "repos": []}})
    assert github_repo_violation(blocked, "gh pr view 1 --repo me/humux") == "me/humux"
    assert github_repo_violation(blocked, "gh api user") is None  # no --repo target

    # Prompts: hopper sees gh + browser, with identity notes; lingua's gh is hidden.
    hp = "\n".join(active_tool_prompts(cfg, hopper))
    assert 'agent "hopper"' in hp and "OWN GitHub identity" in hp and "hop" in hp
    lp = "\n".join(active_tool_prompts(cfg, lingua))
    assert 'name="gh"' not in lp and 'name="browser"' in lp
    # No agent → plain blocks, no identity notes.
    nop = "\n".join(active_tool_prompts(cfg))
    assert 'name="gh"' in nop and "Running as agent" not in nop

    print("tools.py self-check OK")
