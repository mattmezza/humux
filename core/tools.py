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
    from core.personae import Persona

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
How to use it well:
- Put the ENTIRE task in one `--task`, as numbered steps, with every concrete
  value the flow needs (product, dates/times, name, email, phone, full card
  number/expiry/CVC/ZIP). It cannot ask you mid-run, so give it everything up
  front. Example for a booking:
  python3 /app/tools/browser.py explore --url https://shop.example/book --task \
    "1) click 'Book now'. 2) select the Single Kayak product. 3) set pickup and
     return date 2026-06-26, start 09:00, end 10:00. 4) click Next through each
     step. 5) fill name Matteo Merola, email m@x.com, phone +41770000000.
     6) at payment fill card 4242424242424242 exp 03/29 cvc 736 zip 8000 and
     click Pay. 7) report the confirmation."
- It runs autonomously for a few minutes and returns ONE JSON result with an
  `answer`. That is expected — do NOT treat the wait as a hang, do NOT split the
  task, do NOT retry, and do NOT fall back to `read`/`act` (they only see the top
  page, never the widget/iframe, and will mislead you with stale content).
- Quote its returned `answer` (and screenshot path) back to the user. If it
  reports a pending/awaiting-approval status, say so — don't upgrade it to
  "confirmed".
- If the result has `done:false` and a `reason`, the flow could NOT be completed
  (stuck, control not found, dead end). Tell the user what blocked it and share
  the screenshot — do NOT claim success and do NOT blindly re-run the same task;
  the loop already gave up on purpose to avoid wasting effort.
`read`/`screenshot` run without asking. `act` changes state (click/fill/submit)
so it asks for approval each time; on chat channels the approval shows a
screenshot of the page. `--steps` is an ordered JSON array of single-key objects:
  [{"fill":["#user","alice"]},{"fill":["#pass","s3cr3t"]},{"click":"#login"}]
Steps: fill[sel,val], click[sel], select[sel,val], press[sel,key], wait[ms|sel], goto[url].
Guided first-time login: screenshot the login page so the user can follow along,
ask the user for credentials (never store them), then `act` to fill+submit. If
2FA appears, screenshot it and ask the user for the code, then `act` again. After
login the `--profile` session persists, so later visits skip the login.
Some sites with strong bot-management may still block headless automation.
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


def _gh_env(config: Config) -> dict[str, str]:
    gh = config.tools.gh
    if gh.enabled and gh.token:
        # `gh` reads GH_TOKEN (preferred) / GITHUB_TOKEN for non-interactive auth.
        return {"GH_TOKEN": gh.token}
    return {}


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
)


# Env vars the tool registry manages. When a per-persona env override is applied
# (#93), any managed key absent from the override is stripped from the inherited
# process environment too — so a persona that switched `gh` off can never inherit
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


def active_tool_prompts(config: Config, persona: Persona | None = None) -> list[str]:
    """Return system-prompt advertisement blocks for every *enabled* tool.

    When ``persona`` is active and has per-tool config (#93), a tool it opted out
    of is dropped, and a note about its own identity (own ``gh`` token, own browser
    profile) is injected into the tool block so the model authenticates as itself.
    """
    blocks: list[str] = []
    for spec in _REGISTRY:
        if not _is_enabled(config, spec.key):
            continue
        setting = persona.tool_setting(spec.key) if persona else None
        if setting is not None and not setting.get("enabled"):
            continue  # this persona has the tool switched off
        block = spec.prompt(config).strip()
        if not block:
            continue
        note = _persona_tool_note(spec.key, setting, persona)
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


# -- Per-persona tool identity (#93) ----------------------------------------


def gh_token_secret_name(persona_name: str) -> str:
    """Infra-vault name holding a persona's own GitHub token.

    Namespaced per persona so each agent authenticates as a distinct GitHub user.
    Stored in the *infra* vault (machine-key, boot-unsealed) so it works headless
    and in scheduled jobs — same on-disk posture as the system-wide ``GH_TOKEN``.

    The slug is kept verbatim (case preserved; infra names are case-sensitive) so
    two personae whose slugs differ only by case don't collide on one token.
    """
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", (persona_name or "").strip())
    return f"GH_TOKEN_{slug}"


def _persona_tool_note(key: str, setting: dict | None, persona: Persona | None) -> str:
    """A one-line identity note injected into a tool's prompt block for a persona."""
    if persona is None or setting is None:
        return ""
    if key == "gh":
        if (setting.get("token_secret") or "").strip():
            return (
                f'Running as persona "{persona.name}": `gh` is authenticated with the '
                "GitHub token configured for this persona. Every gh/git action appears "
                "as that token's GitHub user."
            )
        return (
            f'Running as persona "{persona.name}": `gh` is authenticated with this '
            "persona's OWN GitHub token (a distinct identity from the owner). Every "
            "gh/git action appears as this persona's GitHub user."
        )
    if key == "browser":
        profile = (setting.get("profile") or persona.name).strip()
        if profile:
            return (
                f'Running as persona "{persona.name}": your browser profile is '
                f'"{profile}" — it is used by default, so your logged-in sessions are '
                "isolated from other personae. Pass `--profile " + profile + "` "
                "explicitly when a command needs it."
            )
    return ""


def effective_tool_env(
    config: Config,
    persona: Persona | None,
    resolve_secret: Callable[[str], str | None],
) -> dict[str, str]:
    """The tool environment for a turn, adjusted for the active persona (#93).

    Starts from the system-wide :func:`tool_env` and, for a persona that has its
    own tool config, swaps in its own identity:

    * ``gh`` — replace ``GH_TOKEN`` with the persona's own token (resolved from the
      infra vault). If the persona switched ``gh`` off, ``GH_TOKEN`` is *removed*
      so it can never act as the owner; if it has no own token, it also falls back
      to no token rather than silently borrowing the owner's.
    * ``browser`` — set ``BROWSER_PROFILE`` to the persona's isolated profile.

    A persona with no entry for a tool inherits the system config unchanged, so
    existing setups keep working (migration §6).
    """
    env = tool_env(config)
    if persona is None:
        return env

    gh = persona.tool_setting("gh")
    if gh is not None:
        # Persona has an explicit gh policy → never inherit the owner's token.
        env.pop("GH_TOKEN", None)
        if gh.get("enabled") and config.tools.gh.enabled:
            # ``token_secret`` lets a persona reuse an existing infra-vault secret
            # (e.g. the system GH_TOKEN, or a shared PAT) instead of storing its own
            # copy; otherwise its own namespaced token is used (#93).
            name = (gh.get("token_secret") or "").strip() or gh_token_secret_name(persona.name)
            token = resolve_secret(name)
            if token:
                env["GH_TOKEN"] = token

    browser = persona.tool_setting("browser")
    if browser is not None and browser.get("enabled") and config.tools.browser.enabled:
        profile = (browser.get("profile") or persona.name).strip()
        if profile:
            env["BROWSER_PROFILE"] = profile

    return env


if __name__ == "__main__":
    # ponytail: one runnable check covering per-persona env swap + prompt notes.
    from core.personae import Persona

    cfg = Config()
    cfg.tools.gh.enabled = True
    cfg.tools.gh.token = "owner-token"
    cfg.tools.browser.enabled = True

    assert gh_token_secret_name("coding-helper") == "GH_TOKEN_coding-helper"
    assert gh_token_secret_name("Hopper") != gh_token_secret_name("hopper")  # case-distinct

    vault = {"GH_TOKEN_hopper": "hopper-token", "SHARED_PAT": "shared-token"}
    resolve = vault.get  # Callable[[str], str | None]

    # No persona → system token, no profile.
    base = effective_tool_env(cfg, None, resolve)
    assert base["GH_TOKEN"] == "owner-token" and "BROWSER_PROFILE" not in base

    # Persona with no tool_config → inherits system config unchanged (migration).
    plain = Persona(name="plain")
    assert effective_tool_env(cfg, plain, resolve)["GH_TOKEN"] == "owner-token"

    # Persona with its own gh token + browser profile → own identity.
    hopper = Persona(
        name="hopper",
        tool_config={"gh": {"enabled": True}, "browser": {"enabled": True, "profile": "hop"}},
    )
    env = effective_tool_env(cfg, hopper, resolve)
    assert env["GH_TOKEN"] == "hopper-token", env.get("GH_TOKEN")
    assert env["BROWSER_PROFILE"] == "hop"

    # gh enabled but no own token stored → no token (never borrows the owner's).
    atlas = Persona(name="atlas", tool_config={"gh": {"enabled": True}})
    assert "GH_TOKEN" not in effective_tool_env(cfg, atlas, resolve)

    # token_secret reuses an existing vault secret instead of the namespaced one.
    ref = Persona(name="atlas", tool_config={"gh": {"enabled": True, "token_secret": "SHARED_PAT"}})
    assert effective_tool_env(cfg, ref, resolve)["GH_TOKEN"] == "shared-token"

    # gh explicitly disabled → GH_TOKEN removed.
    lingua = Persona(name="lingua", tool_config={"gh": {"enabled": False}})
    assert "GH_TOKEN" not in effective_tool_env(cfg, lingua, resolve)

    # Prompts: hopper sees gh + browser, with identity notes; lingua's gh is hidden.
    hp = "\n".join(active_tool_prompts(cfg, hopper))
    assert 'persona "hopper"' in hp and "OWN GitHub token" in hp and "hop" in hp
    lp = "\n".join(active_tool_prompts(cfg, lingua))
    assert 'name="gh"' not in lp and 'name="browser"' in lp
    # No persona → plain blocks, no identity notes.
    nop = "\n".join(active_tool_prompts(cfg))
    assert 'name="gh"' in nop and "Running as persona" not in nop

    print("tools.py self-check OK")
