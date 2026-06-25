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

from collections.abc import Callable
from dataclasses import dataclass

from core.config import Config

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
  python3 /app/tools/browser.py screenshot --url URL            # save a PNG to ~/Downloads
  python3 /app/tools/browser.py act --url URL --profile P --steps JSON
  python3 /app/tools/browser.py explore --url URL --task "..."  # self-driving loop
  python3 /app/tools/browser.py profiles                        # list saved sessions
PREFER `explore` for ANYTHING interactive — clicking buttons, opening modals/
widgets, multi-step forms, bookings, checkouts, anything inside an iframe. One
browser stays open and an inner loop sees every frame and clicks/types on its
own until done, then returns an answer. It is the ONLY verb that can drive
embedded widgets and payment iframes; `read`/`act` only see the top page and
will fail on them.
How to use it well:
- Put the ENTIRE task in one `--task`, with every concrete value the flow needs
  (product, dates/times, name, email, phone, full card number/expiry/CVC/ZIP).
  It cannot ask you mid-run, so give it everything up front.
- It runs autonomously for up to a few minutes and returns ONE JSON result with
  an `answer`. That is expected — do NOT treat the wait as a hang, do NOT retry
  it, and do NOT fall back to `read`/`act`. Call it once and wait.
- Quote its returned `answer` (and screenshot path) back to the user.
Use `read`/`screenshot` only for static pages where you already know the URL.
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


def registry() -> tuple[ToolSpec, ...]:
    """Return all known optional tools."""
    return _REGISTRY


def _is_enabled(config: Config, key: str) -> bool:
    sub = getattr(config.tools, key, None)
    return bool(getattr(sub, "enabled", False))


def active_tool_prompts(config: Config) -> list[str]:
    """Return system-prompt advertisement blocks for every *enabled* tool."""
    blocks: list[str] = []
    for spec in _REGISTRY:
        if _is_enabled(config, spec.key):
            block = spec.prompt(config).strip()
            if block:
                blocks.append(block)
    return blocks


def tool_env(config: Config) -> dict[str, str]:
    """Return the merged environment for every *enabled* tool (auth tokens, etc.)."""
    env: dict[str, str] = {}
    for spec in _REGISTRY:
        if _is_enabled(config, spec.key):
            env.update(spec.env(config))
    return env
