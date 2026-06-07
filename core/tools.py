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


_REGISTRY: tuple[ToolSpec, ...] = (
    ToolSpec(
        key="gh",
        label="GitHub CLI (gh)",
        summary="Let the agent query and act on GitHub (issues, PRs, repos, API).",
        env=_gh_env,
        prompt=lambda _cfg: _GH_PROMPT,
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
