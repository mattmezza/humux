"""Configuration loading and validation.

Loads settings from config.yml with environment variable interpolation.
Secrets (API keys, tokens) come from environment variables; structural
config (which channels are enabled, scheduler jobs, etc.) lives in YAML.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator


def _resolve_env_vars(obj: object) -> object:
    """Recursively resolve ${ENV_VAR} references in strings.

    ``${vault:NAME}`` references contain a ``:`` and so do not match ``\\w+`` —
    they pass through here untouched and are resolved later against the encrypted
    infra vault (see :func:`resolve_vault_vars`).
    """
    if isinstance(obj, str):
        return re.sub(
            r"\$\{(\w+)\}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            obj,
        )
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    return obj


_VAULT_RE = re.compile(r"\$\{vault:([A-Za-z0-9_:-]+)\}")


def resolve_vault_vars(obj: object, resolve: Callable[[str], str | None]) -> object:
    """Recursively resolve ``${vault:NAME}`` references via ``resolve``.

    ``resolve(name)`` returns the decrypted infra-vault value, falling back to
    the environment (so ``.env`` stays a fallback). A miss leaves the reference
    literally in place rather than blanking the field.
    """
    if isinstance(obj, str):
        return _VAULT_RE.sub(lambda m: resolve(m.group(1)) or m.group(0), obj)
    if isinstance(obj, dict):
        return {k: resolve_vault_vars(v, resolve) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_vault_vars(v, resolve) for v in obj]
    return obj


# --- Config models ---


class AgentConfig(BaseModel):
    name: str = "Clio"
    owner_name: str = "Matteo"
    llm_provider: str = "deepseek"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = ""
    google_api_key: str = ""
    google_base_url: str = ""
    grok_api_key: str = ""
    grok_base_url: str = ""
    deepseek_api_key: str = ""
    deepseek_base_url: str = ""
    model: str = "deepseek-v4-flash"
    thinking_level: str = ""  # "" (off) | "low" | "medium" | "high" — only for reasoning models
    # Hard ceiling on tokens the model may emit per response. The agentic loop
    # truncates mid-tool-call when this is too small for the output (e.g. a large
    # file written via write_file), so keep it generous. 8192 is safe across providers;
    # raise it on the LLM admin tab for capable models (Claude allows up to
    # 128000 — note large non-streaming outputs can approach provider timeouts).
    max_tokens: int = 8192
    # Sampling temperature for the main agent loop. Lower = steadier tool calls;
    # higher = more varied prose. Skipped automatically for reasoning calls. Tune
    # on the LLM admin tab. (#12)
    temperature: float = 0.5
    timezone: str = "Europe/Zurich"
    skills_dir: str = "skills/"
    skills_db_path: str = "data/skills.db"
    # How the skills index reaches the model (#50):
    #   "inject"    — the full index rides every turn's preamble (default; unchanged)
    #   "on_demand" — the preamble omits it; the model calls search_skills/list_skills
    # Any unrecognised value falls back to "inject" (the safe default).
    skills_index_mode: str = "inject"
    agents_dir: str = "agents/"
    agents_db_path: str = "data/agents.db"
    active_agent: str = ""  # empty = default identity (character below)
    character: str = ""  # identity + tone (legacy `personalia` was merged in — #98)
    # Account bindings for the DEFAULT identity — used when no agent is active
    # (#110). Same shape as the per-agent facets: email entries carry
    # {account, access_level, is_sender_identity}, calendar/contacts drop the
    # sender flag. Empty = the default agent is unscoped (legacy behaviour: it may
    # use any configured account); non-empty = it is scoped to these accounts,
    # enforced exactly like an agent's.
    email_accounts: list[dict] = Field(default_factory=list)
    calendar_accounts: list[dict] = Field(default_factory=list)
    contacts_accounts: list[dict] = Field(default_factory=list)


class GroupChatConfig(BaseModel):
    """Multi-agent group rooms — turn-taking + loop guard + speaker tags (#30).

    Lets several agent-bots share one Telegram group without the raw misbehaviour
    (every bot answering every message; bots looping replies at each other). In a
    group/supergroup a bot:

    - replies only when **addressed** — @mentioned or replying to one of its own
      messages (``reply_when_addressed_only``); otherwise it stays silent but still
      records the turn for context,
    - **ignores other bots** so two assistants never loop replying to each other
      (``ignore_bots``); their messages are still recorded for context,
    - records every message it sees with a ``[from <author>]`` speaker tag, so a
      agent is never confused about who said what in the shared history.

    Receiving the unaddressed messages that feed the shared context requires the
    bot's Telegram **privacy mode to be OFF** (set via BotFather). With privacy
    mode on (the default) a bot only receives messages addressed to it, so the
    gate is moot and no shared context accumulates. Telegram-only: WhatsApp uses a
    single number, so multi-bot rooms don't apply there.

    Off by default (like ``topics_enabled``) so existing single-bot group flows
    are unchanged on upgrade: enabling it makes a bot reply only when addressed
    and re-keys a group's history from per-sender to per-group (no migration —
    prior group history is simply not carried forward).
    """

    enabled: bool = False
    # Respond-gate: only reply when addressed. False = reply to every human message
    # in the group (still ignoring bots / tagging speakers).
    reply_when_addressed_only: bool = True
    # Loop guard: never reply to a message authored by another bot (record only).
    ignore_bots: bool = True


class TelegramConfig(BaseModel):
    enabled: bool = False
    bot_token: str = ""
    allowed_user_ids: list[int] = Field(default_factory=list)
    # Opt-in: fold forum topics into separate contexts (one agent per topic).
    # Off by default so the plain 1:1 DM flow is unchanged.
    topics_enabled: bool = False
    # Group multi-agent room behaviour (#30); inherited by per-agent bots.
    group_chat: GroupChatConfig = Field(default_factory=GroupChatConfig)

    @field_validator("allowed_user_ids", mode="before")
    @classmethod
    def parse_comma_separated_ints(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return [int(x.strip()) for x in v.split(",") if x.strip()] if v else []
        return v


class ChannelsConfig(BaseModel):
    telegram: TelegramConfig = TelegramConfig()


class CalendarProvider(BaseModel):
    name: str
    url: str
    username: str = ""
    password: str = ""


class CalendarConfig(BaseModel):
    providers: list[CalendarProvider] = Field(default_factory=list)


class KokoroConfig(BaseModel):
    model_path: str = "models/kokoro/kokoro-v1.0.onnx"
    voices_path: str = "models/kokoro/voices-v1.0.bin"
    default_voice: str = "af_bella"


class VoiceConfig(BaseModel):
    stt_model: str = "base"
    tts_voice: str = "en-US-AvaNeural"
    tts_enabled: bool = True
    backend: str = "edge-tts"  # "edge-tts" | "kokoro"
    kokoro: KokoroConfig = KokoroConfig()


class SchedulerJob(BaseModel):
    id: str
    cron: str
    task: str
    channel: str = "telegram"
    type: str = "agent"
    agent: str = ""  # for type="subagent": the agent the run adopts


class SchedulerConfig(BaseModel):
    jobs: list[SchedulerJob] = Field(default_factory=list)


class AdminConfig(BaseModel):
    port: int = 8000
    api_key: str = ""
    capture_prompts: bool = False


class HistoryConfig(BaseModel):
    db_path: str = "data/history.db"
    max_turns: int = 10  # number of user-assistant pairs to include
    mode: str = "injection"  # "injection" (windowed history) or "session" (sticky per channel)


class EmbeddingConfig(BaseModel):
    """Tier 2 — semantic similarity + relevance-ranked injection.

    Disabled by default so the system runs on Tier-1 lexical retrieval with no
    extra dependency or network call. When enabled, vectors are fetched from an
    OpenAI-compatible ``/embeddings`` endpoint and stored as a blob alongside
    each long-term memory (brute-force cosine in Python — fine at <1k rows, no
    native extension required, identical on local and container SQLite).
    """

    enabled: bool = True
    provider: str = "local"  # "local" (fastembed, on-device) or an OpenAI-compatible API
    model: str = "BAAI/bge-small-en-v1.5"  # local model id; for API use e.g. text-embedding-3-small
    cache_dir: str = "models"  # where local models are stored (bundled in the Docker image)
    api_key: str = ""  # API providers only; falls back to the agent provider key when empty
    base_url: str = ""  # API providers only; falls back to the agent provider base URL when empty
    dimensions: int = 0  # 0 = provider default (API providers only)
    injection_top_k: int = 12  # relevance-ranked memories injected per turn
    recall_top_k: int = 10  # max memories returned by the recall_memory tool (full-store lookup)


class MemoryConfig(BaseModel):
    db_path: str = "data/memory.db"
    long_term_limit: int = 50
    extraction_provider: str = "deepseek"
    extraction_model: str = "deepseek-v4-flash"
    consolidation_provider: str = "deepseek"
    consolidation_model: str = "deepseek-v4-flash"
    extraction_thinking_level: str = ""  # "" (off) | "low" | "medium" | "high"
    consolidation_thinking_level: str = ""  # "" (off) | "low" | "medium" | "high"
    extraction_cooldown_seconds: int = 120  # minimum seconds between extractions

    embedding: EmbeddingConfig = EmbeddingConfig()

    # Tier 3 — forgetting / importance / reinforcement
    default_importance: float = 5.0  # 1-10 scale assigned to new long-term memories
    archive_after_days: int = 90  # min age before a cold memory may be archived
    archive_max_importance: float = 4.0  # only archive memories at/below this importance
    archive_min_idle_days: int = 45  # require this long since last access/creation

    # Tier 4 — long-term hygiene pass (cluster + merge near-duplicates)
    hygiene_enabled: bool = True
    hygiene_similarity_threshold: float = 0.45  # min similarity to cluster two memories


class GoalDecompositionConfig(BaseModel):
    enabled: bool = True
    provider: str = "deepseek"
    model: str = "deepseek-v4-flash"
    thinking_level: str = ""  # "" (off) | "low" | "medium" | "high"


class TaskReflectionConfig(BaseModel):
    enabled: bool = True
    provider: str = "deepseek"
    model: str = "deepseek-v4-flash"
    thinking_level: str = ""  # "" (off) | "low" | "medium" | "high"
    db_path: str = "data/reflections.db"
    max_reflections: int = 12  # injected per turn; kept small to cut prompt bloat (#5, was 50)


class ReplyDecisionConfig(BaseModel):
    """Decide whether to reply in shared/group chats (#36).

    Off by default: 1:1 chats always warrant a reply, and the extra LLM call
    adds latency. Turn it on for group chats that mix multiple bots/people,
    where the agent should stay quiet for messages aimed at someone else or
    caught in a bot-to-bot reaction loop.
    """

    enabled: bool = False
    provider: str = "deepseek"
    model: str = "deepseek-v4-flash"  # fast + cheap is ideal for the yes/no reply call
    thinking_level: str = ""  # "" (off) | "low" | "medium" | "high"
    group_only: bool = True  # only gate group chats; DMs always get a reply
    # Hard backstop: never send more than this many auto-replies into one chat
    # per rolling window — guarantees a runaway loop terminates even if the LLM
    # gate keeps voting "reply".
    max_replies_per_window: int = 6
    window_seconds: int = 120


class CompactionConfig(BaseModel):
    """Conversation compaction — summarise old turns when the context grows.

    Only applies in session history mode. The threshold is evaluated against
    the real token usage reported by the provider after each turn.
    """

    enabled: bool = True
    provider: str = "deepseek"
    model: str = "deepseek-v4-flash"
    thinking_level: str = ""  # "" (off) | "low" | "medium" | "high"
    threshold_type: str = "percent"  # "percent" (of context window) or "tokens" (absolute)
    threshold_percent: int = 80  # trigger at this % of the model's context window
    threshold_tokens: int = 150000  # absolute trigger when threshold_type == "tokens"
    context_window: int = 200000  # fallback window for % mode when the model is unknown
    keep_recent_turns: int = 4  # most-recent user turns kept verbatim after compaction


class SearchConfig(BaseModel):
    enabled: bool = False
    provider: str = "tavily"
    api_key: str = ""
    max_results: int = 5


class VisionConfig(BaseModel):
    """Vision fallback — caption images via a secondary vision-capable model
    when the active model can't see images. Off by default; engages only when
    the active model lacks vision and an image is present."""

    enabled: bool = False
    provider: str = "anthropic"
    model: str = "claude-haiku-4-5"


class YouConfig(BaseModel):
    personalia: str = ""


class PromptConfig(BaseModel):
    tool_usage_override: str = ""
    history_handling_override: str = ""


class GhToolConfig(BaseModel):
    """GitHub CLI (`gh`) tool — auth via a Personal Access Token or a GitHub App.

    When ``app_id`` + ``installation_id`` + ``private_key`` are all set, a
    short-lived installation token is minted and injected as ``GH_TOKEN`` so
    ``gh`` acts as the App's bot identity (``<app>[bot]``) with its own
    permissions and rate limit (#111). The App takes precedence over the PAT;
    if it is not configured (or a mint fails) the PAT is used as a fallback.

    ``auth`` makes that choice explicit: ``"app"`` uses the App (PAT only as a
    mint-failure fallback), ``"pat"`` forces the PAT even if App fields linger,
    and ``""`` infers (App if its fields are filled, else PAT) for back-compat.
    """

    # app_id / installation_id are numeric on GitHub; YAML and the App-install
    # API hand them over as ints, so coerce numbers to str instead of rejecting
    # them (fixes "Input should be a valid string" on restart, #111).
    model_config = ConfigDict(coerce_numbers_to_str=True)

    enabled: bool = False
    auth: str = ""  # "" (infer) | "pat" | "app"
    token: str = ""  # GitHub PAT, injected as GH_TOKEN when running `gh`
    # GitHub App (#111). app_id + installation_id are non-secret; the PEM private
    # key is the sensitive credential → keep it in the vault
    # (${vault:GITHUB_APP_PRIVATE_KEY}), same posture as the PAT.
    app_id: str = ""
    installation_id: str = ""
    private_key: str = ""  # RSA PEM


class BrowserToolConfig(BaseModel):
    """Headless browser automation (Playwright/Chromium) — see tools/browser.py."""

    enabled: bool = False
    headless: bool = True
    # Optional CDP endpoint of a sidecar Chromium. Empty = launch locally (works
    # in the local REPL with no extra services). Set to keep the main image lean.
    cdp_url: str = ""
    # Override the browser User-Agent. Empty = built-in desktop Chrome UA.
    user_agent: str = ""


class ImageGenToolConfig(BaseModel):
    """On-demand image generation (issue #55) — see core/imagegen.py."""

    enabled: bool = False
    provider: str = "openrouter"  # openrouter | fal | openai
    model: str = ""  # blank = provider default (core.imagegen.DEFAULT_MODELS)
    # Blank reuses the LLM key for openai/openrouter; else stored in the vault.
    api_key: str = ""
    daily_budget: int = 0  # max images/day; 0 = unlimited
    monthly_budget: int = 0  # max images/month; 0 = unlimited
    db_path: str = "data/imagegen.db"  # usage counter store


class WhatsAppToolConfig(BaseModel):
    """WhatsApp via the local `wacli` CLI (issue #97) — a tool, not a channel.

    The agent reads and sends WhatsApp by running `wacli` through `run_command`.
    Linking (QR scan), sync and logout are managed from the Tools tab.
    """

    enabled: bool = False
    # WACLI_STORE override (which linked account). Blank = wacli default (~/.wacli).
    # ponytail: one global store; per-agent accounts ride #93's tool_env override.
    store: str = ""
    device_label: str = ""  # WACLI_DEVICE_LABEL; blank = wacli default ("MPA")


class ToolsConfig(BaseModel):
    """Optional external CLI tools the agent can use (see core/tools.py)."""

    gh: GhToolConfig = GhToolConfig()
    browser: BrowserToolConfig = BrowserToolConfig()
    imagegen: ImageGenToolConfig = ImageGenToolConfig()
    whatsapp: WhatsAppToolConfig = WhatsAppToolConfig()


class WorkspaceConfig(BaseModel):
    """Coding harness — confined file read/write/edit/list/grep tools (issue #76).

    Off by default. When enabled, the agent gets ``read_file``/``write_file``/
    ``edit_file``/``list_dir``/``grep``/``run_command_in_dir`` tools, all confined
    to ``directory``. Reads are pre-approved; writes ask first. An empty/blank
    ``directory`` keeps the tools inert even if ``enabled`` is true — there is no
    default root, so the agent can never touch the filesystem until the owner
    points it at one (e.g. ``~/projects``). See core/coding.py.
    """

    enabled: bool = False
    directory: str = ""  # allowed workspace root; blank = no access


class ArtifactsConfig(BaseModel):
    """Public serving toggle for agent-published web artifacts (issue #82).

    Artifacts are files the agent writes under ``{workspace}/artifacts/{slug}/``
    with the coding-harness ``write_file`` tool — there is no separate storage,
    TTL or directory here. This flag only gates the public, no-auth
    ``/artifacts/`` route; serving also requires the workspace harness to be on
    (it provides the write path). See core/artifacts.py.
    """

    enabled: bool = True


class SubagentsConfig(BaseModel):
    """Subagents — scoped sub-loops the agent can delegate to (see core/subagents.py).

    Defaults are deliberately conservative so spawning works out of the box
    without runaway recursion or cost: a top-level spawn is depth 1, a subagent
    spawning a subagent is depth 2, and so on up to ``recursion_depth``.
    """

    enabled: bool = True
    recursion_depth: int = 3  # max nesting; spawns are refused beyond this
    max_steps: int = 12  # max tool-call rounds per run (hard stop)
    token_budget: int = 100_000  # approx token ceiling per run (best-effort)
    max_concurrent: int = 3  # max background runs at once


class SubagentSummaryConfig(BaseModel):
    """Summarise a finished background subagent batch (issue #15).

    Instead of dumping a subagent's raw output to the chat and the agent's
    context, a small inference distils each batch into a one-sentence chat
    *notification* and a concise *digest* for the agent's context. Mirrors the
    other background inferences (memory / compaction / reflection).
    """

    enabled: bool = True
    provider: str = "deepseek"  # fast + cheap is ideal for this distillation
    model: str = "deepseek-v4-flash"
    thinking_level: str = ""  # "" (off) | "low" | "medium" | "high"


class Config(BaseModel):
    agent: AgentConfig = AgentConfig()
    channels: ChannelsConfig = ChannelsConfig()
    calendar: CalendarConfig = CalendarConfig()
    voice: VoiceConfig = VoiceConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    admin: AdminConfig = AdminConfig()
    history: HistoryConfig = HistoryConfig()
    memory: MemoryConfig = MemoryConfig()
    goal_decomposition: GoalDecompositionConfig = GoalDecompositionConfig()
    task_reflection: TaskReflectionConfig = TaskReflectionConfig()
    reply_decision: ReplyDecisionConfig = ReplyDecisionConfig()
    compaction: CompactionConfig = CompactionConfig()
    search: SearchConfig = SearchConfig()
    vision: VisionConfig = VisionConfig()
    you: YouConfig = YouConfig()
    prompt: PromptConfig = PromptConfig()
    tools: ToolsConfig = ToolsConfig()
    workspace: WorkspaceConfig = WorkspaceConfig()
    artifacts: ArtifactsConfig = ArtifactsConfig()
    subagents: SubagentsConfig = SubagentsConfig()
    subagent_summary: SubagentSummaryConfig = SubagentSummaryConfig()


def load_config(path: str | Path = "config.yml") -> Config:
    """Load and validate config from a YAML file.

    Loads .env file first, then resolves ${VAR_NAME} references in the
    YAML against environment variables.
    """
    load_dotenv()

    path = Path(path)
    if not path.exists():
        return Config()

    raw = yaml.safe_load(path.read_text()) or {}
    resolved = _resolve_env_vars(raw)
    return Config.model_validate(resolved)
