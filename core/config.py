"""Configuration loading and validation.

Loads settings from config.yml with environment variable interpolation.
Secrets (API keys, tokens) come from environment variables; structural
config (which channels are enabled, scheduler jobs, etc.) lives in YAML.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


def _resolve_env_vars(obj: object) -> object:
    """Recursively resolve ${ENV_VAR} references in strings."""
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


# --- Config models ---


class AgentConfig(BaseModel):
    name: str = "Clio"
    owner_name: str = "Matteo"
    llm_provider: str = "anthropic"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = ""
    google_api_key: str = ""
    google_base_url: str = ""
    grok_api_key: str = ""
    grok_base_url: str = ""
    deepseek_api_key: str = ""
    deepseek_base_url: str = ""
    model: str = "claude-4-6-sonnet"
    timezone: str = "Europe/Zurich"
    skills_dir: str = "skills/"
    skills_db_path: str = "data/skills.db"
    character: str = ""
    personalia: str = ""


class TelegramConfig(BaseModel):
    enabled: bool = False
    bot_token: str = ""
    allowed_user_ids: list[int] = Field(default_factory=list)

    @field_validator("allowed_user_ids", mode="before")
    @classmethod
    def parse_comma_separated_ints(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return [int(x.strip()) for x in v.split(",") if x.strip()] if v else []
        return v


class WhatsAppConfig(BaseModel):
    enabled: bool = False
    bridge_url: str = "local-wacli"
    allowed_numbers: list[str] = Field(default_factory=list)

    @field_validator("allowed_numbers", mode="before")
    @classmethod
    def parse_comma_separated_strings(cls, v):
        if isinstance(v, (int, float)):
            return [f"+{int(v)}"]
        if isinstance(v, str):
            v = v.strip()
            return [x.strip() for x in v.split(",") if x.strip()] if v else []
        return v


class ChannelsConfig(BaseModel):
    telegram: TelegramConfig = TelegramConfig()
    whatsapp: WhatsAppConfig = WhatsAppConfig()


class CalendarProvider(BaseModel):
    name: str
    url: str
    username: str = ""
    password: str = ""


class CalendarConfig(BaseModel):
    providers: list[CalendarProvider] = Field(default_factory=list)


class VoiceConfig(BaseModel):
    stt_model: str = "base"
    tts_voice: str = "en-US-AvaNeural"
    tts_enabled: bool = True


class SchedulerJob(BaseModel):
    id: str
    cron: str
    task: str
    channel: str = "telegram"
    type: str = "agent"


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


class MemoryConfig(BaseModel):
    db_path: str = "data/memory.db"
    long_term_limit: int = 50
    extraction_provider: str = "anthropic"
    extraction_model: str = "claude-haiku-4-5"
    consolidation_provider: str = "anthropic"
    consolidation_model: str = "claude-haiku-4-5"
    extraction_cooldown_seconds: int = 120  # minimum seconds between extractions


class GoalDecompositionConfig(BaseModel):
    enabled: bool = True
    provider: str = "anthropic"
    model: str = "claude-haiku-4-5"


class TaskReflectionConfig(BaseModel):
    enabled: bool = True
    provider: str = "anthropic"
    model: str = "claude-haiku-4-5"
    db_path: str = "data/reflections.db"
    max_reflections: int = 50  # max reflections to keep for prompt injection


class CompactionConfig(BaseModel):
    """Conversation compaction — summarise old turns when the context grows.

    Only applies in session history mode. The threshold is evaluated against
    the real token usage reported by the provider after each turn.
    """

    enabled: bool = True
    provider: str = "anthropic"
    model: str = "claude-haiku-4-5"
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


class YouConfig(BaseModel):
    personalia: str = ""


class PromptConfig(BaseModel):
    tool_usage_override: str = ""
    history_handling_override: str = ""


class GhToolConfig(BaseModel):
    """GitHub CLI (`gh`) tool — auth via a Personal Access Token."""

    enabled: bool = False
    token: str = ""  # GitHub PAT, injected as GH_TOKEN when running `gh`


class ToolsConfig(BaseModel):
    """Optional external CLI tools the agent can use (see core/tools.py)."""

    gh: GhToolConfig = GhToolConfig()


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
    compaction: CompactionConfig = CompactionConfig()
    search: SearchConfig = SearchConfig()
    you: YouConfig = YouConfig()
    prompt: PromptConfig = PromptConfig()
    tools: ToolsConfig = ToolsConfig()


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
