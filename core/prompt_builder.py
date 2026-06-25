"""System prompt builder shared by runtime and admin preview."""

from __future__ import annotations

from dataclasses import dataclass

from core.config import Config
from core.goal_decomposition import DecomposedGoal
from core.personae import Persona
from core.tools import active_tool_prompts

DEFAULT_TOOL_USAGE_BLOCK = """For write actions
(sending emails, replying to emails, sending messages, creating calendar events,
scheduling tasks), ALWAYS use the dedicated structured tools: `send_email`, `reply_email`,
`send_message`, `create_calendar_event`, `manage_jobs`. NEVER use `run_command` for these — the
structured tools handle quoting, piping, and permissions correctly.

For scheduling, use the `manage_jobs` tool to create, list, and cancel jobs. For more advanced
operations (editing jobs, pausing, viewing details), use the `jobs.py` CLI via `run_command`
after loading the `scheduling` skill.

Use `run_command` only for read/query operations (listing emails, reading messages, searching,
managing flags/folders, contacts, memory, etc.).
Always use the skill documentation to construct the correct command.
If you don't have the skill content in context, call `load_skill` with the skill name to load it.
Parse JSON output when available (himalaya supports -o json, sqlite3 supports -json).
If a command fails, read the error and try to fix it.
Never guess at command syntax — always refer to the skill file.

You may create or update skills using the `skills.py` CLI
after loading the `skill-creator` skill."""

DEFAULT_HISTORY_HANDLING_BLOCK = """Previous messages in this conversation
have already been handled.
Always focus exclusively on the latest user message as the current, active request.
Use earlier messages only to understand context, resolve references (e.g. "that", "it",
"the one I mentioned"), and maintain conversational continuity."""


def resolve_prompt_block(default_text: str, override_text: str | None) -> str:
    """Resolve a prompt block, using override when non-empty."""
    if override_text and override_text.strip():
        return override_text.strip()
    return default_text


@dataclass(slots=True)
class PromptSections:
    intro: str
    personalia: str
    character: str
    about_user: str
    tool_usage: str
    tools: str
    memory_instruction: str
    history_handling: str
    memories: str
    available_skills: str
    task_reflections: str
    execution_plan: str

    @property
    def full_prompt(self) -> str:
        parts = [
            self.intro,
            self.personalia,
            self.character,
            self.about_user,
            self.tool_usage,
        ]
        if self.tools:
            parts.append(self.tools)
        parts.append(self.memory_instruction)
        if self.history_handling:
            parts.append(self.history_handling)
        if self.memories:
            parts.append(self.memories)
        if self.available_skills:
            parts.append(self.available_skills)
        if self.task_reflections:
            parts.append(self.task_reflections)
        if self.execution_plan:
            parts.append(self.execution_plan)
        return "\n\n".join(p.strip("\n") for p in parts if p)

    def as_dict(self) -> dict[str, str]:
        return {
            "intro": self.intro,
            "personalia": self.personalia,
            "character": self.character,
            "about_user": self.about_user,
            "tool_usage": self.tool_usage,
            "tools": self.tools,
            "memory_instruction": self.memory_instruction,
            "history_handling": self.history_handling,
            "memories": self.memories,
            "available_skills": self.available_skills,
            "task_reflections": self.task_reflections,
            "execution_plan": self.execution_plan,
        }


def build_prompt_sections(
    *,
    config: Config,
    history_mode: str,
    skills_index: str,
    memories: str,
    reflections: str,
    decomposed_goal: DecomposedGoal | None,
    persona: Persona | None = None,
    include_memories: bool = True,
    include_reflections: bool = True,
) -> PromptSections:
    """Build all prompt sections with current config and dynamic context.

    The prompt is intentionally **static** (no current date/time): it forms the
    cacheable prefix sent to the LLM. The live date/time is injected per turn at
    the start of each user message instead (see ``AgentCore._turn_preamble``).
    """
    cfg = config.agent

    about_user_block = config.you.personalia.strip()
    tool_usage_text = resolve_prompt_block(
        DEFAULT_TOOL_USAGE_BLOCK,
        getattr(config.prompt, "tool_usage_override", ""),
    )
    history_handling_text = resolve_prompt_block(
        DEFAULT_HISTORY_HANDLING_BLOCK,
        getattr(config.prompt, "history_handling_override", ""),
    )

    # When a persona is active it supplies its own identity (personalia +
    # character); otherwise the configured defaults are used, so first-run
    # behaviour with no persona is unchanged.
    personalia_text = persona.personalia if persona else cfg.personalia
    character_text = persona.character if persona else cfg.character
    # A persona may go by its own name; otherwise the globally-configured name.
    agent_name = persona.agent_name if persona and persona.agent_name else cfg.name

    intro = (
        f"You are {agent_name}, a personal AI assistant for {cfg.owner_name}.\n\n"
        f"Your timezone is {cfg.timezone}. The current date and time is provided at the "
        f"start of each user message — always use that as 'now'."
    )
    if persona and persona.role:
        intro += f"\n\nYou are currently acting as the **{persona.role}** persona."

    personalia = f"<personalia>\n{personalia_text}\n</personalia>"
    character = f"<character>\n{character_text}\n</character>"
    about_user = f"<about_user>\n{about_user_block}\n</about_user>" if about_user_block else ""
    tool_usage = f"<tool_usage>\n{tool_usage_text}\n</tool_usage>"

    tool_blocks = active_tool_prompts(config)
    tools_section = ""
    if tool_blocks:
        tools_section = "<tools>\n" + "\n\n".join(tool_blocks) + "\n</tools>"
    memory_instruction = (
        "You can store and recall memories using the sqlite3 CLI (see the memory skill).\n"
        "Proactively remember important facts about the user and their contacts.\n"
        "Before inserting a new long-term memory, check if it already exists to avoid duplicates."
    )

    history_handling = ""
    if history_mode != "session":
        history_handling = f"<history_handling>\n{history_handling_text}\n</history_handling>"

    memory_section = ""
    if include_memories and memories:
        memory_section = f"<memories>\n{memories}\n</memories>"

    skills_section = ""
    if skills_index:
        skills_section = f"<available_skills>\n{skills_index}\n</available_skills>"

    reflections_section = ""
    if include_reflections and reflections:
        reflections_section = f"<task_reflections>\n{reflections}\n</task_reflections>"

    execution_plan = ""
    if decomposed_goal:
        execution_plan = (
            "<execution_plan>\n"
            "The user's request has been analysed and broken into the following sub-goals.\n"
            "Follow this plan step-by-step, completing each sub-goal in order (respecting\n"
            "dependencies). Report progress as you go.\n\n"
            f"{decomposed_goal.format_for_prompt()}\n"
            "</execution_plan>"
        )

    return PromptSections(
        intro=intro,
        personalia=personalia,
        character=character,
        about_user=about_user,
        tool_usage=tool_usage,
        tools=tools_section,
        memory_instruction=memory_instruction,
        history_handling=history_handling,
        memories=memory_section,
        available_skills=skills_section,
        task_reflections=reflections_section,
        execution_plan=execution_plan,
    )
