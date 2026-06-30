# Personal Agent System — Architecture Design

## 1. Overview

A lightweight, self-hosted personal AI agent that runs in a single Docker container on a VPS. The agent acts as a unified interface across messaging channels, email, and calendars — capable of autonomous action, scheduled tasks, and voice interaction.

### Design Principles

- **Single container** — everything runs in one Docker image, orchestrated by a single Python process
- **Python orchestrator + CLI tools** — Python glues everything together; battle-tested CLI tools handle protocol complexity (IMAP, CalDAV, CardDAV)
- **Skills over code** — instead of hardcoded integrations, the LLM learns to use CLI tools via markdown "skill" files, making the system easy to extend
- **SQLite for storage** — no database server, just files on disk accessed via `sqlite3` CLI
- **Two-tier memory** — long-term memories (permanent facts) and short-term context (sliding window with configurable TTL), both stored in SQLite and queried by the LLM via skill files
- **Character** — agent identity and personality are defined in an editable `character.md` (this absorbed the former separate `personalia.md` identity file)
- **Explicit permissions** — nothing happens without your approval (or a pre-approved rule)

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Docker Container                           │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                      Agent Core                            │   │
│  │                                                            │   │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐  │   │
│  │  │  Brain    │  │  Memory  │  │  Permission Engine     │  │   │
│  │  │  (LLM)   │  │ (SQLite) │  │                        │  │   │
│  │  └────┬─────┘  └────┬─────┘  └───────────┬────────────┘  │   │
│  │       │              │                    │               │   │
│  │  ┌────┴──────────────┴────────────────────┴────────┐      │   │
│  │  │              Skills Engine                       │      │   │
│  │  │  Loads markdown skill files into LLM context     │      │   │
│  │  │  to teach it how to use each CLI tool            │      │   │
│  │  └──────────────────────┬──────────────────────────┘      │   │
│  │                         │                                  │   │
│  │              ┌──────────┴──────────┐                       │   │
│  │              │   Tool Executor     │                       │   │
│  │              │  (subprocess.run)   │                       │   │
│  │              └──────────┬──────────┘                       │   │
│  └─────────────────────────┼─────────────────────────────────┘   │
│                            │                                      │
│    ┌───────────┬───────────┼───────────┬─────────────┐           │
│    │           │           │           │             │           │
│    ▼           ▼           ▼           ▼             ▼           │
│  ┌─────┐  ┌───────┐  ┌─────────┐  ┌────────┐  ┌──────────┐    │
│  │ TG  │  │  WA   │  │Himalaya │  │ CalDAV │  │Scheduler │    │
│  │ Bot │  │Bridge │  │  (CLI)  │  │  (py)  │  │  (APS)   │    │
│  └─────┘  └───────┘  │         │  │        │  └──────────┘    │
│                       │ email   │  │calendar│                   │
│                       │ read    │  │contacts│                   │
│                       │ send    │  │ contacts CLI │           │
│                       │ search  │  │        │                   │
│                       └─────────┘  └────────┘                   │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                    Voice Pipeline                           │   │
│  │       STT (Whisper)  ◄──►  Agent  ◄──►  TTS               │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                Admin API (FastAPI)                          │   │
│  │     /health  /permissions  /memory  /config  /logs         │   │
│  └───────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Language | **Python 3.12** | Orchestrator; simplest for newcomers |
| LLM | **Claude API** (Anthropic) | Best reasoning, tool-use support, long context |
| Telegram | **python-telegram-bot** | Mature, async-native, well-documented |
| WhatsApp | **whatsapp-web.js** via bridge OR **Twilio** | See §6.2 for tradeoffs |
| Email | **Himalaya CLI** (Rust binary) | Stateless CLI, JSON output, multi-account, OAuth2 |
| Calendar | **python-caldav** | Stable Python lib, CalDAV is a simple protocol |
| Contacts | **contacts CLI** | Direct provider API (Google People + CardDAV), JSON output |
| Voice → Text | **faster-whisper** | Fast, local, offline-capable STT |
| Text → Voice | **edge-tts** or **Coqui TTS** | Free, no API key needed |
| Scheduler | **APScheduler** | Cron-like scheduling, persistent job store |
| Database | **SQLite** via **sqlite3 CLI** | Zero config, single file, LLM queries it directly via skill file |
| Web / Admin | **FastAPI** | Modern, auto-docs, async, easy to learn |
| Container | **Docker** | Single `docker compose up` to run everything |

### Why CLI Tools Over Python Libraries?

The traditional approach is to write IMAP/SMTP/CardDAV code directly in Python. The CLI approach inverts this:

| Concern | Python Library | CLI Tool |
|---|---|---|
| Protocol complexity | You own it (IMAP quirks, OAuth flows, connection pooling) | The tool owns it |
| Auth management | Implement per-provider | Himalaya + contacts CLI handle it (OAuth2/app passwords) |
| Configuration | Scattered across Python code | One TOML file per tool |
| Debugging | Step through Python code | Run the CLI command manually in your terminal |
| Teaching the LLM | Hardcoded tool schemas | Markdown skill files the LLM reads at runtime |
| Adding a new tool | Write a new Python integration class | Write a new skill markdown file |

The agent becomes a **thin orchestrator**: it reads skill files, passes them to the LLM as context, and executes the CLI commands the LLM constructs. Python handles the parts that benefit from it (CalDAV, which is a simple protocol; async orchestration; the Telegram bot), and CLI tools handle the rest.

---

## 4. The Skills System

This is the central design idea. Instead of hardcoding how each integration works, the agent loads **skill docs** — markdown documents stored in SQLite that teach the LLM how to use each CLI tool. Seed files from `skills/` are inserted into the DB at startup. Skills are injected into the system prompt at runtime based on which tools are available.

### 4.1 How It Works

```python
# core/skills.py
from pathlib import Path

class SkillsEngine:
    """Loads and manages skill files that teach the LLM to use CLI tools."""

    def __init__(self, skills_dir: str = "skills/"):
        self.skills_dir = Path(skills_dir)
        self.skills: dict[str, str] = {}
        self._load_all()

    def _load_all(self):
        for skill_file in self.skills_dir.glob("*.md"):
            self.skills[skill_file.stem] = skill_file.read_text()

    def get_skill(self, name: str) -> str | None:
        return self.skills.get(name)

    def get_all_skills(self) -> str:
        """Concatenate all skills into a single context block."""
        sections = []
        for name, content in self.skills.items():
            sections.append(f"<skill name=\"{name}\">\n{content}\n</skill>")
        return "\n\n".join(sections)

    def get_skills_for_tools(self, tool_names: list[str]) -> str:
        """Get only the skills relevant to the active tools."""
        sections = []
        for name in tool_names:
            if name in self.skills:
                sections.append(f"<skill name=\"{name}\">\n{self.skills[name]}\n</skill>")
        return "\n\n".join(sections)
```

### 4.2 System Prompt Integration

Skills, character, and memories all get injected into the system prompt:

```python
# core/agent.py (excerpt)

def _build_system_prompt(self, user_context: str) -> str:
    skills_block = self.skills.get_all_skills()
    character = self.config.agent.character
    memories = self.memory.format_for_prompt()

    return f"""You are {self.config.agent.name}, a personal AI assistant for {self.config.agent.owner_name}.

Today is {datetime.now().strftime('%A, %B %d, %Y')}. Timezone: {self.config.agent.timezone}.

<character>
{character}
</character>

<memories>
{memories}
</memories>

<available_skills>
{skills_block}
</available_skills>

When you need to perform an action, use the `run_command` tool to execute CLI commands.
Always use the skill documentation to construct the correct command.
Parse JSON output when available (himalaya supports -o json, sqlite3 supports -json).
If a command fails, read the error and try to fix it.
Never guess at command syntax — always refer to the skill file.

You can store and recall memories using the sqlite3 CLI (see the memory skill).
Proactively remember important facts about the user and their contacts.
Before inserting a new long-term memory, check if it already exists to avoid duplicates."""
```

### 4.3 The Generic Tool Executor

Instead of many specific tools, the agent gets one powerful meta-tool — `run_command` — plus a few structured tools for safety-critical actions:

```python
# core/executor.py
import subprocess, json, shlex

class ToolExecutor:
    """Executes CLI commands on behalf of the LLM."""

    # Commands the agent is allowed to run (prefix whitelist)
    ALLOWED_PREFIXES = [
        "himalaya",
        "python3 /app/tools/contacts.py",
        "sqlite3",
        "python3 /app/tools/",
    ]

    async def run_command(self, command: str, timeout: int = 30) -> dict:
        """Execute a shell command and return its output."""
        # Security: validate against whitelist
        parts = shlex.split(command)
        if not any(command.startswith(p) for p in self.ALLOWED_PREFIXES):
            return {"error": f"Command not in the allowed list"}

        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=timeout
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"error": f"Command timed out after {timeout}s"}

    def parse_json_output(self, output: str) -> list | dict:
        """Parse JSON output from CLI tools (himalaya -o json, etc.)."""
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"raw": output}
```

### 4.4 Tool Definitions for the LLM

The LLM gets both the generic `run_command` and structured tools for actions that need permission gating:

```python
def _build_tool_definitions(self) -> list[dict]:
    return [
        # Generic CLI executor — the LLM constructs commands using skill knowledge
        {
            "name": "run_command",
            "description": "Execute a CLI command. Use skill documentation to construct correct syntax. Returns stdout, stderr, and exit_code.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The full CLI command to run"},
                    "purpose": {"type": "string", "description": "Brief explanation of what this command does (for permission checking and audit logging)"},
                },
                "required": ["command", "purpose"]
            }
        },
        # Structured tools for permission-gated write actions
        {
            "name": "send_email",
            "description": "Send an email on behalf of the user. Requires approval.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "account": {"type": "string"},
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["account", "to", "subject", "body"]
            }
        },
        {
            "name": "send_message",
            "description": "Send a message to a contact via Telegram or WhatsApp. Requires approval.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string", "enum": ["telegram", "whatsapp"]},
                    "to": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["channel", "to", "text"]
            }
        },
        {
            "name": "create_calendar_event",
            "description": "Create a calendar event or send an invite. Requires approval.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "calendar": {"type": "string"},
                    "summary": {"type": "string"},
                    "start": {"type": "string", "description": "ISO datetime"},
                    "end": {"type": "string", "description": "ISO datetime"},
                    "attendees": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["calendar", "summary", "start", "end"]
            }
        },
        # Safe read-only tools
        {"name": "web_search", "description": "Search the web", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
        {"name": "schedule_task", "description": "Schedule a one-time future task", "input_schema": {"type": "object", "properties": {"task": {"type": "string"}, "run_at": {"type": "string"}, "channel": {"type": "string"}}, "required": ["task", "run_at"]}},
        # Note: memory (remember/recall) is handled via run_command + sqlite3 CLI — see skills/memory.md
    ]
```

---

## 5. Skill Files

Each skill file is a self-contained markdown document that teaches the LLM how to use a specific CLI tool. These live in `skills/` and are loaded at startup.

### 5.1 `skills/himalaya-email.md`

```markdown
# Himalaya Email CLI

You have access to the `himalaya` CLI to manage emails. Himalaya is a stateless CLI
email client — each command is independent, no session state.

## Configuration

Himalaya is pre-configured with these accounts:
- `personal` — Alex's personal Gmail
- `work` — Alex's work Fastmail

Always specify the account with `-a <account_name>`.

## Reading Emails

### List recent emails (envelopes)
```bash
# List last 10 emails in INBOX (default folder)
himalaya -a personal envelope list -s 10 -o json

# List emails in a specific folder
himalaya -a work envelope list --folder "Archives" -s 20 -o json
```

The JSON output is an array of envelope objects:
```json
[
  {
    "id": "123",
    "subject": "Meeting tomorrow",
    "from": {"name": "Alice", "addr": "alice@example.com"},
    "date": "2025-02-17T10:30:00Z",
    "flags": ["Seen"]
  }
]
```

### Read a specific email
```bash
# Read email by ID (returns plain text body)
himalaya -a personal message read 123

# Read as raw MIME (useful for attachments)
himalaya -a personal message read 123 --raw
```

### Search emails
```bash
# Search by subject
himalaya -a work envelope list --folder INBOX -o json -- "subject:invoice"

# Search by sender
himalaya -a personal envelope list -o json -- "from:elena"

# Combined search
himalaya -a work envelope list -o json -- "from:nordicfurnishings subject:contract"
```

## Sending Emails

### Send a new email
```bash
# Using MML (MIME Meta Language) format via stdin
echo 'From: alex@example.com
To: recipient@example.com
Subject: Hello from the agent

This is the body of the email.' | himalaya -a personal message send
```

### Reply to an email
```bash
# Pipe the reply body to the reply command
echo 'Thank you for your email.

Best regards,
Alex' | himalaya -a personal message reply 123
```

### Forward an email
```bash
himalaya -a work message forward 123
```

## Managing Emails

```bash
# Move to folder
himalaya -a personal message move 123 "Archives"

# Delete
himalaya -a personal message delete 123

# Flag/unflag
himalaya -a personal flag add 123 Seen
himalaya -a personal flag remove 123 Seen

# List folders
himalaya -a personal folder list -o json
```

## Important Notes
- Always use `-o json` when you need to parse results programmatically
- Email IDs are relative to the current folder — always specify --folder when not using INBOX
- For multi-line email bodies, construct the full MML template and pipe it via echo
- The `personal` account is for personal correspondence, `work` for professional
```

### 5.2 `skills/contacts.md`

```markdown
# Contacts CLI

Use the contacts helper script to query providers directly (Google + CardDAV).
Providers are configured in the admin UI (Contacts tab). Avoid hardcoded names.

## List all contacts

```bash
python3 /app/tools/contacts.py list --provider <NAME> --output json
```

## Search by name, phone, or email

```bash
python3 /app/tools/contacts.py search --provider <NAME> --query "Marco" --output json
```

## Get full details for a specific contact

```bash
python3 /app/tools/contacts.py get --provider <NAME> --id <CONTACT_ID> --output json
```

## Output format (JSON)

```json
[
  {"id": "...", "full_name": "Marco Rossi", "emails": ["marco@example.com"], "phones": ["+49 151 5551234"]}
]
```

## Important notes

- If multiple contacts match, present options and ask which to use.
```

### 5.3 `skills/caldav-calendar.md`

```markdown
# Calendar Management (CalDAV)

Calendar operations use helper scripts that wrap Python's caldav library.

## Available Calendars
- `google` — Alex's Google Calendar (primary, work events)
- `icloud` — Shared family calendar

## Reading Events
```bash
# Get today's events
python3 /app/tools/calendar_read.py --calendar google --today -o json

# Get events for a date range
python3 /app/tools/calendar_read.py --calendar google --from 2025-02-17 --to 2025-02-24 -o json

# Get next N events
python3 /app/tools/calendar_read.py --calendar google --next 5 -o json
```

JSON output:
```json
[
  {
    "uid": "abc123",
    "summary": "Team standup",
    "start": "2025-02-17T09:00:00+01:00",
    "end": "2025-02-17T09:30:00+01:00",
    "location": "Google Meet",
    "attendees": ["alice@nordicfurnishings.com", "bob@nordicfurnishings.com"]
  }
]
```

## Creating Events
Use the `create_calendar_event` structured tool (requires permission). Provide:
- `calendar`: "google" or "icloud"
- `summary`: event title
- `start`: ISO datetime with timezone (e.g. "2025-02-20T14:00:00+01:00")
- `end`: ISO datetime with timezone
- `attendees`: optional list of email addresses (sends invites automatically)

## Important Notes
- Always include timezone (Europe/Berlin = UTC+1, UTC+2 during DST)
- For all-day events, use date only: "2025-02-20"
- Use `google` calendar for work events, `icloud` for family/personal
```

### 5.4 `skills/voice.md`

```markdown
# Voice Interaction

## Receiving Voice Messages
Voice messages are automatically transcribed using Whisper before being passed to you.
You see the transcript as regular text, with a `[voice]` prefix.

## Sending Voice Responses
Add `[respond_with_voice]` at the end of your response to trigger TTS. Tag the reply's language as an ISO-639-1 code after a colon (`[respond_with_voice:it]`) so the audio uses the right pronunciation; a bare marker falls back to the configured voice's language.

Use voice responses when:
- The user sent a voice message (mirror the medium)
- The user explicitly asks for a voice reply
- The response is short and conversational (< 3 sentences)

Do NOT use voice responses when:
- The response contains code, links, or structured data
- The response is long or complex
```

### 5.5 `character.md` — Agent Character (Editable)

A top-level markdown file (not in `skills/`) that defines the agent's personality, tone, and behavioral rules. This file is **freely editable** — you can change the agent's character at any time by modifying this file. It is loaded into the system prompt on every conversation turn.

```markdown
# character.md

## Personality
- Be concise in chat. Telegram/WhatsApp messages should be short and direct.
- When acting on Alex's behalf (sending emails, messages), match his communication
  style: professional but warm, slightly informal with close contacts.
- Write messages in first person as if from Alex. Do not add assistant signatures
  unless Alex explicitly requests it.
- When unsure about an action, ask. When confident and pre-approved, just do it.

## Contact Resolution
When Alex refers to someone by first name:
1. Look up the contact using the contacts CLI
2. If multiple matches, ask which one
3. Use the contact's preferred channel (check notes field for preferences)

## Language
- Default to English
- Switch to Italian when Alex speaks Italian or when messaging Italian contacts
- Use German for formal correspondence if appropriate

## Proactive Behaviors (Scheduled Tasks)
When running scheduled tasks (morning briefing, email checks), be:
- Brief and scannable
- Only flag truly important items
- Group related information together
```

### 5.5.1 Agent Identity (part of `character.md`)

The agent's fixed identity attributes — name, owner, capabilities, strengths, and other facts that don't change — live in `character.md` under headings like `## Identity`. (These were once a separate, append-only `personalia.md`; that file was merged into `character.md` so identity and behaviour live in one place — a legacy `personalia` key is still folded in on load for backward compatibility.)

`character.md` therefore covers both _what_ the agent is and _how_ it behaves. Example identity sections:

```markdown
## Identity
- Name: Jarvis
- Owner: Alex
- Role: Personal AI assistant

## Strengths
- Email management across multiple accounts (personal Gmail, work Fastmail)
- Calendar awareness and scheduling (Google Calendar, iCloud)
- Contact resolution and cross-channel messaging (Telegram, WhatsApp)
- Voice interaction (understands voice messages, can respond with voice)
- Proactive daily briefings and monitoring

## Capabilities
- Can read, search, and send emails via Himalaya CLI
- Can look up contacts via contacts CLI
- Can read and create calendar events via CalDAV
- Can send messages on Telegram and WhatsApp
- Can transcribe voice messages and respond with synthesized speech
- Can schedule future tasks and reminders
- Can remember facts long-term and track short-term context

## Limitations
- Cannot make phone calls
- Cannot access websites or browse the internet (except via web_search)
- Cannot access files on Alex's personal devices
- Always needs permission before sending messages or emails on Alex's behalf

## History
- 2025-02-17: Initial deployment with email, calendar, contacts, messaging, and voice support
```

### 5.5.2 How character.md Differs from Skills

| | Skills (`skills/*.md`) | `character.md` |
|---|---|---|
| **Purpose** | Teach the LLM how to use a specific CLI tool | Define identity, personality, and behavioral rules |
| **Mutability** | Replaced when tool changes | Freely editable at any time |
| **Loaded** | Into `<available_skills>` block | Into `<character>` block |
| **Example content** | "Run `himalaya -a personal envelope list`" | "Name: Jarvis" / "Be concise in chat" |

### 5.6 `skills/memory.md`

```markdown
# Memory System (sqlite3)

You have access to a SQLite database at `/app/data/memory.db` via the `sqlite3` CLI.
This database stores your memories in two tiers:

## Database Schema

### long_term — permanent memories
Columns: id, category, subject, content, source, confidence, created_at, updated_at

Categories: "preference", "relationship", "fact", "routine", "work", "health", "travel"

### short_term — temporary context
Columns: id, content, context, expires_at, created_at

## Storing Memories

### Long-term memory (things that stay true)
```bash
sqlite3 /app/data/memory.db "INSERT INTO long_term (category, subject, content, source) VALUES ('preference', 'alex', 'Allergic to shellfish', 'conversation');"
```

### Short-term fact (temporary context, default 24h expiry)
```bash
sqlite3 /app/data/memory.db "INSERT INTO short_term (content, context, expires_at) VALUES ('Working from home today', 'morning chat', datetime('now', '+24 hours'));"
```

### Update an existing memory
```bash
sqlite3 /app/data/memory.db "UPDATE long_term SET content = 'New value', updated_at = datetime('now') WHERE id = 42;"
```

## Querying Memories

### Search by subject
```bash
sqlite3 -json /app/data/memory.db "SELECT * FROM long_term WHERE subject = 'alex';"
```

### Search by category
```bash
sqlite3 -json /app/data/memory.db "SELECT * FROM long_term WHERE category = 'preference';"
```

### Full-text search
```bash
sqlite3 -json /app/data/memory.db "SELECT * FROM long_term WHERE content LIKE '%coffee%';"
```

### Get all active short-term facts
```bash
sqlite3 -json /app/data/memory.db "SELECT * FROM short_term WHERE expires_at > datetime('now');"
```

### Get all long-term memories (summary)
```bash
sqlite3 -json /app/data/memory.db "SELECT id, category, subject, content FROM long_term ORDER BY updated_at DESC;"
```

## Deleting Memories

### Delete a specific memory by ID
```bash
sqlite3 /app/data/memory.db "DELETE FROM long_term WHERE id = 42;"
```

### Delete a short-term fact
```bash
sqlite3 /app/data/memory.db "DELETE FROM short_term WHERE id = 7;"
```

## Important Notes
- Always use `-json` flag when you need to parse results programmatically
- Use LIKE with % wildcards for fuzzy content search
- For long-term memories, always set `category` and `subject` — these are used for filtering
- Short-term facts are auto-cleaned every 8 hours; set `expires_at` appropriately
- When you learn something new that contradicts an existing memory, UPDATE the old one rather than inserting a duplicate
- Before inserting a long-term memory, check if a similar one already exists to avoid duplicates
- Use `source = 'conversation'` for things the user told you, `source = 'inferred'` for things you deduced
```

### 5.7 Adding New Skills

To add any new capability:

1. Install the CLI tool in the Dockerfile
2. Add its prefix to `ALLOWED_PREFIXES` in `executor.py`
3. Write a `skills/tool-name.md` file teaching the LLM how to use it
4. (Optional) Add permission rules for write operations
5. Done — no Python integration code needed

Example: to add GitHub, install `gh`, add `"gh"` to prefixes, write `skills/github.md`:

```markdown
# skills/github.md
# GitHub CLI (gh)

## Check notifications
```bash
gh api notifications --jq '.[].subject.title'
```

## Create an issue
```bash
gh issue create --repo owner/repo --title "Bug" --body "Description"
```
```

The agent can now manage GitHub immediately — no Python code changes.

---

## 6. Module Design

### 6.1 Channel: Telegram

The primary channel. Telegram bots are free, have excellent API support, and handle text, voice, files, and inline keyboards natively.

```python
# channels/telegram.py
from telegram import Update
from telegram.ext import Application, MessageHandler, filters

class TelegramChannel:
    def __init__(self, token: str, agent_core):
        self.app = Application.builder().token(token).build()
        self.agent = agent_core
        self.app.add_handler(MessageHandler(filters.TEXT, self.on_text))
        self.app.add_handler(MessageHandler(filters.VOICE, self.on_voice))

    async def on_text(self, update: Update, context):
        user_id = update.effective_user.id
        if not self.agent.permissions.is_allowed_user(user_id):
            return
        response = await self.agent.process(update.message.text, channel="telegram", user_id=user_id)
        await update.message.reply_text(response.text)
        if response.voice:
            await update.message.reply_voice(response.voice)

    async def on_voice(self, update: Update, context):
        voice_file = await update.message.voice.get_file()
        audio_bytes = await voice_file.download_as_bytearray()
        transcript = await self.agent.voice.transcribe(audio_bytes)
        response = await self.agent.process(f"[voice] {transcript}", channel="telegram", ...)

    async def send(self, chat_id: str, text: str, voice: bytes = None):
        await self.app.bot.send_message(chat_id=chat_id, text=text)
        if voice:
            await self.app.bot.send_voice(chat_id=chat_id, voice=voice)
```

### 6.2 Channel: WhatsApp

| Option | Pros | Cons |
|---|---|---|
| **Twilio WhatsApp API** | Official, reliable, simple REST API | Costs money (~$0.005/msg), requires business verification |
| **wacli** (Go CLI) | Local, no Node sidecar, small footprint | Unofficial, can get banned, uses WhatsApp Web |

Use the local wacli CLI with a minimal admin API integration:

```python
# channels/whatsapp.py
import httpx

class WhatsAppChannel:
    def __init__(self, agent_core):
        self.agent = agent_core

    async def on_message(self, payload: dict):
        text = payload["body"]
        sender = payload["from"]
        response = await self.agent.process(text, channel="whatsapp", user_id=sender)
        await self.send(sender, response.text)

    async def send(self, to: str, text: str, voice: bytes = None):
        async with httpx.AsyncClient() as client:
            await client.post(
                "http://localhost:8000/channels/whatsapp/send",
                json={"to": to, "text": text},
            )
```

```
wacli auth --json
wacli sync
```

### 6.3 Voice Pipeline

```python
# voice/pipeline.py
from faster_whisper import WhisperModel
import edge_tts, io

class VoicePipeline:
    def __init__(self, whisper_model: str = "base", tts_voice: str = "en-US-GuyNeural"):
        self.stt = WhisperModel(whisper_model, compute_type="int8")
        self.tts_voice = tts_voice

    async def transcribe(self, audio_bytes: bytes) -> str:
        segments, _ = self.stt.transcribe(io.BytesIO(audio_bytes))
        return " ".join(s.text for s in segments)

    async def synthesize(self, text: str) -> bytes:
        communicate = edge_tts.Communicate(text, self.tts_voice)
        buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.write(chunk["data"])
        return buffer.getvalue()
```

### 6.4 Scheduler

```python
# core/scheduler.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

class AgentScheduler:
    def __init__(self, db_path: str, agent_core):
        self.scheduler = AsyncIOScheduler(
            jobstores={"default": SQLAlchemyJobStore(url=f"sqlite:///{db_path}")}
        )
        self.agent = agent_core

    def load_jobs(self, config):
        """Register cron jobs from config. Three job types:
        - "agent": natural-language task → agent.process() → deliver to channel
        - "system": raw CLI command → executor.run_command_trusted()
        - "memory_consolidation": review short-term memories via LLM,
          promote worthy ones to long-term, delete expired entries
        """
        for job in config.jobs:
            cron_kwargs = _parse_cron(job.cron)

            if job.type == "system":
                self.scheduler.add_job(
                    self._run_system_command, "cron",
                    id=job.id, kwargs={"command": job.task},
                    replace_existing=True, **cron_kwargs,
                )
            elif job.type == "memory_consolidation":
                self.scheduler.add_job(
                    self._run_memory_consolidation, "cron",
                    id=job.id, replace_existing=True, **cron_kwargs,
                )
            else:
                self.scheduler.add_job(
                    self._run_agent_task, "cron",
                    id=job.id, kwargs={"task": job.task, "channel": job.channel},
                    replace_existing=True, **cron_kwargs,
                )

    def add_one_shot(self, job_id: str, run_at: datetime, task: str, channel: str):
        self.scheduler.add_job(
            self._run_agent_task, "date",
            id=job_id, run_date=run_at, kwargs={"task": task, "channel": channel},
            replace_existing=True,
        )

    async def _run_memory_consolidation(self):
        """Review short-term memories, promote worthy ones, delete expired."""
        result = await self.agent.memory.consolidate_and_cleanup(
            llm=self.agent.llm,
            model=self.agent.config.memory.consolidation_model,
        )
```

---

## 7. Agent Core (The Brain)

```python
# core/agent.py
from anthropic import AsyncAnthropic
from core.skills import SkillsEngine
from core.executor import ToolExecutor

class AgentCore:
    def __init__(self, config):
        self.config = config
        self.llm = AsyncAnthropic(api_key=config.anthropic_key)
        self.memory = MemoryStore(config.memory.db_path)
        self.permissions = PermissionEngine(config.db_path)
        self.skills = SkillsEngine(config.skills_dir)
        self.executor = ToolExecutor()
        self.channels = {}
        self.scheduler = AgentScheduler(config.db_path, self)
        self.voice = VoicePipeline()

    async def process(self, message: str, channel: str, user_id: str) -> AgentResponse:
        # 1. Load conversation history (stored in agent.db, separate from memories)
        conversation = await self._get_conversation(user_id, channel)

        # 2. Build tools and system prompt (skills, character, memories injected here)
        tools = self._build_tool_definitions()
        system = self._build_system_prompt()

        # 3. Call LLM
        response = await self.llm.messages.create(
            model="claude-sonnet-4-5-20250514",
            max_tokens=4096,
            system=system,
            messages=conversation + [{"role": "user", "content": message}],
            tools=tools,
        )

        # 4. Agentic loop — handle tool calls with permission checks
        while response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = await self._execute_tool(block, user_id, channel)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })
            response = await self.llm.messages.create(
                model="claude-sonnet-4-5-20250514",
                max_tokens=4096,
                system=system,
                messages=conversation + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response.content},
                    {"role": "user", "content": tool_results},
                ],
                tools=tools,
            )

        # 5. Save conversation turn + extract memories
        final_text = self._extract_text(response)
        await self._save_turn(user_id, channel, message, final_text)
        await self._extract_and_save_memories(message, final_text)

        # 6. Check if voice response requested
        voice_bytes = None
        if "[respond_with_voice]" in final_text:
            clean_text = final_text.replace("[respond_with_voice]", "").strip()
            voice_bytes = await self.voice.synthesize(clean_text)
            final_text = clean_text

        return AgentResponse(text=final_text, voice=voice_bytes)

    async def _execute_tool(self, tool_call, user_id, channel):
        action = tool_call.name
        params = tool_call.input

        if action == "run_command":
            # Check command-level permissions via glob patterns
            if not self.permissions.is_approved(user_id, "run_command", params):
                approved = await self._request_permission(user_id, channel, action, params)
                if not approved:
                    return {"error": "Permission denied"}
            return await self.executor.run_command(params["command"])

        elif action in ("send_email", "send_message", "create_calendar_event"):
            # Write actions always go through permission check
            if not self.permissions.is_approved(user_id, action, params):
                approved = await self._request_permission(user_id, channel, action, params)
                if not approved:
                    return {"error": "Permission denied"}
            return await self._dispatch_structured_tool(action, params)

        else:
            return await self._dispatch_structured_tool(action, params)
```

---

## 8. Memory System

The agent has a two-tier memory system accessed via the `sqlite3` CLI. This follows the same "skills over code" philosophy as the rest of the system — the LLM learns to query and write memories using the `skills/memory.md` skill file, and the Python orchestrator just runs the commands via `subprocess.run`. No Python ORM, no aiosqlite — just a SQLite database on disk and the `sqlite3` binary.

### 8.1 Memory Tiers

| Tier | Purpose | Lifetime | Examples |
|---|---|---|---|
| **Long-term** | Facts worth keeping forever | Permanent (never auto-deleted) | "Alex's wife is Elena", "Accountant's name is Dr. Novak", "Alex prefers window seats" |
| **Short-term** | Transient context worth keeping briefly | Configurable sliding window (default 24h), cleaned up periodically (default every 8h) | "Alex is at the airport right now", "Alex asked me to remind him about the report after lunch", "Elena is visiting her parents this weekend" |

The distinction: if it would still be useful next month, it's long-term. If it's situational context that will be stale in a day or two, it's short-term.

### 8.2 Schema

The database lives at `data/memory.db` (separate from `data/agent.db` for conversations/permissions). The schema is initialized on first startup:

```sql
-- data/memory.db

CREATE TABLE IF NOT EXISTS long_term (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,        -- e.g. "preference", "relationship", "fact", "routine"
    subject TEXT NOT NULL,         -- who/what this is about, e.g. "alex", "elena", "work"
    content TEXT NOT NULL,         -- the actual memory, natural language
    source TEXT,                   -- where this came from: "conversation", "email", "inferred"
    confidence TEXT DEFAULT 'stated',  -- "stated" (user said it), "inferred" (agent deduced it)
    created_at DATETIME DEFAULT (datetime('now')),
    updated_at DATETIME DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS short_term (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,         -- the fact, natural language
    context TEXT,                  -- why this was stored, e.g. "user mentioned during morning chat"
    expires_at DATETIME NOT NULL,  -- when this should be cleaned up
    created_at DATETIME DEFAULT (datetime('now'))
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_lt_category ON long_term(category);
CREATE INDEX IF NOT EXISTS idx_lt_subject ON long_term(subject);
CREATE INDEX IF NOT EXISTS idx_st_expires ON short_term(expires_at);
```

### 8.3 Access via sqlite3 CLI

The agent reads and writes memories by constructing `sqlite3` commands, taught by the `skills/memory.md` skill file (see §5.6). The `sqlite3` binary is added to `ALLOWED_PREFIXES` in the executor.

```bash
# Example: store a long-term memory
sqlite3 /app/data/memory.db "INSERT INTO long_term (category, subject, content, source) VALUES ('preference', 'alex', 'Prefers oat milk in coffee', 'conversation');"

# Example: query long-term memories about a subject
sqlite3 -json /app/data/memory.db "SELECT * FROM long_term WHERE subject = 'alex' AND category = 'preference';"

# Example: store a short-term fact (expires in 24h)
sqlite3 /app/data/memory.db "INSERT INTO short_term (content, context, expires_at) VALUES ('Alex is working from home today', 'mentioned in morning chat', datetime('now', '+24 hours'));"

# Example: search memories by content
sqlite3 -json /app/data/memory.db "SELECT * FROM long_term WHERE content LIKE '%coffee%';"

# Example: update a long-term memory
sqlite3 /app/data/memory.db "UPDATE long_term SET content = 'Prefers almond milk in coffee', updated_at = datetime('now') WHERE id = 42;"
```

### 8.4 Memory Consolidation & Cleanup

A scheduled job of type `memory_consolidation` runs on a configurable cron schedule (default: every 8 hours). It does two things:

1. **Consolidation** — reviews all active (non-expired) short-term memories via a lightweight LLM call, and promotes any that contain durable facts to long-term memory. The LLM compacts aggressively: strips temporal context, deduplicates against existing long-term memories, and only promotes facts that would still be useful weeks or months later.

2. **Cleanup** — deletes all expired short-term memories regardless of whether the LLM call succeeded.

#### Unified long-term write pipeline

Every long-term write — both the automatic per-turn extraction and consolidation's promotions — flows through a single path, `MemoryStore.update_memory`. For each candidate fact it retrieves the most lexically similar existing long-term memories (cheap token-overlap ranking, no embeddings or extra dependencies), then makes one LLM call that decides exactly one operation:

- **ADD** — genuinely new information.
- **UPDATE** — refines, corrects, or re-words one existing memory (the LLM returns the merged content to keep).
- **DELETE** — the candidate says an existing memory is no longer true, with nothing worth keeping in its place.
- **NOOP** — duplicate or not worth keeping.

The decision prompt includes each candidate's `created_at`/`updated_at`, so the model can prefer recent facts when resolving contradictions. When nothing similar exists the fact is added directly without an LLM call. This replaces the earlier exact-subject + substring + "longer content wins" heuristic, which missed semantic duplicates and never resolved contradictions. Malformed model output is a safe no-op.

This is configured as a regular scheduled job in `config.yml`:

```yaml
scheduler:
  jobs:
    - id: "memory_consolidation"
      cron: "0 */8 * * *"
      task: "memory_consolidation"
      type: "memory_consolidation"
```

The model used for the consolidation LLM call is configured in the `memory` section:

```yaml
memory:
  consolidation_model: "claude-haiku-4-5"
```

You can also trigger consolidation manually via the admin API: `POST /memory/consolidate`.

#### Semantic retrieval & relevance-ranked injection (Tier 2)

Each long-term memory gets a vector embedding, stored as a packed float32 blob in the `embedding` column. Similarity is brute-force cosine in Python (no native SQLite extension, so it behaves identically locally and in the container; trivial at <1k rows).

**Backends** (`memory.embedding.provider`):

- `local` (default) — runs a small on-device model via `fastembed` (`BAAI/bge-small-en-v1.5`, 384-dim, ~130MB ONNX/CPU). Private (memory never leaves the box), no API key, free. The model is **prefetched at Docker build** into `/app/models` (outside the data volume) so the first call has no download latency and the container works offline. The model loads lazily on first use, in a worker thread. A "Download model" button in the admin Memory tab (and `python -m core.embeddings prefetch`) fetches it on demand.
- `openai` / `google` (or any OpenAI-compatible `/embeddings` endpoint via `base_url`) — calls a remote API. Needs a key (falls back to the matching agent provider key); a few cents/year at typical volume, but memory text is sent to the provider. Note: DeepSeek has no embeddings endpoint.

Set `memory.embedding.enabled: false` to fall back to Tier-1 lexical (word-overlap) retrieval — still works, no model needed. All of this is configurable from the **admin Memory tab** (enable toggle, backend, model, top-k, download/test buttons) and applies live to the running agent.

With embeddings on:

- `update_memory` retrieves ADD/UPDATE/DELETE/NOOP candidates by cosine similarity (with a lexical fallback for any memory that has no vector yet).
- Prompt injection becomes **relevance-ranked**: instead of dumping the most recent `long_term_limit` rows, only the `injection_top_k` memories most relevant to the current message are injected, scored Generative-Agents style (relevance + importance + recency). The inbound message is threaded into prompt building as the query. In session mode (where the static prompt is snapshotted once), the first message of the session is used as the query.

#### Forgetting, importance & reinforcement (Tier 3)

`long_term` carries `importance` (1–10), `last_accessed`, `access_count`, and an `archived` flag. Recalled memories are reinforced (their `access_count`/`last_accessed` are bumped), and re-mentioning a fact (an UPDATE through the unified pipeline) raises its importance. The consolidation job archives **cold** memories — old, low-importance, and not accessed recently — via the `archived` flag (a soft delete, not a hard delete), so long-term memory stops growing without bound. Thresholds are configurable (`archive_after_days`, `archive_max_importance`, `archive_min_idle_days`).

#### Long-term hygiene pass (Tier 4)

Each consolidation run also clusters near-duplicate long-term memories (by embedding or lexical similarity, threshold `hygiene_similarity_threshold`) and resolves each cluster with one LLM call that merges duplicates and drops contradictions, keeping the most recent fact. This compacts memories that accumulated over time, not just at write time.

These behaviours are configured in the `memory` section (see `config.yml.example` for the full set):

```yaml
memory:
  embedding:
    enabled: true
    provider: "local"                # "local" | "openai" | "google"
    model: "BAAI/bge-small-en-v1.5"  # local model; API e.g. text-embedding-3-small
    cache_dir: "models"              # local model store (bundled in the image)
    injection_top_k: 12
  default_importance: 5.0
  archive_after_days: 90
  hygiene_enabled: true
```

The consolidation summary returned by `consolidate_and_cleanup` reports `active_reviewed`, `promoted_to_long_term`, `expired_deleted`, `hygiene_merged`, and `archived`. The new columns are added to existing databases by an in-place `ALTER TABLE` migration in `MemoryStore._ensure_schema`, so upgrading needs no manual steps.

### 8.5 Memory in the Agent Loop

On each conversation turn, the orchestrator queries both memory tiers and injects them into the system prompt as context. This happens before the LLM call:

```python
# core/memory.py
import subprocess, json

class MemoryStore:
    def __init__(self, db_path: str = "data/memory.db"):
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self):
        """Run schema creation on startup."""
        schema = open("schema/memory.sql").read()
        subprocess.run(["sqlite3", self.db_path], input=schema, text=True, check=True)

    def get_long_term_context(self, limit: int = 50) -> list[dict]:
        """Retrieve long-term memories for system prompt injection."""
        result = subprocess.run(
            ["sqlite3", "-json", self.db_path,
             f"SELECT category, subject, content FROM long_term ORDER BY updated_at DESC LIMIT {limit};"],
            capture_output=True, text=True
        )
        return json.loads(result.stdout) if result.stdout.strip() else []

    def get_short_term_context(self) -> list[dict]:
        """Retrieve active (non-expired) short-term memories."""
        result = subprocess.run(
            ["sqlite3", "-json", self.db_path,
             "SELECT content, context FROM short_term WHERE expires_at > datetime('now') ORDER BY created_at DESC;"],
            capture_output=True, text=True
        )
        return json.loads(result.stdout) if result.stdout.strip() else []

    def format_for_prompt(self) -> str:
        """Format both tiers into a context block for the system prompt."""
        sections = []

        long_term = self.get_long_term_context()
        if long_term:
            lines = [f"- [{m['category']}] {m['subject']}: {m['content']}" for m in long_term]
            sections.append("## Long-term memories\n" + "\n".join(lines))

        short_term = self.get_short_term_context()
        if short_term:
            lines = [f"- {m['content']}" + (f" ({m['context']})" if m.get('context') else "") for m in short_term]
            sections.append("## Current context (short-term)\n" + "\n".join(lines))

        return "\n\n".join(sections) if sections else "No memories stored yet."

    def cleanup_expired(self):
        """Delete expired short-term memories. Called by scheduler."""
        subprocess.run(
            ["sqlite3", self.db_path, "DELETE FROM short_term WHERE expires_at < datetime('now');"],
            check=True
        )
```

### 8.6 When the Agent Stores Memories

The agent decides when to store memories as part of its normal reasoning. After each conversation turn, the orchestrator can optionally run a lightweight "memory extraction" LLM call:

```python
async def extract_and_save_memories(self, user_msg: str, agent_msg: str):
    """Ask a fast model to identify facts worth remembering from the conversation."""
    prompt = f"""Given this conversation exchange, identify any facts worth remembering.

User: {user_msg}
Assistant: {agent_msg}

For each fact, classify it:
- LONG_TERM: preferences, relationships, routines, biographical facts — things that stay true
- SHORT_TERM: situational context, temporary states, time-bound info — things that expire

Return a JSON array. Example:
[
  {{"tier": "LONG_TERM", "category": "preference", "subject": "alex", "content": "Prefers window seats on flights"}},
  {{"tier": "SHORT_TERM", "content": "Has a dentist appointment at 17:30 today", "context": "mentioned in morning chat", "ttl_hours": 12}}
]

If nothing is worth remembering, return an empty array: []"""

    # Use a fast/cheap model for extraction
    response = await self.llm.messages.create(
        model="claude-haiku-4-20250514", max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    # Parse and store via sqlite3 CLI commands...
```

The agent can also store memories explicitly during tool use — the `skills/memory.md` skill file teaches it the exact `sqlite3` commands (see §5.6).

---

## 9. Permission System

```python
# core/permissions.py

class PermissionEngine:
    """
    Permission levels:
      ALWAYS — pre-approved, no confirmation needed
      ASK    — agent must ask before executing
      NEVER  — blocked entirely

    Rules use glob patterns for flexible matching.
    """

    DEFAULT_PERMISSIONS = {
        # Read operations — safe by default
        "run_command:himalaya*list*":                         "ALWAYS",
        "run_command:himalaya*read*":                         "ALWAYS",
        "run_command:himalaya*envelope*":                     "ALWAYS",
        "run_command:himalaya*folder*":                       "ALWAYS",
        "run_command:python3 /app/tools/contacts.py*":       "ALWAYS",
        "run_command:python3 /app/tools/calendar_read.py*":  "ALWAYS",
        "run_command:sqlite3*/app/data/memory.db*SELECT*":   "ALWAYS",  # memory reads
        "run_command:sqlite3*/app/data/memory.db*INSERT*":   "ALWAYS",  # memory writes
        "run_command:sqlite3*/app/data/memory.db*UPDATE*":   "ALWAYS",  # memory updates
        "run_command:sqlite3*/app/data/memory.db*DELETE*":   "ALWAYS",  # memory deletes
        "web_search":                                         "ALWAYS",

        # Write operations — ask first
        "send_email":                    "ASK",
        "send_message":                  "ASK",
        "create_calendar_event":         "ASK",
        "run_command:himalaya*send*":    "ASK",
        "run_command:himalaya*delete*":  "ASK",
        "run_command:himalaya*move*":    "ASK",
        "schedule_task":                 "ASK",

        # Dangerous memory operations — never allow schema changes
        "run_command:sqlite3*DROP*":     "NEVER",
        "run_command:sqlite3*ALTER*":    "NEVER",
    }
```

User can manage rules in chat:
```
You: "Always allow sending emails to elena@example.com"
Agent: ✅ Added rule: send_email to elena@example.com → ALWAYS
```

---

## 10. Project Structure

```
personal-agent/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── config.yml                  # Agent config (channels, scheduler, etc.)
├── character.md                # Agent identity, personality & behavior (editable)
│
├── core/
│   ├── agent.py                # AgentCore — the brain
│   ├── skills.py               # Skills engine — loads skill markdown files
│   ├── executor.py             # Tool executor — subprocess with whitelist
│   ├── memory.py               # Memory store — queries sqlite3 CLI, formats for prompt
│   ├── permissions.py          # Permission engine with glob patterns
│   ├── scheduler.py            # APScheduler wrapper
│   └── models.py               # Shared data models
│
├── channels/
│   ├── telegram.py             # Telegram bot channel
│   └── whatsapp.py             # WhatsApp channel (calls Node bridge)
│
├── skills/                     # ← THE KEY DIRECTORY
│   ├── himalaya-email.md       # Teaches LLM to use himalaya CLI
│   ├── contacts.md             # Teaches LLM to use contacts CLI
│   ├── caldav-calendar.md      # Teaches LLM to use calendar helpers
│   ├── memory.md               # Teaches LLM to use sqlite3 for memories
│   └── voice.md                # Voice interaction guidelines
│
├── schema/
│   └── memory.sql              # Memory DB schema (long_term + short_term tables)
│
├── tools/                      # Python helper scripts callable via run_command
│   ├── calendar_read.py        # CalDAV query helper (JSON output)
│   ├── calendar_write.py       # CalDAV create/update helper
│   └── contacts.py             # Contacts CLI (Google People + CardDAV)
│
├── voice/
│   └── pipeline.py             # Whisper STT + edge-tts
│
├── api/
│   └── admin.py                # FastAPI admin endpoints
│
│   # wacli (WhatsApp CLI) is installed from pinned upstream, not vendored
│
├── cli-configs/                # Config files for CLI tools (mounted into container)
│   ├── himalaya.toml           # → ~/.config/himalaya/config.toml
│   └── (no contacts CLI config files)
│
└── data/                       # Persistent volume
    ├── agent.db                # SQLite database (conversations, permissions)
    ├── memory.db               # SQLite database (long-term + short-term memories)
    ├── whisper-models/         # Cached Whisper model files
    ├── (no contacts sync folder)
    └── wa-session/             # WhatsApp session persistence
```

---

## 11. Configuration

### Agent Config (`config.yml`)

```yaml
agent:
  name: "Jarvis"
  owner_name: "Alex"
  anthropic_api_key: "${ANTHROPIC_API_KEY}"
  model: "claude-sonnet-4-5-20250514"
  timezone: "Europe/Berlin"
  skills_dir: "skills/"

memory:
  db_path: "data/memory.db"
  long_term_limit: 50                    # max long-term memories injected into prompt
  extraction_model: "claude-haiku-4-5"   # cheap model for post-turn memory extraction
  consolidation_model: "claude-haiku-4-5" # model for scheduled consolidation reviews

channels:
  telegram:
    enabled: true
    bot_token: "${TELEGRAM_BOT_TOKEN}"
    allowed_user_ids: [123456789]
  whatsapp:
    enabled: true
    bridge_url: "local-wacli"
    allowed_numbers: ["+49..."]

calendar:
  providers:
    - name: "google"
      url: "https://apidata.googleusercontent.com/caldav/v2/..."
      username: "${GOOGLE_EMAIL}"
      password: "${GOOGLE_APP_PASSWORD}"
    - name: "icloud"
      url: "https://caldav.icloud.com/"
      username: "${ICLOUD_EMAIL}"
      password: "${ICLOUD_APP_PASSWORD}"

voice:
  stt_model: "base"
  tts_voice: "en-US-GuyNeural"
  tts_enabled: true

scheduler:
  jobs:
    - id: "morning_briefing"
      cron: "0 7 * * *"
      task: "Give me a morning briefing: weather in Berlin, today's calendar, unread emails summary"
      channel: "telegram"
    - id: "email_check"
      cron: "*/15 * * * *"
      task: "Check for urgent unread emails across all accounts and notify me if any"
      channel: "telegram"
    - id: "memory_consolidation"
      cron: "0 */8 * * *"
      task: "memory_consolidation"
      type: "memory_consolidation"

admin:
  enabled: true
  port: 8000
  api_key: "${ADMIN_API_KEY}"
```

### Himalaya Config (`cli-configs/himalaya.toml`)

```toml
[accounts.personal]
email = "alex@example.com"
display-name = "Alex Chen"
default = true

backend.type = "imap"
backend.host = "imap.gmail.com"
backend.port = 993
backend.login = "alex@example.com"
backend.auth.type = "password"
backend.auth.raw = "app-password-here"

message.send.backend.type = "smtp"
message.send.backend.host = "smtp.gmail.com"
message.send.backend.port = 587
message.send.backend.starttls = true
message.send.backend.login = "alex@example.com"
message.send.backend.auth.type = "password"
message.send.backend.auth.raw = "app-password-here"

[accounts.work]
email = "alex@work.com"
display-name = "Alex Chen"

backend.type = "imap"
backend.host = "imap.fastmail.com"
backend.port = 993
backend.login = "alex@work.com"
backend.auth.type = "password"
backend.auth.raw = "app-password-here"

message.send.backend.type = "smtp"
message.send.backend.host = "smtp.fastmail.com"
message.send.backend.port = 587
message.send.backend.starttls = true
message.send.backend.login = "alex@work.com"
message.send.backend.auth.type = "password"
message.send.backend.auth.raw = "app-password-here"
```


---

## 12. Docker Setup

### Dockerfile

```dockerfile
FROM python:3.12-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl ca-certificates sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Install Himalaya (pre-built Rust binary)
RUN curl -sSL https://raw.githubusercontent.com/pimalaya/himalaya/master/install.sh \
    | PREFIX=/usr/local sh

# Contacts CLI is provided by /app/tools/contacts.py

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# CLI config directories
RUN mkdir -p /root/.config/himalaya

EXPOSE 8000
CMD ["python", "-m", "core.main"]
```

### docker-compose.yml

```yaml
version: "3.8"

services:
  agent:
    build: .
    container_name: personal-agent
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./config.yml:/app/config.yml:ro
      - ./character.md:/app/character.md
      - ./skills:/app/skills:ro
      - ./cli-configs/himalaya.toml:/root/.config/himalaya/config.toml:ro
    env_file: .env
```

### Resource Requirements

| Component | RAM | CPU | Disk |
|---|---|---|---|
| Python agent + FastAPI | ~100 MB | minimal | — |
| Himalaya binaries | ~20 MB | per-call | ~50 MB |
| Whisper `base` model | ~300 MB | 1 core during STT | 150 MB |
| WhatsApp (wacli) | ~60 MB | minimal | — |
| SQLite DB + vCards | ~10 MB | minimal | grows |
| **Total** | **~530 MB** | **2 cores** | **~500 MB** |

Runs comfortably on a **2 vCPU / 2 GB RAM** VPS ($5-10/month on Hetzner, Contabo, etc.).

---

## 13. Security

### Command Execution Safety

- `ALLOWED_PREFIXES` whitelist — the executor only runs approved CLI tools
- Glob-based permission patterns on command strings
- Write operations always require explicit user approval
- Command timeout (30s default) prevents hangs
- `purpose` field in run_command provides audit trail

### Network

- Telegram/WhatsApp connections are outbound-only
- Admin API protected by API key + optional IP whitelist
- All secrets in `.env` file, never in code or config

### Authentication

- Telegram: user ID whitelist (immutable, spoofing-proof)
- WhatsApp: phone number whitelist
- Email: app-specific passwords via himalaya config
- Calendar: app-specific passwords via caldav config
- Admin API: Bearer token

### Data

- SQLite DB on an encrypted volume (LUKS or VPS-level encryption)
- Conversation history can be auto-pruned after N days
- No data leaves the VPS except to Anthropic API and your configured providers

---

## 14. Proactive / Unsolicited Interactions

Scheduled jobs pass natural language tasks to the agent. The agent uses its skills to figure out which CLI commands to run — you don't hardcode the briefing logic:

```python
async def morning_briefing(agent):
    response = await agent.process(
        "Give me a morning briefing: weather in Berlin, today's calendar, unread emails summary",
        channel="system", user_id="scheduler"
    )
    await agent.channels["telegram"].send(owner_chat_id, response.text)
```

The LLM reads its skills and decides to run `python3 /app/tools/calendar_read.py --today -o json`, then `himalaya -a personal envelope list -s 5 -o json`, then composes the briefing.

---

## 15. Messaging Other Contacts

```
You: "Send a WhatsApp message to Marco asking if he's free for dinner Saturday"

Agent thinks: I need Marco's phone number
→ run_command("python3 /app/tools/contacts.py search --provider <NAME> --query Marco --output json", "Look up Marco's contact info")
→ Returns: "+49 170 ..."

→ send_message(channel="whatsapp", to="+49170...",
    text="Hey Marco! Are you free for dinner Saturday?")
→ Permission check → ASK

Agent: I'd like to send this to Marco (+49 151 5551234):
       "Hey Marco! Are you free for dinner Saturday?"
       [Approve] [Edit] [Deny]

You: [Approve]
Agent: ✅ Sent.
```

---

## 16. Startup Flow

```python
# core/main.py
import asyncio
from core.agent import AgentCore
from core.config import load_config
from channels.telegram import TelegramChannel
from channels.whatsapp import WhatsAppChannel
from api.admin import create_admin_app
import uvicorn

async def main():
    config = load_config("config.yml")

    # 1. Initialize core (loads skills, character, and memory schema automatically)
    agent = AgentCore(config)

    # 2. Register channels
    if config.channels.telegram.enabled:
        tg = TelegramChannel(config.channels.telegram.bot_token, agent)
        agent.channels["telegram"] = tg
    if config.channels.whatsapp.enabled:
        wa = WhatsAppChannel(agent)
        agent.channels["whatsapp"] = wa

    # 3. Start scheduler
    agent.scheduler.start()

    # 4. Start admin API + channels concurrently
    admin_app = create_admin_app(agent)
    await asyncio.gather(
        tg.app.run_polling() if tg else asyncio.sleep(0),
        uvicorn.Server(uvicorn.Config(admin_app, host="0.0.0.0", port=8000)).serve(),
    )

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 17. Implementation Roadmap

| Phase | What | Time Estimate |
|---|---|---|
| **1. Foundation** | Agent core + skills engine + executor + Telegram + Claude tool-use loop | 2-3 days |
| **2. Identity** | `character.md` + system prompt injection | 0.5 day |
| **3. Memory** | SQLite schema + `skills/memory.md` + sqlite3 integration + consolidation (LLM-based promotion + expired cleanup) | 1-2 days |
| **4. Email** | Install himalaya, write config + `himalaya-email.md` skill | 1 day |
| **5. Calendar** | CalDAV helper scripts + `caldav-calendar.md` skill | 1 day |
| **6. Contacts** | Add contacts CLI + `contacts.md` skill | 0.5 day |
| **7. Scheduler** | APScheduler + morning briefing + periodic email check + memory consolidation + contact sync | 0.5 day |
| **8. Permissions** | Permission engine + glob patterns + Telegram inline approval | 1 day |
| **9. Voice** | Whisper STT + edge-tts + `voice.md` skill | 1 day |
| **10. WhatsApp** | wacli auth + channel integration | 1-2 days |
| **11. Polish** | Admin API, logging, error handling, Docker optimization | 1-2 days |

**Total: ~10-15 days**

### Extending the Agent

To add any new capability:
1. Install the CLI tool in the Dockerfile
2. Add its prefix to `ALLOWED_PREFIXES`
3. Write a `skills/tool-name.md` file
4. (Optional) Add permission rules for write operations
5. No Python code changes needed

---

## 18. Example Interactions

**Morning briefing (proactive, scheduled):**
> 🤖 Good morning Alex! Here's your Tuesday briefing:
> 📅 3 events today: standup at 9:00, 1:1 with Dana at 14:00, dentist at 17:30
> 📧 12 unread emails — 1 flagged urgent from Nordic Furnishings AG RE: contract terms
> 🌤️ Berlin: 4°C, partly cloudy

**Email management:**
> You: "Check my work email for anything from Meridian Foods"
> Agent: *(runs: `himalaya -a work envelope list -o json -- "from:meridianfoods"`)*
> Found 2 emails from Meridian Foods:
> 1. "Q1 Budget Review" (Jan 15) — finance@meridianfoods.com
> 2. "Office Move Update" (Jan 12) — facilities@meridianfoods.com
> Want me to read either of them?

**Contact-aware messaging:**
> You: "Text Elena that I'll be home late tonight"
> Agent: *(runs: `python3 /app/tools/contacts.py search --provider <NAME> --query Elena --output json` → resolves number)*
> I'll send this WhatsApp message to Elena (+49 170 ...):
> "Hey, I'll be home a bit late tonight"
> [Approve] [Edit] [Deny]

**Sending on your behalf:**
> You: "Email my accountant asking for the 2025 tax filing deadline"
> Agent: *(runs: `python3 /app/tools/contacts.py search --provider <NAME> --query accountant --output json` → resolves email)*
> I'll send this from your personal email:
> **To:** accountant@example.de
> **Subject:** Question re: 2025 tax filing deadline
> **Body:** Dear [name], I wanted to check — what is the filing deadline for...
> [Approve] [Edit] [Deny]

**Memory — learning and recalling:**
> You: "By the way, I'm lactose intolerant. Keep that in mind for restaurant suggestions."
> Agent: *(runs: `sqlite3 -json /app/data/memory.db "SELECT * FROM long_term WHERE subject='alex' AND content LIKE '%lactose%';"` → no existing memory)*
> *(runs: `sqlite3 /app/data/memory.db "INSERT INTO long_term (category, subject, content, source) VALUES ('health', 'alex', 'Lactose intolerant — avoid dairy in restaurant suggestions', 'conversation');"` )*
> Got it, I'll remember that.

> *(Later...)*
> You: "Find me a good restaurant for dinner tonight near Market Street"
> Agent: *(checks memory, finds lactose intolerance fact)*
> *(searches for restaurants)*
> Here are 3 options near Market Street with good dairy-free choices: ...

**Memory — short-term context:**
> You: "I'm at the airport, flight to Vienna boards at 15:40"
> Agent: *(runs: `sqlite3 /app/data/memory.db "INSERT INTO short_term (content, context, expires_at) VALUES ('At Berlin airport, flight to Vienna boards at 15:40', 'telegram message', datetime('now', '+12 hours'));"` )*
> Safe travels! I'll keep that in mind. Want me to check your Vienna calendar for tomorrow?
