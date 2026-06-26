# MPA — My Personal Agent

A self-hosted personal AI agent that runs in a single Docker container. MPA acts as a unified interface across messaging channels (Telegram, WhatsApp), email, calendars, and contacts — capable of autonomous action, scheduled tasks, voice interaction, and persistent memory.

## Features

- **Messaging** — Telegram (and WhatsApp) with text and voice messages
- **Email** — Read, compose, and manage emails via [Himalaya](https://github.com/pimalaya/himalaya) CLI
- **Calendar** — CalDAV integration (Google Calendar, iCloud, etc.)
- **Contacts** — CardDAV providers via the built-in contacts CLI
- **Memory** — Two-tier system: permanent long-term facts and expiring short-term context, both extracted automatically from conversations
- **Scheduled tasks** — Cron-based jobs for morning briefings, email checks, contact sync, and custom tasks
- **Voice** — Speech-to-text (faster-whisper) and text-to-speech (edge-tts)
- **Web search** — Tavily integration for real-time information
- **Browser automation** — Optional headless browser (Playwright) to read JS-heavy pages and act on sites, with persistent logged-in profiles and per-domain approval (off by default)
- **Secrets vault** — Encrypted, two-tier secrets store: infrastructure keys (machine-key sealed, served into config via `${vault:NAME}`) and per-persona login/agent secrets (admin-password sealed, used by reference as `{{secret:NAME}}` in commands — values never enter the model's context). Bitwarden import + secure-link credential requests
- **Permissions** — Glob-pattern rules (ALWAYS/ASK/NEVER) with interactive Telegram approval for write actions
- **Admin UI** — Web dashboard for configuration, skills editing, memory inspection, job management, and agent lifecycle control
- **Skills** — Teach the agent new capabilities by writing markdown files instead of code
- **Setup wizard** — Step-by-step first-boot configuration via the admin UI

## Architecture

MPA follows a **Python orchestrator + CLI tools** design. Python glues everything together, while battle-tested CLI tools handle protocol complexity:

| Concern | Tool |
|---------|------|
| LLM | Anthropic Claude, OpenAI, Grok (xAI), DeepSeek |
| Email | Himalaya CLI (Rust) |
| Contacts | Built-in contacts CLI |
| Calendar | python-caldav |
| Storage | SQLite (4 databases) |
| Voice | faster-whisper (STT) + edge-tts (TTS) |
| Admin UI | FastAPI + HTMX + Tailwind CSS |

Instead of hardcoded integrations, the agent learns to use CLI tools via markdown "skill" files in `skills/`. Adding a new capability means writing a markdown file and adding the tool's command prefix to the executor whitelist.

## Quick start

### Prerequisites

- Docker and Docker Compose
- An [Anthropic API key](https://console.anthropic.com/)
- A [Telegram bot token](https://core.telegram.org/bots#botfather)

### 1. Clone and configure

```bash
git clone https://github.com/mattmezza/mpa.git
cd mpa
cp .env.example .env
cp config.yml.example config.yml
cp character.md.example character.md
cp personalia.md.example personalia.md
```

Edit `.env` with your API keys and secrets. Edit `config.yml` to customize the agent name, owner, channels, calendar providers, and scheduled jobs.

### 2. Run with Docker Compose

```bash
docker compose up -d
```

The admin UI will be available at `http://localhost:8000`. On first boot, MPA starts in **setup mode** — a wizard walks you through the initial configuration.

### 3. Run without Docker

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
make setup       # creates venv, installs deps, copies example configs
# edit .env and config.yml
make run         # starts the agent
```

## Configuration

MPA uses a dual-layer config system:

- **`config.yml`** + **`.env`** — File-based seed config loaded on first boot. Supports `${ENV_VAR}` interpolation.
- **SQLite config store** (`data/config.db`) — Becomes the source of truth after setup. Managed through the admin UI.

### Key files

| File | Purpose |
|------|---------|
| `.env` | API keys and secrets |
| `config.yml` | Agent settings, channels, calendar, scheduler jobs |
| `character.md` | Agent personality and communication style (editable) |
| `personalia.md` | Agent identity facts — name, owner, context (append-only) |
| `skills/*.md` | Skill documents that teach the agent how to use tools |
| `personae/*.md` | Persona definitions — swappable agent identities (starter gallery) |

## Project structure

```
core/           Core agent modules
  agent.py        LLM tool-use loop
  config.py       Pydantic config models, YAML loader
  config_store.py SQLite-backed config store + setup wizard
  executor.py     CLI command executor with prefix whitelist
  history.py      Conversation history persistence
  main.py         Entry point, lifecycle management
  memory.py       Two-tier memory extraction + consolidation
  permissions.py  Permission engine with approval flow
  scheduler.py    APScheduler wrapper for cron/one-shot jobs
  skills.py       Skills store + lazy loading for LLM
channels/       Communication channels
  telegram.py     Telegram bot (text, voice, approvals)
api/            Admin web interface
  admin.py        FastAPI routes + HTMX partials
  templates/      Jinja2 templates
  static/         CSS (Tailwind)
voice/          Voice pipeline
  pipeline.py     Whisper STT + edge-tts TTS
tools/          CLI helper scripts
  calendar_read.py   CalDAV event reader
  calendar_write.py  CalDAV event creator
  wacli/              WhatsApp CLI (vendor)
skills/         Markdown skill files
personae/       Persona definitions (starter gallery)
schema/         Database schemas
tests/          Test suite
data/           Runtime SQLite databases (gitignored)
```

## Development

```bash
make install-dev   # install all dependencies including dev tools
make dev           # auto-restart on code changes + CSS watch
make test          # run tests
make lint          # lint with ruff
make format        # format with ruff
make css           # build minified CSS
```

### Running tests

```bash
uv run pytest          # run all tests
uv run pytest -n auto  # run in parallel
```

## Skills

Skills are markdown documents stored in SQLite that teach the agent how to use CLI tools. Seed files in `skills/` are loaded into the DB at startup. The agent loads skills on-demand during conversations.

Example skills included:
- `himalaya-email.md` — Email management via Himalaya CLI
- `contacts.md` — Contact lookup and management
- `caldav-calendar.md` — Calendar event reading and creation
- `memory.md` — Memory querying via sqlite3
- `voice.md` — Voice response conventions
- `weather.md` — Weather lookups
- `jq.md` — JSON processing
- `browser.md` — Headless browser: read JS-heavy pages and act on sites

Create new skills by adding `.md` files to `skills/`, through the admin UI's skill editor, or via the skills CLI:

```bash
python3 /app/tools/skills.py upsert --name my-skill --stdin
```

Behavior and identity are configured in `character.md.example` and `personalia.md.example`.

## WhatsApp

MPA uses [wacli](https://github.com/openclaw/wacli) to authenticate and sync WhatsApp locally. The admin UI starts auth, displays the QR code, and manages sync.
The binary is installed from a pinned upstream tag (`WACLI_VERSION` in the `Dockerfile`; `make dev-wa` for local dev) — not vendored. See `docs/content/docs/channels.mdx` for upgrade/re-auth notes.

## Tech stack

- **Python 3.14** with **uv** for package management
- **Anthropic Claude**, **OpenAI**, **Grok (xAI)**, or **DeepSeek** as the LLM backend
- **SQLite** via aiosqlite for all persistence
- **FastAPI** + **Jinja2** + **HTMX** + **Alpine.js** + **Tailwind CSS v4** for the admin UI
- **python-telegram-bot** for the Telegram channel
- **APScheduler** for cron jobs
- **faster-whisper** + **edge-tts** for voice
- **ruff** for linting and formatting
- **pytest** with asyncio and xdist for testing
- **Docker** for production deployment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `make lint` and `make test`
5. Open a pull request

## License

See [LICENSE](LICENSE) for details.
