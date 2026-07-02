<p align="center">
  <img src=".github/humux.png" alt="humux" width="160" style="border-radius: 8px; vertical-align: middle; margin-right: 20px;">
  <br>
  <span style="display: inline-block; vertical-align: middle;">
    <strong style="font-size: 1.8em; font-family: monospace;">humux</strong><br>
    <em>Human Multiplexer</em> — your self-hosted personal AI agent.<br>
    <a href="https://humux.dev">humux.dev</a> ·
    <a href="#-features">Features</a> ·
    <a href="#-quick-start">Quick Start</a> ·
    <a href="https://mattmezza.github.io/humux">Docs</a><br>
    <a href="https://github.com/mattmezza/humux/actions/workflows/release.yml">
      <img src="https://github.com/mattmezza/humux/actions/workflows/release.yml/badge.svg" alt="CI">
    </a>
    <a href="https://github.com/mattmezza/humux/blob/main/pyproject.toml">
      <img src="https://img.shields.io/badge/python-3.14-blue" alt="Python 3.14+">
    </a>
    <a href="https://github.com/mattmezza/humux">
      <img src="https://img.shields.io/github/license/mattmezza/humux" alt="License">
    </a>
  </span>
</p>

---

**humux** is a self-hosted personal AI agent that runs in a **single Docker container**. It multiplexes across all the channels of your digital life — Telegram, email, calendar, contacts, WhatsApp — into one unified, autonomous intelligence. It remembers, plans, acts, and speaks.

No cloud dependency. No data leaving your server. One `docker compose up` and you have your own AI.

---

## ✨ Features
## Monorepo structure

| Directory | Contents |
|-----------|----------|
| [`humux/`](./humux) | The agent application (Python, FastAPI, Docker) |
| [`docs/`](./docs) | Documentation site (Next.js, Fumadocs) |
| [`www/`](./www) | Marketing website (HTML, Tailwind CSS v4) |


<details open>

<summary><strong>Messaging</strong> — Talk to your agent wherever you are</summary>

- **Telegram** — full bot with text, voice messages, reactions, inline approvals
- **WhatsApp** — read and send via [wacli](https://github.com/openclaw/wacli) CLI, link once and it stays authenticated
- **Multi-agent groups** — several agent-bots share one Telegram group, each replies only when addressed, never loops with other bots
- **Reply decision** — in group chats the agent decides per message whether to reply, with a hard rate cap that guarantees runaway loops end
- **Per-chat settings** — gate per Telegram chat who can trigger an agent and who may DM it

</details>

<details>

<summary><strong>Agents</strong> — Swappable identities, each with its own bot</summary>

- Each agent has its own **character**, **skill/tool scope**, **voice**, and **email/calendar/contacts accounts**
- Each agent runs its own **Telegram bot** — several run concurrently as separate contacts
- Agents are created and configured through the admin UI, no code needed
- Per-agent **tool identities** — own `gh` token, own browser profile (#93)

</details>

<details>

<summary><strong>Email</strong> — Read, compose, and route</summary>

- Powered by [Himalaya CLI](https://github.com/pimalaya/himalaya) (Rust — fast, stateless, JSON output)
- Multi-account: Gmail, Fastmail, iCloud, or any IMAP/SMTP provider
- Each agent can own a dedicated mailbox or be granted read/read-write access
- Credentials resolve from the encrypted vault — never reach the model's context

</details>

<details>

<summary><strong>Calendar & Contacts</strong> — Your schedule, your address book</summary>

- **CalDAV** — Google Calendar, iCloud, any CalDAV server
- **Contacts** — CardDAV (Purelymail, iCloud, Fastmail) and Google Contacts
- Both bindable per-agent with read / read-write access levels

</details>

<details>

<summary><strong>Memory</strong> — Four-tier persistent memory that learns and forgets</summary>

| Tier | What | How |
|------|------|-----|
| **T1 Lexical** | Word-overlap retrieval | Always-on, zero deps |
| **T2 Semantic** | Embedding vectors (fastembed, on-device) | Relevance-ranked injection |
| **T3 Forgetting** | Importance score + access recency | Cold memories archive automatically |
| **T4 Hygiene** | Cluster + merge near-duplicates | Self-healing compaction |

Memories are extracted automatically from conversations. The agent reads AND writes them via `sqlite3` CLI through the same skill system.

</details>

<details>

<summary><strong>Scheduled tasks</strong> — Proactive, not just reactive</summary>

- Cron-based jobs for morning briefings, email checks, memory consolidation
- Subagent jobs: delegate recurring work to a named agent
- One-shot tasks via Telegram (`/jobs`) or the admin UI

</details>

<details>

<summary><strong>Subagents</strong> — Delegate subtasks to scoped sub-loops</summary>

- Spawn a sub-loop under any agent, on demand or scheduled
- Scope is a subset of the caller's — **inherit-never-widen** for tools, skills, secrets, and GitHub repo access
- Runs sync (result returned in-turn) or background (distilled summary)
- Monitor and cancel from Telegram or admin UI

</details>

<details>

<summary><strong>Voice</strong> — Speak to your agent, hear it reply</summary>

- **STT**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — local, offline, multi-language
- **TTS**: [edge-tts](https://github.com/rany2/edge-tts) (cloud) or **Kokoro 82M** (fully offline, multilingual)
- Voice marker syntax lets the model request a spoken reply per-turn
- Per-agent voice selection

</details>

<details>

<summary><strong>Secrets vault</strong> — Encrypted, two-tier, never in context</summary>

| Vault | Key | Unseals |
|-------|-----|---------|
| **Infra vault** | Machine key (`HUMUX_MASTER_KEY` / `data/master.key`) | At boot, headless — for provider keys, bot tokens |
| **Agent vault** | Admin password (envelope encryption) | On login — for website logins, payment keys |

Secrets are referenced as `${vault:NAME}` in config and `{{secret:NAME}}` in commands — **the model never sees the value**. Bitwarden import + secure-link credential requests included.

</details>

<details>

<summary><strong>Permissions</strong> — You're always in control</summary>

- Glob-pattern rules: `ALWAYS` / `ASK` / `NEVER`
- Write actions ask for Telegram approval with context preview
- Per-agent tool scoping — an agent can only use what you give it
- Per-agent GitHub repo allowlist — an agent can only touch repos you authorize

</details>

<details>

<summary><strong>Browser automation</strong> — The agent can browse the web</summary>

- Optional headless Chromium (Playwright) for JS-heavy pages
- **Self-driving explore mode** — an inner LLM loop navigates sites, fills forms, clicks buttons until done
- Persistent logged-in profiles (cookies survive between calls)
- Per-domain action rules (Allow / Ask / Block)

</details>

<details>

<summary><strong>Web artifacts</strong> — Publish pages, dashboards, documents</summary>

- The agent writes files under `{workspace}/artifacts/<slug>/` with the coding harness
- Served as shareable links at `/artifacts/<slug>/` behind a sandbox CSP
- Multi-file sites, PDFs, images — anything you can write to disk

</details>

<details>

<summary><strong>Image generation</strong> — Visual answers</summary>

- Optional `generate_image` tool (OpenRouter, fal.ai, or OpenAI)
- Reuses your existing LLM API key for OpenRouter/OpenAI
- Daily/monthly budget caps

</details>

<details>

<summary><strong>Coding harness</strong> — The agent works on real code</summary>

- `read_file`, `write_file`, `edit_file`, `list_dir`, `grep`, `run_command_in_dir`
- Confined to one configurable workspace directory — path traversal blocked
- Reads pre-approved, writes ask permission

</details>

<details>

<summary><strong>Admin UI</strong> — Full web dashboard</summary>

- Configuration, agents, skills editor, memory inspection, job management
- Per-agent log streams, filterable by stream / level / time / text
- Agent lifecycle control (start/stop/restart)
- Setup wizard for first boot
- Built with **FastAPI + HTMX + Alpine.js + Tailwind CSS v4**

</details>

---

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- An [Anthropic](https://console.anthropic.com/), [OpenAI](https://platform.openai.com/), or [DeepSeek](https://platform.deepseek.com/) API key
- A [Telegram bot token](https://core.telegram.org/bots#botfather) (optional but recommended)

### 1. Clone and configure

```bash
git clone https://github.com/mattmezza/mpa.git
cd mpa/humux
cp .env.example .env
cp config.yml.example config.yml
cp character.md.example character.md
```

Edit `.env` with your API keys. Edit `config.yml` to customize the agent name, owner, channels, and scheduled jobs.

### 2. Run with Docker Compose

```bash
cd mpa/humux
docker compose up -d
```

The admin UI is at `http://localhost:8000`. On first boot, humux starts in **setup mode** — a wizard walks you through the initial configuration.

### 3. Run without Docker (development)

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
cd mpa/humux
make setup       # creates venv, installs deps, copies example configs
make run         # starts the agent
```

### 4. Chat from the terminal (no Telegram needed)

```bash
cd mpa/humux
make repl        # interactive REPL — type your messages, see the agent think
make repl AGENT=my-agent  # chat as a specific agent
make repl YOLO=1          # auto-approve all permissions (local testing)
```

---

## 🏗️ Architecture

humux follows a **Python orchestrator + CLI tools** design. Python handles the async LLM loop, the admin web UI, and orchestration. Battle-tested CLI tools handle protocol complexity:

| Concern | Tool |
|---------|------|
| **LLM** | Anthropic Claude, OpenAI, Grok (xAI), Google, DeepSeek, OpenRouter |
| **Email** | [Himalaya](https://github.com/pimalaya/himalaya) CLI (Rust) |
| **Contacts** | Built-in contacts CLI (CardDAV + Google People) |
| **Calendar** | python-caldav |
| **WhatsApp** | [wacli](https://github.com/openclaw/wacli) (Go) |
| **Browser** | Playwright (Chromium) |
| **Voice STT** | faster-whisper (CTranslate2) |
| **Voice TTS** | edge-tts / Kokoro 82M |
| **Web search** | Tavily |
| **Scheduler** | APScheduler |
| **Storage** | SQLite (8 databases) |
| **Admin UI** | FastAPI + Jinja2 + HTMX + Alpine.js + Tailwind CSS v4 |

### The skill system

Instead of hardcoded integrations, the agent learns to use CLI tools via **markdown skill files** stored in SQLite. Skills are injected into the LLM's context on-demand during conversations. Adding a new capability means:

1. Install the CLI tool
2. Write a markdown file teaching the agent how to use it
3. Add the command prefix to the executor whitelist

No Python code. No redeploy. The agent picks it up on the next turn.

### Project structure

```
humux/
├── core/             Core agent modules
│   ├── agent.py          LLM tool-use loop with agentic reasoning
│   ├── llm.py            Multi-provider LLM client abstraction
│   ├── memory.py         Four-tier memory extraction + consolidation
│   ├── config.py         Pydantic config models, YAML/env loader
│   ├── config_store.py   SQLite-backed config store + setup wizard
│   ├── executor.py       CLI command executor with prefix whitelist
│   ├── permissions.py    Glob-pattern permission engine
│   ├── skills.py         SQLite-backed skills store + lazy loading
│   ├── scheduler.py      APScheduler wrapper for cron/one-shot jobs
│   ├── subagents.py      Scoped sub-loop delegation
│   ├── vault.py          Encryption primitives + key management
│   ├── secret_store.py   SQLite-backed secrets vault storage + ACL
│   ├── artifacts.py      Web artifact serving (sandboxed)
│   ├── coding.py         Confined workspace file tools
│   ├── compaction.py     Conversation compaction for session history
│   ├── github_app.py     GitHub App JWT minting + installation tokens
│   ├── history.py        Conversation history persistence
│   ├── imagegen.py       Image generation with budget caps
│   ├── job_store.py      Scheduled job persistence
│   ├── log_streams.py    Per-agent structured log streaming
│   ├── agents.py         Agent definitions, CRUD, markdown parsing
│   ├── embeddings.py     Local (fastembed) + remote embeddings
│   ├── goal_decomposition.py  Task breakdown for complex requests
│   ├── task_reflection.py     Post-task reflection store
│   └── reply_decision.py      Group-chat reply gate
├── channels/         Communication channels
│   └── telegram.py       Telegram bot (text, voice, approvals)
├── api/              Admin web interface
│   ├── admin.py          FastAPI routes + HTMX partials
│   ├── templates/        Jinja2 templates
│   └── static/           Tailwind CSS
├── voice/            Voice pipeline
│   └── pipeline.py       Whisper STT + edge-tts/Kokoro TTS
├── tools/            CLI helper scripts
│   ├── calendar_read.py  CalDAV event reader
│   ├── calendar_write.py CalDAV event creator
│   ├── contacts.py       CardDAV/Google Contacts client
│   ├── browser.py        Headless browser automation (Playwright)
│   └── skills.py         Skills management CLI
├── skills/           Markdown skill files (seed → SQLite)
├── schema/           SQL schema files
└── tests/            Test suite (pytest + asyncio + xdist)
docs/             Documentation site (Next.js + Fumadocs)
www/              Marketing site (humux.dev)
```

---

## ⚙️ Configuration

humux uses a dual-layer config system:

1. **`config.yml`** + **`.env`** — File-based seed config loaded on first boot. Supports `${ENV_VAR}` interpolation and `${vault:NAME}` for secrets.
2. **SQLite config store** (`data/config.db`) — Becomes the source of truth after setup. Managed through the admin UI.

### Key files

| File | Purpose |
|------|---------|
| `.env` | API keys and secrets |
| `config.yml` | Agent settings, channels, calendar, scheduler jobs |
| `character.md` | Agent identity, personality, and communication style |
| `skills/*.md` | Skill documents that teach the agent how to use tools |
| `agents/*.md` | Optional agent-definition seed files |

---

## 📦 Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.14+ with [uv](https://docs.astral.sh/uv/) |
| **LLM providers** | Anthropic Claude, OpenAI, Google Gemini, Grok (xAI), DeepSeek, OpenRouter (any OpenAI-compatible) |
| **Messaging** | python-telegram-bot, wacli (WhatsApp) |
| **Persistence** | SQLite (8 databases: config, skills, agents, history, memory, reflections, jobs, imagegen) |
| **Admin UI** | FastAPI + Jinja2 + HTMX + Alpine.js + Tailwind CSS v4 |
| **Voice** | faster-whisper (STT), edge-tts / Kokoro 82M (TTS) |
| **Browser** | Playwright (Chromium), headless or CDP sidecar |
| **Scheduler** | APScheduler |
| **Search** | Tavily |
| **Container** | Docker (single image, multi-stage) |
| **CI/CD** | GitHub Actions (lint, test, build, publish to ghcr.io) |

---

## 💬 Community & Support

- **[humux.dev](https://humux.dev)** — Marketing site
- **[Documentation](https://mattmezza.github.io/humux)** — Full docs with guides and API reference
- **[GitHub Issues](https://github.com/mattmezza/humux/issues)** — Bug reports and feature requests
- **[Discussions](https://github.com/mattmezza/humux/discussions)** — Questions and community help

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `make lint` and `make test`
5. Open a pull request

See [Development docs](https://mattmezza.github.io/humux/docs/development) for setup instructions.

---

<div align="center">
  <p>
    <strong>humux — Human Multiplexer</strong>
  </p>
  <p>
    Built with ❤️ by <a href="https://github.com/mattmezza">Matteo Merola</a>
  </p>
</div>
