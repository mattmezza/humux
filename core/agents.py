"""Agent (profile) engine — SQLite-backed store of swappable agent identities.

An **agent** is a first-class identity the assistant can take on (fitness coach,
finance assistant, …). It is exactly four things (issue #13):

- its own system-prompt identity (``character`` — who it is + its tone),
- a **skill allowlist** (which skills it may load — empty means *all*),
- a **tool scope** (which function-tools are advertised — empty means *all*),
- a **secret scope** (vault namespaces it may use — stored now, enforced by #19),
- **tool identities** (``tool_config``: own ``gh`` token / browser profile per
  external CLI tool — #93; empty means *inherit the system-wide config*).

The store mirrors :mod:`core.skills`: markdown files with YAML frontmatter,
seeded into SQLite at startup, editable in the admin UI. When no agent is
selected the assistant falls back to the configured default ``character`` so
first-run behaviour is unchanged. (A legacy ``personalia`` field was merged into
``character`` in #98; old frontmatter/rows are folded in on load.)

(Formerly the "persona" concept — the store, tables and DB file were renamed to
"agent" in #115; existing ``personae`` DBs are migrated in place on first open.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import aiosqlite
import yaml

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    name TEXT PRIMARY KEY,
    agent_name TEXT DEFAULT '',
    role TEXT DEFAULT '',
    emoji TEXT DEFAULT '',
    voice TEXT DEFAULT '',
    character TEXT DEFAULT '',
    skills TEXT DEFAULT '',
    tools TEXT DEFAULT '',
    secrets TEXT DEFAULT '',
    bot_token TEXT DEFAULT '',
    allowed_user_ids TEXT DEFAULT '',
    tool_config TEXT DEFAULT '',
    email_accounts TEXT DEFAULT '',
    calendar_accounts TEXT DEFAULT '',
    contacts_accounts TEXT DEFAULT '',
    chat_settings TEXT DEFAULT '',
    created_at DATETIME DEFAULT (datetime('now')),
    updated_at DATETIME DEFAULT (datetime('now'))
);
-- Slugs deliberately removed (deleted, or renamed away from) so seeding does not
-- resurrect them from their markdown file on the next list (#102). Re-creating a
-- slug via upsert/rename clears its tombstone.
CREATE TABLE IF NOT EXISTS agent_tombstones (
    name TEXT PRIMARY KEY,
    created_at DATETIME DEFAULT (datetime('now'))
);
"""

# #115: rename the legacy "persona" tables to "agent" in place, preserving every
# row. Skipped (OperationalError) on a fresh DB or once already renamed.
_TABLE_RENAMES = (
    "ALTER TABLE personae RENAME TO agents",
    "ALTER TABLE persona_tombstones RENAME TO agent_tombstones",
)

# Columns added after the table first shipped — applied to existing DBs on open.
_MIGRATIONS = (
    "ALTER TABLE agents ADD COLUMN bot_token TEXT DEFAULT ''",  # #29
    "ALTER TABLE agents ADD COLUMN allowed_user_ids TEXT DEFAULT ''",  # #29
    "ALTER TABLE agents ADD COLUMN tool_config TEXT DEFAULT ''",  # #93
    "ALTER TABLE agents ADD COLUMN email_accounts TEXT DEFAULT ''",  # #110
    "ALTER TABLE agents ADD COLUMN calendar_accounts TEXT DEFAULT ''",  # #110
    "ALTER TABLE agents ADD COLUMN contacts_accounts TEXT DEFAULT ''",  # #110 (contacts)
    "ALTER TABLE agents ADD COLUMN chat_settings TEXT DEFAULT ''",  # #129
    # #98: personalia merged into character. Prepend any existing personalia to
    # character (idempotent — the WHERE guard empties it, so a re-run is a no-op),
    # then drop the now-unused column. On a fresh DB (no such column) both raise
    # OperationalError and are skipped, like the ADD COLUMNs above.
    "UPDATE agents SET character = TRIM(personalia || char(10) || char(10) || character), "
    "personalia = '' WHERE TRIM(COALESCE(personalia, '')) != ''",
    "ALTER TABLE agents DROP COLUMN personalia",
)


@dataclass(slots=True)
class Agent:
    """A swappable agent identity + its scopes."""

    name: str  # slug / identifier (PK)
    agent_name: str = ""  # display name it goes by when active; empty = global agent.name
    role: str = ""
    emoji: str = ""
    voice: str = ""  # TTS voice override; empty = configured default
    character: str = ""  # identity + tone (a legacy `personalia` field folded in here — #98)
    skills: list[str] = field(default_factory=list)  # allowlist; [] = all
    tools: list[str] = field(default_factory=list)  # allowlist; [] = all
    secrets: list[str] = field(default_factory=list)  # vault scope; stored only (#19)
    bot_token: str = ""  # own Telegram bot; empty = reachable only via the default bot (#29)
    allowed_user_ids: list[int] = field(default_factory=list)  # bot ACL; [] = inherit global
    # Per-agent config for the optional external CLI tools (gh, browser) — #93.
    # Shape: {tool_key: {"enabled": bool, **tool-specific cfg}}. Empty = inherit
    # the system-wide tool config (own credentials/profile fall back to shared).
    tool_config: dict = field(default_factory=dict)
    # Per-agent email/calendar account bindings (#110). Each email entry is
    # {account, access_level: read|read_write, is_sender_identity: bool}; each
    # calendar entry drops the sender flag. Empty = NO email/calendar access
    # (safe default — an agent reaches only the accounts it is bound to).
    email_accounts: list[dict] = field(default_factory=list)
    calendar_accounts: list[dict] = field(default_factory=list)
    # Per-agent contacts (CardDAV) account bindings (#110 follow-up). Each entry
    # is {account, access_level: read|read_write}; read = search/list, read_write =
    # also create contacts. Empty = no contacts access (safe default).
    contacts_accounts: list[dict] = field(default_factory=list)
    # Per-agent, per-Telegram-chat trigger/DM permissions (#129). Keyed by the
    # runtime chat id (group id in a group, sender id in a DM). Each value is
    # {"mode": "everyone"|"nobody"|"users", "users": [int]}. A chat with no entry
    # (or mode "everyone") is unrestricted — so the default is unchanged.
    chat_settings: dict = field(default_factory=dict)

    def chat_permits(self, chat_id: str, sender_id: int) -> bool:
        """Whether ``sender_id`` may trigger this agent (or DM it) in ``chat_id``.

        No stored setting = everyone allowed (unchanged behaviour). ``nobody``
        blocks all; ``users`` allows only the listed Telegram ids (#129).
        """
        setting = self.chat_settings.get(chat_id) if isinstance(self.chat_settings, dict) else None
        if not isinstance(setting, dict):
            return True
        mode = setting.get("mode", "everyone")
        if mode == "nobody":
            return False
        if mode == "users":
            return sender_id in setting.get("users", [])
        return True  # "everyone" / unknown → allow

    def allows_skill(self, name: str) -> bool:
        return not self.skills or name in self.skills

    def allows_tool(self, name: str) -> bool:
        return not self.tools or name in self.tools

    def email_access(self, account: str) -> str | None:
        """This agent's access level ('read'/'read_write') on ``account``, or
        None if it is not bound to that email account (#110)."""
        for e in self.email_accounts:
            if e.get("account") == account:
                return e.get("access_level")
        return None

    def calendar_access(self, account: str) -> str | None:
        """This agent's access level on calendar ``account``, or None (#110)."""
        for e in self.calendar_accounts:
            if e.get("account") == account:
                return e.get("access_level")
        return None

    def contacts_access(self, account: str) -> str | None:
        """This agent's access level on contacts ``account``, or None (#110)."""
        for e in self.contacts_accounts:
            if e.get("account") == account:
                return e.get("access_level")
        return None

    def sender_identity(self) -> str | None:
        """The email account this agent sends from (its is_sender_identity
        binding), or None if it has no send identity (#110)."""
        for e in self.email_accounts:
            if e.get("is_sender_identity"):
                return e.get("account")
        return None

    def tool_setting(self, key: str) -> dict | None:
        """This agent's config for external tool ``key`` (gh/browser), or None
        if it has none — in which case the system-wide config applies (#93)."""
        cfg = self.tool_config.get(key) if isinstance(self.tool_config, dict) else None
        return cfg if isinstance(cfg, dict) else None


def _as_list(value: object) -> list[str]:
    """Coerce a frontmatter / form value into a clean list of strings."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        # Comma- or newline-separated (form input).
        return [p.strip() for p in value.replace("\n", ",").split(",") if p.strip()]
    if isinstance(value, (list, tuple)):
        return [str(p).strip() for p in value if str(p).strip()]
    return []


def _as_tool_config(value: object) -> dict:
    """Coerce a frontmatter / DB-column value into a per-tool config dict (#93).

    Accepts an already-parsed dict (frontmatter) or a JSON string (DB column).
    Anything malformed degrades to ``{}`` so a broken value never breaks load.
    """
    if isinstance(value, dict):
        return {k: v for k, v in value.items() if isinstance(v, dict)}
    if isinstance(value, str) and value.strip():
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError, ValueError:
            return {}
        return _as_tool_config(loaded)
    return {}


def _as_chat_settings(value: object) -> dict:
    """Coerce a frontmatter / form / DB-column value into per-chat settings (#129).

    Shape: ``{chat_id: {"mode": everyone|nobody|users, "users": [int]}}``. Accepts
    a parsed dict (frontmatter/form) or a JSON string (DB column). ``everyone``
    entries are dropped (that is the default — an absent chat is unrestricted), so
    the stored dict stays lean. Non-numeric user ids and malformed entries are
    discarded so a broken value never breaks load.
    """
    if isinstance(value, str) and value.strip():
        try:
            value = json.loads(value)
        except json.JSONDecodeError, ValueError:
            return {}
    if not isinstance(value, dict):
        return {}
    out: dict = {}
    for cid, spec in value.items():
        if not isinstance(spec, dict):
            continue
        mode = str(spec.get("mode", "everyone") or "everyone").strip().lower()
        if mode not in ("everyone", "nobody", "users"):
            mode = "everyone"
        if mode == "everyone":
            continue  # the default — no need to store it
        users: list[int] = []
        for u in spec.get("users") or []:
            try:
                users.append(int(u))
            except TypeError, ValueError:
                continue
        out[str(cid)] = {"mode": mode, "users": users}
    return out


_ACCESS_LEVELS = ("read", "read_write")


def _as_account_list(value: object, *, sender: bool = False) -> list[dict]:
    """Coerce a frontmatter / DB-column value into account-binding dicts (#110).

    Accepts a parsed list (frontmatter) or a JSON string (DB column). Each entry
    normalises to ``{account, access_level}``; for email (``sender=True``) an
    ``is_sender_identity`` flag is kept too. A bare string is read as a
    read-only binding. Unknown access levels default to ``read`` (safe). A sender
    identity is forced to ``read_write`` — you cannot send from a read-only
    account — and at most one sender identity survives (later ones demoted).
    Anything malformed drops out so a broken value never breaks load.
    """
    if isinstance(value, str) and value.strip():
        try:
            value = json.loads(value)
        except json.JSONDecodeError, ValueError:
            return []
    if not isinstance(value, (list, tuple)):
        return []
    out: list[dict] = []
    seen: set[str] = set()
    sender_taken = False
    for item in value:
        if isinstance(item, str):
            item = {"account": item}
        if not isinstance(item, dict):
            continue
        account = str(item.get("account", "") or "").strip()
        if not account or account in seen:
            continue
        seen.add(account)
        level = str(item.get("access_level", "") or "").strip().lower()
        if level not in _ACCESS_LEVELS:
            level = "read"
        entry: dict = {"account": account, "access_level": level}
        if sender:
            is_sender = bool(item.get("is_sender_identity")) and not sender_taken
            if is_sender:
                entry["access_level"] = "read_write"  # sending needs write
                sender_taken = True
            entry["is_sender_identity"] = is_sender
        out.append(entry)
    return out


def _as_int_list(value: object) -> list[int]:
    """Coerce a frontmatter / form value into a clean list of ints (drops non-numeric)."""
    out: list[int] = []
    for item in _as_list(value):
        try:
            out.append(int(item))
        except ValueError:
            continue
    return out


def parse_markdown(text: str, *, name: str) -> Agent:
    """Parse an agent markdown doc (YAML frontmatter + optional body).

    Identity lives in the frontmatter (``character`` block scalar); any markdown
    body after the frontmatter is appended to ``character`` so authors can write
    free-form prose too. A legacy ``personalia`` key is folded into ``character``
    (prepended, so nothing is lost — #98).
    """
    fm: dict = {}
    body = text
    stripped = text.lstrip()
    if stripped.startswith("---"):
        rest = stripped[3:]
        end = rest.find("\n---")
        if end != -1:
            # Closing fence present: frontmatter then a markdown body.
            fm_text, body = rest[:end], rest[end + 4 :].lstrip("-\n")
        else:
            # No closing fence: the whole doc is frontmatter (block scalars).
            fm_text, body = rest, ""
        loaded = yaml.safe_load(fm_text)
        if isinstance(loaded, dict):
            fm = loaded
        else:
            body = text  # not real frontmatter — treat the whole doc as body
    if not isinstance(fm, dict):
        fm = {}

    character = str(fm.get("character", "") or "")
    body = body.strip()
    if body:
        character = f"{character}\n\n{body}".strip()
    legacy_personalia = str(fm.get("personalia", "") or "").strip()
    if legacy_personalia:  # #98: fold old personalia in, prepended so nothing is lost
        character = f"{legacy_personalia}\n\n{character}".strip()

    return Agent(
        name=name,
        agent_name=str(fm.get("agent_name", "") or ""),
        role=str(fm.get("role", "") or ""),
        emoji=str(fm.get("emoji", "") or ""),
        voice=str(fm.get("voice", "") or ""),
        character=character,
        skills=_as_list(fm.get("skills")),
        tools=_as_list(fm.get("tools")),
        secrets=_as_list(fm.get("secrets")),
        bot_token=str(fm.get("bot_token", "") or ""),
        allowed_user_ids=_as_int_list(fm.get("allowed_user_ids")),
        tool_config=_as_tool_config(fm.get("tool_config")),
        email_accounts=_as_account_list(fm.get("email_accounts"), sender=True),
        calendar_accounts=_as_account_list(fm.get("calendar_accounts")),
        contacts_accounts=_as_account_list(fm.get("contacts_accounts")),
        chat_settings=_as_chat_settings(fm.get("chat_settings")),
    )


def to_markdown(a: Agent) -> str:
    """Serialise an agent back to frontmatter markdown (the raw power-user view)."""
    fm = {
        "agent_name": a.agent_name,
        "role": a.role,
        "emoji": a.emoji,
        "voice": a.voice,
        "bot_token": a.bot_token,
        "allowed_user_ids": a.allowed_user_ids,
        "skills": a.skills,
        "tools": a.tools,
        "secrets": a.secrets,
        "tool_config": a.tool_config,
        "email_accounts": a.email_accounts,
        "calendar_accounts": a.calendar_accounts,
        "contacts_accounts": a.contacts_accounts,
        "chat_settings": a.chat_settings,
        "character": a.character,
    }
    dumped = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True, default_flow_style=False)
    return f"---\n{dumped}---\n"


class AgentStore:
    """SQLite-backed store for agents, seeded from a markdown directory."""

    def __init__(self, db_path: str = "data/agents.db", seed_dir: str | Path = "agents/"):
        self.db_path = db_path
        self.seed_dir = Path(seed_dir) if seed_dir else None
        self._ready = False

    def _migrate_legacy_db_file(self) -> None:
        """#115: adopt a pre-rename ``personae.db`` sitting next to the new path.

        If the configured DB file is absent but a sibling ``personae.db`` exists,
        move it into place so an upgraded deployment keeps its agents (the tables
        inside are renamed on connect below). A wholly custom filename won't match
        and just gets a fresh DB — acceptable for the default deployment.
        """
        new = Path(self.db_path)
        if new.exists():
            return
        legacy = new.with_name("personae.db")
        if legacy.exists():
            legacy.rename(new)

    async def _ensure_schema(self) -> None:
        if self._ready:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_db_file()
        async with aiosqlite.connect(self.db_path) as db:
            for stmt in _TABLE_RENAMES:
                try:
                    await db.execute(stmt)
                except aiosqlite.OperationalError:
                    pass  # fresh DB or already renamed
            await db.executescript(_SCHEMA)
            for stmt in _MIGRATIONS:
                try:
                    await db.execute(stmt)
                except aiosqlite.OperationalError:
                    pass  # column already present
            await db.commit()
        self._ready = True

    async def ensure_seeded(self) -> bool:
        """Seed missing agents from the seed directory (idempotent)."""
        await self._ensure_schema()
        if not self.seed_dir or not self.seed_dir.exists():
            return False
        inserted = 0
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT name FROM agents")
            existing = {row[0] for row in await cursor.fetchall()}
            cursor = await db.execute("SELECT name FROM agent_tombstones")
            tombstoned = {row[0] for row in await cursor.fetchall()}
            for f in sorted(self.seed_dir.glob("*.md")):
                content = f.read_text().strip()
                if not content or f.stem in existing or f.stem in tombstoned:
                    continue
                await self._upsert(db, parse_markdown(content, name=f.stem))
                inserted += 1
            await db.commit()
        return inserted > 0

    @staticmethod
    async def _upsert(db: aiosqlite.Connection, a: Agent) -> None:
        await db.execute(
            "INSERT INTO agents "
            "(name, agent_name, role, emoji, voice, character, skills, tools, "
            "secrets, bot_token, allowed_user_ids, tool_config, "
            "email_accounts, calendar_accounts, contacts_accounts, chat_settings) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET "
            "agent_name=excluded.agent_name, role=excluded.role, emoji=excluded.emoji, "
            "voice=excluded.voice, character=excluded.character, "
            "skills=excluded.skills, tools=excluded.tools, secrets=excluded.secrets, "
            "bot_token=excluded.bot_token, allowed_user_ids=excluded.allowed_user_ids, "
            "tool_config=excluded.tool_config, email_accounts=excluded.email_accounts, "
            "calendar_accounts=excluded.calendar_accounts, "
            "contacts_accounts=excluded.contacts_accounts, "
            "chat_settings=excluded.chat_settings, updated_at=datetime('now')",
            (
                a.name,
                a.agent_name,
                a.role,
                a.emoji,
                a.voice,
                a.character,
                "\n".join(a.skills),
                "\n".join(a.tools),
                "\n".join(a.secrets),
                a.bot_token,
                "\n".join(str(i) for i in a.allowed_user_ids),
                json.dumps(a.tool_config) if a.tool_config else "",
                json.dumps(a.email_accounts) if a.email_accounts else "",
                json.dumps(a.calendar_accounts) if a.calendar_accounts else "",
                json.dumps(a.contacts_accounts) if a.contacts_accounts else "",
                json.dumps(a.chat_settings) if a.chat_settings else "",
            ),
        )

    @staticmethod
    def _row_to_agent(row: aiosqlite.Row) -> Agent:
        return Agent(
            name=row["name"],
            agent_name=row["agent_name"] or "",
            role=row["role"] or "",
            emoji=row["emoji"] or "",
            voice=row["voice"] or "",
            character=row["character"] or "",
            skills=_as_list(row["skills"]),
            tools=_as_list(row["tools"]),
            secrets=_as_list(row["secrets"]),
            bot_token=(row["bot_token"] if "bot_token" in row.keys() else "") or "",
            allowed_user_ids=_as_int_list(
                row["allowed_user_ids"] if "allowed_user_ids" in row.keys() else ""
            ),
            tool_config=_as_tool_config(row["tool_config"] if "tool_config" in row.keys() else ""),
            email_accounts=_as_account_list(
                row["email_accounts"] if "email_accounts" in row.keys() else "", sender=True
            ),
            calendar_accounts=_as_account_list(
                row["calendar_accounts"] if "calendar_accounts" in row.keys() else ""
            ),
            contacts_accounts=_as_account_list(
                row["contacts_accounts"] if "contacts_accounts" in row.keys() else ""
            ),
            chat_settings=_as_chat_settings(
                row["chat_settings"] if "chat_settings" in row.keys() else ""
            ),
        )

    async def list_agents(self) -> list[Agent]:
        await self.ensure_seeded()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM agents ORDER BY name")
            return [self._row_to_agent(r) for r in await cursor.fetchall()]

    async def get(self, name: str) -> Agent | None:
        await self.ensure_seeded()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM agents WHERE name = ?", (name,))
            row = await cursor.fetchone()
            return self._row_to_agent(row) if row else None

    async def upsert(self, a: Agent) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            await self._upsert(db, a)
            # Deliberately (re)creating a slug clears any tombstone on it (#102).
            await db.execute("DELETE FROM agent_tombstones WHERE name = ?", (a.name,))
            await db.commit()

    async def delete(self, name: str) -> bool:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("DELETE FROM agents WHERE name = ?", (name,))
            if cursor.rowcount > 0:
                # Tombstone so seeding doesn't resurrect a deleted seed agent (#102).
                await db.execute("INSERT OR IGNORE INTO agent_tombstones(name) VALUES (?)", (name,))
            await db.commit()
            return cursor.rowcount > 0

    async def rename(self, old: str, new: str) -> bool:
        """Change an agent's slug (its PRIMARY KEY). Returns False if ``old`` is
        missing; raises ``ValueError`` if ``new`` already names another agent.

        This only moves the agents row. The slug is referenced from other
        stores (per-chat bindings, memory scope, jobs, the active-agent config,
        and the ``telegram:<slug>`` bot channel) — the admin rename route cascades
        the new slug to those so nothing is orphaned.
        """
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT 1 FROM agents WHERE name = ?", (new,))
            if await cursor.fetchone():
                raise ValueError(f"An agent named '{new}' already exists")
            cursor = await db.execute(
                "UPDATE agents SET name = ?, updated_at = datetime('now') WHERE name = ?",
                (new, old),
            )
            if cursor.rowcount > 0:
                # Old slug must not resurrect from its seed file; new slug is now
                # live, so clear any tombstone it carried (#102).
                await db.execute("INSERT OR IGNORE INTO agent_tombstones(name) VALUES (?)", (old,))
                await db.execute("DELETE FROM agent_tombstones WHERE name = ?", (new,))
            await db.commit()
            return cursor.rowcount > 0


async def bind_existing_accounts(
    store: AgentStore,
    email_names: list[str],
    calendar_names: list[str],
    contacts_names: list[str] | None = None,
) -> int:
    """One-time #110 compatibility: bind existing accounts to existing agents.

    Before #110 any agent could use any configured email/calendar/contacts
    account (the tools took an ``account`` argument with no per-agent gate). #110
    makes an empty binding mean *no access*, so on upgrade every agent that has
    **no** bindings yet is granted full (``read_write``) access to all existing
    accounts — the first email account becomes its sender identity — preserving
    prior behaviour exactly. Agents created afterwards start empty (safe default).
    Idempotent: an agent that already has bindings is left untouched, so a re-run
    (or an agent configured post-migration) is a no-op. Returns the count updated.
    """
    contacts_names = contacts_names or []
    updated = 0
    for a in await store.list_agents():
        # An agent that already carries any binding was configured deliberately —
        # leave it alone (so a re-run, or a post-migration agent, is a no-op).
        if a.email_accounts or a.calendar_accounts or a.contacts_accounts:
            continue
        if not email_names and not calendar_names and not contacts_names:
            continue
        a.email_accounts = [
            {"account": n, "access_level": "read_write", "is_sender_identity": (i == 0)}
            for i, n in enumerate(email_names)
        ]
        a.calendar_accounts = [{"account": n, "access_level": "read_write"} for n in calendar_names]
        a.contacts_accounts = [{"account": n, "access_level": "read_write"} for n in contacts_names]
        await store.upsert(a)
        updated += 1
    return updated


if __name__ == "__main__":
    # ponytail: one runnable check covering the parse/serialise round-trip + scopes.
    md = """---
agent_name: Forge
role: Fitness coach
emoji: "🏋️"
skills: [scheduling, memory]
tools:
  - run_command
  - send_message
secrets: []
bot_token: "123:ABC"
allowed_user_ids: [111, 222]
tool_config:
  gh:
    enabled: true
  browser:
    enabled: true
    profile: forge
email_accounts:
  - account: fitness-agent
    access_level: read_write
    is_sender_identity: true
  - account: personal
    access_level: read
calendar_accounts:
  - account: fitness-agent
    access_level: read_write
contacts_accounts:
  - account: fitness-agent
    access_level: read_write
  - account: shared
    access_level: read
personalia: |
  You are Forge, a strength coach.
character: |
  Direct and motivating.
---
Extra prose in the body.
"""
    a = parse_markdown(md, name="fitness-coach")
    assert a.agent_name == "Forge", a.agent_name
    assert a.role == "Fitness coach", a.role
    assert a.emoji == "🏋️"
    assert a.skills == ["scheduling", "memory"], a.skills
    assert a.tools == ["run_command", "send_message"], a.tools
    assert a.bot_token == "123:ABC", a.bot_token
    assert a.allowed_user_ids == [111, 222], a.allowed_user_ids
    # #98: legacy personalia folded into character (prepended), body still appended.
    assert "Forge" in a.character and "motivating" in a.character and "Extra prose" in a.character
    assert a.character.index("Forge") < a.character.index("motivating")  # personalia first
    assert a.allows_skill("scheduling") and not a.allows_skill("email")
    assert a.allows_tool("run_command") and not a.allows_tool("send_email")
    assert a.tool_setting("gh") == {"enabled": True}, a.tool_setting("gh")
    assert a.tool_setting("browser") == {"enabled": True, "profile": "forge"}
    assert a.tool_setting("weather") is None  # no entry = inherit system config

    # #110: email/calendar account bindings + access levels.
    assert a.email_access("fitness-agent") == "read_write", a.email_accounts
    assert a.email_access("personal") == "read"
    assert a.email_access("work") is None  # not bound = no access
    assert a.sender_identity() == "fitness-agent", a.sender_identity()
    assert a.calendar_access("fitness-agent") == "read_write"
    assert a.calendar_access("personal") is None
    # Contacts bindings (read + read_write), no sender concept.
    assert a.contacts_access("fitness-agent") == "read_write"
    assert a.contacts_access("shared") == "read"
    assert a.contacts_access("work") is None

    # A read-only account marked as sender is force-upgraded to read_write, and
    # only the first sender identity survives.
    coerced = _as_account_list(
        [
            {"account": "a", "access_level": "read", "is_sender_identity": True},
            {"account": "b", "access_level": "read_write", "is_sender_identity": True},
            "c",  # bare string → read-only binding
        ],
        sender=True,
    )
    assert coerced[0] == {"account": "a", "access_level": "read_write", "is_sender_identity": True}
    assert coerced[1]["is_sender_identity"] is False  # second sender demoted
    assert coerced[2] == {"account": "c", "access_level": "read", "is_sender_identity": False}

    # Empty allowlists = allow everything (default agent semantics)…
    blank = Agent(name="default")
    assert blank.allows_skill("anything") and blank.allows_tool("anything")
    assert blank.tool_setting("gh") is None  # no per-tool config by default
    # …but no account bindings = NO email/calendar access (safe default, #110).
    assert blank.email_access("personal") is None and blank.sender_identity() is None

    # Round-trip through markdown preserves the structured fields.
    a2 = parse_markdown(to_markdown(a), name="fitness-coach")
    assert a2.agent_name == a.agent_name and a2.skills == a.skills and a2.tools == a.tools
    assert a2.character.strip() == a.character.strip()
    assert a2.bot_token == a.bot_token and a2.allowed_user_ids == a.allowed_user_ids
    assert a2.tool_config == a.tool_config, a2.tool_config
    assert a2.email_accounts == a.email_accounts, a2.email_accounts
    assert a2.calendar_accounts == a.calendar_accounts, a2.calendar_accounts
    assert a2.contacts_accounts == a.contacts_accounts, a2.contacts_accounts
    print("agents.py self-check OK")
