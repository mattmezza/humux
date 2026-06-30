"""Persona (profile) engine — SQLite-backed store of swappable agent identities.

A **persona** is a first-class identity the agent can take on (fitness coach,
finance assistant, …). It is exactly four things (issue #13):

- its own system-prompt identity (``personalia`` + ``character``),
- a **skill allowlist** (which skills it may load — empty means *all*),
- a **tool scope** (which function-tools are advertised — empty means *all*),
- a **secret scope** (vault namespaces it may use — stored now, enforced by #19),
- **tool identities** (``tool_config``: own ``gh`` token / browser profile per
  external CLI tool — #93; empty means *inherit the system-wide config*).

The store mirrors :mod:`core.skills`: markdown files with YAML frontmatter,
seeded into SQLite at startup, editable in the admin UI. When no persona is
active the agent falls back to the configured ``character``/``personalia`` so
first-run behaviour is unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import aiosqlite
import yaml

_SCHEMA = """
CREATE TABLE IF NOT EXISTS personae (
    name TEXT PRIMARY KEY,
    agent_name TEXT DEFAULT '',
    role TEXT DEFAULT '',
    emoji TEXT DEFAULT '',
    voice TEXT DEFAULT '',
    personalia TEXT DEFAULT '',
    character TEXT DEFAULT '',
    skills TEXT DEFAULT '',
    tools TEXT DEFAULT '',
    secrets TEXT DEFAULT '',
    bot_token TEXT DEFAULT '',
    allowed_user_ids TEXT DEFAULT '',
    tool_config TEXT DEFAULT '',
    created_at DATETIME DEFAULT (datetime('now')),
    updated_at DATETIME DEFAULT (datetime('now'))
);
-- Slugs deliberately removed (deleted, or renamed away from) so seeding does not
-- resurrect them from their markdown file on the next list (#102). Re-creating a
-- slug via upsert/rename clears its tombstone.
CREATE TABLE IF NOT EXISTS persona_tombstones (
    name TEXT PRIMARY KEY,
    created_at DATETIME DEFAULT (datetime('now'))
);
"""

# Columns added after the table first shipped — applied to existing DBs on open.
_MIGRATIONS = (
    "ALTER TABLE personae ADD COLUMN bot_token TEXT DEFAULT ''",  # #29
    "ALTER TABLE personae ADD COLUMN allowed_user_ids TEXT DEFAULT ''",  # #29
    "ALTER TABLE personae ADD COLUMN tool_config TEXT DEFAULT ''",  # #93
)


@dataclass(slots=True)
class Persona:
    """A swappable agent identity + its scopes."""

    name: str  # slug / identifier (PK)
    agent_name: str = ""  # name the assistant goes by when active; empty = global agent.name
    role: str = ""
    emoji: str = ""
    voice: str = ""  # TTS voice override; empty = configured default
    personalia: str = ""
    character: str = ""
    skills: list[str] = field(default_factory=list)  # allowlist; [] = all
    tools: list[str] = field(default_factory=list)  # allowlist; [] = all
    secrets: list[str] = field(default_factory=list)  # vault scope; stored only (#19)
    bot_token: str = ""  # own Telegram bot; empty = reachable only via the default bot (#29)
    allowed_user_ids: list[int] = field(default_factory=list)  # bot ACL; [] = inherit global
    # Per-persona config for the optional external CLI tools (gh, browser) — #93.
    # Shape: {tool_key: {"enabled": bool, **tool-specific cfg}}. Empty = inherit
    # the system-wide tool config (own credentials/profile fall back to shared).
    tool_config: dict = field(default_factory=dict)

    def allows_skill(self, name: str) -> bool:
        return not self.skills or name in self.skills

    def allows_tool(self, name: str) -> bool:
        return not self.tools or name in self.tools

    def tool_setting(self, key: str) -> dict | None:
        """This persona's config for external tool ``key`` (gh/browser), or None
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


def _as_int_list(value: object) -> list[int]:
    """Coerce a frontmatter / form value into a clean list of ints (drops non-numeric)."""
    out: list[int] = []
    for item in _as_list(value):
        try:
            out.append(int(item))
        except ValueError:
            continue
    return out


def parse_markdown(text: str, *, name: str) -> Persona:
    """Parse a persona markdown doc (YAML frontmatter + optional body).

    Identity lives in the frontmatter (``personalia``/``character`` block
    scalars); any markdown body after the frontmatter is appended to
    ``character`` so authors can write free-form prose too.
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

    return Persona(
        name=name,
        agent_name=str(fm.get("agent_name", "") or ""),
        role=str(fm.get("role", "") or ""),
        emoji=str(fm.get("emoji", "") or ""),
        voice=str(fm.get("voice", "") or ""),
        personalia=str(fm.get("personalia", "") or ""),
        character=character,
        skills=_as_list(fm.get("skills")),
        tools=_as_list(fm.get("tools")),
        secrets=_as_list(fm.get("secrets")),
        bot_token=str(fm.get("bot_token", "") or ""),
        allowed_user_ids=_as_int_list(fm.get("allowed_user_ids")),
        tool_config=_as_tool_config(fm.get("tool_config")),
    )


def to_markdown(p: Persona) -> str:
    """Serialise a persona back to frontmatter markdown (the raw power-user view)."""
    fm = {
        "agent_name": p.agent_name,
        "role": p.role,
        "emoji": p.emoji,
        "voice": p.voice,
        "bot_token": p.bot_token,
        "allowed_user_ids": p.allowed_user_ids,
        "skills": p.skills,
        "tools": p.tools,
        "secrets": p.secrets,
        "tool_config": p.tool_config,
        "personalia": p.personalia,
        "character": p.character,
    }
    dumped = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True, default_flow_style=False)
    return f"---\n{dumped}---\n"


class PersonaStore:
    """SQLite-backed store for personae, seeded from a markdown directory."""

    def __init__(self, db_path: str = "data/personae.db", seed_dir: str | Path = "personae/"):
        self.db_path = db_path
        self.seed_dir = Path(seed_dir) if seed_dir else None
        self._ready = False

    async def _ensure_schema(self) -> None:
        if self._ready:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(_SCHEMA)
            for stmt in _MIGRATIONS:
                try:
                    await db.execute(stmt)
                except aiosqlite.OperationalError:
                    pass  # column already present
            await db.commit()
        self._ready = True

    async def ensure_seeded(self) -> bool:
        """Seed missing personae from the seed directory (idempotent)."""
        await self._ensure_schema()
        if not self.seed_dir or not self.seed_dir.exists():
            return False
        inserted = 0
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT name FROM personae")
            existing = {row[0] for row in await cursor.fetchall()}
            cursor = await db.execute("SELECT name FROM persona_tombstones")
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
    async def _upsert(db: aiosqlite.Connection, p: Persona) -> None:
        await db.execute(
            "INSERT INTO personae "
            "(name, agent_name, role, emoji, voice, personalia, character, skills, tools, "
            "secrets, bot_token, allowed_user_ids, tool_config) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET "
            "agent_name=excluded.agent_name, role=excluded.role, emoji=excluded.emoji, "
            "voice=excluded.voice, personalia=excluded.personalia, character=excluded.character, "
            "skills=excluded.skills, tools=excluded.tools, secrets=excluded.secrets, "
            "bot_token=excluded.bot_token, allowed_user_ids=excluded.allowed_user_ids, "
            "tool_config=excluded.tool_config, updated_at=datetime('now')",
            (
                p.name,
                p.agent_name,
                p.role,
                p.emoji,
                p.voice,
                p.personalia,
                p.character,
                "\n".join(p.skills),
                "\n".join(p.tools),
                "\n".join(p.secrets),
                p.bot_token,
                "\n".join(str(i) for i in p.allowed_user_ids),
                json.dumps(p.tool_config) if p.tool_config else "",
            ),
        )

    @staticmethod
    def _row_to_persona(row: aiosqlite.Row) -> Persona:
        return Persona(
            name=row["name"],
            agent_name=row["agent_name"] or "",
            role=row["role"] or "",
            emoji=row["emoji"] or "",
            voice=row["voice"] or "",
            personalia=row["personalia"] or "",
            character=row["character"] or "",
            skills=_as_list(row["skills"]),
            tools=_as_list(row["tools"]),
            secrets=_as_list(row["secrets"]),
            bot_token=(row["bot_token"] if "bot_token" in row.keys() else "") or "",
            allowed_user_ids=_as_int_list(
                row["allowed_user_ids"] if "allowed_user_ids" in row.keys() else ""
            ),
            tool_config=_as_tool_config(row["tool_config"] if "tool_config" in row.keys() else ""),
        )

    async def list_personae(self) -> list[Persona]:
        await self.ensure_seeded()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM personae ORDER BY name")
            return [self._row_to_persona(r) for r in await cursor.fetchall()]

    async def get(self, name: str) -> Persona | None:
        await self.ensure_seeded()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM personae WHERE name = ?", (name,))
            row = await cursor.fetchone()
            return self._row_to_persona(row) if row else None

    async def upsert(self, p: Persona) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            await self._upsert(db, p)
            # Deliberately (re)creating a slug clears any tombstone on it (#102).
            await db.execute("DELETE FROM persona_tombstones WHERE name = ?", (p.name,))
            await db.commit()

    async def delete(self, name: str) -> bool:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("DELETE FROM personae WHERE name = ?", (name,))
            if cursor.rowcount > 0:
                # Tombstone so seeding doesn't resurrect a deleted seed persona (#102).
                await db.execute(
                    "INSERT OR IGNORE INTO persona_tombstones(name) VALUES (?)", (name,)
                )
            await db.commit()
            return cursor.rowcount > 0

    async def rename(self, old: str, new: str) -> bool:
        """Change a persona's slug (its PRIMARY KEY). Returns False if ``old`` is
        missing; raises ``ValueError`` if ``new`` already names another persona.

        This only moves the personae row. The slug is referenced from other
        stores (per-chat bindings, memory scope, jobs, the active-persona config,
        and the ``telegram:<slug>`` bot channel) — the admin rename route cascades
        the new slug to those so nothing is orphaned.
        """
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT 1 FROM personae WHERE name = ?", (new,))
            if await cursor.fetchone():
                raise ValueError(f"A persona named '{new}' already exists")
            cursor = await db.execute(
                "UPDATE personae SET name = ?, updated_at = datetime('now') WHERE name = ?",
                (new, old),
            )
            if cursor.rowcount > 0:
                # Old slug must not resurrect from its seed file; new slug is now
                # live, so clear any tombstone it carried (#102).
                await db.execute(
                    "INSERT OR IGNORE INTO persona_tombstones(name) VALUES (?)", (old,)
                )
                await db.execute("DELETE FROM persona_tombstones WHERE name = ?", (new,))
            await db.commit()
            return cursor.rowcount > 0


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
personalia: |
  You are Forge, a strength coach.
character: |
  Direct and motivating.
---
Extra prose in the body.
"""
    p = parse_markdown(md, name="fitness-coach")
    assert p.agent_name == "Forge", p.agent_name
    assert p.role == "Fitness coach", p.role
    assert p.emoji == "🏋️"
    assert p.skills == ["scheduling", "memory"], p.skills
    assert p.tools == ["run_command", "send_message"], p.tools
    assert p.bot_token == "123:ABC", p.bot_token
    assert p.allowed_user_ids == [111, 222], p.allowed_user_ids
    assert "Forge" in p.personalia
    assert "motivating" in p.character and "Extra prose" in p.character  # body appended
    assert p.allows_skill("scheduling") and not p.allows_skill("email")
    assert p.allows_tool("run_command") and not p.allows_tool("send_email")
    assert p.tool_setting("gh") == {"enabled": True}, p.tool_setting("gh")
    assert p.tool_setting("browser") == {"enabled": True, "profile": "forge"}
    assert p.tool_setting("weather") is None  # no entry = inherit system config

    # Empty allowlists = allow everything (default persona semantics).
    blank = Persona(name="default")
    assert blank.allows_skill("anything") and blank.allows_tool("anything")
    assert blank.tool_setting("gh") is None  # no per-tool config by default

    # Round-trip through markdown preserves the structured fields.
    p2 = parse_markdown(to_markdown(p), name="fitness-coach")
    assert p2.agent_name == p.agent_name and p2.skills == p.skills and p2.tools == p.tools
    assert p2.personalia.strip() == p.personalia.strip()
    assert p2.bot_token == p.bot_token and p2.allowed_user_ids == p.allowed_user_ids
    assert p2.tool_config == p.tool_config, p2.tool_config
    print("personae.py self-check OK")
