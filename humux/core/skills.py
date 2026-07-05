"""Skills engine — loads skill docs from a SQLite-backed store."""

from __future__ import annotations

import hashlib
from pathlib import Path

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    name TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    summary TEXT DEFAULT '',
    seed_hash TEXT DEFAULT NULL,
    created_at DATETIME DEFAULT (datetime('now')),
    updated_at DATETIME DEFAULT (datetime('now'))
);
"""

# Migration: add seed_hash to existing tables (safe no-op if already present).
_MIGRATE_SEED_HASH = "ALTER TABLE skills ADD COLUMN seed_hash TEXT DEFAULT NULL;"


# One-line instruction prepended to the skills index (#178), telling the model
# skills are read on demand via bash. Shared by the runtime prompt and the admin
# prompt-preview so the preview can't drift from what the model actually sees.
SKILLS_INDEX_HEADER = (
    "Skills are reusable instructions for specific tasks. Before acting on a task "
    "a skill covers, read it with bash: `python3 /app/tools/skills.py show <name>`."
)


def render_skills_index(entries: list[dict]) -> str:
    """Render index ``{name, summary}`` rows as the ``<available_skills>`` body
    (header line + one ``<skill>`` element each). Empty rows → empty string."""
    if not entries:
        return ""
    lines = [SKILLS_INDEX_HEADER]
    lines += [
        f'<skill name="{e["name"]}">{(e.get("summary") or "").strip()}</skill>' for e in entries
    ]
    return "\n".join(lines)


def _extract_summary(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        return stripped[:120]
    return ""


class SkillsStore:
    """SQLite-backed store for skill documents."""

    def __init__(self, db_path: str = "data/skills.db", seed_dir: str | Path = "skills/"):
        self.db_path = db_path
        self.seed_dir = Path(seed_dir) if seed_dir else None
        self._ready = False

    async def _ensure_schema(self) -> None:
        if self._ready:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(_SCHEMA)
        self._ready = True

    async def _count(self) -> int:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM skills")
            row = await cursor.fetchone()
            return int(row[0]) if row else 0

    async def ensure_seeded(self) -> bool:
        """Seed / re-seed skills from markdown files (#182).

        Uses a content hash (``seed_hash``) to detect when a source file has
        changed. Only overwrites rows whose ``seed_hash`` matches the current
        file hash — user-edited rows (``seed_hash IS NULL``) are never touched.
        """
        await self._ensure_schema()
        # Run migration (safe no-op if column already exists).
        async with aiosqlite.connect(self.db_path) as db:
            try:
                await db.execute(_MIGRATE_SEED_HASH)
            except aiosqlite.OperationalError:  # already present
                pass
            await db.commit()

        if not self.seed_dir or not self.seed_dir.exists():
            return False

        changed = 0
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT name, content, seed_hash FROM skills")
            existing_rows = {row[0]: {"content": row[1], "seed_hash": row[2]} for row in await cursor.fetchall()}

            for skill_file in sorted(self.seed_dir.glob("*.md")):
                content = skill_file.read_text().strip()
                if not content:
                    continue
                name = skill_file.stem
                file_hash = hashlib.sha256(content.encode()).hexdigest()
                summary = _extract_summary(content)

                if name not in existing_rows:
                    # New skill — insert.
                    await db.execute(
                        "INSERT INTO skills (name, content, summary, seed_hash) VALUES (?, ?, ?, ?)",
                        (name, content, summary, file_hash),
                    )
                    changed += 1
                else:
                    row = existing_rows[name]
                    if row["seed_hash"] is None:
                        # Pre-migration or user-edited. If content still matches
                        # the seed file, adopt the hash so future re-seeds work.
                        if row["content"] == content:
                            await db.execute(
                                "UPDATE skills SET seed_hash = ? WHERE name = ?",
                                (file_hash, name),
                            )
                    elif row["seed_hash"] != file_hash:
                        # Seed file changed — re-seed.
                        await db.execute(
                            "UPDATE skills SET content = ?, summary = ?, "
                            "seed_hash = ?, updated_at = datetime('now') "
                            "WHERE name = ?",
                            (content, summary, file_hash, name),
                        )
                        changed += 1
                    # Same hash → no-op.

            await db.commit()
        return changed > 0

    async def list_skills(self) -> list[dict]:
        await self.ensure_seeded()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT name, summary, content, updated_at FROM skills ORDER BY name"
            )
            return [dict(row) for row in await cursor.fetchall()]

    async def get_skill(self, name: str) -> dict | None:
        await self.ensure_seeded()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT name, content, summary, updated_at FROM skills WHERE name = ?",
                (name,),
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def upsert_skill(self, name: str, content: str) -> None:
        """Upsert a skill, clearing its seed_hash (user edit = no longer pristine)."""
        await self._ensure_schema()
        summary = _extract_summary(content)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO skills (name, content, summary) VALUES (?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET "
                "content = excluded.content, summary = excluded.summary, "
                "seed_hash = NULL, updated_at = datetime('now')",
                (name, content, summary),
            )
            await db.commit()

    async def delete_skill(self, name: str) -> bool:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("DELETE FROM skills WHERE name = ?", (name,))
            await db.commit()
            return cursor.rowcount > 0


class SkillsEngine:
    """Skill index + lazy loader for LLM usage."""

    def __init__(self, db_path: str = "data/skills.db", seed_dir: str | Path = "skills/"):
        self.store = SkillsStore(db_path=db_path, seed_dir=seed_dir)

    async def index_entries(self, allow: list[str] | None = None) -> list[dict]:
        """The skills index as ``{name, summary}`` rows, scoped to ``allow``
        (an agent's allowlist; ``None``/empty = all). Backs the index block,
        the admin skills view, and the Telegram skill commands."""
        skills = await self.store.list_skills()
        if allow:
            allowed = set(allow)
            skills = [s for s in skills if s["name"] in allowed]
        return [{"name": s["name"], "summary": (s.get("summary") or "").strip()} for s in skills]

    async def get_index_block(self, allow: list[str] | None = None) -> str:
        """Render the skills index as an XML-style listing (#178). When ``allow``
        is given (an agent's allowlist), only those skills are advertised;
        ``None``/empty = all.

        Skills are domain knowledge, not tools: the index carries name + summary
        + how to load, and the model pulls a body on demand via bash (the
        ``skills.py show`` read is pre-approved).
        """
        return render_skills_index(await self.index_entries(allow=allow))

    async def get_skill_content(self, name: str) -> str:
        skill = await self.store.get_skill(name)
        if not skill:
            return ""
        return str(skill.get("content", ""))
