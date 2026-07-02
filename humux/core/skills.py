"""Skills engine — loads skill docs from a SQLite-backed store."""

from __future__ import annotations

from pathlib import Path

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    name TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    summary TEXT DEFAULT '',
    created_at DATETIME DEFAULT (datetime('now')),
    updated_at DATETIME DEFAULT (datetime('now'))
);
"""


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
        """Seed missing skills from markdown files."""
        await self._ensure_schema()
        if not self.seed_dir or not self.seed_dir.exists():
            return False

        inserted = 0
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT name FROM skills")
            existing = {row[0] for row in await cursor.fetchall()}
            for skill_file in sorted(self.seed_dir.glob("*.md")):
                content = skill_file.read_text().strip()
                if not content:
                    continue
                name = skill_file.stem
                if name in existing:
                    continue
                summary = _extract_summary(content)
                await db.execute(
                    "INSERT INTO skills (name, content, summary) VALUES (?, ?, ?)",
                    (name, content, summary),
                )
                inserted += 1
            await db.commit()
        return inserted > 0

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
        await self._ensure_schema()
        summary = _extract_summary(content)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO skills (name, content, summary) VALUES (?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET "
                "content = excluded.content, summary = excluded.summary, "
                "updated_at = datetime('now')",
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
        (an agent's allowlist; ``None``/empty = all). Backs the index block and
        the ``list_skills``/``search_skills`` discovery tools."""
        skills = await self.store.list_skills()
        if allow:
            allowed = set(allow)
            skills = [s for s in skills if s["name"] in allowed]
        return [{"name": s["name"], "summary": (s.get("summary") or "").strip()} for s in skills]

    async def get_index_block(self, allow: list[str] | None = None) -> str:
        """Render the skills index. When ``allow`` is given (an agent's
        allowlist), only those skills are advertised; ``None``/empty = all."""
        entries = await self.index_entries(allow=allow)
        if not entries:
            return ""
        return "\n".join(
            f"- {e['name']}: {e['summary']}" if e["summary"] else f"- {e['name']}" for e in entries
        )

    async def search_index(
        self, query: str, allow: list[str] | None = None, limit: int = 10
    ) -> list[dict]:
        """Top-``limit`` index entries matching ``query`` (keyword scored over
        name + summary), scoped to ``allow``. An empty query returns the first
        ``limit`` entries (a cheap browse). No match → empty list.

        ponytail: lexical scoring only; the issue defers embedding ranking until
        keyword search measurably falls short.
        """
        entries = await self.index_entries(allow=allow)
        terms = [t for t in query.lower().split() if t]
        if not terms:
            return entries[:limit]
        scored = []
        for e in entries:
            haystack = f"{e['name']} {e['summary']}".lower()
            score = sum(haystack.count(t) for t in terms)
            if any(t in e["name"].lower() for t in terms):
                score += 5  # a name hit beats a summary hit
            if score:
                scored.append((score, e))
        scored.sort(key=lambda se: (-se[0], se[1]["name"]))
        return [e for _, e in scored[:limit]]

    async def get_skill_content(self, name: str) -> str:
        skill = await self.store.get_skill(name)
        if not skill:
            return ""
        return str(skill.get("content", ""))
