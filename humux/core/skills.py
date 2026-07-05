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
        """Seed missing skills from markdown files and adopt hashes for pre-migration rows.

        Does NOT re-seed existing rows when the seed file changes — use
        ``reset_skill_to_seed()`` for that (triggered by a UI button).
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

        inserted = 0
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT name, content, seed_hash FROM skills")
            existing = {row[0]: {"content": row[1], "seed_hash": row[2]} for row in await cursor.fetchall()}

            for skill_file in sorted(self.seed_dir.glob("*.md")):
                content = skill_file.read_text().strip()
                if not content:
                    continue
                name = skill_file.stem
                file_hash = hashlib.sha256(content.encode()).hexdigest()
                summary = _extract_summary(content)

                if name not in existing:
                    # New skill — insert with hash.
                    await db.execute(
                        "INSERT INTO skills (name, content, summary, seed_hash) VALUES (?, ?, ?, ?)",
                        (name, content, summary, file_hash),
                    )
                    inserted += 1
                elif existing[name]["seed_hash"] is None and existing[name]["content"] == content:
                    # Pre-migration row still matching the seed — adopt the hash.
                    await db.execute(
                        "UPDATE skills SET seed_hash = ? WHERE name = ?",
                        (file_hash, name),
                    )
            await db.commit()
        return inserted > 0

    def _seed_path(self, name: str) -> Path | None:
        """Return the seed file path for a skill name, or None."""
        if not self.seed_dir:
            return None
        path = self.seed_dir / f"{name}.md"
        return path if path.exists() else None

    def _seed_hash_for(self, name: str) -> str | None:
        """Compute sha256 of the seed file for this skill, or None."""
        path = self._seed_path(name)
        if path is None:
            return None
        content = path.read_text().strip()
        if not content:
            return None
        return hashlib.sha256(content.encode()).hexdigest()

    async def reset_skill_to_seed(self, name: str) -> bool:
        """Re-seed a single skill from its markdown file. No-op if the skill
        has no seed file or the seed file hasn't changed. Returns True if the
        skill was updated, False otherwise."""
        path = self._seed_path(name)
        if path is None:
            return False
        content = path.read_text().strip()
        if not content:
            return False
        file_hash = hashlib.sha256(content.encode()).hexdigest()
        summary = _extract_summary(content)
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "UPDATE skills SET content = ?, summary = ?, seed_hash = ?, "
                "updated_at = datetime('now') WHERE name = ?",
                (content, summary, file_hash, name),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def list_skills(self) -> list[dict]:
        await self.ensure_seeded()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT name, summary, content, seed_hash, updated_at FROM skills ORDER BY name"
            )
            rows = [dict(row) for row in await cursor.fetchall()]
        # Annotate each row with whether the seed has drifted.
        for r in rows:
            if r.get("seed_hash"):
                current = self._seed_hash_for(r["name"])
                r["stale_seed"] = current is not None and current != r["seed_hash"]
            else:
                r["stale_seed"] = False
        return rows

    async def get_skill(self, name: str) -> dict | None:
        await self.ensure_seeded()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT name, content, summary, seed_hash, updated_at FROM skills WHERE name = ?",
                (name,),
            )
            row = await cursor.fetchone()
            if not row:
                return None
            skill = dict(row)
            if skill.get("seed_hash"):
                current = self._seed_hash_for(skill["name"])
                skill["stale_seed"] = current is not None and current != skill["seed_hash"]
            else:
                skill["stale_seed"] = False
            return skill

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


# ── Skill file validator ──────────────────────────────────────────────────────

# The canonical allowlist for command prefix validation.  Imported at module
# level so the validator stays in sync with the executor without an import
# cycle (executor → skills → executor would loop).  We replicate the list
# here; keep it in sync with ToolExecutor.ALLOWED_PREFIXES.
_ALLOWED_PREFIXES: list[str] = [
    "curl",
    "himalaya",
    "jq",
    "wacli",
    "python3",
    "sqlite3",
    "gh",
    "git",
    "w3m",
    "pandoc",
    "pdftotext",
    "rg",
    "yt-dlp",
    "cal",
    "cp",
]

_SAFE_FILTERS: set[str] = {
    "jq", "head", "tail", "rg", "cat", "sort", "uniq", "wc", "grep", "cut", "tr", "column",
}

# Tools that are registered as callable tools (not CLI commands) — their
# names are valid as tool-names inside bash code blocks.
_KNOWN_TOOL_NAMES: set[str] = {
    # Communication tools
    "send_email",
    "reply_email",
    "send_message",
    "set_reaction",
    # Calendar
    "create_calendar_event",
    # Contacts
    "create_contact",
    "search_contacts",
    # Web / search
    "web_search",
    "generate_image",
    # Workspace file tools
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "grep",
    "run_command_in_dir",
    # Memory
    "remember",
    "recall_memory",
    "list_secrets",
    "request_secret",
    # Scheduling
    "manage_jobs",
    # Agent
    "spawn_subagent",
    "load_skill",
}


class ValidationError:
    """A single validation finding — either an error or a warning."""

    def __init__(self, file: str, line: int, message: str, severity: str = "error"):
        self.file = file
        self.line = line
        self.message = message
        self.severity = severity  # "error" | "warning"

    def __str__(self) -> str:
        prefix = "E" if self.severity == "error" else "W"
        return f"{prefix}: {self.file}:{self.line}: {self.message}"


def _validate_h1_title(path: Path, content: str) -> list[ValidationError]:
    """Check that the file has a valid H1 title matching the filename."""
    errors: list[ValidationError] = []
    lines = content.splitlines()
    h1_found = False
    for i, line in enumerate(lines):
        if line.startswith("# ") and not line.startswith("## "):
            h1_found = True
            break
    if not h1_found:
        errors.append(ValidationError(
            str(path), 1,
            "Missing H1 title (first line should be '# Title')",
        ))
    return errors


def _validate_code_blocks(path: Path, content: str) -> list[ValidationError]:
    """Check that fenced code blocks are properly opened and closed."""
    errors: list[ValidationError] = []
    lines = content.splitlines()
    fence_open = False
    fence_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            if fence_open:
                fence_open = False
            else:
                fence_open = True
                fence_start = i + 1
    if fence_open:
        errors.append(ValidationError(
            str(path), fence_start,
            "Unclosed fenced code block (opening ``` without closing ```)",
        ))
    return errors


def _validate_bash_commands(path: Path, content: str) -> list[ValidationError]:
    """Check that every command in bash code blocks complies with the prefix
    allowlist.  Ignores lines that look like commentary, blank lines, or
    variable assignments.  Handles backslash and pipe continuations."""
    errors: list[ValidationError] = []
    lines = content.splitlines()
    in_bash = False
    buf: list[str] = []
    buf_start: int = 0
    continued: bool = False  # True when the previous line ended with a backslash
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_bash and buf:
                _check_command_buffer(buf, buf_start, path, errors)
                buf = []
                continued = False
            lang = stripped.removeprefix("```").strip().lower()
            if in_bash:
                in_bash = False
            elif lang in ("bash", "sh"):
                in_bash = True
            continue
        if not in_bash:
            continue
        if not stripped or stripped.startswith("#"):
            if buf:
                _check_command_buffer(buf, buf_start, path, errors)
                buf = []
                continued = False
            continue
        if "=" in stripped and not stripped.startswith(" "):
            lhs = stripped.split("=", 1)[0]
            if lhs.isidentifier() or lhs.isupper():
                continue
        ends_with_bs = stripped.endswith("\\")
        is_pipe_cont = stripped.startswith("|")
        if buf:
            if continued or is_pipe_cont:
                buf.append(line)
                continued = ends_with_bs
                if not continued and not is_pipe_cont:
                    _check_command_buffer(buf, buf_start, path, errors)
                    buf = []
                    continued = False
                continue
            else:
                _check_command_buffer(buf, buf_start, path, errors)
                buf = []
                continued = False
        if ends_with_bs:
            buf = [line]
            buf_start = i + 1
            continued = True
        else:
            _check_single_command(stripped, i + 1, path, errors)
    if in_bash and buf:
        _check_command_buffer(buf, buf_start, path, errors)
    return errors


def _first_token(stripped: str) -> str:
    """Extract the first token from a stripped command line, stripping any
    opening parenthesis so that ``write_file(path=...)`` is recognised as
    the tool name ``write_file``."""
    token = stripped.split(maxsplit=1)[0] if stripped else ""
    # Strip opening parenthesis and everything after for tool-name matching
    if "(" in token:
        token = token.split("(")[0]
    return token


def _command_allowed(first_token: str) -> bool:
    """Check whether *first_token* looks like an allowed command start."""
    if not first_token:
        return True
    if first_token in _KNOWN_TOOL_NAMES:
        return True
    if first_token in _SAFE_FILTERS:
        return True
    if first_token.startswith("$"):
        return True
    if first_token in ("echo", "printf", "cd", "export", "source", "."):
        return True
    for prefix in _ALLOWED_PREFIXES:
        if first_token.startswith(prefix):
            return True
    return False


def _check_single_command(stripped: str, line_no: int, path: Path, errors: list[ValidationError]) -> None:
    """Check a single (non-continuation) command line."""
    token = _first_token(stripped)
    if token and not _command_allowed(token):
        errors.append(ValidationError(
            str(path), line_no,
            f"Command '{token}' is not in the allowed prefix list "
            f"({_ALLOWED_PREFIXES})",
        ))


def _check_command_buffer(buf: list[str], start_line: int, path: Path, errors: list[ValidationError]) -> None:
    """Check a logical command assembled from continuation lines by
    examining the first token of the first line."""
    if not buf:
        return
    first = buf[0].strip()
    token = _first_token(first)
    if token and not _command_allowed(token):
        errors.append(ValidationError(
            str(path), start_line,
            f"Command '{token}' is not in the allowed prefix list "
            f"({_ALLOWED_PREFIXES})",
        ))


def _validate_cross_references(path: Path, content: str) -> list[ValidationError]:
    """Check that referenced tool scripts or file paths actually exist, using
    the seed directory as the root.  Limited to obvious script references."""
    errors: list[ValidationError] = []
    lines = content.splitlines()
    seed_dir = path.parent
    for i, line in enumerate(lines):
        if "tools/" in line and ("/" in line or "`" in line):
            for part in line.split():
                clean = part.strip("`\"'(),.")
                if clean.startswith("./tools/") or clean.startswith("tools/"):
                    ref = seed_dir.parent / clean
                    if not ref.exists():
                        errors.append(ValidationError(
                            str(path), i + 1,
                            f"Referenced path '{clean}' does not exist",
                            severity="warning",
                        ))
    return errors


def validate_skill_file(path: Path, strict: bool = False) -> list[ValidationError]:
    """Validate a single skill markdown file.  Returns a list of findings (empty
    = file is valid)."""
    if not path.exists():
        return [ValidationError(str(path), 1, "File not found")]
    content = path.read_text()
    errors: list[ValidationError] = []
    errors.extend(_validate_h1_title(path, content))
    errors.extend(_validate_code_blocks(path, content))
    errors.extend(_validate_bash_commands(path, content))
    errors.extend(_validate_cross_references(path, content))

    if strict:
        lines = content.splitlines()
        in_code = False
        has_description = False
        for line in lines[1:5]:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue
            if stripped and not stripped.startswith("#"):
                has_description = True
                break
        if not has_description:
            errors.append(ValidationError(
                str(path), 2,
                "Missing description paragraph after the title",
                severity="warning",
            ))
    return errors


def validate_skill_content(content: str, strict: bool = False) -> list[ValidationError]:
    """Validate skill content in-memory, without a file on disk.

    Reuses the same internal validators as ``validate_skill_file`` but
    passes a synthetic path so line-numbered errors still make sense.
    """
    dummy = Path("<inline>")
    errors: list[ValidationError] = []
    errors.extend(_validate_h1_title(dummy, content))
    errors.extend(_validate_code_blocks(dummy, content))
    errors.extend(_validate_bash_commands(dummy, content))
    errors.extend(_validate_cross_references(dummy, content))
    if strict:
        lines = content.splitlines()
        in_code = False
        has_description = False
        for line in lines[1:5]:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue
            if stripped and not stripped.startswith("#"):
                has_description = True
                break
        if not has_description:
            errors.append(ValidationError(
                str(dummy), 2,
                "Missing description paragraph after the title",
                severity="warning",
            ))
    return errors


def validate_skill_dir(seed_dir: str | Path, strict: bool = False) -> list[ValidationError]:
    """Validate all skill markdown files in *seed_dir*."""
    d = Path(seed_dir)
    if not d.exists():
        return [ValidationError(str(d), 1, f"Seed directory '{seed_dir}' not found")]
    errors: list[ValidationError] = []
    for skill_file in sorted(d.glob("*.md")):
        errors.extend(validate_skill_file(skill_file, strict=strict))
    return errors


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m core.skills validate``."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m core.skills", description="Skills CLI")
    sub = parser.add_subparsers(dest="command")

    validate_parser = sub.add_parser("validate", help="Validate skill files")
    validate_parser.add_argument(
        "--seed-dir", default="skills/",
        help="Path to the skills seed directory (default: skills/)",
    )
    validate_parser.add_argument(
        "--strict", action="store_true",
        help="Enable additional style warnings",
    )

    args = parser.parse_args(argv)
    if args.command == "validate":
        errors = validate_skill_dir(args.seed_dir, strict=args.strict)
        for err in errors:
            print(str(err), file=__import__("sys").stderr)
        return 1 if any(e.severity == "error" for e in errors) else 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
