"""Skills engine — loads skill docs from a SQLite-backed store."""

from __future__ import annotations

import asyncio
import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import aiosqlite
import yaml

_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    name TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    summary TEXT DEFAULT '',
    seed_hash TEXT DEFAULT NULL,
    origin TEXT DEFAULT NULL,
    created_at DATETIME DEFAULT (datetime('now')),
    updated_at DATETIME DEFAULT (datetime('now'))
);
"""

# Migrations: add columns to existing tables (safe no-op if already present).
_MIGRATE_SEED_HASH = "ALTER TABLE skills ADD COLUMN seed_hash TEXT DEFAULT NULL;"
# origin = "<repo-url>#<path-in-repo>" for skills installed from a git repo
# (#65). Origin-tagged skills are read-only: update = re-install from origin.
_MIGRATE_ORIGIN = "ALTER TABLE skills ADD COLUMN origin TEXT DEFAULT NULL;"

# Marker file inside an installed skill directory recording where it came from,
# so a rebuilt DB re-seeds with provenance intact (#65). One line: "url#path".
ORIGIN_MARKER = ".origin"

# Skill name rules (mirrors tools/skills.py): safe as a directory name.
_NAME_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")


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


def split_frontmatter(raw: str) -> tuple[dict, str]:
    """Split Agent Skills YAML frontmatter from the body (#65).

    Returns ``(meta, body)``; ``meta`` is ``{}`` when there is no frontmatter
    or it fails to parse (the whole text is then the body).
    """
    if not raw.startswith("---"):
        return {}, raw
    parts = raw.split("\n---", 2)
    if len(parts) < 2:
        return {}, raw
    try:
        meta = yaml.safe_load(parts[0][3:])
    except yaml.YAMLError:
        return {}, raw
    if not isinstance(meta, dict):
        return {}, raw
    body = parts[1]
    if len(parts) == 3:
        body += "\n---" + parts[2]
    return meta, body.lstrip("\n")


def _seed_summary(raw: str) -> str:
    """Summary for a seed file: frontmatter ``description`` beats the
    first-line heuristic. Agent Skills caps descriptions at 1024 chars; the
    index wants one line, so trim harder."""
    meta, body = split_frontmatter(raw)
    desc = str(meta.get("description") or "").strip()
    return " ".join(desc.split())[:300] if desc else _extract_summary(body)


def _bundled_files_header(name: str, base_dir: Path) -> str:
    """One-line pointer prepended to a directory skill's content so the model
    can reach the files bundled next to SKILL.md (#65). Relative paths inside
    the skill body resolve against this directory."""
    return (
        f"> Bundled files for this skill live in `{base_dir}/`. Relative paths in "
        f"this document resolve there. Read one with: "
        f"`python3 /app/tools/skills.py cat {name} <relative-path>`\n\n"
    )


def _load_seed(name: str, path: Path) -> tuple[str, str] | None:
    """Load a seed file into ``(stored_content, summary)``; None if empty.

    Directory skills (``<dir>/SKILL.md``) get the bundled-files header
    prepended; the seed hash is computed over this final stored content so
    drift detection keeps working for both formats.
    """
    raw = path.read_text().strip()
    if not raw:
        return None
    summary = _seed_summary(raw)
    if path.name == "SKILL.md":
        raw = _bundled_files_header(name, path.parent.resolve()) + raw
    return raw, summary


class SkillsStore:
    """SQLite-backed store for skill documents."""

    def __init__(
        self,
        db_path: str = "data/skills.db",
        seed_dir: str | Path = "skills/",
        installed_dir: str | Path = "data/skills",
    ):
        self.db_path = db_path
        self.seed_dir = Path(seed_dir) if seed_dir else None
        # Skills installed from git repos land here (#65) — a writable dir
        # (the seed dir is mounted read-only in Docker).
        self.installed_dir = Path(installed_dir) if installed_dir else None
        self._ready = False

    async def _ensure_schema(self) -> None:
        if self._ready:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(_SCHEMA)
            # Migrations for pre-existing tables (safe no-op when current).
            for migration in (_MIGRATE_SEED_HASH, _MIGRATE_ORIGIN):
                try:
                    await db.execute(migration)
                except aiosqlite.OperationalError:  # already present
                    pass
            await db.commit()
        self._ready = True

    async def _count(self) -> int:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM skills")
            row = await cursor.fetchone()
            return int(row[0]) if row else 0

    def _iter_seed_files(self) -> list[tuple[str, Path]]:
        """All seed files as ``(name, path)`` — flat ``<seed_dir>/<name>.md``,
        spec dirs ``<seed_dir>/<name>/SKILL.md`` (#65), and installed dirs
        ``<installed_dir>/<name>/SKILL.md``. First occurrence of a name wins."""
        out: list[tuple[str, Path]] = []
        seen: set[str] = set()
        roots = [d for d in (self.seed_dir, self.installed_dir) if d and d.exists()]
        for root in roots:
            for path in sorted(root.glob("*.md")) + sorted(root.glob("*/SKILL.md")):
                name = path.stem if path.name != "SKILL.md" else path.parent.name
                if name in seen:
                    continue
                seen.add(name)
                out.append((name, path))
        return out

    def _origin_for(self, path: Path) -> str | None:
        """The install origin ("url#path") recorded next to SKILL.md, if any."""
        if path.name != "SKILL.md":
            return None
        marker = path.parent / ORIGIN_MARKER
        if not marker.exists():
            return None
        return marker.read_text().strip() or None

    async def ensure_seeded(self) -> bool:
        """Seed missing skills from markdown files and adopt hashes for pre-migration rows.

        Does NOT re-seed existing rows when the seed file changes — use
        ``reset_skill_to_seed()`` for that (triggered by a UI button).
        """
        await self._ensure_schema()
        seeds = self._iter_seed_files()
        if not seeds:
            return False

        inserted = 0
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT name, content, seed_hash FROM skills")
            existing = {
                row[0]: {"content": row[1], "seed_hash": row[2]} for row in await cursor.fetchall()
            }

            for name, skill_file in seeds:
                loaded = _load_seed(name, skill_file)
                if loaded is None:
                    continue
                content, summary = loaded
                file_hash = hashlib.sha256(content.encode()).hexdigest()

                if name not in existing:
                    # New skill — insert with hash (+ origin for installed ones).
                    await db.execute(
                        "INSERT INTO skills"
                        " (name, content, summary, seed_hash, origin) VALUES (?, ?, ?, ?, ?)",
                        (name, content, summary, file_hash, self._origin_for(skill_file)),
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
        """Return the seed file path for a skill name, or None. Checks the flat
        seed file, the spec-format seed dir, and the installed dir (#65)."""
        candidates = []
        if self.seed_dir:
            candidates += [self.seed_dir / f"{name}.md", self.seed_dir / name / "SKILL.md"]
        if self.installed_dir:
            candidates.append(self.installed_dir / name / "SKILL.md")
        for path in candidates:
            if path.exists():
                return path
        return None

    def skill_base_dir(self, name: str) -> Path | None:
        """Directory holding a skill's bundled files, or None for flat skills."""
        path = self._seed_path(name)
        if path is None or path.name != "SKILL.md":
            return None
        return path.parent

    def _seed_hash_for(self, name: str) -> str | None:
        """Compute sha256 of the seeded content for this skill, or None."""
        path = self._seed_path(name)
        if path is None:
            return None
        loaded = _load_seed(name, path)
        if loaded is None:
            return None
        return hashlib.sha256(loaded[0].encode()).hexdigest()

    async def reset_skill_to_seed(self, name: str) -> bool:
        """(Re-)seed a single skill from its file on disk. Upserts, so it also
        covers a freshly installed skill not yet in the DB (#65). Returns True
        if a row was written."""
        path = self._seed_path(name)
        if path is None:
            return False
        loaded = _load_seed(name, path)
        if loaded is None:
            return False
        content, summary = loaded
        file_hash = hashlib.sha256(content.encode()).hexdigest()
        origin = self._origin_for(path)
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO skills (name, content, summary, seed_hash, origin)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(name) DO UPDATE SET"
                " content = excluded.content, summary = excluded.summary,"
                " seed_hash = excluded.seed_hash, origin = excluded.origin,"
                " updated_at = datetime('now')",
                (name, content, summary, file_hash, origin),
            )
            await db.commit()
            return True

    async def list_skills(self, offset: int | None = None, limit: int | None = None) -> list[dict]:
        await self.ensure_seeded()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            query = (
                "SELECT name, summary, content, seed_hash, origin, updated_at"
                " FROM skills ORDER BY name"
            )
            params: list[int] = []
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            if offset is not None:
                query += " OFFSET ?"
                params.append(offset)
            cursor = await db.execute(query, params)
            rows = [dict(row) for row in await cursor.fetchall()]
        # Annotate each row with whether the seed has drifted.
        for r in rows:
            if r.get("seed_hash"):
                current = self._seed_hash_for(r["name"])
                r["stale_seed"] = current is not None and current != r["seed_hash"]
            else:
                r["stale_seed"] = False
        return rows

    async def count_skills(self) -> int:
        """Return the total number of skills in the store."""
        await self.ensure_seeded()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM skills")
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def get_skill(self, name: str) -> dict | None:
        await self.ensure_seeded()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT name, content, summary, seed_hash, origin, updated_at"
                " FROM skills WHERE name = ?",
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
        """Upsert a skill, clearing its seed_hash (user edit = no longer pristine).

        Raises ``ValueError`` for installed (origin-tagged) skills — they are
        read-only; update by re-installing, customize by copying (#65).
        """
        await self._ensure_schema()
        existing = await self.get_skill(name)
        if existing and existing.get("origin"):
            raise ValueError(
                f"Skill '{name}' was installed from {existing['origin']} and is "
                "read-only. Delete it and create your own copy to customize."
            )
        summary = _seed_summary(content)
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
            deleted = cursor.rowcount > 0
        # Installed skills also own a directory under installed_dir — remove it
        # so ensure_seeded can't resurrect the row (#65). Never touches seed_dir.
        if self.installed_dir:
            skill_dir = self.installed_dir / name
            if (skill_dir / "SKILL.md").exists():
                shutil.rmtree(skill_dir)
                deleted = True
        return deleted

    async def install_from_git(self, repo_url: str, skill_path: str = "") -> dict:
        """Install (or update) a skill from a git repo into ``installed_dir`` (#65).

        ``skill_path`` is the directory inside the repo holding SKILL.md
        ("" = repo root is the skill). Clones shallow, copies the dir, records
        the origin marker, and (re-)seeds the DB row. Returns
        ``{name, origin, warnings}``; raises ``ValueError`` on anything wrong.
        """
        if not self.installed_dir:
            raise ValueError("No installed-skills directory configured")
        skill_path = skill_path.strip().strip("/")
        if ".." in skill_path.split("/"):
            raise ValueError("Skill path cannot contain '..'")
        with tempfile.TemporaryDirectory(prefix="skill-install-") as tmp:
            # to_thread: keep the event loop free during the clone (called from
            # the admin API).
            proc = await asyncio.to_thread(
                subprocess.run,
                ["git", "clone", "--depth", "1", repo_url, tmp],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode != 0:
                raise ValueError(f"git clone failed: {proc.stderr.strip()[-500:]}")
            src = Path(tmp) / skill_path if skill_path else Path(tmp)
            skill_md = src / "SKILL.md"
            if not skill_md.exists():
                raise ValueError(f"No SKILL.md at '{skill_path or '.'}' in {repo_url}")
            meta, _ = split_frontmatter(skill_md.read_text())
            name = str(meta.get("name") or src.name).strip()
            # Same name rules as tools/skills.py — the name becomes a directory
            # under installed_dir, so this is also the path-safety check.
            if not _NAME_PATTERN.match(name) or "--" in name:
                raise ValueError(
                    f"Invalid skill name {name!r} (lowercase letters, digits, hyphens)"
                )
            findings = validate_skill_file(skill_md)
            errors = [str(f) for f in findings if f.severity == "error"]
            if errors:
                raise ValueError("Skill failed validation: " + "; ".join(errors))
            dest = self.installed_dir / name
            if dest.exists():
                shutil.rmtree(dest)
            self.installed_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dest)
            shutil.rmtree(dest / ".git", ignore_errors=True)
            origin = f"{repo_url}#{skill_path}"
            (dest / ORIGIN_MARKER).write_text(origin + "\n")
        await self.reset_skill_to_seed(name)
        return {
            "name": name,
            "origin": origin,
            "warnings": [str(f) for f in findings if f.severity == "warning"],
        }

    async def update_installed_skill(self, name: str) -> dict:
        """Re-install a skill from its recorded origin (#65)."""
        skill = await self.get_skill(name)
        if not skill:
            raise ValueError(f"Skill not found: {name}")
        origin = skill.get("origin") or ""
        if "#" not in origin:
            raise ValueError(f"Skill '{name}' has no install origin — nothing to update")
        repo_url, _, skill_path = origin.partition("#")
        return await self.install_from_git(repo_url, skill_path)


class SkillsEngine:
    """Skill index + lazy loader for LLM usage."""

    def __init__(
        self,
        db_path: str = "data/skills.db",
        seed_dir: str | Path = "skills/",
        installed_dir: str | Path = "data/skills",
    ):
        self.store = SkillsStore(db_path=db_path, seed_dir=seed_dir, installed_dir=installed_dir)

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
    "python",  # community Agent Skills say `python`; executor allows it for skill scripts
    "sqlite3",
    "gh",
    "git",
    "w3m",
    "pandoc",
    "pdftotext",
    "pdftoppm",
    "rg",
    "yt-dlp",
    "cal",
    "cp",
]

_SAFE_FILTERS: set[str] = {
    "jq",
    "head",
    "tail",
    "rg",
    "cat",
    "sort",
    "uniq",
    "wc",
    "grep",
    "cut",
    "tr",
    "column",
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
        errors.append(
            ValidationError(
                str(path),
                1,
                "Missing H1 title (first line should be '# Title')",
            )
        )
    return errors


def _validate_frontmatter(path: Path, content: str) -> list[ValidationError]:
    """Agent Skills frontmatter checks for SKILL.md files (#65): ``name`` and
    ``description`` required, spec length caps, name matches the directory."""
    errors: list[ValidationError] = []
    meta, _ = split_frontmatter(content)
    if not meta:
        errors.append(
            ValidationError(str(path), 1, "SKILL.md requires YAML frontmatter (name, description)")
        )
        return errors
    name = str(meta.get("name") or "").strip()
    desc = str(meta.get("description") or "").strip()
    if not name:
        errors.append(ValidationError(str(path), 1, "Frontmatter is missing 'name'"))
    elif len(name) > 64:
        errors.append(ValidationError(str(path), 1, "Frontmatter 'name' exceeds 64 characters"))
    elif not _NAME_PATTERN.match(name):
        errors.append(
            ValidationError(str(path), 1, f"Frontmatter name {name!r} must be lowercase [a-z0-9-]")
        )
    elif path.name == "SKILL.md" and path.parent.name and name != path.parent.name:
        errors.append(
            ValidationError(
                str(path),
                1,
                f"Frontmatter name {name!r} does not match directory '{path.parent.name}'",
            )
        )
    if not desc:
        errors.append(ValidationError(str(path), 1, "Frontmatter is missing 'description'"))
    elif len(desc) > 1024:
        errors.append(
            ValidationError(str(path), 1, "Frontmatter 'description' exceeds 1024 characters")
        )
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
        errors.append(
            ValidationError(
                str(path),
                fence_start,
                "Unclosed fenced code block (opening ``` without closing ```)",
            )
        )
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


def _check_single_command(
    stripped: str, line_no: int, path: Path, errors: list[ValidationError]
) -> None:
    """Check a single (non-continuation) command line."""
    token = _first_token(stripped)
    if token and not _command_allowed(token):
        errors.append(
            ValidationError(
                str(path),
                line_no,
                f"Command '{token}' is not in the allowed prefix list ({_ALLOWED_PREFIXES})",
            )
        )


def _check_command_buffer(
    buf: list[str], start_line: int, path: Path, errors: list[ValidationError]
) -> None:
    """Check a logical command assembled from continuation lines by
    examining the first token of the first line."""
    if not buf:
        return
    first = buf[0].strip()
    token = _first_token(first)
    if token and not _command_allowed(token):
        errors.append(
            ValidationError(
                str(path),
                start_line,
                f"Command '{token}' is not in the allowed prefix list ({_ALLOWED_PREFIXES})",
            )
        )


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
                        errors.append(
                            ValidationError(
                                str(path),
                                i + 1,
                                f"Referenced path '{clean}' does not exist",
                                severity="warning",
                            )
                        )
    return errors


def validate_skill_file(path: Path, strict: bool = False) -> list[ValidationError]:
    """Validate a single skill markdown file.  Returns a list of findings (empty
    = file is valid)."""
    if not path.exists():
        return [ValidationError(str(path), 1, "File not found")]
    content = path.read_text()
    errors: list[ValidationError] = []
    is_spec = path.name == "SKILL.md"
    if is_spec:
        errors.extend(_validate_frontmatter(path, content))
        # Spec skills carry frontmatter, not an H1; and community skills freely
        # reference commands outside humux's executor allowlist — the executor
        # blocks those at runtime, so here they're a heads-up, not a failure.
        for finding in _validate_bash_commands(path, content):
            finding.severity = "warning"
            errors.append(finding)
    else:
        errors.extend(_validate_h1_title(path, content))
        errors.extend(_validate_bash_commands(path, content))
    errors.extend(_validate_code_blocks(path, content))
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
            errors.append(
                ValidationError(
                    str(path),
                    2,
                    "Missing description paragraph after the title",
                    severity="warning",
                )
            )
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
            errors.append(
                ValidationError(
                    str(dummy),
                    2,
                    "Missing description paragraph after the title",
                    severity="warning",
                )
            )
    return errors


def validate_skill_dir(seed_dir: str | Path, strict: bool = False) -> list[ValidationError]:
    """Validate all skill markdown files in *seed_dir*."""
    d = Path(seed_dir)
    if not d.exists():
        return [ValidationError(str(d), 1, f"Seed directory '{seed_dir}' not found")]
    errors: list[ValidationError] = []
    for skill_file in sorted(d.glob("*.md")) + sorted(d.glob("*/SKILL.md")):
        errors.extend(validate_skill_file(skill_file, strict=strict))
    return errors


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m core.skills validate``."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m core.skills", description="Skills CLI")
    sub = parser.add_subparsers(dest="command")

    validate_parser = sub.add_parser("validate", help="Validate skill files")
    validate_parser.add_argument(
        "--seed-dir",
        default="skills/",
        help="Path to the skills seed directory (default: skills/)",
    )
    validate_parser.add_argument(
        "--strict",
        action="store_true",
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
