#!/usr/bin/env python3
"""Skills CLI — list, show, create, and delete skills.

Usage examples:
  python3 /app/tools/skills.py list --output json
  python3 /app/tools/skills.py show memory --output json
  python3 /app/tools/skills.py upsert --name weather --stdin
  python3 /app/tools/skills.py delete weather
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.skills import SkillsStore  # noqa: E402

NAME_PATTERN = re.compile(r"^[a-z0-9-]+$")


def _default_db_path() -> str:
    if Path("/app/data").exists():
        return "/app/data/skills.db"
    return "data/skills.db"


def _default_seed_dir() -> str:
    if Path("/app/skills").exists():
        return "/app/skills"
    return "skills"


def _validate_name(name: str) -> str:
    value = name.strip()
    if not value:
        raise ValueError("Skill name is required")
    if "/" in value or "\\" in value:
        raise ValueError("Skill name cannot include path separators")
    if not NAME_PATTERN.match(value):
        raise ValueError("Skill name must be lowercase letters, digits, and hyphens")
    if "--" in value or value.startswith("-") or value.endswith("-"):
        raise ValueError("Skill name cannot start/end with hyphen or contain consecutive hyphens")
    return value


def _read_content(args: argparse.Namespace) -> str:
    sources = [bool(args.content), bool(args.file), bool(args.stdin)]
    if sum(sources) != 1:
        raise ValueError("Provide exactly one of --content, --file, or --stdin")
    if args.content:
        return args.content
    if args.file:
        content_path = Path(args.file)
        if not content_path.exists():
            raise ValueError(f"File not found: {content_path}")
        return content_path.read_text()
    return sys.stdin.read()


def _output(data, fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(data, indent=2, default=str))
        return
    if isinstance(data, list):
        for item in data:
            summary = (item.get("summary") or "").strip()
            if summary:
                print(f"- {item.get('name')}: {summary}")
            else:
                print(f"- {item.get('name')}")
        return
    if isinstance(data, dict) and "content" in data:
        print(data.get("content", ""))
        return
    print(data)


async def _list_skills(store: SkillsStore, fmt: str) -> None:
    skills = await store.list_skills()
    _output(skills, fmt)


async def _show_skill(store: SkillsStore, name: str, fmt: str) -> None:
    skill = await store.get_skill(name)
    if not skill:
        print(json.dumps({"error": f"Skill not found: {name}"}))
        sys.exit(1)
    _output(skill, fmt)


async def _upsert_skill(store: SkillsStore, name: str, content: str) -> None:
    cleaned = content.strip()
    if not cleaned:
        raise ValueError("Skill content is required")
    await store.upsert_skill(name, cleaned)


async def _delete_skill(store: SkillsStore, name: str) -> None:
    deleted = await store.delete_skill(name)
    if not deleted:
        print(json.dumps({"error": f"Skill not found: {name}"}))
        sys.exit(1)


def _write_seed_file(seed_dir: str, name: str, content: str) -> Path:
    seed_path = Path(seed_dir)
    if not seed_path.exists():
        raise ValueError(f"Seed directory not found: {seed_path}")
    dest = seed_path / f"{name}.md"
    dest.write_text(content.strip() + "\n")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage humux skills.")
    parser.add_argument("--db", default=_default_db_path(), help="Path to skills DB")
    parser.add_argument(
        "--seed-dir",
        default=_default_seed_dir(),
        help="Seed directory for skills markdown files",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List skills")
    list_parser.add_argument("--output", default="json", choices=["json", "text"])

    show_parser = subparsers.add_parser("show", help="Show a skill")
    show_parser.add_argument("name")
    show_parser.add_argument("--output", default="json", choices=["json", "text"])

    upsert_parser = subparsers.add_parser("upsert", help="Create or update a skill")
    upsert_parser.add_argument("--name", required=True)
    upsert_parser.add_argument("--content")
    upsert_parser.add_argument("--file")
    upsert_parser.add_argument("--stdin", action="store_true")
    upsert_parser.add_argument(
        "--write-seed",
        action="store_true",
        help="Also write/update the seed markdown file",
    )

    delete_parser = subparsers.add_parser("delete", help="Delete a skill")
    delete_parser.add_argument("name")

    args = parser.parse_args()

    store = SkillsStore(db_path=args.db, seed_dir=args.seed_dir)

    try:
        if args.command == "list":
            asyncio.run(_list_skills(store, args.output))
        elif args.command == "show":
            name = _validate_name(args.name)
            asyncio.run(_show_skill(store, name, args.output))
        elif args.command == "upsert":
            name = _validate_name(args.name)
            content = _read_content(args)
            asyncio.run(_upsert_skill(store, name, content))
            if args.write_seed:
                dest = _write_seed_file(args.seed_dir, name, content)
                print(json.dumps({"status": "seed_written", "path": str(dest)}))
        elif args.command == "delete":
            name = _validate_name(args.name)
            asyncio.run(_delete_skill(store, name))
        else:
            parser.print_help()
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
