# Skill Creator

Create or update skills in the skills database. Use this when you need to add a
new skill, revise an existing skill, or inspect available skills. Skills are
stored in SQLite and seeded from markdown files at startup.

## Critical rules

- Always read existing skills before editing them.
- Use `--output json` when you need to parse results programmatically.
- Keep skills concise and task-focused; only include non-obvious guidance.
- Skill names must be lowercase letters, digits, and hyphens only.
- Use `--write-seed` when you want the skill to be available in the seed
  directory for future bootstraps.

## Skill creation workflow

1. Check existing skills to avoid duplicates.
2. Identify required tools and permissions.
3. Create or update the skill in the DB.
4. Optionally update the seed file for new deployments.

## Skill format (humux)

- Skills are plain markdown files stored in the DB.
- The first non-empty line becomes the summary shown in the UI.
- Prefer a short title line, then sections like Overview, Commands, Notes.
- Include exact CLI commands in fenced code blocks.
- Avoid long tutorials; rely on examples and short rule lists.

### Minimal template

Use this outline when creating new skills:

Skill Title
  One-line summary for the UI.

Overview
  What this skill enables.

Commands
  example --flags --here

Notes
  - Safety rules
  - Edge cases

## Tool onboarding checklist

When a new skill depends on a new CLI tool or script:

- Add the CLI script to `tools/`.
- Allow the command prefix in `core/executor.py`.
- Add permission rules in `core/permissions.py`.
- Write the skill markdown and upsert it into the DB.

## List skills

```bash
python3 /app/tools/skills.py list --output json
```

## Read a skill

```bash
python3 /app/tools/skills.py show memory --output json
```

## Create or update a skill

Provide content via stdin to avoid shell quoting issues:

```bash
python3 /app/tools/skills.py upsert --name weather --stdin
```

If you also want to write/update the seed markdown file:

```bash
python3 /app/tools/skills.py upsert --name weather --stdin --write-seed
```

## Delete a skill

```bash
python3 /app/tools/skills.py delete weather
```
