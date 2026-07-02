# Scheduling — Jobs Management

You can create, list, edit, and cancel scheduled jobs. Jobs run automatically
at the configured time and deliver results to the user's messaging channel.

There are two ways to manage jobs:

1. **`manage_jobs` tool** — for common operations (create, list, cancel)
2. **`jobs.py` CLI** — for advanced operations (edit, pause, remove, detailed view)

## Quick reference: `manage_jobs` tool

### Create a recurring job (cron)

```json
{
  "action": "create",
  "job_id": "morning-brief",
  "task": "Send me a morning briefing with weather, calendar, and unread emails",
  "cron": "30 7 * * 1-5",
  "channel": "telegram",
  "description": "Morning briefing on weekdays at 07:30"
}
```

### Create a one-time job

```json
{
  "action": "create",
  "job_id": "remind-nick-email",
  "task": "Remind me to reply to Nick's email about the project deadline",
  "run_at": "2026-02-21T09:00:00+01:00",
  "channel": "telegram",
  "description": "One-time reminder"
}
```

### List all active jobs

```json
{
  "action": "list"
}
```

### Cancel a job

```json
{
  "action": "cancel",
  "job_id": "morning-brief"
}
```

## Cron expression format

Standard 5-field cron: `minute hour day month weekday`

| Expression       | Meaning                        |
|------------------|--------------------------------|
| `30 7 * * 1-5`  | Weekdays at 07:30              |
| `0 9 * * *`     | Every day at 09:00             |
| `0 */2 * * *`   | Every 2 hours                  |
| `0 8 1 * *`     | First of month at 08:00        |
| `0 20 * * 0`    | Every Sunday at 20:00          |
| `*/15 * * * *`  | Every 15 minutes               |

## Advanced: `jobs.py` CLI

Use `run_command` for operations not covered by the `manage_jobs` tool.

### List jobs

```bash
python3 /app/tools/jobs.py list --output json
```

### List all jobs including done/cancelled

```bash
python3 /app/tools/jobs.py list --all --output json
```

### Show a single job

```bash
python3 /app/tools/jobs.py show <job_id> --output json
```

### Create a cron job via CLI

```bash
python3 /app/tools/jobs.py create --id daily-standup \
  --cron "0 9 * * 1-5" \
  --type agent \
  --task "Prepare a standup summary from yesterday's activity" \
  --channel telegram \
  --description "Daily standup prep"
```

### Create a one-shot job via CLI

```bash
python3 /app/tools/jobs.py create --id remind-call \
  --once "2026-02-21T14:30:00+01:00" \
  --type agent \
  --task "Remind me about the call with the design team" \
  --channel telegram
```

### Edit an existing job

```bash
# Change the schedule
python3 /app/tools/jobs.py edit morning-brief --cron "0 8 * * 1-5"

# Change the task
python3 /app/tools/jobs.py edit morning-brief --task "New briefing instructions"

# Pause a job
python3 /app/tools/jobs.py edit morning-brief --status paused

# Resume a paused job
python3 /app/tools/jobs.py edit morning-brief --status active
```

### Cancel a job (keeps history)

```bash
python3 /app/tools/jobs.py cancel <job_id>
```

### Remove a job permanently

```bash
python3 /app/tools/jobs.py remove <job_id>
```

## Job types

| Type                   | Behavior                                             |
|------------------------|------------------------------------------------------|
| `agent`                | Runs the task through the agent, sends result to user |
| `agent_silent`         | Same as `agent`, but suppresses empty results         |
| `system`               | Executes a raw CLI command (no agent involved)        |
| `memory_consolidation` | Reviews and consolidates memories (internal)          |

When creating jobs for the user, use `agent` type. Use `agent_silent` for
background checks where "no news is good news" (e.g. email monitoring).

## Job statuses

| Status     | Meaning                        |
|------------|--------------------------------|
| `active`   | Job is scheduled and will run  |
| `paused`   | Job exists but won't run       |
| `done`     | One-shot job completed         |
| `cancelled`| Job was cancelled by user/agent|

## Important notes

- **Job IDs** should be lowercase, short, and descriptive (dashes ok). Example:
  `morning-brief`, `email-check`, `remind-dentist`.
- **One-shot jobs** are automatically marked `done` after they execute.
- **Channel** is where the result gets delivered. Usually `telegram` or `whatsapp`.
- When the user says "stop" or "don't do that anymore", use the `cancel` action
  on the relevant job.
- When the user says "remind me in X minutes", calculate the ISO datetime from
  the current time and create a one-shot job.
- Prefer the `manage_jobs` tool for simple create/list/cancel. Use the CLI only
  when you need to edit, pause/resume, or inspect job details.
- The CLI `list` and `show` subcommands are pre-approved (no user confirmation
  needed). The `create`, `edit`, `remove`, and `cancel` subcommands require
  user approval.
