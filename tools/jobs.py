#!/usr/bin/env python3
"""Jobs CLI — manage scheduled jobs from the command line.

Usage examples:
  python3 /app/tools/jobs.py list --output json
  python3 /app/tools/jobs.py show <job_id> --output json
  python3 /app/tools/jobs.py create --id morning-brief --cron "30 7 * * 1-5" \\
      --type agent --task "Send me a morning briefing" --channel telegram
  python3 /app/tools/jobs.py create --id remind-email --once "2026-02-21T09:00:00" \\
      --type agent --task "Reply to Nick's email agreeing with him" --channel telegram
  python3 /app/tools/jobs.py edit <job_id> --status paused
  python3 /app/tools/jobs.py edit <job_id> --task "New task text"
  python3 /app/tools/jobs.py remove <job_id>
  python3 /app/tools/jobs.py cancel <job_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.job_store import VALID_STATUSES, VALID_TYPES, JobStore  # noqa: E402

DB_PATH = "data/jobs.db"
# Also check the Docker path
if Path("/app/data").exists():
    DB_PATH = "/app/data/jobs.db"


def _get_store() -> JobStore:
    return JobStore(db_path=DB_PATH)


def _output(data, fmt: str = "json") -> None:
    """Print output in the requested format."""
    if fmt == "json":
        print(json.dumps(data, indent=2, default=str))
    elif fmt == "table":
        if isinstance(data, list):
            if not data:
                print("No jobs found.")
                return
            # Print a simple table
            headers = ["id", "type", "schedule", "cron", "run_at", "status", "channel", "task"]
            widths = {h: len(h) for h in headers}
            for row in data:
                for h in headers:
                    val = str(row.get(h, "") or "")
                    if h == "task":
                        val = val[:50] + ("..." if len(val) > 50 else "")
                    widths[h] = max(widths[h], len(val))
            # Header
            line = "  ".join(h.upper().ljust(widths[h]) for h in headers)
            print(line)
            print("-" * len(line))
            for row in data:
                vals = []
                for h in headers:
                    val = str(row.get(h, "") or "")
                    if h == "task":
                        val = val[:50] + ("..." if len(val) > 50 else "")
                    vals.append(val.ljust(widths[h]))
                print("  ".join(vals))
        elif isinstance(data, dict):
            for k, v in data.items():
                print(f"{k}: {v}")
        else:
            print(data)
    else:
        print(json.dumps(data, indent=2, default=str))


def cmd_list(args) -> None:
    """List all jobs."""
    store = _get_store()
    jobs = store.list_jobs_sync(
        status=args.status,
        include_done=args.all,
    )
    _output(jobs, args.output)


def cmd_show(args) -> None:
    """Show details of a single job."""
    store = _get_store()
    job = store.get_job_sync(args.job_id)
    if not job:
        print(json.dumps({"error": f"Job not found: {args.job_id}"}))
        sys.exit(1)
    _output(job, args.output)


def cmd_create(args) -> None:
    """Create a new job."""
    store = _get_store()

    # Validate
    if args.type not in VALID_TYPES:
        print(json.dumps({"error": f"Invalid type: {args.type}. Must be one of: {VALID_TYPES}"}))
        sys.exit(1)

    if args.once:
        schedule = "once"
        cron = None
        run_at = args.once
    elif args.cron:
        schedule = "cron"
        cron = args.cron
        run_at = None
    else:
        print(json.dumps({"error": "Must specify --cron or --once"}))
        sys.exit(1)

    # Block only a live (active/paused) id; done/cancelled may be recreated. (issue #11)
    existing = store.get_job_sync(args.id)
    if existing and existing["status"] in ("active", "paused"):
        msg = f"Job already exists and is {existing['status']}: {args.id}. Use 'edit' to modify."
        print(json.dumps({"error": msg}))
        sys.exit(1)

    job = store.upsert_job_sync(
        job_id=args.id,
        type=args.type,
        schedule=schedule,
        cron=cron,
        run_at=run_at,
        task=args.task or "",
        channel=args.channel,
        status="active",
        created_by=args.created_by,
        description=args.description or "",
        agent=args.agent or "",
    )
    _output(job, args.output)


def cmd_edit(args) -> None:
    """Edit an existing job."""
    store = _get_store()

    existing = store.get_job_sync(args.job_id)
    if not existing:
        print(json.dumps({"error": f"Job not found: {args.job_id}"}))
        sys.exit(1)

    # Build updated fields from flags (only update what's specified)
    updates = {}
    if args.type is not None:
        if args.type not in VALID_TYPES:
            print(json.dumps({"error": f"Invalid type: {args.type}"}))
            sys.exit(1)
        updates["type"] = args.type
    if args.cron is not None:
        updates["schedule"] = "cron"
        updates["cron"] = args.cron
        updates["run_at"] = None
    if args.once is not None:
        updates["schedule"] = "once"
        updates["run_at"] = args.once
        updates["cron"] = None
    if args.task is not None:
        updates["task"] = args.task
    if args.channel is not None:
        updates["channel"] = args.channel
    if args.status is not None:
        if args.status not in VALID_STATUSES:
            print(json.dumps({"error": f"Invalid status: {args.status}"}))
            sys.exit(1)
        updates["status"] = args.status
    if args.description is not None:
        updates["description"] = args.description
    if args.agent is not None:
        updates["agent"] = args.agent

    if not updates:
        print(json.dumps({"error": "No updates specified"}))
        sys.exit(1)

    # Merge with existing
    merged = {
        "type": updates.get("type", existing["type"]),
        "schedule": updates.get("schedule", existing["schedule"]),
        "cron": updates.get("cron", existing["cron"]),
        "run_at": updates.get("run_at", existing["run_at"]),
        "task": updates.get("task", existing["task"]),
        "channel": updates.get("channel", existing["channel"]),
        "status": updates.get("status", existing["status"]),
        "created_by": existing["created_by"],
        "description": updates.get("description", existing["description"]),
        "agent": updates.get("agent", existing.get("agent", "")),
    }

    job = store.upsert_job_sync(job_id=args.job_id, **merged)
    _output(job, args.output)


def cmd_remove(args) -> None:
    """Remove a job permanently."""
    store = _get_store()
    deleted = store.delete_job_sync(args.job_id)
    if deleted:
        _output({"ok": True, "deleted": args.job_id}, args.output)
    else:
        print(json.dumps({"error": f"Job not found: {args.job_id}"}))
        sys.exit(1)


def cmd_cancel(args) -> None:
    """Cancel a job (set status to 'cancelled')."""
    store = _get_store()
    existing = store.get_job_sync(args.job_id)
    if not existing:
        print(json.dumps({"error": f"Job not found: {args.job_id}"}))
        sys.exit(1)

    store.upsert_job_sync(
        job_id=args.job_id,
        type=existing["type"],
        schedule=existing["schedule"],
        cron=existing["cron"],
        run_at=existing["run_at"],
        task=existing["task"],
        channel=existing["channel"],
        status="cancelled",
        created_by=existing["created_by"],
        description=existing.get("description", ""),
    )
    _output({"ok": True, "cancelled": args.job_id}, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="jobs",
        description="Manage scheduled jobs (cron and one-shot tasks).",
    )
    parser.add_argument(
        "--output",
        "-o",
        choices=["json", "table"],
        default="json",
        help="Output format (default: json)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- list --
    p_list = subparsers.add_parser("list", help="List all jobs")
    p_list.add_argument("--status", choices=list(VALID_STATUSES), help="Filter by status")
    p_list.add_argument("--all", action="store_true", help="Include done/cancelled jobs")
    p_list.set_defaults(func=cmd_list)

    # -- show --
    p_show = subparsers.add_parser("show", help="Show a single job")
    p_show.add_argument("job_id", help="Job ID to show")
    p_show.set_defaults(func=cmd_show)

    # -- create --
    p_create = subparsers.add_parser("create", help="Create a new job")
    p_create.add_argument("--id", required=True, help="Unique job ID")
    p_create.add_argument(
        "--type",
        default="agent",
        choices=list(VALID_TYPES),
        help="Job type (default: agent)",
    )
    p_create.add_argument("--cron", help="Cron schedule (5-field: min hour day month weekday)")
    p_create.add_argument("--once", help="One-shot datetime (ISO format)")
    p_create.add_argument("--task", help="Task description or command")
    p_create.add_argument(
        "--channel", default="telegram", help="Delivery channel (default: telegram)"
    )
    p_create.add_argument("--description", help="Human-readable description of the job")
    p_create.add_argument("--agent", help="Agent this job runs as (blank = default identity)")
    p_create.add_argument("--created-by", default="agent", help="Who created this job")
    p_create.set_defaults(func=cmd_create)

    # -- edit --
    p_edit = subparsers.add_parser("edit", help="Edit an existing job")
    p_edit.add_argument("job_id", help="Job ID to edit")
    p_edit.add_argument("--type", choices=list(VALID_TYPES), help="New job type")
    p_edit.add_argument("--cron", help="New cron schedule")
    p_edit.add_argument("--once", help="Change to one-shot with this datetime")
    p_edit.add_argument("--task", help="New task description")
    p_edit.add_argument("--channel", help="New delivery channel")
    p_edit.add_argument("--status", choices=list(VALID_STATUSES), help="New status")
    p_edit.add_argument("--description", help="New description")
    p_edit.add_argument("--agent", help="New agent this job runs as")
    p_edit.set_defaults(func=cmd_edit)

    # -- remove --
    p_remove = subparsers.add_parser("remove", help="Remove a job permanently")
    p_remove.add_argument("job_id", help="Job ID to remove")
    p_remove.set_defaults(func=cmd_remove)

    # -- cancel --
    p_cancel = subparsers.add_parser("cancel", help="Cancel a job (keep in history)")
    p_cancel.add_argument("job_id", help="Job ID to cancel")
    p_cancel.set_defaults(func=cmd_cancel)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
