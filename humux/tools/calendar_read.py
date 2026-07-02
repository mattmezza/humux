#!/usr/bin/env python3
"""CalDAV calendar reader — CLI helper for the agent.

Reads events from a CalDAV server and outputs them as JSON or plain text.
Calendar providers are configured in config.yml under `calendar.providers`.

Usage:
    python3 /app/tools/calendar_read.py --calendar google --today -o json
    python3 /app/tools/calendar_read.py --calendar google --from 2025-02-17 --to 2025-02-24 -o json
    python3 /app/tools/calendar_read.py --calendar google --next 5 -o json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import caldav
import yaml
from dotenv import load_dotenv

# Reuse the config env-var resolver
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import _resolve_env_vars  # noqa: E402
from tools.calendar_auth import connect  # noqa: E402


def load_calendar_providers(config_path: str = "config.yml") -> dict[str, dict]:
    """Load calendar providers from config.yml, keyed by name."""
    load_dotenv()
    path = Path(config_path)
    if not path.exists():
        print("Error: config.yml not found", file=sys.stderr)
        sys.exit(1)

    raw = yaml.safe_load(path.read_text()) or {}
    resolved = _resolve_env_vars(raw)
    providers = resolved.get("calendar", {}).get("providers", [])
    return {p["name"]: p for p in providers}


def event_to_dict(event: caldav.Event) -> dict:
    """Convert a caldav Event to a plain dict."""
    vevent = event.vobject_instance.vevent
    result: dict = {}

    result["uid"] = str(vevent.uid.value) if hasattr(vevent, "uid") else ""
    result["summary"] = str(vevent.summary.value) if hasattr(vevent, "summary") else ""

    # Start / end
    if hasattr(vevent, "dtstart"):
        dt = vevent.dtstart.value
        result["start"] = dt.isoformat() if isinstance(dt, datetime) else dt.isoformat()
    if hasattr(vevent, "dtend"):
        dt = vevent.dtend.value
        result["end"] = dt.isoformat() if isinstance(dt, datetime) else dt.isoformat()

    # Optional fields
    if hasattr(vevent, "location"):
        result["location"] = str(vevent.location.value)
    if hasattr(vevent, "description"):
        result["description"] = str(vevent.description.value)

    # Attendees
    if hasattr(vevent, "attendee"):
        attendees_raw = vevent.attendee
        if not isinstance(attendees_raw, list):
            attendees_raw = [attendees_raw]
        result["attendees"] = [str(a.value).replace("mailto:", "") for a in attendees_raw]

    return result


def format_event_text(ev: dict) -> str:
    """Format a single event dict as human-readable text."""
    start = ev.get("start", "?")
    end = ev.get("end", "")
    summary = ev.get("summary", "(no title)")
    location = ev.get("location", "")

    line = f"  {start}"
    if end:
        # Show just the time portion of end if same day
        line += f" — {end}"
    line += f"  {summary}"
    if location:
        line += f"  ({location})"
    return line


def main():
    parser = argparse.ArgumentParser(description="Read CalDAV calendar events")
    parser.add_argument("--calendar", "-c", help="Calendar provider name")
    parser.add_argument("--config", default="config.yml", help="Path to config.yml")
    parser.add_argument(
        "--list", action="store_true", help="List available calendar providers and exit"
    )

    # Query modes (mutually exclusive)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--today", action="store_true", help="Get today's events")
    group.add_argument("--from", dest="from_date", metavar="DATE", help="Start date (YYYY-MM-DD)")
    group.add_argument("--next", type=int, metavar="N", help="Get next N upcoming events")

    parser.add_argument(
        "--to", dest="to_date", metavar="DATE", help="End date (YYYY-MM-DD), used with --from"
    )
    parser.add_argument(
        "-o", "--output", choices=["json", "text"], default="text", help="Output format"
    )

    args = parser.parse_args()

    # Load providers
    providers = load_calendar_providers(args.config)

    if args.list:
        if not providers:
            print("No calendar providers configured.")
        elif args.output == "json":
            print(json.dumps(list(providers.keys())))
        else:
            for name in providers:
                print(f"  - {name}")
        sys.exit(0)

    if not args.calendar:
        parser.error("--calendar/-c is required (use --list to see available providers)")

    if not any([args.today, args.from_date, args.next is not None]):
        parser.error("one of --today, --from, --next is required (or use --list)")

    if args.calendar not in providers:
        available = ", ".join(providers.keys()) if providers else "(none configured)"
        print(
            f"Error: calendar '{args.calendar}' not found. Available: {available}", file=sys.stderr
        )
        sys.exit(1)

    cal = connect(providers[args.calendar])

    # Determine date range
    today = date.today()
    events = []

    if args.today:
        start = datetime.combine(today, datetime.min.time(), tzinfo=UTC)
        end = start + timedelta(days=1)
        results = cal.search(start=start, end=end, expand=True)
        events = [event_to_dict(e) for e in results]

    elif args.from_date:
        start = datetime.combine(
            date.fromisoformat(args.from_date), datetime.min.time(), tzinfo=UTC
        )
        if args.to_date:
            end = datetime.combine(
                date.fromisoformat(args.to_date), datetime.min.time(), tzinfo=UTC
            ) + timedelta(days=1)
        else:
            # Default to 7 days if no --to
            end = start + timedelta(days=7)
        results = cal.search(start=start, end=end, expand=True)
        events = [event_to_dict(e) for e in results]

    elif args.next is not None:
        # Fetch events for the next 90 days and take the first N
        start = datetime.combine(today, datetime.min.time(), tzinfo=UTC)
        end = start + timedelta(days=90)
        results = cal.search(start=start, end=end, expand=True)
        all_events = [event_to_dict(e) for e in results]
        # Sort by start time
        all_events.sort(key=lambda e: e.get("start", ""))
        events = all_events[: args.next]

    # Sort by start time
    events.sort(key=lambda e: e.get("start", ""))

    # Output
    if args.output == "json":
        print(json.dumps(events, indent=2, ensure_ascii=False))
    else:
        if not events:
            print("No events found.")
        else:
            for ev in events:
                print(format_event_text(ev))


if __name__ == "__main__":
    main()
