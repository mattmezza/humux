#!/usr/bin/env python3
"""CalDAV calendar writer — CLI helper for the agent.

Creates events on a CalDAV server. Calendar providers are configured in
config.yml under `calendar.providers`.

Usage:
    python3 /app/tools/calendar_write.py --calendar google \
        --summary "Team standup" \
        --start "2025-02-20T09:00:00+01:00" \
        --end "2025-02-20T09:30:00+01:00"

    python3 /app/tools/calendar_write.py --calendar google \
        --summary "Lunch with Alice" \
        --start "2025-02-20T12:00:00+01:00" \
        --end "2025-02-20T13:00:00+01:00" \
        --attendee alice@example.com \
        --location "Café Sprüngli"
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

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


def build_vcalendar(
    summary: str,
    start: str,
    end: str,
    location: str | None = None,
    description: str | None = None,
    attendees: list[str] | None = None,
) -> str:
    """Build a VCALENDAR with times normalized to UTC for maximum compatibility."""
    uid = str(uuid.uuid4())
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    dtstart = _to_utc_ical(start)
    dtend = _to_utc_ical(end)

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//humux//CalDAV Writer//EN",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{now}",
        f"DTSTART:{dtstart}",
        f"DTEND:{dtend}",
        f"SUMMARY:{summary}",
    ]

    if location:
        lines.append(f"LOCATION:{location}")
    if description:
        lines.append(f"DESCRIPTION:{description}")
    if attendees:
        for addr in attendees:
            lines.append(f"ATTENDEE;RSVP=TRUE:mailto:{addr}")

    lines.extend(["END:VEVENT", "END:VCALENDAR"])
    return "\r\n".join(lines)


def _to_utc_ical(iso_str: str) -> str:
    """Convert ISO 8601 to UTC iCalendar format."""
    # Date-only (all-day event): use VALUE=DATE format
    if len(iso_str) == 10:
        return iso_str.replace("-", "")

    dt = datetime.fromisoformat(iso_str)
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC)
    return dt.strftime("%Y%m%dT%H%M%SZ")


def main():
    parser = argparse.ArgumentParser(description="Create CalDAV calendar events")
    parser.add_argument("--calendar", "-c", required=True, help="Calendar provider name")
    parser.add_argument("--config", default="config.yml", help="Path to config.yml")
    parser.add_argument("--summary", required=True, help="Event title")
    parser.add_argument("--start", required=True, help="Start time (ISO 8601 with timezone)")
    parser.add_argument("--end", required=True, help="End time (ISO 8601 with timezone)")
    parser.add_argument("--location", help="Event location")
    parser.add_argument("--description", help="Event description")
    parser.add_argument(
        "--attendee", action="append", dest="attendees", help="Attendee email (repeatable)"
    )

    args = parser.parse_args()

    # Load provider
    providers = load_calendar_providers(args.config)
    if args.calendar not in providers:
        available = ", ".join(providers.keys()) if providers else "(none configured)"
        print(
            f"Error: calendar '{args.calendar}' not found. Available: {available}", file=sys.stderr
        )
        sys.exit(1)

    cal = connect(providers[args.calendar])

    # Build iCalendar data (UTC-normalized for compatibility)
    ical_data = build_vcalendar(
        summary=args.summary,
        start=args.start,
        end=args.end,
        location=args.location,
        description=args.description,
        attendees=args.attendees,
    )

    # Create the event
    try:
        cal.save_event(ical_data)
        result = {
            "ok": True,
            "calendar": args.calendar,
            "summary": args.summary,
            "start": args.start,
            "end": args.end,
        }
        if args.location:
            result["location"] = args.location
        if args.attendees:
            result["attendees"] = args.attendees
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
