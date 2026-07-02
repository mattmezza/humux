# Calendar Management (CalDAV)

Calendar operations use helper scripts that wrap Python's caldav library.
Calendar providers are configured via the admin UI.

## Discovering available calendars

```bash
python3 ./tools/calendar_read.py --list -o json
```

Returns a JSON array of provider names, e.g. `["google", "icloud"]`.
Always run this first if you are unsure which calendars are configured.

## Reading events

```bash
# Get today's events
python3 ./tools/calendar_read.py --calendar <NAME> --today -o json

# Get events for a date range
python3 ./tools/calendar_read.py --calendar <NAME> --from YYYY-MM-DD --to YYYY-MM-DD -o json

# Get next N events
python3 ./tools/calendar_read.py --calendar <NAME> --next N -o json
```

Replace `<NAME>` with a provider name from `--list`.

JSON output:

```json
[
  {
    "uid": "abc123",
    "summary": "Team standup",
    "start": "2025-02-17T09:00:00+01:00",
    "end": "2025-02-17T09:30:00+01:00",
    "location": "Google Meet",
    "attendees": ["alice@example.com", "bob@example.com"]
  }
]
```

## Creating events

Use the `create_calendar_event` structured tool (requires user permission). Provide:
- `calendar`: provider name from `--list`
- `summary`: event title
- `start`: ISO 8601 datetime with timezone (e.g. "2025-02-20T14:00:00+01:00")
- `end`: ISO 8601 datetime with timezone
- `attendees`: optional list of email addresses

## Important notes

- Always include timezone offset in datetimes (Europe/Zurich = +01:00, +02:00 during DST).
- For all-day events, use date only: "2025-02-20".
