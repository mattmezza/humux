# Contacts CLI

Use the contacts helper script to query providers directly (Google + CardDAV).
Providers are configured in the admin UI (Contacts tab). Avoid hardcoded names.

## List all contacts

```bash
python3 /app/tools/contacts.py list --provider <NAME> --output json
```

## Search by name, phone, or email

```bash
python3 /app/tools/contacts.py search --provider <NAME> --query "Alice" --output json
```

The query matches full name, phones, and emails.

## Get full details for a specific contact

```bash
python3 /app/tools/contacts.py get --provider <NAME> --id <CONTACT_ID> --output json
```

`CONTACT_ID` comes from the `id` field in list/search results.

## Output format

`--output json` returns an array of objects (or a single object for `get`).
Fields include:
- `id`
- `full_name`
- `phones`
- `emails`
- `source`

## Notes

- If no providers are configured, ask the user to add one in the admin UI.
- If multiple matches appear, present the options and ask which to use.
