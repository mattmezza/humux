# Himalaya Email CLI

You have access to the `himalaya` CLI (**v1.2.0**) to manage emails. Himalaya is a
stateless CLI email client — each command is independent, no session state.

## Configuration

Himalaya is pre-configured. The available accounts are defined in its TOML config file.
Always specify the account with `-a <account_name>`.

```bash
# List configured accounts
himalaya account list

# List folders for an account (use exact names from here for move/copy)
himalaya folder list -a personal -o json
```

## Reading emails

### List recent emails (envelopes)

```bash
# List last 10 emails in INBOX (default folder)
himalaya envelope list -a personal -s 10 -o json

# List emails in a specific folder
himalaya envelope list -a personal --folder "Archives" -s 20 -o json

# Page through results (page 2)
himalaya envelope list -a personal -s 10 -p 2 -o json
```

JSON output is an array of envelope objects with fields like `id`, `subject`, `from`,
`date`, and `flags`.

### Read a specific email

```bash
# Read email by ID — AUTO-MARKS the message as Seen
himalaya message read -a personal 123

# Preview WITHOUT marking as Seen
himalaya message read -a personal -p 123

# Read specific headers only
himalaya message read -a personal 123 --header From --header Subject --header Date
```

### Search emails (query DSL)

v1.x uses a **positional query DSL** — no `--`, no raw IMAP syntax.

- Conditions: `date`, `before`, `after`, `from`, `to`, `subject`, `body`, `flag`
- Operators: `not`, `and`, `or`
- Ordering: `order by <date|from|to|subject> [asc|desc]`

```bash
# Unread emails
himalaya envelope list -a personal -o json "not flag seen"

# By subject
himalaya envelope list -a personal -o json "subject invoice"

# By sender
himalaya envelope list -a personal -o json "from alice@example.com"

# Combined, newest first
himalaya envelope list -a personal -o json "from ikea and subject contract and not flag seen order by date desc"
```

## Sending emails

`message read/reply/forward/write/edit` open `$EDITOR` interactively — **not**
automation-safe. For automation use the non-interactive `template` path, or pipe a raw
message into `message send`.

### Send a new email

```bash
printf 'From: matteo@merola.co\nTo: bob@example.com\nSubject: Hello\n\nBody text here.\n' \
  | himalaya message send -a personal
```

### Reply (and reply-all)

```bash
# Reply to sender
himalaya template reply -a personal 123 "Thanks, got it." | himalaya template send -a personal

# Reply-all: add -A
himalaya template reply -a personal -A 123 "Thanks all." | himalaya template send -a personal
```

### Forward

```bash
himalaya template forward -a personal 123 "FYI — see below." | himalaya template send -a personal
```

Always include a correct `From:` header matching the account email. Before sending,
present the draft to the user for approval.

## Managing emails

`message move`/`copy` take the **TARGET folder FIRST**, then the ID. `-f` sets the
SOURCE folder.

```bash
# Move to a folder (TARGET first)
himalaya message move -a personal Archives 123

# Move from a non-INBOX source folder
himalaya message move -a personal -f Spam INBOX 123

# Copy to a folder (TARGET first)
himalaya message copy -a personal Important 123

# Mark as spam — no spam flag exists; move to the Spam folder
# (run `folder list` for the exact name)
himalaya message move -a personal Spam 123

# Delete (moves to Trash)
himalaya message delete -a personal 123

# Mark read / unread
himalaya flag add -a personal 123 Seen
himalaya flag remove -a personal 123 Seen

# Other flags
himalaya flag add -a personal 123 Flagged
```

## Important notes

- Use `-o json` whenever you need to parse results programmatically.
- Email IDs are folder-relative — specify `--folder` when not operating on INBOX.
- `message read` marks Seen; use `-p`/`--preview` to avoid changing flags.
- Sending uses SMTP auth from the account config. The personal account reuses
  `$MAIL_MEROLA_CO_APP_PASSWORD` for both IMAP and SMTP — if sending fails with an auth
  error, confirm that env var is exported.
- Use `printf` with `\n` for newlines when building raw messages; never use bare `echo`
  with literal newlines.
- Accounts are identified by NAME (`-a personal`), which is all any command needs. The
  config file holding the addresses lives outside the workspace and is not readable —
  don't `cat`/`grep`/`read_file` it. If you genuinely need an account's own address,
  read it off any message: `himalaya envelope list -a NAME -s 1 -o json` (the `to`/`from`
  field). To just see which accounts exist, run `himalaya account list`.
- Server-side search lags delivery: a message you just sent may not match a `subject`/
  `from` query for a few seconds even though it has already arrived. To find just-sent
  mail, list recent envelopes (`envelope list -s 20 -o json`) rather than searching — the
  envelope listing reflects new mail immediately.
