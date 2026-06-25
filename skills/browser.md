# Browser Automation (headless)

A headless browser (Playwright/Chromium) for reading JS-heavy pages and acting on
sites on the user's behalf. **Disabled by default** — only available when the user
enables it in Settings → Tools.

**Last resort.** Prefer an existing API or CLI (e.g. `gh`, `himalaya`, calendar
tools) whenever one exists. Reach for the browser only when there is no better way.

Every command is a fresh process: cookies/sessions persist via `--profile`, but the
open page does **not**. So a multi-step interaction (e.g. a login) must be a single
`act` call.

## Reading a page

```bash
# Readable text (waits for JS to settle) — runs without asking
python3 ./tools/browser.py read --url https://example.com

# Save a screenshot (PNG) — runs without asking
python3 ./tools/browser.py screenshot --url https://example.com -o /app/data/browser/shot.png
```

`read` returns `{"url", "title", "text"}`. `screenshot` returns `{"url", "title", "path"}`.

## Acting on a page

```bash
python3 ./tools/browser.py act --url https://site/login --profile acme \
  --steps '[{"fill":["#user","alice"]},{"fill":["#pass","s3cr3t"]},{"click":"#login"}]'
```

`act` changes state, so it **asks for approval each time**. On chat channels the
approval shows a screenshot of the page — so always `screenshot` the page first, so
the user can follow along. `--steps` is an ordered JSON array of single-key objects:

| Step | Meaning |
|------|---------|
| `{"fill":["sel","value"]}` | type `value` into the element |
| `{"click":"sel"}` | click the element |
| `{"select":["sel","value"]}` | choose a `<select>` option |
| `{"press":["sel","Key"]}` | press a key (e.g. `Enter`) in the element |
| `{"wait": 1000}` | wait N milliseconds |
| `{"wait": "sel"}` | wait until the element appears |
| `{"goto":"url"}` | navigate within the same call |

`act` returns `{"url", "title", "steps", "screenshot"}`.

## Profiles (logged-in sessions)

A `--profile NAME` keeps its own cookies/session under `data/browser/profiles/NAME`,
so you log in once and reuse it. List them:

```bash
python3 ./tools/browser.py profiles   # [{"name","authenticated","updated"}]
```

## Guided first-time login (mobile-followable)

1. `screenshot` the login page and send it to the user so they can follow along.
2. Ask the user for their credentials. **Never store or log credentials.**
3. `act` to fill the username/password and submit (this asks for approval; the user
   sees the screenshot + Approve/Deny).
4. If 2FA appears: `screenshot` it, ask the user for the code, then `act` to enter it.
5. Done — the `--profile` session is saved; later visits to that site skip the login.

A user can also pre-seed a profile by dropping an exported Playwright
`storage_state.json` into `data/browser/profiles/NAME/`.

## Limitations

Sites behind strong bot-management or interactive challenges may block headless
automation. Persistent sessions help with the common cases but not the hardest tier.
Tell the user plainly when a site looks blocked rather than retrying endlessly.
