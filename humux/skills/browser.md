# Browser Automation (headless)

A headless browser (Playwright/Chromium) for reading JS-heavy pages and acting on
sites on the user's behalf. **Disabled by default** — only available when the user
enables it in Settings → Tools.

**Last resort.** Prefer an existing API or CLI (e.g. `gh`, `himalaya`, calendar
tools) whenever one exists. Reach for the browser only when there is no better way.

**CRITICAL:** every command starts a BRAND-NEW browser that reloads `--url` from
scratch — cookies/sessions persist via `--profile`, but the open page does
**not**. You CANNOT do a flow step-by-step across several commands; each call
restarts from the beginning and loses all progress. A multi-step interaction
(e.g. a login, a booking) must be a single `act` or `explore` call carrying the
whole flow.

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

## Exploring a page (self-driving loop)

```bash
python3 ./tools/browser.py explore --url https://site/booking --profile acme \
  --task "1. Search flights ZRH->LHR on 2026-09-01. 2. Pick the cheapest. 3. Fill passenger name Alice Smith, email a@b.com. 4. Stop before payment and report the total."
```

PREFER `explore` for ANYTHING interactive: clicking buttons, opening modals or
widgets, multi-step forms, bookings, checkouts, anything inside an iframe. One
browser stays open and an inner LLM loop sees every frame and clicks/types on
its own until done, then returns a JSON `{"answer", ...}`. It is the ONLY verb
that can drive embedded widgets and payment iframes — `read`/`act` only see the
top page and will fail on them.

Put the ENTIRE flow in one `--task` as numbered steps with every value it
needs (product, dates, name, email, phone, card details) — it cannot ask
mid-run. It runs for a few minutes; do not treat the wait as a hang, split the
task, retry, or fall back to `read`/`act`. Quote the `answer` (and screenshot)
back to the user; if it reports pending/awaiting-approval don't upgrade that
to "confirmed"; if it returns `done:false` with a `reason`, report what
blocked it and don't blindly re-run.

`explore` **self-drives to completion under a SINGLE approval** — there is no
per-step gate once it starts. Do NOT use it to spend money or submit
irreversible actions on its own: for purchases/payments, confirm the exact
details with the owner first, and prefer guided `act` steps instead (each
fill/submit is approved separately).

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
