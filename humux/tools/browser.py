#!/usr/bin/env python3
"""Browser CLI — headless web automation via Playwright (Chromium).

The agent drives this with the ``run_command`` tool, same as the other CLIs.
Three verbs:

  read       Load a page and return its readable text (for JS-heavy sites).
  screenshot Load a page and save a PNG (the agent/you can then view it).
  act        Load a page and run an ordered list of steps (click/fill/...).

``read`` and ``screenshot`` are read-only (pre-approved). ``act`` changes state
(it clicks/submits) so it asks for approval each time — and because every command
carries ``--url``, per-domain permission rules work out of the box, e.g.
``run_command:python3 tools/browser.py act*github.com*`` -> ALWAYS.

Persistent profiles
  Each ``--profile NAME`` reuses a Chromium ``user-data-dir`` under
  ``data/browser/profiles/NAME``, so cookies/sessions survive between calls. Log
  in once (via the guided ``act`` flow), then the session is reused.

Why one verb does goto+steps: every ``run_command`` is a fresh process, so page
state (the open page, form fields) does NOT survive between calls — only cookies
do, via the profile. So a multi-step interaction (fill user, fill pass, submit)
must be a single ``act`` invocation.
  # ponytail: no live browser kept between calls — each verb re-navigates.
  # Upgrade path if iterative "click around" sessions matter: a per-profile
  # browser daemon / the BROWSER_CDP_URL sidecar below.

Sidecar (optional, off by default)
  If ``BROWSER_CDP_URL`` is set, connect to a remote Chromium over CDP instead of
  launching one locally. Keeps the main image lean (no bundled Chromium). With
  CDP the profile lives on the sidecar, so ``--profile`` is informational only.

Usage examples
  python3 tools/browser.py read --url https://example.com
  python3 tools/browser.py screenshot --url https://example.com -o shot.png
  python3 tools/browser.py act --url https://site/login --profile acme \\
      --steps '[{"fill":["#user","alice"]},{"fill":["#pass","s3cr3t"]},{"click":"#login"}]'
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_DEFAULT_UA = (
    "Mozilla/5.0 (macOS; AArch64) Ladybird/1.0 Chrome/146.0.0.0 AppleWebKit/537.36 Safari/537.36"
)


def _user_agent() -> str:
    # BROWSER_USER_AGENT (from config.tools.browser.user_agent) overrides the
    # built-in default, so the UA is choosable without code changes.
    return os.environ.get("BROWSER_USER_AGENT", "").strip() or _DEFAULT_UA


def _data_dir() -> Path:
    # Mirror the convention used by the other tools (tools/skills.py).
    return Path("/app/data") if Path("/app/data").exists() else Path("data")


def _validate_profile(name: str) -> str:
    value = (name or "").strip().lower()
    if not value:
        raise ValueError("Profile name is required")
    if not NAME_PATTERN.match(value):
        raise ValueError("Profile name must be lowercase letters, digits, '-' or '_'")
    return value


def _profile_dir(name: str) -> Path:
    return _data_dir() / "browser" / "profiles" / name


def _state_file(name: str) -> Path:
    # Playwright storage_state (cookies + localStorage). The user-data-dir alone
    # drops *session* cookies on close, so we also persist them here — and a user
    # can drop their own exported storage_state.json here to import a live login.
    return _profile_dir(name) / "storage_state.json"


def _preview_path(name: str) -> Path:
    # Latest screenshot per profile — the approval flow attaches this so the user
    # sees the page before approving an `act`. Kept outside the user-data-dir.
    return _data_dir() / "browser" / "last" / f"{name}.png"


def _status_path() -> Path:
    # Live one-line progress for an in-flight `explore` run. The REPL spinner tails
    # this so the user sees step-by-step progress during the (multi-minute) loop.
    return _data_dir() / "browser" / "last" / "explore.status"


def _shot_dir() -> Path:
    """Where saved screenshots go.

    Local dev/REPL: ~/Downloads so you can open the file. Server/Docker (/app/data
    present): the data volume instead — nobody browses the container's home, and
    Downloads isn't mounted, so a Downloads default would just bloat the container
    layer invisibly.
    """
    downloads = Path.home() / "Downloads"
    if not Path("/app/data").exists() and downloads.is_dir():
        return downloads
    return _data_dir() / "browser" / "screenshots"


def _prune_old_shots(max_age_h: int = 24) -> None:
    """Delete data-dir screenshots older than max_age_h so a long-running server
    doesn't accumulate PNGs forever. Best-effort; only touches our own dir, never
    a user's ~/Downloads."""
    cutoff = time.time() - max_age_h * 3600
    d = _data_dir() / "browser" / "screenshots"
    if not d.is_dir():
        return
    for p in d.glob("*.png"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass


def _write_status(text: str) -> None:
    try:
        p = _status_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    except OSError:
        pass  # progress display is best-effort; never let it break a run.


def _parse_steps(raw: str) -> list[dict]:
    try:
        steps = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--steps is not valid JSON: {exc}") from exc
    if not isinstance(steps, list) or not steps:
        raise ValueError("--steps must be a non-empty JSON array")
    for step in steps:
        if not isinstance(step, dict) or len(step) != 1:
            raise ValueError('each step must be a single-key object, e.g. {"click":"#btn"}')
    return steps


# -- Playwright glue ---------------------------------------------------------


def _settle(page, timeout_ms: int) -> None:
    """Best-effort wait for the page to stop loading (JS-heavy sites)."""
    try:
        page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 10_000))
    except Exception:
        pass  # networkidle never fires on some sites (long-poll/websockets) — fine.


def _run_step(page, step: dict, timeout_ms: int) -> str:
    verb, arg = next(iter(step.items()))
    if verb == "click":
        page.click(arg, timeout=timeout_ms)
        return f"click {arg}"
    if verb == "fill":
        sel, val = arg
        page.fill(sel, val, timeout=timeout_ms)
        return f"fill {sel}"
    if verb == "select":
        sel, val = arg
        page.select_option(sel, val, timeout=timeout_ms)
        return f"select {sel}={val}"
    if verb == "press":
        sel, key = arg
        page.press(sel, key, timeout=timeout_ms)
        return f"press {sel} {key}"
    if verb == "wait":
        if isinstance(arg, int | float):
            page.wait_for_timeout(arg)
            return f"wait {arg}ms"
        page.wait_for_selector(arg, timeout=timeout_ms)
        return f"wait {arg}"
    if verb == "goto":
        page.goto(arg, timeout=timeout_ms, wait_until="domcontentloaded")
        return f"goto {arg}"
    raise ValueError(f"unknown step verb: {verb!r}")


class _Session:
    """A Chromium page bound to a persistent profile (or a CDP sidecar)."""

    def __init__(self, profile: str, headless: bool):
        self.profile = profile
        # Headed Chromium needs an X/Wayland display. On a headless server (e.g.
        # the Docker image) there is none, so a "show the browser" setting would
        # crash the launch with "Missing X server or $DISPLAY". Fall back to
        # headless instead of crashing. (CDP sidecar ignores this — it's remote.)
        if (
            not headless
            and sys.platform.startswith("linux")
            and not os.environ.get("DISPLAY")
            and not os.environ.get("WAYLAND_DISPLAY")
        ):
            print("[browser] no display available — running headless", file=sys.stderr)
            headless = True
        self.headless = headless
        self.is_cdp = False
        self._pw = None
        self._ctx = None
        self._browser = None
        self.page = None

    def __enter__(self):
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self.is_cdp = bool(os.environ.get("BROWSER_CDP_URL", "").strip())
        ua = _user_agent()
        if self.is_cdp:
            # ponytail: CDP profile lives on the sidecar; --profile is advisory here.
            cdp = os.environ["BROWSER_CDP_URL"].strip()
            self._browser = self._pw.chromium.connect_over_cdp(cdp)
            # A reused sidecar context keeps its own UA; only a fresh one can take ours.
            self._ctx = (
                self._browser.contexts[0]
                if self._browser.contexts
                else self._browser.new_context(user_agent=ua)
            )
        else:
            user_data_dir = _profile_dir(self.profile) / "udd"
            user_data_dir.mkdir(parents=True, exist_ok=True)
            self._ctx = self._pw.chromium.launch_persistent_context(
                str(user_data_dir),
                headless=self.headless,
                user_agent=ua,
                viewport={"width": 1280, "height": 800},
            )
            self._restore_session()
        self.page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
        return self

    def _restore_session(self) -> None:
        """Re-add saved cookies (incl. session cookies the user-data-dir drops)."""
        state = _state_file(self.profile)
        if not state.exists():
            return
        try:
            cookies = json.loads(state.read_text()).get("cookies", [])
            if cookies:
                self._ctx.add_cookies(cookies)
        except Exception:
            pass  # corrupt/foreign state file — fall back to whatever the udd has.

    def __exit__(self, *exc):
        try:
            if not self.is_cdp and self._ctx:
                # Persist cookies + localStorage (captures session cookies too).
                try:
                    self._ctx.storage_state(path=str(_state_file(self.profile)))
                except Exception:
                    pass
            if self.is_cdp:
                if self._browser:
                    self._browser.close()
            elif self._ctx:
                self._ctx.close()
        finally:
            if self._pw:
                self._pw.stop()

    def goto(self, url: str, timeout_ms: int):
        self.page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
        _settle(self.page, timeout_ms)

    def snapshot(self, dest: Path, full_page: bool = True) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=str(dest), full_page=full_page)
        # Mirror to the per-profile preview the approval flow reads.
        preview = _preview_path(self.profile)
        preview.parent.mkdir(parents=True, exist_ok=True)
        preview.write_bytes(dest.read_bytes())
        return dest


# -- Verbs -------------------------------------------------------------------


def cmd_read(args) -> dict:
    profile = _validate_profile(args.profile)
    with _Session(profile, args.headless) as s:
        s.goto(args.url, args.timeout)
        text = s.page.inner_text("body")
        return {"url": s.page.url, "title": s.page.title(), "text": text.strip()}


def cmd_screenshot(args) -> dict:
    profile = _validate_profile(args.profile)
    _prune_old_shots()
    if args.output:
        dest = Path(args.output)
    else:
        # Spottable name in the env-appropriate dir (~/Downloads locally, the data
        # volume on a server — see _shot_dir).
        host = urlparse(args.url).netloc or "page"
        dest = _shot_dir() / f"clio-{host}-{int(time.time())}.png"
    with _Session(profile, args.headless) as s:
        s.goto(args.url, args.timeout)
        s.snapshot(dest, full_page=args.full_page)
        return {"url": s.page.url, "title": s.page.title(), "path": str(dest)}


def cmd_act(args) -> dict:
    profile = _validate_profile(args.profile)
    _prune_old_shots()
    steps = _parse_steps(args.steps)
    done: list[str] = []
    with _Session(profile, args.headless) as s:
        s.goto(args.url, args.timeout)
        for step in steps:
            done.append(_run_step(s.page, step, args.timeout))
        _settle(s.page, args.timeout)
        dest = _data_dir() / "browser" / "screenshots" / f"{profile}-{int(time.time())}.png"
        s.snapshot(dest, full_page=True)
        return {"url": s.page.url, "title": s.page.title(), "steps": done, "screenshot": str(dest)}


# -- explore: one process, one live browser, deepseek drives the loop ---------
#
# The other verbs are one-shot: a fresh process re-navigates every call, so an
# outer agent that wants to "look, decide, click, look again" ping-pongs and
# re-loads the page each round. `explore` instead keeps ONE browser open and
# runs an inner LLM loop (the project's own cheap model, vision OFF) that
# observes the page as indexed elements and issues ONE action per step until it
# reports `done`. Built for openai-compatible providers (deepseek default).

# JS: per-frame, tag every visible/enabled interactive element with data-bu-idx
# (numbered from `offset` so indices are unique across frames) and return
# {count, elements}. Selectors are then [data-bu-idx="N"] within that frame.
_ENUM_JS = """
(offset) => {
  const sel = 'a,button,input,textarea,select,[role=button],[role=link],[role=radio],'
    + '[role=checkbox],[role=tab],[onclick],[contenteditable=true]';
  const out = []; let i = 0;
  for (const el of document.querySelectorAll(sel)) {
    const rects = el.getClientRects();
    if (!rects.length) continue;
    const st = getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none') continue;
    const tag = el.tagName.toLowerCase();
    let label = (el.innerText || el.getAttribute('aria-label') || el.getAttribute('placeholder') ||
      el.getAttribute('name') || el.getAttribute('title') || el.value ||
      '').trim().replace(/\\s+/g, ' ').slice(0, 80);
    // Skip pure-decoration anchors with no label/href target text.
    if (!label && tag === 'a') continue;
    const idx = offset + i;
    el.setAttribute('data-bu-idx', String(idx));
    const rec = {idx, tag, type: el.getAttribute('type') || '', label};
    // Keep disabled controls visible (marked) — a disabled "Next" button tells the
    // model a prerequisite field is still missing, instead of vanishing silently.
    if (el.disabled) rec.disabled = true;
    if (tag === 'input' || tag === 'textarea') rec.value = (el.value || '').slice(0, 40);
    if (el.type === 'radio' || el.type === 'checkbox') rec.checked = el.checked;
    if (tag === 'select') {
      rec.value = el.value;
      // Show the option VALUE, plus its label in parens only when they differ —
      // a "value:text" join breaks for values that contain ':' (e.g. times).
      rec.options = [...el.options].slice(0, 12).map(o => {
        const v = o.value, t = o.text.trim().slice(0, 24);
        return v === t ? v : `${v} (${t})`;
      });
    }
    const exp = el.getAttribute('aria-expanded');
    if (exp !== null) rec.expanded = exp;
    // Spatial signal (cheap stand-in for vision): on-screen position + whether the
    // element is scrolled out of view + whether it currently has focus. Lets the
    // model reason about "top-right", "scroll down to it", "which field is active".
    const r = rects[0];
    rec.rect = [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)];
    if (r.bottom <= 0 || r.top >= innerHeight) rec.offscreen = true;
    if (document.activeElement === el) rec.focused = true;
    out.push(rec);
    i++;
  }
  return {count: i, elements: out};
}
"""

_EXPLORE_SYSTEM = """You are a web-automation agent driving a real browser.
Each turn you receive the current page: URL, title, a text excerpt, a list of
the actions you took so far, and a numbered list of interactive ELEMENTS
(across all frames, including embedded widgets and payment iframes). Achieve the
user's TASK by calling `browser_action` exactly ONCE per turn.

Actions: click(index), fill(index,text), select(index,text), goto(url),
scroll(text="down"|"up"), done(answer).

Rules:
- Element indices are RENUMBERED every turn. Always use indices from the LATEST
  ELEMENTS list, never an old one.
- To advance a multi-step flow, click the button that moves forward (e.g.
  "Next", "Continue", "Book now", "Pay").
- Fill every required field before clicking the step's continue button.
- For input(date) fill YYYY-MM-DD (pick a date a few days out); for input(time)
  fill HH:MM. A dropdown showing options like ["Select date first..."] is
  DISABLED until the date it depends on is filled — set the date input first,
  then on the next turn its real options appear and you can select one.
- A select shows its options inline as options=[value (label), ...]; pass the
  bare VALUE (the part before any parenthesis) to select(index,value).
- An input shows its current text as value=...; if it already holds what you
  need, don't refill it.
- Each element may show its on-screen position as @x,y, FOCUSED if it currently
  has focus, and off-screen if it is scrolled out of view. Use these for spatial
  choices ("the button on the right" = larger x) and scroll toward off-screen
  targets before acting on them.
- An element marked DISABLED cannot be clicked yet — it is gated on a missing
  field. Do not click it; find and complete the missing required field first
  (an empty value=, an unselected radio/option), which will enable it.
- If a total/price stays 0 or reads "will be calculated"/"unavailable" after you
  filled the fields, your CHOICE is invalid (e.g. that date range or duration is
  not available), not the form — change the selection (try a shorter or
  different date/time range) until a real price appears and the button enables.
- If a NOTE says the page did not change, your last action had no effect — do
  NOT repeat it. Pick a different element (you may need to scroll, set a
  prerequisite field first, or click a radio/option). Do NOT navigate away with
  goto to restart — stay in the flow and fix the current step.
- Payment fields (card number, expiry, CVC) live in iframes but appear in the
  ELEMENTS list like any other input — fill them normally.
- Call done(answer) when the task is finished (e.g. a confirmation page shows),
  describing the outcome. Don't call done prematurely."""

_ACTION_TOOL = {
    "name": "browser_action",
    "description": "Take exactly one action on the page, or finish with done.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["click", "fill", "select", "goto", "scroll", "done"],
            },
            "index": {"type": "integer", "description": "target element index"},
            "text": {"type": "string", "description": "text to fill / option / scroll dir"},
            "url": {"type": "string", "description": "url for goto"},
            "answer": {"type": "string", "description": "final result when action=done"},
        },
        "required": ["action"],
    },
}


def _format_state(url: str, title: str, excerpt: str, elements: list[dict]) -> str:
    lines = [f"URL: {url}", f"TITLE: {title}", "", "EXCERPT:", excerpt.strip()[:1400]]
    lines += ["", "ELEMENTS:"]
    last_frame = None
    for e in elements:
        fr = e.get("frame")
        if fr and fr != last_frame:
            lines.append(f"-- in frame: {fr} --")
            last_frame = fr
        t = f"{e['tag']}" + (f"({e['type']})" if e.get("type") else "")
        extra = " DISABLED" if e.get("disabled") else ""
        if e.get("checked") is not None:
            extra += " CHECKED" if e["checked"] else ""
        if e.get("value"):
            extra += f" value={e['value']!r}"
        if e.get("expanded") is not None:
            extra += f" expanded={e['expanded']}"
        if e.get("options"):
            extra += f" options={e['options']}"
        if e.get("focused"):
            extra += " FOCUSED"
        if e.get("offscreen"):
            extra += " off-screen"
        if e.get("rect"):
            extra += f" @{e['rect'][0]},{e['rect'][1]}"
        lines.append(f"[{e['idx']}] {t} {e.get('label', '')!r}{extra}")
    if not elements:
        lines.append("(none found)")
    return "\n".join(lines)


def _observe(page) -> tuple[str, dict]:
    """Enumerate interactive elements across ALL frames (main + embedded widgets +
    payment iframes). Returns (state_text, {index: frame}) so actions dispatch into
    the frame that owns each element."""
    elements: list[dict] = []
    frame_map: dict = {}
    texts: list[str] = []
    offset = 0
    for fr in page.frames:
        try:
            res = fr.evaluate(_ENUM_JS, offset)
        except Exception:
            continue  # frame detached / not ready — skip it this turn
        items = res.get("elements", [])
        # Label frames other than the main one so the model sees the widget split.
        frame_label = "" if fr is page.main_frame else (fr.url.split("//")[-1][:40] or "iframe")
        for it in items:
            it["frame"] = frame_label
            elements.append(it)
            frame_map[it["idx"]] = fr
        offset += res.get("count", 0)
        # The real content often lives in the widget iframe, not the host page —
        # pull a short text excerpt from each frame that has interactive elements.
        if items:
            try:
                snippet = fr.inner_text("body").strip().replace("\n\n", "\n")[:600]
            except Exception:
                snippet = ""
            if snippet:
                texts.append((f"[{frame_label}]\n" if frame_label else "") + snippet)
    excerpt = "\n---\n".join(texts)[:1400]
    return _format_state(page.url, page.title(), excerpt, elements), frame_map


def _apply_action(frame_map: dict, action: dict, timeout_ms: int, page=None) -> str:
    """Execute one model action in the frame that owns the target element."""
    verb = action.get("action")
    idx = action.get("index")
    target = frame_map.get(idx)
    sel = f'[data-bu-idx="{idx}"]'
    if verb == "click":
        # force=True skips Playwright's actionability wait (visibility/stability/
        # overlay-intercept/in-viewport). Real sites bury the true control under a
        # label or image and use sr-only radios — without force every click hangs
        # the full timeout then fails. The JS-level click still fires change events.
        target.click(sel, timeout=timeout_ms, force=True)
        return f"clicked [{idx}]"
    if verb == "fill":
        target.fill(sel, action.get("text", ""), timeout=timeout_ms)
        return f"filled [{idx}]"
    if verb == "select":
        val = action.get("text", "")
        try:
            target.select_option(sel, value=val, timeout=timeout_ms)
        except Exception:
            # Model may have passed the visible label rather than the option value.
            target.select_option(sel, label=val, timeout=timeout_ms)
        return f"selected [{idx}]"
    if verb == "goto":
        page.goto(action.get("url", ""), timeout=timeout_ms, wait_until="domcontentloaded")
        return f"went to {action.get('url')}"
    if verb == "scroll":
        page.mouse.wheel(0, -800 if action.get("text", "").startswith("up") else 800)
        return "scrolled"
    raise ValueError(f"unknown action: {verb!r}")


def _load_agent_config():
    """Pull the agent's LLM config from the same store the app uses.

    Resolves ``${vault:NAME}`` references so a migrated LLM key (issue #35)
    reaches the explore loop's own LLM calls. This subprocess inherits the
    machine key (``HUMUX_MASTER_KEY`` / ``data/master.key``) and ``.env``, so the
    infra vault unseals headlessly with a plain env fallback.
    """
    import asyncio

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.config_store import ConfigStore
    from core.secret_store import SecretStore

    db = str(_data_dir() / "config.db")

    async def _load():
        secret_store = SecretStore(db_path=db)
        await secret_store.load_infra_cache()
        return await ConfigStore(db).export_to_config(vault_resolve=secret_store.infra_resolve)

    return asyncio.run(_load()).agent


class _Aio:
    """Run coroutines on a persistent background loop.

    Playwright's sync API holds an event loop on the main thread, so `asyncio.run`
    there raises "cannot be called from a running event loop". One long-lived loop
    in its own thread sidesteps that and keeps the AsyncOpenAI client bound to a
    single loop across calls.
    """

    def __init__(self):
        import asyncio
        import threading

        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

    def run(self, coro):
        import asyncio

        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)


def cmd_explore(args) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.llm import LLMClient

    profile = _validate_profile(args.profile)
    _prune_old_shots()
    agent_cfg = _load_agent_config()
    llm = LLMClient.from_agent_config(agent_cfg)
    # Each step is a small mechanical action pick over rich state — high "thinking"
    # just makes every step slow (and the whole loop time out). Force it low so a
    # 25-step booking finishes in the time budget.
    llm.thinking_level = "low"
    model = agent_cfg.model
    aio = _Aio()
    verbose = bool(os.environ.get("BROWSER_EXPLORE_VERBOSE"))

    trail: list[str] = []

    def trail_block() -> str:
        return "STEPS SO FAR: " + (", ".join(trail) if trail else "(none yet)")

    try:
        _write_status("loading page…")
        with _Session(profile, args.headless) as s:
            s.goto(args.url, args.timeout)
            state, frame_map = _observe(s.page)
            messages: list[dict] = [
                {"role": "user", "content": f"TASK: {args.task}\n\n{trail_block()}\n\n{state}"}
            ]
            answer = None
            reason = None
            prev_state = state
            stuck = 0  # consecutive steps with no page change → bail out (see below)
            STUCK_LIMIT = 5
            for step in range(args.max_steps):
                resp = aio.run(
                    llm.generate(
                        model=model,
                        system=_EXPLORE_SYSTEM,
                        messages=messages,
                        tools=[_ACTION_TOOL],
                        max_tokens=1024,
                    )
                )
                if not resp.tool_calls:
                    answer = resp.text  # model answered in prose — treat as done
                    break
                call = resp.tool_calls[0]
                action = call.arguments or {}
                # openai-compatible assistant turn carrying exactly one tool_call, so
                # the required tool-result message pairs up cleanly. ponytail: deepseek
                # path only; anthropic tool threading differs (this verb is deepseek).
                messages.append(
                    {
                        "role": "assistant",
                        "content": resp.text or "",
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {"name": call.name, "arguments": json.dumps(action)},
                            }
                        ],
                    }
                )
                if action.get("action") == "done":
                    answer = action.get("answer", "")
                    trail.append("done")
                    break
                # Short per-action timeout: a buried/wrong element should fail in
                # seconds so the model can recover, not burn the whole budget on one
                # 30s actionability wait. Navigation keeps the full timeout.
                act_timeout = args.timeout if action.get("action") == "goto" else 6000
                try:
                    note = _apply_action(frame_map, action, act_timeout, page=s.page)
                except Exception as exc:
                    msg = str(exc).split("\nCall log:")[0]  # drop Playwright's verbose log
                    note = f"error: {type(exc).__name__}: {msg}"
                if note.startswith("error"):
                    label = note
                elif action.get("action") in ("goto", "scroll"):
                    label = note
                else:
                    label = f"{action.get('action')}[{action.get('index')}]"
                trail.append(label)
                _settle(s.page, 3500)
                state, frame_map = _observe(s.page)
                # No-progress guard: if the page looks identical, tell the model so
                # it stops hammering the same dead element.
                changed = state != prev_state
                prev_state = state
                hint = (
                    ""
                    if changed
                    else (
                        "\nNOTE: the page did NOT change after your action — it had no "
                        "effect. Do not repeat it; try a different element (you may need "
                        "to scroll, or click a radio/option first)."
                    )
                )
                tag = "" if changed else " (no change)"
                _write_status(f"step {step + 1}/{args.max_steps} · {label}{tag}")
                if verbose:
                    print(f"[step {step}] {label}{tag}", file=sys.stderr)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": f"{note}\n\n{trail_block()}{hint}\n\n{state}",
                    }
                )
                # Feasibility guard: if nothing the model tries changes the page for
                # STUCK_LIMIT steps in a row, the flow is wedged (element not found,
                # gated step, dead end). Bail with a reason instead of burning the
                # whole budget — an impossible task must fail fast and cheap.
                stuck = stuck + 1 if not changed else 0
                if stuck >= STUCK_LIMIT:
                    reason = (
                        f"Stopped after {stuck} consecutive steps with no effect on the "
                        f"page — the flow appears stuck (last actions: "
                        f"{', '.join(trail[-stuck:])}). The task may not be completable "
                        f"as described, or a required control could not be found."
                    )
                    break

            host = urlparse(s.page.url).netloc or "page"
            shot = _shot_dir() / f"clio-{host}-{int(time.time())}.png"
            s.snapshot(shot, full_page=True)
            # Return only what the calling agent needs — the full step trail is
            # debugging noise in its context (use BROWSER_EXPLORE_VERBOSE to see it).
            result = {
                "answer": answer,
                "done": answer is not None,
                "steps_taken": len(trail),
                "url": s.page.url,
                "screenshot": str(shot),
            }
            if reason:  # stuck-abort verdict, or hit the step cap without finishing
                result["reason"] = reason
            elif answer is None:
                result["reason"] = (
                    f"Ran out of steps ({args.max_steps}) before the task reported "
                    f"completion. Increase --max-steps or simplify the task."
                )
            return result
    finally:
        aio.close()
        try:
            _status_path().unlink(missing_ok=True)  # done — let the spinner revert
        except OSError:
            pass


def cmd_profiles(_args) -> dict:
    """List saved profiles with a rough 'authenticated' hint (has a cookies DB)."""
    root = _data_dir() / "browser" / "profiles"
    out = []
    if root.exists():
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            state = d / "storage_state.json"
            authed = False
            if state.exists():
                try:
                    authed = bool(json.loads(state.read_text()).get("cookies"))
                except Exception:
                    authed = False
            out.append(
                {
                    "name": d.name,
                    # ponytail: "logged in" approximated by having saved cookies.
                    "authenticated": authed,
                    "updated": int(state.stat().st_mtime)
                    if state.exists()
                    else int(d.stat().st_mtime),
                }
            )
    return {"profiles": out}


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless browser automation (Playwright).")
    # Default headless from the env injected by core/tools.py (config.tools.browser).
    headless_default = os.environ.get("BROWSER_HEADLESS", "1") != "0"
    # Default profile from the env injected per active agent (#93): when an agent
    # omits --profile, it still gets its OWN isolated session dir, not a shared one.
    profile_default = os.environ.get("BROWSER_PROFILE", "default") or "default"
    parser.add_argument(
        "--headless", dest="headless", action="store_true", default=headless_default
    )
    parser.add_argument("--headed", dest="headless", action="store_false")
    parser.add_argument("--timeout", type=int, default=30_000, help="per-action timeout (ms)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_read = sub.add_parser("read", help="Load a page, return readable text")
    p_read.add_argument("--url", required=True)
    p_read.add_argument("--profile", default=profile_default)

    p_shot = sub.add_parser("screenshot", help="Load a page, save a PNG")
    p_shot.add_argument("--url", required=True)
    p_shot.add_argument("--profile", default=profile_default)
    p_shot.add_argument("-o", "--output", help="output PNG path")
    p_shot.add_argument("--full-page", action="store_true", default=True)
    p_shot.add_argument("--viewport-only", dest="full_page", action="store_false")

    p_act = sub.add_parser("act", help="Load a page and run ordered steps")
    p_act.add_argument("--url", required=True)
    p_act.add_argument("--profile", default=profile_default)
    p_act.add_argument("--steps", required=True, help="JSON array of single-key step objects")

    p_exp = sub.add_parser("explore", help="LLM-driven loop: one live browser, act until done")
    p_exp.add_argument("--url", required=True)
    p_exp.add_argument("--task", required=True, help="what to accomplish on the site")
    p_exp.add_argument("--profile", default=profile_default)
    p_exp.add_argument("--max-steps", type=int, default=35, dest="max_steps")

    sub.add_parser("profiles", help="List saved profiles + auth hint")

    args = parser.parse_args()
    handlers = {
        "read": cmd_read,
        "screenshot": cmd_screenshot,
        "act": cmd_act,
        "explore": cmd_explore,
        "profiles": cmd_profiles,
    }
    try:
        result = handlers[args.command](args)
        print(json.dumps(result, indent=2, default=str))
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)
    except Exception as exc:  # Playwright errors, navigation failures, etc.
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
