"""Deep research — autonomous multi-step web research and report synthesis (#293).

The ``deep_research`` tool runs a deterministic pipeline instead of a free-form
agent loop: plan (decompose the question into sub-questions) → search (the
configured web-search provider) → read (plain HTTP fetch + HTML-to-text; the
browser tool is overkill for articles) → iterate (spot gaps, search again, up
to ``depth`` cycles) → synthesize (a structured, cited report).

Sub-calls (planning / per-source extraction / gap analysis) run on a fast,
cheap model; the synthesis defaults to the main agent model. Both are
configurable in the Tools tab. A hard ``token_budget`` bounds the whole run.

The pipeline is pure orchestration: everything with side effects (LLM clients,
web search, progress delivery, cancellation) is passed in by the tool handler
in ``core/agent.py``, so this module stays independently testable.
"""

from __future__ import annotations

import asyncio
import html
import html.parser
import logging
import re
from collections.abc import Awaitable, Callable

log = logging.getLogger(__name__)

# Cap on the page text handed to the extraction model — enough for most
# articles; keeps a single huge page from eating the whole token budget.
PAGE_TEXT_CAP = 8_000
FETCH_TIMEOUT = 15.0
# Sub-questions per plan and gap queries per cycle. Fixed: the knobs that
# matter for cost (depth, max_sources, token_budget) are already configurable.
MAX_SUBQUESTIONS = 5
MAX_GAP_QUERIES = 3


class ResearchCancelled(Exception):
    """Raised between pipeline phases when the user stopped the turn."""


class _TextExtractor(html.parser.HTMLParser):
    """Strip an HTML document to its visible text (script/style/nav dropped)."""

    _SKIP = {"script", "style", "noscript", "template", "svg", "head"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth and data.strip():
            self._chunks.append(data)

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._chunks)).strip()


def html_to_text(raw: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(raw)
        parser.close()
    except Exception:  # tolerate broken markup — keep whatever was parsed
        pass
    return parser.text()


async def fetch_page(url: str) -> str:
    """Fetch a URL and return its visible text (capped), or '' on any failure.

    Plain HTTP GET — deliberately not the JS-enabled browser tool; most
    articles don't need it and a failed fetch just drops one source.
    """
    import httpx

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(FETCH_TIMEOUT),
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; humux-research)"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if "html" not in ctype and "text" not in ctype:
                return ""
            text = html_to_text(resp.text) if "html" in ctype else resp.text
            return text[:PAGE_TEXT_CAP]
    except Exception as exc:
        log.info("deep_research: fetch failed for %s: %s", url, exc)
        return ""


def parse_lines(raw: str, cap: int) -> list[str]:
    """Parse an LLM 'one item per line' reply: strip bullets/numbering, cap."""
    out = []
    for line in (raw or "").splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if line and line.upper() != "NONE":
            out.append(line)
    return out[:cap]


# ── prompts ─────────────────────────────────────────────────────────────────

_PLAN_PROMPT = (
    "Decompose this research question into {n} focused, self-contained web search "
    "queries covering its distinct angles. One query per line, nothing else.\n\n"
    "Research question: {query}"
)

_EXTRACT_PROMPT = (
    "You are extracting research notes. From the page text below, pull every fact, "
    "figure, date, name or claim relevant to the research question. Terse bullet "
    "points with specifics. If the page has nothing relevant, reply exactly "
    "IRRELEVANT.\n\nResearch question: {query}\n\nPage ({url}):\n{text}"
)

_GAP_PROMPT = (
    "Research question: {query}\n\nNotes gathered so far:\n{notes}\n\n"
    "What important angles are still missing or unverified? Reply with up to "
    "{n} web search queries that would fill the gaps, one per line. If the notes "
    "already cover the question well, reply exactly NONE."
)

_SYNTH_PROMPT = (
    "Write a structured research report in Markdown answering the research "
    "question from the numbered source notes below. Requirements:\n"
    "- Start with a '## Summary' section (a few tight paragraphs).\n"
    "- Then 2-5 thematic '## <heading>' finding sections.\n"
    "- End with a '## Conclusions' section.\n"
    "- Cite sources inline as [n] matching the source numbers.\n"
    "- Only claims supported by the notes; note disagreements between sources.\n"
    "- No preamble before the first heading, no source list (appended separately).\n\n"
    "Research question: {query}\n\nSource notes:\n{notes}"
)


async def run(
    query: str,
    *,
    depth: int,
    max_sources: int,
    token_budget: int,
    gen_fast: Callable[[str, int], Awaitable[tuple[str, int]]],
    gen_synth: Callable[[str, int], Awaitable[tuple[str, int]]],
    search: Callable[[str], Awaitable[dict]],
    fetch: Callable[[str], Awaitable[str]] = fetch_page,
    progress: Callable[[str], Awaitable[None]] | None = None,
    cancelled: Callable[[], bool] = lambda: False,
) -> dict:
    """Run the research pipeline; returns {query, report, sources, tokens_used}.

    ``gen_fast``/``gen_synth`` are (prompt, max_tokens) → (text, tokens_spent)
    callables; ``search`` returns the web_search tool's {results: [...]} shape.
    ``cancelled`` is polled between phases (→ ResearchCancelled). The token
    budget is enforced between phases: when exceeded, the loop stops early and
    the report is synthesized from whatever was gathered.
    """
    tokens = 0

    async def note_progress(msg: str) -> None:
        if progress:
            try:
                await progress(msg)
            except Exception:
                log.exception("deep_research: progress delivery failed")

    def check_cancel() -> None:
        if cancelled():
            raise ResearchCancelled()

    # Plan
    text, spent = await gen_fast(_PLAN_PROMPT.format(n=MAX_SUBQUESTIONS, query=query), 1000)
    tokens += spent
    queries = parse_lines(text, MAX_SUBQUESTIONS) or [query]
    await note_progress(
        "🔍 Researching: " + "; ".join(queries[:3]) + ("…" if len(queries) > 3 else "")
    )

    sources: list[dict] = []  # {n, url, title, notes}
    seen_urls: set[str] = set()

    for cycle in range(max(1, depth)):
        check_cancel()
        if tokens >= token_budget or len(sources) >= max_sources:
            break

        # Search all this cycle's queries concurrently; dedupe URLs across cycles.
        results = await asyncio.gather(*(search(q) for q in queries), return_exceptions=True)
        candidates: list[dict] = []
        cycle_urls: set[str] = set()
        for res in results:
            if not isinstance(res, dict):
                continue
            for item in res.get("results", []):
                url = (item.get("url") or "").strip()
                if url and url not in seen_urls and url not in cycle_urls:
                    cycle_urls.add(url)
                    candidates.append({**item, "url": url})  # normalized url everywhere below
        candidates = candidates[: max_sources - len(sources)]
        # Only URLs actually read count as seen — a candidate dropped by the
        # cap above stays eligible for a later cycle.
        seen_urls.update(c["url"] for c in candidates)
        if not candidates:
            break

        check_cancel()
        await note_progress(f"📖 Reading {len(candidates)} sources (cycle {cycle + 1}/{depth})…")

        # Fetch + extract per source in one independent chain, concurrently —
        # no barrier, so one slow site doesn't stall the others' extraction.
        # A failed fetch falls back to the search snippet.
        async def read_source(cand: dict) -> tuple[dict, str, int]:
            body = (await fetch(cand["url"])) or cand.get("content", "")
            if not body:
                return cand, "", 0
            out, spent = await gen_fast(
                _EXTRACT_PROMPT.format(query=query, url=cand["url"], text=body[:PAGE_TEXT_CAP]),
                1500,
            )
            return cand, out, spent

        extracted = await asyncio.gather(
            *(read_source(c) for c in candidates), return_exceptions=True
        )
        for item in extracted:
            if not isinstance(item, tuple):
                log.warning("deep_research: source extraction failed: %s", item)
                continue
            cand, notes, spent = item
            tokens += spent
            if notes and notes.strip().upper() != "IRRELEVANT":
                sources.append(
                    {
                        "n": len(sources) + 1,
                        "url": cand["url"],
                        "title": cand.get("title", "") or cand["url"],
                        "notes": notes.strip(),
                    }
                )

        # Gap analysis feeds the next cycle's queries.
        if cycle + 1 >= depth or tokens >= token_budget or len(sources) >= max_sources:
            break
        check_cancel()
        notes_blob = _notes_blob(sources)
        text, spent = await gen_fast(
            _GAP_PROMPT.format(query=query, notes=notes_blob, n=MAX_GAP_QUERIES), 500
        )
        tokens += spent
        queries = parse_lines(text, MAX_GAP_QUERIES)
        if not queries:
            break
        await note_progress("🕳️ Following up on gaps: " + "; ".join(queries))

    if not sources:
        return {
            "query": query,
            "error": "No readable sources found — try rephrasing the question.",
            "tokens_used": tokens,
        }

    check_cancel()
    await note_progress(f"✍️ Synthesizing report from {len(sources)} sources…")
    report, spent = await gen_synth(
        _SYNTH_PROMPT.format(query=query, notes=_notes_blob(sources)), 8192
    )
    tokens += spent
    return {
        "query": query,
        "report": report.strip(),
        "sources": [{"n": s["n"], "url": s["url"], "title": s["title"]} for s in sources],
        "tokens_used": tokens,
    }


def _notes_blob(sources: list[dict]) -> str:
    return "\n\n".join(f"[{s['n']}] {s['title']} ({s['url']})\n{s['notes']}" for s in sources)


# ── artifact rendering ──────────────────────────────────────────────────────
# One shared theme so every deep-research artifact has the same look and feel.

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font-family: system-ui, -apple-system, sans-serif; line-height: 1.65;
  max-width: 46rem; margin: 0 auto; padding: 2rem 1.25rem 4rem;
  color: #1f2430; background: #fbfbfd; }}
header {{ border-bottom: 2px solid #6366f1; padding-bottom: 1rem; margin-bottom: 2rem; }}
header .kicker {{ color: #6366f1; font-size: .8rem; text-transform: uppercase;
  letter-spacing: .12em; font-weight: 600; }}
h1 {{ font-size: 1.6rem; margin: .3rem 0 .4rem; }}
header .meta {{ color: #6b7280; font-size: .85rem; }}
h2 {{ font-size: 1.2rem; margin-top: 2.2rem; border-bottom: 1px solid #e5e7eb;
  padding-bottom: .3rem; }}
h3 {{ font-size: 1.05rem; margin-top: 1.6rem; }}
a {{ color: #4f46e5; }}
code {{ background: #eef0f4; padding: .1em .35em; border-radius: 4px; font-size: .9em; }}
ol.sources {{ font-size: .9rem; color: #4b5563; }}
ol.sources li {{ margin-bottom: .35rem; overflow-wrap: anywhere; }}
sup a {{ text-decoration: none; }}
@media (prefers-color-scheme: dark) {{
  body {{ color: #e2e5ec; background: #14161c; }}
  h2 {{ border-color: #2c303a; }}
  header .meta {{ color: #9aa1ad; }}
  a {{ color: #8b90ff; }}
  code {{ background: #232733; }}
}}
</style>
</head>
<body>
<header>
  <div class="kicker">Deep research</div>
  <h1>{title}</h1>
  <div class="meta">{meta}</div>
</header>
{body}
<h2>Sources</h2>
<ol class="sources">
{sources}
</ol>
</body>
</html>
"""


def _md_inline(text: str) -> str:
    """Escape + render the inline Markdown subset the synthesis prompt yields."""
    text = html.escape(text, quote=False)
    text = re.sub(r"\[(\d+)\]", r'<sup><a href="#src-\1">[\1]</a></sup>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)
    return text


def _md_to_html(md: str) -> str:
    """Minimal Markdown→HTML for the report structure (headings, lists, paras).

    ponytail: covers exactly what _SYNTH_PROMPT asks for; swap in a real
    Markdown dependency if reports ever need tables/quotes/code fences.
    """
    out: list[str] = []
    for block in re.split(r"\n{2,}", md.strip()):
        lines = block.splitlines()
        if all(re.match(r"\s*[-*•]\s+", ln) for ln in lines if ln.strip()):
            out.append("<ul>")
            for ln in lines:
                item = re.sub(r"^\s*[-*•]\s+", "", ln).strip()
                if item:
                    out.append(f"<li>{_md_inline(item)}</li>")
            out.append("</ul>")
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", lines[0])
        if m:
            # md '##' → h2 (the styled section heading); h1 stays the page title.
            level = min(max(len(m.group(1)), 2), 4)
            out.append(f"<h{level}>{_md_inline(m.group(2))}</h{level}>")
            rest = "\n".join(lines[1:]).strip()
            if rest:
                out.append(f"<p>{_md_inline(rest)}</p>")
            continue
        out.append(f"<p>{_md_inline(' '.join(ln.strip() for ln in lines))}</p>")
    return "\n".join(out)


def render_report_html(query: str, report_md: str, sources: list[dict], meta: str = "") -> str:
    """The themed artifact page for one research run."""
    src_items = "\n".join(
        f'<li id="src-{s["n"]}" value="{s["n"]}"><a href="{html.escape(s["url"], quote=True)}" '
        f'rel="noreferrer">{html.escape(s["title"] or s["url"], quote=False)}</a></li>'
        for s in sources
    )
    return _PAGE.format(
        title=html.escape(query, quote=False),
        meta=html.escape(meta, quote=False),
        body=_md_to_html(report_md),
        sources=src_items,
    )


def slugify(query: str, suffix: str) -> str:
    """Artifact slug: charset-safe, readable, collision-suffixed."""
    base = re.sub(r"[^A-Za-z0-9]+", "-", query.lower()).strip("-")[:40].strip("-")
    return f"research-{base}-{suffix}" if base else f"research-{suffix}"
