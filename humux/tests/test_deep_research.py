"""Deep research pipeline tests (issue #293)."""

from __future__ import annotations

import pytest

from core import deep_research
from core.agent import apply_feature_gates
from core.deep_research import (
    ResearchCancelled,
    html_to_text,
    parse_lines,
    render_report_html,
    slugify,
)

# ── helpers ─────────────────────────────────────────────────────────────────


def make_gen(replies: list[str], tokens_per_call: int = 100):
    """A gen_fast/gen_synth stub yielding canned replies in order (last repeats)."""
    calls: list[str] = []

    async def gen(prompt: str, max_tokens: int) -> tuple[str, int]:
        calls.append(prompt)
        reply = replies[min(len(calls) - 1, len(replies) - 1)]
        return reply, tokens_per_call

    gen.calls = calls
    return gen


async def fake_search(query: str) -> dict:
    return {
        "query": query,
        "results": [
            {"title": f"Result for {query}", "url": f"https://ex.com/{query}", "content": "snip"}
        ],
    }


async def fake_fetch(url: str) -> str:
    return f"page text of {url}"


# ── unit helpers ────────────────────────────────────────────────────────────


def test_html_to_text_strips_script_and_style():
    raw = (
        "<html><head><style>b{}</style></head>"
        "<body><script>x()</script><p>Hello <b>world</b></p></body></html>"
    )
    assert html_to_text(raw) == "Hello world"


def test_parse_lines_strips_bullets_numbering_and_none():
    raw = "1. first query\n- second query\n• third\nNONE\n\n"
    assert parse_lines(raw, 5) == ["first query", "second query", "third"]
    assert parse_lines("NONE", 5) == []
    assert parse_lines("a\nb\nc", 2) == ["a", "b"]


def test_slugify_charset_and_suffix():
    slug = slugify("Impact of X on Y? (2026)", "abc123")
    assert slug == "research-impact-of-x-on-y-2026-abc123"
    # servable by the artifacts route charset
    from core.artifacts import valid_id

    assert valid_id(slug)
    assert valid_id(slugify("???", "abc123"))


def test_render_report_html_escapes_and_links_citations():
    page = render_report_html(
        "<query> & co",
        "## Summary\nA **bold** claim [1].\n\n- item one\n- item two",
        [{"n": 1, "url": "https://ex.com/a?x=1&y=2", "title": "Source <A>"}],
        meta="July 10, 2026 · 1 sources",
    )
    assert "&lt;query&gt; &amp; co" in page
    assert '<a href="#src-1">[1]</a>' in page
    assert "<strong>bold</strong>" in page
    assert "<li>item one</li>" in page
    assert 'id="src-1"' in page
    assert "Source &lt;A&gt;" in page


# ── pipeline ────────────────────────────────────────────────────────────────


async def test_run_happy_path_single_cycle():
    gen_fast = make_gen(["q-one\nq-two", "fact about q-one", "fact about q-two"])
    gen_synth = make_gen(["## Summary\nAnswer [1][2].\n\n## Conclusions\nDone."])
    progress_msgs: list[str] = []

    async def progress(msg: str) -> None:
        progress_msgs.append(msg)

    result = await deep_research.run(
        "the question",
        depth=1,
        max_sources=10,
        token_budget=100_000,
        gen_fast=gen_fast,
        gen_synth=gen_synth,
        search=fake_search,
        fetch=fake_fetch,
        progress=progress,
    )
    assert "error" not in result
    assert result["report"].startswith("## Summary")
    assert [s["n"] for s in result["sources"]] == [1, 2]
    assert result["tokens_used"] == 400  # plan + 2 extracts + synthesis
    assert any("Synthesizing" in m for m in progress_msgs)
    # depth=1 → no gap-analysis call: plan + 2 extracts only on the fast model
    assert len(gen_fast.calls) == 3


async def test_run_iterates_on_gaps_and_dedupes_urls():
    # plan → one query; extract; gap → same query again (dupe URL) + a new one;
    # extract for the new one; second gap never runs (depth=2).
    gen_fast = make_gen(["alpha", "notes alpha", "alpha\nbeta", "notes beta"])
    gen_synth = make_gen(["## Summary\nok"])
    result = await deep_research.run(
        "q",
        depth=2,
        max_sources=10,
        token_budget=100_000,
        gen_fast=gen_fast,
        gen_synth=gen_synth,
        search=fake_search,
        fetch=fake_fetch,
    )
    urls = [s["url"] for s in result["sources"]]
    assert urls == ["https://ex.com/alpha", "https://ex.com/beta"]  # dupe dropped


async def test_run_irrelevant_sources_dropped_and_empty_errors():
    gen_fast = make_gen(["only-q", "IRRELEVANT"])
    gen_synth = make_gen(["never"])
    result = await deep_research.run(
        "q",
        depth=1,
        max_sources=5,
        token_budget=100_000,
        gen_fast=gen_fast,
        gen_synth=gen_synth,
        search=fake_search,
        fetch=fake_fetch,
    )
    assert "error" in result and "report" not in result


async def test_run_token_budget_stops_iteration_but_still_synthesizes():
    gen_fast = make_gen(["a\nb", "notes", "notes"], tokens_per_call=60_000)
    gen_synth = make_gen(["## Summary\npartial"])
    result = await deep_research.run(
        "q",
        depth=3,
        max_sources=10,
        token_budget=100_000,
        gen_fast=gen_fast,
        gen_synth=gen_synth,
        search=fake_search,
        fetch=fake_fetch,
    )
    # plan (60k) + 2 extracts (120k) blows the budget → no gap cycle, but the
    # report is still written from what was gathered.
    assert result["report"] == "## Summary\npartial"
    assert len(gen_fast.calls) == 3


async def test_run_cancelled_between_phases():
    gen_fast = make_gen(["a", "notes"])
    gen_synth = make_gen(["never"])
    with pytest.raises(ResearchCancelled):
        await deep_research.run(
            "q",
            depth=2,
            max_sources=10,
            token_budget=100_000,
            gen_fast=gen_fast,
            gen_synth=gen_synth,
            search=fake_search,
            fetch=fake_fetch,
            cancelled=lambda: True,
        )


async def test_run_max_sources_cap():
    async def many_results(query: str) -> dict:
        return {
            "results": [
                {"title": f"t{i}", "url": f"https://ex.com/{i}", "content": "c"} for i in range(20)
            ]
        }

    gen_fast = make_gen(["only-q"] + ["notes"] * 20)
    gen_synth = make_gen(["## Summary\nok"])
    result = await deep_research.run(
        "q",
        depth=1,
        max_sources=3,
        token_budget=100_000,
        gen_fast=gen_fast,
        gen_synth=gen_synth,
        search=many_results,
        fetch=fake_fetch,
    )
    assert len(result["sources"]) == 3


async def test_run_fetch_failure_falls_back_to_snippet():
    async def no_fetch(url: str) -> str:
        return ""

    gen_fast = make_gen(["only-q", "notes from snippet"])
    gen_synth = make_gen(["## Summary\nok"])
    result = await deep_research.run(
        "q",
        depth=1,
        max_sources=5,
        token_budget=100_000,
        gen_fast=gen_fast,
        gen_synth=gen_synth,
        search=fake_search,
        fetch=no_fetch,
    )
    assert len(result["sources"]) == 1  # snippet ("snip") was extracted instead


# ── feature gating ──────────────────────────────────────────────────────────


def test_feature_gate_requires_both_flags():
    tools = [{"name": "deep_research"}, {"name": "web_search"}]

    def gated(**kw):
        return [t["name"] for t in apply_feature_gates(tools, secrets_available=True, **kw)]

    assert "deep_research" in gated(search_enabled=True, deep_research_enabled=True)
    assert "deep_research" not in gated(search_enabled=False, deep_research_enabled=True)
    assert "deep_research" not in gated(search_enabled=True, deep_research_enabled=False)
