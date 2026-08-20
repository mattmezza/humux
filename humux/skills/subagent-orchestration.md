# Subagent orchestration — split big work across parallel subagents, then synthesize

You can delegate work to subagents: `spawn_subagent` runs one, `spawn_subagents`
runs 2–6 in parallel. This skill is about *when* that is worth it and how to
brief, size, and reassemble the runs.

## When NOT to fan out

Start here. A subagent is a full agent run: own system prompt, cold start, own
tokens, and no knowledge of this conversation. Three quick web searches, two
file reads, one API call — do those inline. Spawning for them costs more tokens
and more wall-clock than doing the work yourself.

Fan out when **both** hold: each subtask needs several tool calls and its own
reading, and the subtasks are genuinely independent. If only one holds, do it
inline or run a pipeline.

## Independent vs dependent

**Independent → `spawn_subagents`.** "Compare Postgres, MySQL and SQLite for
this workload." Each is researched without knowing the others. One call, three
tasks, all at once; wall-clock ≈ the slowest one.

**Dependent → pipeline.** "Find the slowest endpoint, then optimize it." You
cannot brief the optimizer before you know the endpoint. Spawn the profiler,
read its result, *then* spawn the optimizer in a later step with the finding
baked into the brief. Never fan out a chain — the later briefs would be blanks.

## Writing a brief

The subagent cannot see the conversation, your memory, or the user's last
message. Everything it needs goes in `task`. Include:

- **goal** — one sentence, what "done" means;
- **inputs** — absolute paths, URLs, ids, versions. Never "the file we discussed";
- **constraints** — what not to touch, time/source limits, tone;
- **answer shape** — the exact structure you want back, since you will paste it
  into a synthesis.

> **Bad:** Look into the pricing thing and report back.

> **Good:** Research current list pricing for Vercel Pro (goal). Use
> vercel.com/pricing and the official docs only; ignore blogs (constraints).
> Report: monthly base price, included bandwidth, overage price per GB, and the
> date the page was last updated. Six lines max, no prose intro (shape).

If a brief is hard to write self-contained, that is a signal the task is not
independent — pipeline it.

## Labels

Always give every fan-out task a short distinctive `label`, 1–3 words:
`pricing`, `docs-sweep`, `sqlite-bench`. It names the run in the logs and the
admin UI — it is how four concurrent runs stay tellable apart. Derive it from
the task's first words if nothing better comes to mind.

## Sizing

| Task shape | max_steps | token_budget | thinking_effort | model |
|---|---|---|---|---|
| Mechanical extraction (read file, pull fields) | 5–8 | 15k–25k | `off`/`low` | cheap fast model |
| Focused web research, 3–6 sources | 10–15 | 30k–50k | `low` | cheap fast model |
| Code reading + explanation across a module | 15–25 | 50k–80k | `medium` | parent's model |
| Judgement, design, tradeoff analysis, review | 20–30 | 80k–120k | `high` | parent's model |

Requested values are capped at the configured maxima — asking for 1M tokens
gets the cap, not an error. Undersizing is the commoner mistake: a run that
stops with `stopped_reason: max_steps` wasted everything it spent.

## The patterns

**Map** — same operation, many inputs.

```jsonc
{
  "tasks": [
    {"label": "auth-audit", "task": "Read /srv/app/auth.py; list every function touching a password or token, with line numbers. Bullets only.", "max_steps": 8, "thinking_effort": "low"},
    {"label": "billing-audit", "task": "Read /srv/app/billing.py; list every function touching a password or token, with line numbers. Bullets only.", "max_steps": 8, "thinking_effort": "low"}
  ]
}
```

**Map-reduce** — same, but *you* are the reduce step. Fan out, then write the
comparison yourself from the returned summaries. Do not spawn a child to
synthesize; you already have every result in-turn.

**Pipeline** — sequential, each brief built from the last result. Spawn step
one, read it, then spawn step two with the finding baked in.

```jsonc
{
  "task": "Profile /srv/app with `python -m cProfile` over the test suite at /srv/app/tests; report the 5 slowest functions with cumulative times. No fixes.",
  "max_steps": 15,
  "thinking_effort": "medium"
}
```

**Verification pass** — the highest-value pattern and the most underused. After
producing something substantial (a plan, a migration, a report, a refactor),
spawn one fresh subagent whose only job is to check it against the original
brief. Fresh context means no attachment to your reasoning and no memory of why
you chose what you chose — it sees what the user will see.

```jsonc
{
  "task": "Verify a deliverable. The original request was: 'migrate the settings module from JSON to TOML without changing the public API'. The work is at /srv/app/settings.py and /srv/app/settings.toml. Check: (1) every key present in the old JSON schema at /srv/app/settings.schema.json still exists, (2) the public function signatures are unchanged, (3) nothing else in /srv/app imports json for settings. Report only discrepancies, each with file and line. If there are none, say 'no discrepancies'.",
  "max_steps": 12,
  "thinking_effort": "high"
}
```

Cheap, bounded, and it catches the class of error you are structurally blind to.
Do it before handing anything consequential to the user.

**Best-of-N** — same task to N subagents at different angles, then pick. Only
when the user actually asked for options ("give me a few directions for the
landing page"); otherwise it is N× the cost for one answer you must arbitrate.

## Handing back big results

Subagents share your filesystem. A child producing bulk output — a long report,
extracted data, generated code — should write it to a file and return the
absolute path plus a summary. Put that in the brief:

> Write the full table to /srv/scratch/pricing.md and return only the absolute
> path under a `Files:` line plus a 3-line summary.

Then read the file only if you need the detail. Returning 20k tokens of body
text through the result field burns your turn budget for no gain.

## Partial failure

Results are per-index and order is preserved. You will get 3-of-4. Read `status`
on **every** entry, and `stopped_reason` on any that came back thin —
`max_steps` or `token_budget` means truncated, not wrong-but-complete.

For each failure: retry once with a larger budget if a cap caused it, work
around it (do that piece inline), or tell the user which part is missing. Never
synthesize as though you got everything — a comparison table with one silently
guessed column is worse than a table with a visible gap.

## Budgets

Every spawn — singular, plural, and any nested subagent — draws from one shared
pool for the whole turn: `max_spawns_per_turn` (default 12) and
`turn_token_budget` (default 400000). Each run reserves its `token_budget` up
front and refunds the unused part on completion — so a generous budget on a
short task is cheap; generous budgets on many tasks are not.

Two or three well-briefed subtasks beat six vague ones, every time. A spawn
refused with a budget error is the signal to finish with what you have and tell
the user what you skipped — not to retry the fan-out in smaller pieces until
the pool drains. Note also that only `max_concurrent` (default 4) runs execute
at once per chat, so a width-6 fan-out is two waves, not one.

## Anti-patterns

- **Nested fan-out.** `spawn_subagents` exists only at the top level; a child
  can still `spawn_subagent` serially up to `recursion_depth` (default 3). If
  you want a tree, flatten it into one top-level fan-out.
- **Fanning out to dodge a hard problem.** Four confused children produce four
  confused answers you then have to reconcile. Understand it first, then split.
- **Spawning for something you have a direct tool for.** If one tool call
  answers it, call the tool.
- **Six near-identical briefs.** If they differ by one word, you wanted one
  subagent with a list, or a loop you run yourself.
