"""Memory store — initializes schema, queries memories, and extracts new ones."""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite

from core.embeddings import (
    EmbeddingClient,
    cosine_similarity,
    cosine_to_matrix,
    pack_vector,
    unpack_vector,
)
from core.llm import LLMClient

log = logging.getLogger(__name__)

_SCHEMA_FILE = Path(__file__).resolve().parent.parent / "schema" / "memory.sql"

_CONSOLIDATION_PROMPT = """\
You are reviewing short-term memories stored by a personal AI assistant.
Your job is to decide which short-term memories contain facts worth keeping
permanently, and compact them into long-term memories.

## Existing long-term memories (for deduplication)
{existing_long_term}

## Short-term memories to review
{short_term_entries}

For each short-term memory, decide:
1. PROMOTE — the fact is durable and worth keeping. Compact it aggressively
   into a short, dense long-term memory. Strip dates, times, and situational
   framing. Keep only the core fact.
   Examples:
   - "Matteo is at the airport, flight to Rome boards at 15:40" → discard (ephemeral)
   - "Matteo mentioned he has a standing lunch with Marco every Friday" → promote: \
"Has a standing Friday lunch with Marco"
   - "Simge is visiting her parents this weekend" → discard (time-bound)
   - "Matteo said he switched to a standing desk at work" → promote: \
"Uses a standing desk at work"
2. DISCARD — the fact is time-bound, stale, or already captured in long-term memory.

Return a JSON array of objects to promote. Each object:
  {{"category": "<category>", "subject": "<who/what>", \
"content": "<compacted fact>"}}

Categories: preference, relationship, fact, routine, work, health, travel

Rules:
- Be ruthless. Most short-term memories should be discarded.
- Only promote facts that would still be useful weeks or months from now.
- Compact aggressively: strip temporal context, keep the core fact.
- Do NOT promote anything that duplicates or overlaps with existing long-term memories.
- If a short-term memory refines or updates an existing long-term memory,
  promote it with the updated content (it will replace the old one).
- Use lowercase for subject (e.g. "matteo", "simge").
- If nothing is worth promoting, return an empty array: []

Respond with ONLY the JSON array, no other text."""

_UPDATE_PROMPT = """\
You maintain the long-term memory of a personal AI assistant. Decide what to do
with a new candidate fact relative to the existing memories it most resembles.

Today's date: {today}

## Candidate fact
[{category}] {subject}: {content}

## Existing related memories
{existing}

Choose exactly ONE operation:
- ADD — the candidate is genuinely new information not already covered above.
- UPDATE — the candidate refines, corrects, or re-words ONE existing memory.
  Give its id and the final merged content to keep (prefer the newer fact on
  conflict; keep it short and dense).
- DELETE — the candidate states that an existing memory is no longer true, and
  there is nothing worth keeping in its place. Give the id to remove.
- NOOP — the candidate duplicates an existing memory, or is not worth keeping.

Keep long-term memories short and dense: strip dates, times, and situational
framing. Use lowercase for subject.

Respond with ONLY a JSON object, no other text. One of:
  {{"operation": "ADD"}}
  {{"operation": "UPDATE", "id": <id>, "category": "<cat>", \
"subject": "<subj>", "content": "<merged fact>"}}
  {{"operation": "DELETE", "id": <id>}}
  {{"operation": "NOOP"}}"""

_HYGIENE_PROMPT = """\
You are tidying a cluster of near-duplicate or possibly conflicting long-term
memories for a personal AI assistant.

Today's date: {today}

## Memories in this cluster
{cluster}

Resolve the cluster into the minimal set of correct, non-redundant memories:
- Merge duplicates and overlapping facts into one, keeping the clearest wording.
- On contradictions, keep the most recent fact and drop the stale one.
- Keep each memory short and dense (strip dates, times, situational framing).

Return ONLY a JSON object describing the changes to apply:
  {{"updates": [{{"id": <id>, "category": "<cat>", "subject": "<subj>", \
"content": "<merged fact>"}}],
   "deletes": [<id>, <id>]}}

- Put the surviving memory in "updates" (reuse one of the cluster ids), with the
  final merged content.
- Put every other id in the cluster that should be removed in "deletes".
- If the cluster is already clean, return {{"updates": [], "deletes": []}}.

Respond with ONLY the JSON object, no other text."""

_EXTRACTION_PROMPT = """\
Given this conversation exchange, identify any facts worth remembering.

User: {user_msg}
Assistant: {agent_msg}
{recent_turns_block}
{existing_memories_block}\
For each fact, classify it into ONE of these tiers:

LONG_TERM — durable facts that remain true indefinitely:
  - Personal preferences ("prefers oat milk", "favourite editor is Neovim")
  - Relationships ("Marco is Matteo's colleague", "Simge is Matteo's partner")
  - Biographical facts ("lives in Zurich", "works as a software engineer")
  - Routines ("goes to the gym on Mondays and Thursdays")
  DO NOT use LONG_TERM for:
  - Plans, tasks, or events (even recurring ones that haven't been confirmed as routines)
  - Anything time-bound ("working on project X", "has a deadline Friday")
  - Opinions about transient topics ("thinks the new API is buggy")

SHORT_TERM — situational context that expires:
  - Current activities ("working from home today", "debugging the auth flow")
  - Near-term plans ("dinner with Marco tomorrow", "flight to Rome on Friday")
  - Temporary states ("feeling tired", "waiting for a code review")
  - Active projects or tasks ("refactoring the memory system this week")

When in doubt between LONG_TERM and SHORT_TERM, choose SHORT_TERM. Only use
LONG_TERM for facts you are highly confident will still be true months from now.
{scope_block}
Return a JSON array (max 3 items). Each element must be one of:
  {{"tier": "LONG_TERM", "category": "<category>", \
"subject": "<who/what>", "content": "<the fact>"}}
  {{"tier": "SHORT_TERM", "content": "<the fact>", \
"context": "<why stored>", "ttl_hours": <int>}}

Categories: preference, relationship, fact, routine, work, health, travel

Rules:
- Extract only genuinely useful, non-obvious facts. Skip greetings, filler,
  acknowledgements, and anything already obvious from the conversation.
- Most conversation turns contain NOTHING worth remembering. Return [] liberally.
- Do NOT extract facts that duplicate or overlap with the existing memories
  listed above.
- Use lowercase for subject (e.g. "matteo", "simge").
- For LONG_TERM, always set category and subject.
- For SHORT_TERM, you MUST set ttl_hours:
  - 2-4h: trivial, task-at-hand context ("looking at flights now")
  - 8-12h: day-scoped situations ("working from home today")
  - 24-48h: near-term plans ("dinner with Marco tomorrow")
  - 72-168h: week-scoped context ("Simge visiting parents this week")
- If nothing is worth remembering, return an empty array: []

Respond with ONLY the JSON array, no other text."""

# Regex to match a fenced code block: ```json ... ``` (or just ``` ... ```)
_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def _extract_json_array(raw: str) -> list | None:
    """Best-effort extraction of a JSON array from an LLM response.

    Handles common quirks:
    - Markdown code fences (```json ... ```)
    - Preamble / trailing prose around the JSON
    - Leading/trailing whitespace
    - Empty responses

    Returns the parsed list on success, or ``None`` if no valid JSON
    array could be extracted.
    """
    raw = raw.strip()
    if not raw:
        return None

    # 1. Try parsing the raw response directly (happy path).
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # 2. Try extracting from markdown code fences.
    fence_match = _FENCE_RE.search(raw)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1).strip())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # 3. Find the outermost [ ... ] bracket pair in the response.
    start = raw.find("[")
    if start != -1:
        # Walk forward to find the matching closing bracket.
        depth = 0
        in_string = False
        escape = False
        end = -1
        for i in range(start, len(raw)):
            ch = raw[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            try:
                result = json.loads(raw[start : end + 1])
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

    return None


def _extract_json_object(raw: str) -> dict | None:
    """Best-effort extraction of a single JSON object from an LLM response.

    Mirrors :func:`_extract_json_array` but for ``{ ... }`` payloads. Returns
    the parsed dict on success, or ``None`` if none could be extracted.
    """
    raw = raw.strip()
    if not raw:
        return None

    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    fence_match = _FENCE_RE.search(raw)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1).strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    start = raw.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        end = -1
        for i in range(start, len(raw)):
            ch = raw[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            try:
                result = json.loads(raw[start : end + 1])
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

    return None


# Tokeniser for cheap lexical similarity (no embeddings, no new deps).
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "with",
        "his",
        "her",
        "their",
        "has",
        "have",
        "had",
        "uses",
        "use",
        "that",
        "this",
        "it",
        "as",
        "by",
    }
)


def _normalize_subject(subject: str) -> str:
    """Canonicalise a memory subject (lowercase, trimmed)."""
    return (subject or "").strip().lower()


def _extraction_scope_block(persona_scope: str) -> str:
    """Scope instruction injected into the extraction prompt (#42).

    Empty when no persona is active (everything is shared). When a persona is
    active, lets the model mark domain-specific facts private to it; the default
    stays shared so owner-level facts reach every persona.
    """
    if not persona_scope:
        return ""
    return (
        f'\nYou are extracting for the "{persona_scope}" persona. Add '
        f'`"scope": "private"` to a fact ONLY if it is specific to this persona\'s '
        f"domain and should NOT be shared with the owner's other assistants. "
        f"General facts about the owner (preferences, relationships, biography) are "
        f'shared — omit scope or use `"scope": "shared"`. When unsure, leave it shared.\n'
    )


def _resolve_extracted_scope(mem: dict, persona_scope: str) -> str:
    """Map an extracted item's scope hint to a stored scope key (#42).

    Private only when a persona is active AND the model tagged it private;
    everything else is shared (``''``).
    """
    if persona_scope and str(mem.get("scope", "")).strip().lower() == "private":
        return persona_scope
    return ""


def _scope_filter(scope: str | None) -> tuple[str, tuple]:
    """Build a SQL ``AND`` fragment + params restricting rows by scope (#42).

    - ``None`` → no filter (every scope; for admin listing / owner-level jobs).
    - ``""``   → shared only (``scope = ''``); the default identity's view.
    - ``"x"``  → shared + persona ``x`` (``scope IN ('', 'x')``); never another
      persona's private memory.
    """
    if scope is None:
        return "", ()
    if scope == "":
        return " AND scope = ''", ()
    return " AND scope IN ('', ?)", (scope,)


def _tokens(text: str) -> set[str]:
    """Lowercase content words, dropping stopwords and single characters."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in _STOPWORDS}


def _similarity(a: set[str], b: set[str]) -> float:
    """Jaccard overlap between two token sets (0.0 when either is empty)."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


# Half-life (in days) for the recency component of the retrieval score.
_RECENCY_HALF_LIFE_DAYS = 30.0


def _parse_sqlite_ts(ts: str | None) -> datetime | None:
    """Parse a SQLite ``datetime('now')`` string (UTC, no tz suffix)."""
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    except ValueError, TypeError:
        return None


def _recency_score(ts: str | None) -> float:
    """Exponential-decay recency in [0, 1]; newer timestamps score higher."""
    parsed = _parse_sqlite_ts(ts)
    if parsed is None:
        return 0.0
    age_days = max(0.0, (datetime.now(tz=UTC) - parsed).total_seconds() / 86400.0)
    return 0.5 ** (age_days / _RECENCY_HALF_LIFE_DAYS)


def _batch_relevance(query_vec, rows: list[dict]) -> dict[int, float]:
    """Map row index -> cosine similarity to *query_vec*, computed in one
    vectorised pass. Only rows whose stored embedding matches the query
    dimension are included; the rest are left for a lexical fallback by the
    caller. Returns an empty map when there is no query vector."""
    if query_vec is None:
        return {}
    dim = len(query_vec)
    idxs: list[int] = []
    vecs: list = []
    for i, row in enumerate(rows):
        vec = unpack_vector(row.get("embedding"))
        if vec is not None and vec.shape[0] == dim:
            idxs.append(i)
            vecs.append(vec)
    if not vecs:
        return {}
    sims = cosine_to_matrix(query_vec, vecs)
    return {idx: float(sims[k]) for k, idx in enumerate(idxs)}


def _pair_similarity(a: dict, b: dict) -> float:
    """Similarity between two long-term rows: embedding cosine when both have a
    stored vector, otherwise token overlap on subject + content."""
    va = unpack_vector(a.get("embedding"))
    vb = unpack_vector(b.get("embedding"))
    if va is not None and vb is not None and va.shape == vb.shape:
        return cosine_similarity(va, vb)
    return _similarity(
        _tokens(f"{a['subject']} {a['content']}"),
        _tokens(f"{b['subject']} {b['content']}"),
    )


class MemoryStore:
    """Two-tier memory system backed by SQLite.

    The LLM reads and writes memories via the sqlite3 CLI (taught by
    skills/memory.md).  This class handles schema initialisation,
    provides async helpers to query both tiers for injection into the
    system prompt, and runs automatic memory extraction after each
    conversation turn.
    """

    def __init__(
        self,
        db_path: str = "data/memory.db",
        long_term_limit: int = 50,
        *,
        embedder: EmbeddingClient | None = None,
        injection_top_k: int = 12,
        recall_top_k: int = 10,
        default_importance: float = 5.0,
        archive_after_days: int = 90,
        archive_max_importance: float = 4.0,
        archive_min_idle_days: int = 45,
        hygiene_enabled: bool = True,
        hygiene_similarity_threshold: float = 0.45,
    ):
        self.db_path = db_path
        self.long_term_limit = long_term_limit
        self.embedder = embedder
        self.injection_top_k = injection_top_k
        self.recall_top_k = recall_top_k
        self.default_importance = default_importance
        self.archive_after_days = archive_after_days
        self.archive_max_importance = archive_max_importance
        self.archive_min_idle_days = archive_min_idle_days
        self.hygiene_enabled = hygiene_enabled
        self.hygiene_similarity_threshold = hygiene_similarity_threshold
        self._ready = False
        self._last_extraction: float | None = None  # monotonic timestamp of last extraction
        # Turns skipped by the cooldown, replayed into the next extraction so
        # back-to-back salient turns aren't dropped (issue #7).
        self._pending_turns: list[tuple[str, str]] = []

    async def _ensure_schema(self) -> None:
        if self._ready:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        schema = _SCHEMA_FILE.read_text()
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(schema)
            await self._migrate_long_term(db)
            await self._migrate_short_term(db)
        self._ready = True

    # Columns added after the original two-tier schema shipped. Each is applied
    # via ALTER TABLE on databases created before the column existed, so an
    # existing data/memory.db upgrades in place (defaults are constant, as
    # required by SQLite's ALTER TABLE ADD COLUMN).
    _LONG_TERM_MIGRATIONS = (
        ("embedding", "embedding BLOB"),
        ("importance", "importance REAL NOT NULL DEFAULT 5.0"),
        ("last_accessed", "last_accessed DATETIME"),
        ("access_count", "access_count INTEGER NOT NULL DEFAULT 0"),
        ("archived", "archived INTEGER NOT NULL DEFAULT 0"),
        # #42: scope column — existing rows become '' (shared), the right default.
        ("scope", "scope TEXT NOT NULL DEFAULT ''"),
    )

    _SHORT_TERM_MIGRATIONS = (("scope", "scope TEXT NOT NULL DEFAULT ''"),)

    async def _migrate_long_term(self, db: aiosqlite.Connection) -> None:
        cursor = await db.execute("PRAGMA table_info(long_term)")
        existing = {row[1] for row in await cursor.fetchall()}
        for name, ddl in self._LONG_TERM_MIGRATIONS:
            if name not in existing:
                await db.execute(f"ALTER TABLE long_term ADD COLUMN {ddl}")  # noqa: S608
        # Safe to create now: the archived column is guaranteed to exist (fresh
        # DBs declare it; legacy DBs just had it added above).
        await db.execute("CREATE INDEX IF NOT EXISTS idx_lt_archived ON long_term(archived)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_lt_scope ON long_term(scope)")
        await db.commit()

    async def _migrate_short_term(self, db: aiosqlite.Connection) -> None:
        cursor = await db.execute("PRAGMA table_info(short_term)")
        existing = {row[1] for row in await cursor.fetchall()}
        for name, ddl in self._SHORT_TERM_MIGRATIONS:
            if name not in existing:
                await db.execute(f"ALTER TABLE short_term ADD COLUMN {ddl}")  # noqa: S608
        await db.execute("CREATE INDEX IF NOT EXISTS idx_st_scope ON short_term(scope)")
        await db.commit()

    async def get_long_term(self, scope: str | None = None) -> list[dict]:
        """Retrieve recent (non-archived) long-term memories for injection.

        ``scope`` filters per #42 (see :func:`_scope_filter`); ``None`` = all.
        """
        await self._ensure_schema()
        clause, params = _scope_filter(scope)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT category, subject, content FROM long_term "
                f"WHERE archived = 0{clause} ORDER BY updated_at DESC LIMIT ?",  # noqa: S608
                (*params, self.long_term_limit),
            )
            return [dict(row) for row in await cursor.fetchall()]

    async def get_relevant_long_term(self, query: str, scope: str | None = None) -> list[dict]:
        """Return long-term memories most relevant to *query*, relevance-ranked.

        Uses a Generative-Agents-style score (recency + importance + relevance)
        over embedding cosine similarity, and reinforces the chosen memories
        (bumps ``access_count`` / ``last_accessed``). Falls back to recency
        order when embeddings are unavailable or the query can't be embedded.

        ``scope`` filters per #42 (see :func:`_scope_filter`); ``None`` = all.
        """
        if not self.embedder or not query.strip():
            return await self.get_long_term(scope)

        try:
            query_vec = await self.embedder.embed_one(query)
        except Exception:
            log.exception("Query embedding failed; falling back to recency order")
            return await self.get_long_term(scope)
        if not query_vec:
            return await self.get_long_term(scope)

        await self._ensure_schema()
        clause, params = _scope_filter(scope)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id, category, subject, content, importance, embedding, "
                f"updated_at, last_accessed FROM long_term WHERE archived = 0{clause}",  # noqa: S608
                params,
            )
            rows = [dict(r) for r in await cursor.fetchall()]

        rel_map = _batch_relevance(query_vec, rows)
        scored: list[tuple[float, dict]] = []
        for i, row in enumerate(rows):
            relevance = rel_map.get(i, 0.0)
            importance = (row.get("importance") or self.default_importance) / 10.0
            recency = _recency_score(row.get("last_accessed") or row.get("updated_at"))
            score = relevance + 0.5 * importance + 0.3 * recency
            scored.append((score, row))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = [row for _, row in scored[: self.injection_top_k]]
        await self._reinforce([row["id"] for row in top])
        return [
            {"category": r["category"], "subject": r["subject"], "content": r["content"]}
            for r in top
        ]

    # Upper bound on how many memories one recall_memory call may return, and the
    # minimum relevance a row needs to be worth returning at all.
    _RECALL_MAX_LIMIT = 25
    # ponytail: relevance floor — raise to cut noise, lower to surface more long tail.
    _RECALL_MIN_RELEVANCE = 0.1

    async def recall(
        self, query: str, limit: int | None = None, scope: str | None = None
    ) -> list[dict]:
        """Deliberate semantic search over the FULL long-term store (issue #47).

        This is the agent's on-demand recall tool — the complement to the
        always-injected top-k (:meth:`get_relevant_long_term`). Where injection
        ranks only *non-archived* rows by a recency+importance+relevance blend
        and is capped to a small per-turn budget, recall searches *every*
        long-term memory (archived included), ranks purely by semantic relevance
        to *query*, and returns the best matches above a small relevance floor.

        Recalling reinforces the returned rows and un-archives any that had been
        archived (the agent looked them up and they matched, so they are warm
        again). Falls back to lexical token overlap when embeddings are
        unavailable or the query can't be embedded, so recall always works.

        ``scope`` filters per #42 (see :func:`_scope_filter`) exactly like the
        injection readers, so a persona only recalls shared + its own private
        memories, never another persona's; ``None`` = every scope.
        """
        query = (query or "").strip()
        if not query:
            return []
        limit = self.recall_top_k if not limit or limit < 1 else limit
        limit = min(limit, self._RECALL_MAX_LIMIT)

        await self._ensure_schema()
        clause, params = _scope_filter(scope)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id, category, subject, content, embedding, archived "  # noqa: S608
                f"FROM long_term WHERE 1=1{clause}",
                params,
            )
            rows = [dict(r) for r in await cursor.fetchall()]
        if not rows:
            return []

        # Embedding cosine for rows with a matching-dim vector; lexical for the rest.
        query_vec = await self._safe_embed(query)
        rel_map = _batch_relevance(query_vec, rows)
        query_tokens = _tokens(query)

        scored: list[tuple[float, dict]] = []
        for i, row in enumerate(rows):
            if i in rel_map:
                relevance = rel_map[i]
            else:
                relevance = _similarity(query_tokens, _tokens(f"{row['subject']} {row['content']}"))
            if relevance >= self._RECALL_MIN_RELEVANCE:
                scored.append((relevance, row))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = [row for _, row in scored[:limit]]
        # Recall revives matches: reinforce + un-archive. The returned rows are
        # therefore all non-archived now, so no archived flag is surfaced.
        await self._reinforce([row["id"] for row in top], unarchive=True)
        return [
            {"category": r["category"], "subject": r["subject"], "content": r["content"]}
            for r in top
        ]

    async def _reinforce(self, ids: list[int], *, unarchive: bool = False) -> None:
        """Strengthen recalled memories: bump access_count and last_accessed.

        With *unarchive*, also clear the archived flag — a memory the agent
        deliberately recalled and used is demonstrably warm again (issue #47).
        """
        if not ids:
            return
        await self._ensure_schema()
        archived_clause = ", archived = 0" if unarchive else ""
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(
                "UPDATE long_term SET access_count = access_count + 1, "  # noqa: S608
                f"last_accessed = datetime('now'){archived_clause} WHERE id = ?",
                [(i,) for i in ids],
            )
            await db.commit()

    async def get_short_term(self, scope: str | None = None) -> list[dict]:
        """Retrieve active (non-expired) short-term memories.

        ``scope`` filters per #42 (see :func:`_scope_filter`); ``None`` = all.
        """
        await self._ensure_schema()
        clause, params = _scope_filter(scope)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT content, context FROM short_term "
                f"WHERE expires_at > datetime('now'){clause} "  # noqa: S608
                "ORDER BY created_at DESC",
                params,
            )
            return [dict(row) for row in await cursor.fetchall()]

    async def format_for_prompt(self, query: str | None = None, scope: str | None = None) -> str:
        """Format both tiers into a block for the system prompt.

        When *query* is given and embeddings are enabled, only the long-term
        memories most relevant to the query are injected (relevance-ranked),
        instead of dumping the most recent ``long_term_limit`` rows (issue #5).

        ``scope`` restricts to the active persona's view per #42: ``""`` =
        shared only (default identity), ``"<persona>"`` = shared + that
        persona's private memory, ``None`` = every scope.
        """
        sections: list[str] = []

        if query:
            long_term = await self.get_relevant_long_term(query, scope)
        else:
            long_term = await self.get_long_term(scope)
        if long_term:
            lines = [f"- [{m['category']}] {m['subject']}: {m['content']}" for m in long_term]
            sections.append("## Long-term memories\n" + "\n".join(lines))

        short_term = await self.get_short_term(scope)
        if short_term:
            lines = []
            for m in short_term:
                entry = f"- {m['content']}"
                if m.get("context"):
                    entry += f" ({m['context']})"
                lines.append(entry)
            sections.append("## Current context (short-term)\n" + "\n".join(lines))

        return "\n\n".join(sections) if sections else ""

    # -- Automatic memory extraction --

    # Maximum number of memories to store per extraction call.
    _MAX_PER_TURN = 3

    # Maximum number of cooldown-skipped turns to buffer for the next extraction.
    _MAX_PENDING_TURNS = 6

    # Number of similar long-term memories retrieved as ADD/UPDATE/DELETE candidates.
    _UPDATE_TOP_K = 8

    def _format_pending_turns(self) -> str:
        """Render buffered cooldown turns as a prompt section (empty if none)."""
        if not self._pending_turns:
            return ""
        lines = ["", "Earlier turns since the last review (also consider these):"]
        for user_msg, agent_msg in self._pending_turns:
            lines.append(f"User: {user_msg}")
            lines.append(f"Assistant: {agent_msg}")
        return "\n".join(lines) + "\n"

    async def extract_memories(
        self,
        llm: LLMClient,
        model: str,
        user_msg: str,
        agent_msg: str,
        cooldown_seconds: int = 120,
        persona_scope: str = "",
    ) -> int:
        """Extract facts from a conversation turn and store them.

        Makes a secondary LLM call (cheap/fast model) to identify facts
        worth remembering, then writes them to the appropriate tier.

        If fewer than *cooldown_seconds* have elapsed since the last
        extraction call, the call is skipped entirely (returns 0).

        ``persona_scope`` is the active persona's key (#42). When set, the
        extraction LLM may tag a fact ``"scope": "private"`` to keep it inside
        that persona; everything else is stored shared (``''``). With no active
        persona it is ``""`` and every fact is shared.

        Returns the number of memories stored.
        """
        now = time.monotonic()
        if (
            cooldown_seconds > 0
            and self._last_extraction is not None
            and now - self._last_extraction < cooldown_seconds
        ):
            # Buffer the skipped turn instead of dropping it; it is replayed
            # into the next extraction once the cooldown elapses (issue #7).
            self._pending_turns.append((user_msg, agent_msg))
            del self._pending_turns[: -self._MAX_PENDING_TURNS]
            log.debug(
                "Buffering memory extraction (cooldown: %.0fs remaining, %d pending)",
                cooldown_seconds - (now - self._last_extraction),
                len(self._pending_turns),
            )
            return 0
        self._last_extraction = now

        # Replay any turns buffered during the cooldown, then clear the buffer.
        recent_turns_block = self._format_pending_turns()
        self._pending_turns = []

        # Build existing-memories block so the LLM can avoid duplicates — scoped
        # to what this persona may see (shared + its own private).
        existing_block = await self._existing_memories_block(persona_scope)

        prompt = _EXTRACTION_PROMPT.format(
            user_msg=user_msg,
            agent_msg=agent_msg,
            recent_turns_block=recent_turns_block,
            existing_memories_block=existing_block,
            scope_block=_extraction_scope_block(persona_scope),
        )

        try:
            raw = await llm.generate_text(model=model, prompt=prompt, max_tokens=4096)
        except Exception:
            log.exception("Memory extraction LLM call failed")
            return 0

        memories = _extract_json_array(raw)
        if memories is None:
            log.warning("Memory extraction returned non-JSON: %s", raw[:200])
            return 0

        stored = 0
        for mem in memories[: self._MAX_PER_TURN]:
            try:
                scope = _resolve_extracted_scope(mem, persona_scope)
                tier = mem.get("tier", "").upper()
                if tier == "LONG_TERM":
                    op = await self.update_memory(llm, model, mem, scope=scope)
                    if op in ("ADD", "UPDATE"):
                        stored += 1
                elif tier == "SHORT_TERM":
                    stored += await self._store_short_term(mem, scope=scope)
                else:
                    log.warning("Unknown memory tier: %s", tier)
            except Exception:
                log.exception("Failed to store extracted memory: %s", mem)

        if stored == 0:
            log.info("Memory extraction stored 0 items (model=%s)", model)

        if stored:
            log.info("Extracted and stored %d memories", stored)
        return stored

    async def _existing_memories_block(self, scope: str | None = None) -> str:
        """Build a summary of existing memories for the extraction prompt."""
        long_term = await self.get_long_term(scope)
        short_term = await self.get_short_term(scope)

        if not long_term and not short_term:
            return ""

        parts = ["## Existing memories (do NOT extract duplicates)\n"]
        if long_term:
            for m in long_term:
                parts.append(f"- [LT] {m['subject']}: {m['content']}")
        if short_term:
            for m in short_term:
                parts.append(f"- [ST] {m['content']}")
        parts.append("")  # trailing newline
        return "\n".join(parts) + "\n"

    async def update_memory(
        self, llm: LLMClient, model: str, candidate: dict, scope: str = ""
    ) -> str:
        """Apply a candidate fact to long-term memory via a unified pipeline.

        Retrieves the most lexically similar existing long-term memories, then
        a single LLM call decides ADD / UPDATE / DELETE / NOOP — handling
        semantic duplicates, refinements, and contradictions (issues #1–#4, #8).
        When nothing similar exists the candidate is added directly without an
        LLM call. Malformed model output is a safe no-op.

        ``scope`` (#42) tags an ADD and bounds the dedup/UPDATE/DELETE candidate
        set to shared + that scope, so a private fact never merges into or
        deletes another persona's private memory.

        Returns the operation applied: ``"ADD"``, ``"UPDATE"``, ``"DELETE"``,
        or ``"NOOP"``.
        """
        category = candidate.get("category") or "fact"
        subject = _normalize_subject(candidate.get("subject", ""))
        content = (candidate.get("content") or "").strip()
        if not content:
            return "NOOP"

        # Candidates restricted to shared + own scope: a private fact must not
        # see — and therefore cannot UPDATE/DELETE — another persona's memory.
        similar = await self._retrieve_similar_long_term(subject, content, scope or "")
        if not similar:
            await self._insert_long_term(category, subject, content, scope=scope)
            log.debug("ADD long-term (no similar): [%s] %s: %s", category, subject, content[:80])
            return "ADD"

        existing_lines = []
        for row in similar:
            existing_lines.append(
                f"- id={row['id']} [{row['category']}] {row['subject']}: {row['content']} "
                f"(created {row['created_at']}, updated {row['updated_at']})"
            )
        prompt = _UPDATE_PROMPT.format(
            today=datetime.now(tz=UTC).date().isoformat(),
            category=category,
            subject=subject or "(unknown)",
            content=content,
            existing="\n".join(existing_lines),
        )

        try:
            raw = await llm.generate_text(model=model, prompt=prompt, max_tokens=1024)
        except Exception:
            log.exception("update_memory LLM call failed; skipping candidate")
            return "NOOP"

        decision = _extract_json_object(raw)
        if not isinstance(decision, dict):
            log.warning("update_memory returned non-JSON: %s", raw[:200])
            return "NOOP"

        operation = str(decision.get("operation", "")).upper()
        valid_ids = {row["id"] for row in similar}

        if operation == "ADD":
            await self._insert_long_term(category, subject, content, scope=scope)
            log.debug("ADD long-term: [%s] %s: %s", category, subject, content[:80])
            return "ADD"

        if operation == "UPDATE":
            target_id = decision.get("id")
            if target_id not in valid_ids:
                log.warning("update_memory UPDATE with invalid id %r; no-op", target_id)
                return "NOOP"
            new_content = (decision.get("content") or content).strip()
            new_category = decision.get("category") or category
            new_subject = _normalize_subject(decision.get("subject") or subject)
            blob = await self._embed_blob(f"{new_subject}: {new_content}")
            await self._ensure_schema()
            async with aiosqlite.connect(self.db_path) as db:
                # Re-mentioning a fact reinforces it: bump importance (capped).
                if blob is not None:
                    await db.execute(
                        "UPDATE long_term SET category = ?, subject = ?, content = ?, "
                        "embedding = ?, importance = MIN(10.0, importance + 1.0), "
                        "updated_at = datetime('now') WHERE id = ?",
                        (new_category, new_subject, new_content, blob, target_id),
                    )
                else:
                    await db.execute(
                        "UPDATE long_term SET category = ?, subject = ?, content = ?, "
                        "importance = MIN(10.0, importance + 1.0), "
                        "updated_at = datetime('now') WHERE id = ?",
                        (new_category, new_subject, new_content, target_id),
                    )
                await db.commit()
            log.debug("UPDATE long-term %s: %s", target_id, new_content[:80])
            return "UPDATE"

        if operation == "DELETE":
            target_id = decision.get("id")
            if target_id not in valid_ids:
                log.warning("update_memory DELETE with invalid id %r; no-op", target_id)
                return "NOOP"
            await self._ensure_schema()
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("DELETE FROM long_term WHERE id = ?", (target_id,))
                await db.commit()
            log.debug("DELETE long-term %s (contradicted)", target_id)
            return "DELETE"

        return "NOOP"

    async def _retrieve_similar_long_term(
        self, subject: str, content: str, scope: str | None = None
    ) -> list[dict]:
        """Return the top-k existing (non-archived) long-term memories similar to
        a candidate (subject + content).

        Uses embedding cosine similarity when an embedder is configured (with a
        per-row lexical fallback for memories that have no stored vector yet),
        otherwise pure token overlap. A matching subject adds a fixed boost.
        Cheap and dependency-free at <1k rows.

        ``scope`` (#42) bounds the candidate set: ``""`` = shared only, a persona
        key = shared + that persona, ``None`` = every scope."""
        await self._ensure_schema()
        clause, params = _scope_filter(scope)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id, category, subject, content, created_at, updated_at, embedding "
                f"FROM long_term WHERE archived = 0{clause}",  # noqa: S608
                params,
            )
            rows = [dict(r) for r in await cursor.fetchall()]

        subject_norm = _normalize_subject(subject)
        cand_tokens = _tokens(f"{subject} {content}")
        cand_vec = await self._safe_embed(f"{subject}: {content}")
        # Embedding cosine for rows with a matching-dim vector; lexical for the rest.
        rel_map = _batch_relevance(cand_vec, rows)

        scored: list[tuple[float, dict]] = []
        for i, row in enumerate(rows):
            if i in rel_map:
                base = rel_map[i]
            else:
                base = _similarity(cand_tokens, _tokens(f"{row['subject']} {row['content']}"))
            score = base
            if subject_norm and _normalize_subject(row["subject"]) == subject_norm:
                score += 0.5
            if score > 0:
                row.pop("embedding", None)
                scored.append((score, row))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [row for _, row in scored[: self._UPDATE_TOP_K]]

    async def _safe_embed(self, text: str) -> list[float] | None:
        """Best-effort embedding; returns None if disabled or on failure."""
        if not self.embedder:
            return None
        try:
            vec = await self.embedder.embed_one(text)
        except Exception:
            log.exception("Embedding call failed; proceeding without a vector")
            return None
        return vec or None

    async def _embed_blob(self, text: str) -> bytes | None:
        """Best-effort packed embedding blob (None if disabled or on failure)."""
        vec = await self._safe_embed(text)
        return pack_vector(vec) if vec else None

    async def _insert_long_term(
        self,
        category: str,
        subject: str,
        content: str,
        importance: float | None = None,
        scope: str = "",
    ) -> None:
        """Insert a new long-term memory row (with embedding + importance)."""
        await self._ensure_schema()
        blob = await self._embed_blob(f"{subject}: {content}")
        imp = self.default_importance if importance is None else importance
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO long_term "
                "(category, subject, content, source, confidence, embedding, importance, scope) "
                "VALUES (?, ?, ?, 'conversation', 'stated', ?, ?, ?)",
                (category, subject, content, blob, imp, scope),
            )
            await db.commit()

    async def _store_short_term(self, mem: dict, scope: str = "") -> int:
        """Store a short-term memory with a LLM-determined TTL.

        Skips insertion if an active (non-expired) short-term memory
        already exists with overlapping content. ``scope`` (#42) tags the row
        and bounds the duplicate check to shared + that scope.
        """
        content = mem.get("content", "")
        context = mem.get("context", "")
        ttl_hours = mem.get("ttl_hours")
        if not content:
            return 0
        if not ttl_hours or not isinstance(ttl_hours, int | float):
            log.warning("Short-term memory missing ttl_hours, skipping: %s", content[:80])
            return 0

        expires_at = datetime.now(tz=UTC) + timedelta(hours=ttl_hours)
        # Store in SQLite-compatible format (no timezone suffix, always UTC)
        expires_str = expires_at.strftime("%Y-%m-%d %H:%M:%S")

        await self._ensure_schema()
        clause, params = _scope_filter(scope or "")
        async with aiosqlite.connect(self.db_path) as db:
            # Check for duplicate active short-term memories within this scope.
            cursor = await db.execute(
                f"SELECT id, content FROM short_term WHERE expires_at > datetime('now'){clause}",  # noqa: S608
                params,
            )
            existing = await cursor.fetchall()
            content_lower = content.lower()
            for row in existing:
                if content_lower in row[1].lower() or row[1].lower() in content_lower:
                    log.debug("Skipping duplicate short-term memory: %s", content[:80])
                    return 0

            await db.execute(
                "INSERT INTO short_term (content, context, expires_at, scope) VALUES (?, ?, ?, ?)",
                (content, context, expires_str, scope),
            )
            await db.commit()
            log.debug("Stored short-term memory (TTL %dh): %s", ttl_hours, content[:80])
            return 1

    # -- Consolidation & cleanup --

    async def consolidate_and_cleanup(self, llm: LLMClient, model: str) -> dict:
        """Consolidate short-term memories and clean up expired ones.

        1. Fetch all non-expired short-term memories.
        2. Ask the LLM which ones should be promoted to long-term (compacted).
        3. Store the promotions.
        4. Delete all expired short-term rows.

        Returns a summary dict with counts.
        """
        await self._ensure_schema()

        # Fetch non-expired short-term memories (with IDs for logging)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id, content, context, created_at, expires_at, scope FROM short_term "
                "WHERE expires_at > datetime('now') "
                "ORDER BY created_at ASC",
            )
            active_short_term = [dict(row) for row in await cursor.fetchall()]

        # Promote per scope (#42): each scope's short-term is reviewed and
        # promoted into long-term of the same scope, never mixing two personas'
        # private memory in one consolidation call.
        promoted = 0
        if active_short_term:
            by_scope: dict[str, list[dict]] = {}
            for row in active_short_term:
                by_scope.setdefault(row.get("scope") or "", []).append(row)
            for scope, rows in by_scope.items():
                promoted += await self._run_consolidation_llm(llm, model, rows, scope=scope)

        # Delete all expired short-term memories
        expired_count = await self._delete_expired_short_term()

        # Tier 4: merge near-duplicate / contradictory long-term rows.
        merged = 0
        if self.hygiene_enabled:
            merged = await self._hygiene_pass(llm, model)

        # Tier 3: archive cold, low-importance long-term memories.
        archived = await self._archive_cold_memories()

        summary = {
            "active_reviewed": len(active_short_term),
            "promoted_to_long_term": promoted,
            "expired_deleted": expired_count,
            "hygiene_merged": merged,
            "archived": archived,
        }
        log.info(
            "Memory consolidation complete: %d reviewed, %d promoted, %d expired deleted, "
            "%d merged, %d archived",
            summary["active_reviewed"],
            summary["promoted_to_long_term"],
            summary["expired_deleted"],
            summary["hygiene_merged"],
            summary["archived"],
        )
        return summary

    async def _archive_cold_memories(self) -> int:
        """Archive cold, low-importance long-term memories (Tier 3, issue #9).

        A memory is archived (soft-deleted via the ``archived`` flag, not hard
        deleted) when it is old enough, has low importance, and has not been
        accessed recently. Returns the number archived.
        """
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "UPDATE long_term SET archived = 1 WHERE archived = 0 "
                "AND importance <= ? "
                "AND created_at < datetime('now', ?) "
                "AND COALESCE(last_accessed, created_at) < datetime('now', ?)",
                (
                    self.archive_max_importance,
                    f"-{self.archive_after_days} days",
                    f"-{self.archive_min_idle_days} days",
                ),
            )
            count = cursor.rowcount
            await db.commit()
        if count:
            log.info("Archived %d cold long-term memories", count)
        return count

    # Cap how many clusters one hygiene pass resolves, to bound LLM cost.
    _HYGIENE_MAX_CLUSTERS = 10

    async def _hygiene_pass(self, llm: LLMClient, model: str) -> int:
        """Cluster near-duplicate long-term memories and merge each cluster via
        one LLM call (Tier 4, issue #6). Returns the number of rows removed."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id, category, subject, content, created_at, updated_at, embedding, scope "
                "FROM long_term WHERE archived = 0"
            )
            rows = [dict(r) for r in await cursor.fetchall()]

        if len(rows) < 2:
            return 0

        # Cluster within a scope only (#42): merging must never collapse one
        # persona's private memory into another's (or into shared).
        by_scope: dict[str, list[dict]] = {}
        for row in rows:
            by_scope.setdefault(row.get("scope") or "", []).append(row)
        clusters: list[list[dict]] = []
        for scope_rows in by_scope.values():
            clusters.extend(self._cluster_long_term(scope_rows))
        clusters = clusters[: self._HYGIENE_MAX_CLUSTERS]
        removed = 0
        for cluster in clusters:
            try:
                removed += await self._resolve_cluster(llm, model, cluster)
            except Exception:
                log.exception("Hygiene cluster resolution failed")
        if removed:
            log.info("Hygiene pass merged away %d duplicate long-term memories", removed)
        return removed

    def _cluster_long_term(self, rows: list[dict]) -> list[list[dict]]:
        """Greedily group memories whose pairwise similarity meets the threshold.

        Returns only clusters with two or more members (singletons need no work).
        """
        threshold = self.hygiene_similarity_threshold
        unassigned = list(rows)
        clusters: list[list[dict]] = []
        while unassigned:
            seed = unassigned.pop(0)
            cluster = [seed]
            rest: list[dict] = []
            for row in unassigned:
                if _pair_similarity(seed, row) >= threshold:
                    cluster.append(row)
                else:
                    rest.append(row)
            unassigned = rest
            if len(cluster) >= 2:
                clusters.append(cluster)
        return clusters

    async def _resolve_cluster(self, llm: LLMClient, model: str, cluster: list[dict]) -> int:
        """Ask the LLM to merge one cluster; apply updates/deletes. Returns rows
        removed (deletes that actually matched a cluster member)."""
        cluster_lines = [
            f"- id={row['id']} [{row['category']}] {row['subject']}: {row['content']} "
            f"(created {row['created_at']}, updated {row['updated_at']})"
            for row in cluster
        ]
        prompt = _HYGIENE_PROMPT.format(
            today=datetime.now(tz=UTC).date().isoformat(),
            cluster="\n".join(cluster_lines),
        )
        try:
            raw = await llm.generate_text(model=model, prompt=prompt, max_tokens=1024)
        except Exception:
            log.exception("Hygiene LLM call failed")
            return 0

        plan = _extract_json_object(raw)
        if not isinstance(plan, dict):
            log.warning("Hygiene LLM returned non-JSON: %s", raw[:200])
            return 0

        valid_ids = {row["id"] for row in cluster}
        updates = plan.get("updates") or []
        deletes = plan.get("deletes") or []

        removed = 0
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            for upd in updates:
                if not isinstance(upd, dict):
                    continue
                uid = upd.get("id")
                if uid not in valid_ids:
                    continue
                content = (upd.get("content") or "").strip()
                if not content:
                    continue
                subject = _normalize_subject(upd.get("subject") or "")
                category = upd.get("category") or "fact"
                blob = await self._embed_blob(f"{subject}: {content}")
                if blob is not None:
                    await db.execute(
                        "UPDATE long_term SET category = ?, subject = ?, content = ?, "
                        "embedding = ?, updated_at = datetime('now') WHERE id = ?",
                        (category, subject, content, blob, uid),
                    )
                else:
                    await db.execute(
                        "UPDATE long_term SET category = ?, subject = ?, content = ?, "
                        "updated_at = datetime('now') WHERE id = ?",
                        (category, subject, content, uid),
                    )
            for did in deletes:
                if did in valid_ids:
                    await db.execute("DELETE FROM long_term WHERE id = ?", (did,))
                    removed += 1
            await db.commit()
        return removed

    async def _run_consolidation_llm(
        self, llm: LLMClient, model: str, short_term_rows: list[dict], scope: str = ""
    ) -> int:
        """Ask the LLM which short-term memories to promote to long-term.

        All rows are assumed to share ``scope`` (#42); promotions are stored in
        that scope and deduplicated only against shared + that scope.
        """
        # Build the short-term entries block
        st_lines = []
        for row in short_term_rows:
            entry = f"- {row['content']}"
            if row.get("context"):
                entry += f" (context: {row['context']})"
            st_lines.append(entry)
        st_block = "\n".join(st_lines)

        # Build existing long-term summary for deduplication (same scope view).
        long_term = await self.get_long_term(scope or "")
        if long_term:
            lt_lines = [f"- [{m['category']}] {m['subject']}: {m['content']}" for m in long_term]
            lt_block = "\n".join(lt_lines)
        else:
            lt_block = "(none)"

        prompt = _CONSOLIDATION_PROMPT.format(
            existing_long_term=lt_block,
            short_term_entries=st_block,
        )

        try:
            raw = await llm.generate_text(model=model, prompt=prompt, max_tokens=4096)
        except Exception:
            log.exception("Consolidation LLM call failed")
            return 0

        promotions = _extract_json_array(raw)
        if promotions is None:
            log.warning("Consolidation LLM returned non-JSON: %s", raw[:200])
            return 0

        stored = 0
        for mem in promotions:
            try:
                op = await self.update_memory(
                    llm,
                    model,
                    {
                        "category": mem.get("category", "fact"),
                        "subject": mem.get("subject", ""),
                        "content": mem.get("content", ""),
                    },
                    scope=scope,
                )
                if op in ("ADD", "UPDATE"):
                    stored += 1
            except Exception:
                log.exception("Failed to store promoted memory: %s", mem)

        if stored:
            log.info("Consolidation promoted %d short-term memories to long-term", stored)
        return stored

    async def _delete_expired_short_term(self) -> int:
        """Delete all expired short-term memories. Returns the count deleted."""
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("DELETE FROM short_term WHERE expires_at < datetime('now')")
            count = cursor.rowcount
            await db.commit()
            if count:
                log.info("Deleted %d expired short-term memories", count)
            return count
