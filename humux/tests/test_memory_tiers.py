"""Tests for Tier 2 (embeddings + relevance injection), Tier 3 (forgetting /
importance / reinforcement), and Tier 4 (long-term hygiene) of the memory
system, plus the in-place schema migration (issue #5)."""

from __future__ import annotations

import json
import re
import zlib

import aiosqlite
import numpy as np
import pytest

from core.embeddings import cosine_similarity, cosine_to_matrix, pack_vector, unpack_vector
from core.memory import MemoryStore


class _HashEmbedder:
    """Deterministic bag-of-words embedder for tests (no network).

    Strings that share tokens get non-zero cosine similarity, so the embedding
    retrieval/ranking paths are genuinely exercised.
    """

    DIM = 64

    async def embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            # crc32, not the builtin hash(): str hashing is salted per process
            # (PYTHONHASHSEED), which made the bucketing — and the ranking tests
            # below — flaky across runs. crc32 is stable, keeping this honestly
            # deterministic.
            vec[zlib.crc32(tok.encode()) % self.DIM] += 1.0
        return vec

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_one(t) for t in texts]


@pytest.fixture
async def store(tmp_path):
    s = MemoryStore(db_path=str(tmp_path / "memory.db"), long_term_limit=50)
    await s._ensure_schema()
    return s


@pytest.fixture
async def embed_store(tmp_path):
    s = MemoryStore(
        db_path=str(tmp_path / "memory.db"),
        long_term_limit=50,
        embedder=_HashEmbedder(),
        injection_top_k=2,
    )
    await s._ensure_schema()
    return s


class _StubLLM:
    def __init__(self, response: str):
        self._response = response
        self.calls = 0

    async def generate_text(self, *, model, prompt, max_tokens=1024) -> str:
        self.calls += 1
        return self._response


async def _insert(
    store: MemoryStore,
    subject: str,
    content: str,
    *,
    category: str = "fact",
    importance: float = 5.0,
    created_offset_days: int = 0,
    idle_days: int | None = None,
    embedding: list[float] | None = None,
) -> int:
    blob = pack_vector(embedding) if embedding is not None else None
    last_accessed = None if idle_days is None else f"datetime('now', '-{idle_days} days')"
    async with aiosqlite.connect(store.db_path) as db:
        cur = await db.execute(
            f"INSERT INTO long_term "  # noqa: S608
            "(category, subject, content, importance, embedding, created_at, last_accessed) "
            f"VALUES (?, ?, ?, ?, ?, datetime('now', '-{created_offset_days} days'), "
            f"{last_accessed if last_accessed else 'NULL'})",
            (category, subject, content, importance, blob),
        )
        await db.commit()
        return cur.lastrowid


async def _row(store: MemoryStore, rid: int) -> dict:
    async with aiosqlite.connect(store.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM long_term WHERE id = ?", (rid,))
        row = await cur.fetchone()
        return dict(row) if row else {}


# -- vector helpers --


class TestVectorHelpers:
    def test_pack_unpack_roundtrip(self):
        vec = [0.1, -2.0, 3.5, 0.0]
        out = unpack_vector(pack_vector(vec))
        assert out.tolist() == pytest.approx(vec, abs=1e-6)

    def test_unpack_none(self):
        assert unpack_vector(None) is None
        assert unpack_vector(b"") is None

    def test_cosine(self):
        assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
        assert cosine_similarity([], [1]) == 0.0
        assert cosine_similarity([0, 0], [1, 1]) == 0.0
        assert cosine_similarity([1, 2, 3], [1, 2]) == 0.0  # shape mismatch

    def test_cosine_to_matrix(self):
        q = [1.0, 0.0]
        rows = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
        out = cosine_to_matrix(q, rows)
        assert out.shape == (3,)
        assert out[0] == pytest.approx(1.0)
        assert out[1] == pytest.approx(0.0)
        assert out[2] == pytest.approx(0.7071, abs=1e-3)
        # Matches the scalar implementation row-by-row.
        for i, r in enumerate(rows):
            assert out[i] == pytest.approx(cosine_similarity(q, r), abs=1e-5)

    def test_cosine_to_matrix_empty(self):
        assert cosine_to_matrix([1.0, 0.0], []).shape == (0,)


# -- Tier 2: embeddings --


class TestEmbeddingWritePath:
    async def test_insert_stores_embedding_blob(self, embed_store):
        await embed_store._insert_long_term("fact", "matteo", "lives in zurich")
        rows = await _row_all(embed_store)
        assert rows[0]["embedding"] is not None
        assert unpack_vector(rows[0]["embedding"]) is not None

    async def test_retrieval_uses_embeddings(self, embed_store):
        # Two stored memories; candidate shares tokens with one of them.
        await embed_store._insert_long_term("health", "matteo", "allergic to shellfish")
        await embed_store._insert_long_term("fact", "simge", "speaks turkish fluently")

        similar = await embed_store._retrieve_similar_long_term("matteo", "cannot eat shellfish")

        assert similar
        assert "shellfish" in similar[0]["content"]

    async def test_relevant_injection_ranks_and_reinforces(self, embed_store):
        rid_shell = await _insert(embed_store, "matteo", "allergic to shellfish", embedding=None)
        await _insert(embed_store, "simge", "speaks turkish fluently", embedding=None)
        # Give the shellfish row an embedding aligned with the query tokens.
        emb = _HashEmbedder()
        async with aiosqlite.connect(embed_store.db_path) as db:
            for rid, text in [
                (rid_shell, "allergic to shellfish"),
            ]:
                blob = pack_vector(await emb.embed_one(text))
                await db.execute("UPDATE long_term SET embedding = ? WHERE id = ?", (blob, rid))
            await db.commit()

        out = await embed_store.get_relevant_long_term("what foods is he allergic to shellfish")

        assert out  # relevance-ranked subset
        assert out[0]["content"] == "allergic to shellfish"
        # Reinforcement bumped access_count on the recalled row.
        assert (await _row(embed_store, rid_shell))["access_count"] >= 1

    async def test_relevant_injection_respects_top_k(self, embed_store):
        for i in range(5):
            await embed_store._insert_long_term("fact", "matteo", f"likes hobby number {i}")
        out = await embed_store.get_relevant_long_term("matteo likes hobby")
        assert len(out) <= embed_store.injection_top_k

    async def test_format_for_prompt_without_query_uses_recency(self, embed_store):
        await embed_store._insert_long_term("fact", "matteo", "lives in zurich")
        block = await embed_store.format_for_prompt()
        assert "lives in zurich" in block


# -- recall_memory: deliberate full-store semantic lookup (issue #47) --


async def _set_archived(store: MemoryStore, rid: int) -> None:
    async with aiosqlite.connect(store.db_path) as db:
        await db.execute("UPDATE long_term SET archived = 1 WHERE id = ?", (rid,))
        await db.commit()


class TestRecall:
    async def test_empty_query_returns_nothing(self, embed_store):
        await embed_store._insert_long_term("fact", "matteo", "lives in zurich")
        assert await embed_store.recall("   ") == []

    async def test_semantic_match_ranks_first(self, embed_store):
        await embed_store._insert_long_term("health", "matteo", "allergic to shellfish")
        await embed_store._insert_long_term("fact", "simge", "speaks turkish fluently")

        out = await embed_store.recall("food allergies and shellfish")

        assert out
        assert out[0]["content"] == "allergic to shellfish"

    async def test_searches_and_unarchives_archived_rows(self, embed_store):
        # An archived memory is invisible to injection but recall must still find
        # it — and recalling it brings it back (un-archives + reinforces).
        emb = await _HashEmbedder().embed_one("allergic to shellfish")
        rid = await _insert(
            embed_store, "matteo", "allergic to shellfish", category="health", embedding=emb
        )
        await _set_archived(embed_store, rid)
        assert await embed_store.get_long_term() == []  # archived → not injected

        out = await embed_store.recall("shellfish allergy")

        assert any(m["content"] == "allergic to shellfish" for m in out)
        row = await _row(embed_store, rid)
        assert row["archived"] == 0  # un-archived on recall
        assert row["access_count"] >= 1  # reinforced

    async def test_respects_limit(self, embed_store):
        for i in range(6):
            await embed_store._insert_long_term("fact", "matteo", f"likes hobby number {i}")
        out = await embed_store.recall("matteo likes hobby", limit=3)
        # All 6 rows clear the floor, so the limit slice must cap at exactly 3.
        assert len(out) == 3

    async def test_lexical_fallback_without_embedder(self, store):
        # No embedder configured → recall falls back to token overlap, still works.
        await store._insert_long_term("health", "matteo", "allergic to shellfish")
        await store._insert_long_term("fact", "simge", "speaks turkish")
        out = await store.recall("shellfish allergy")
        assert any("shellfish" in m["content"] for m in out)
        # The relevance floor drops the zero-overlap row (deterministic on the
        # lexical path — guards _RECALL_MIN_RELEVANCE against being lowered to 0).
        assert all("turkish" not in m["content"] for m in out)

    async def test_scope_isolation(self, embed_store):
        # Recall must honour agent scope (#42): an agent sees shared + its own
        # private memories, never another agent's private rows.
        await embed_store._insert_long_term("fact", "matteo", "allergic to dust", scope="")
        await embed_store._insert_long_term(
            "health", "matteo", "allergic to shellfish", scope="coach"
        )
        await embed_store._insert_long_term(
            "health", "matteo", "allergic to peanuts", scope="finance"
        )

        coach = {
            m["content"] for m in await embed_store.recall("allergic allergies", scope="coach")
        }
        assert "allergic to shellfish" in coach  # coach's own private
        assert "allergic to dust" in coach  # shared
        assert "allergic to peanuts" not in coach  # finance's private — never crosses

        # The default identity (scope="") sees shared only.
        owner = {m["content"] for m in await embed_store.recall("allergic allergies", scope="")}
        assert owner == {"allergic to dust"}


# -- Tier 3: forgetting / importance / reinforcement --


class TestForgetting:
    async def test_get_long_term_excludes_archived(self, store):
        keep = await _insert(store, "matteo", "keeps this")
        gone = await _insert(store, "matteo", "archived one")
        async with aiosqlite.connect(store.db_path) as db:
            await db.execute("UPDATE long_term SET archived = 1 WHERE id = ?", (gone,))
            await db.commit()

        rows = await store.get_long_term()
        contents = {r["content"] for r in rows}
        assert "keeps this" in contents
        assert "archived one" not in contents
        assert keep  # silence unused

    async def test_archive_cold_low_importance(self, store):
        store.archive_after_days = 60
        store.archive_min_idle_days = 30
        store.archive_max_importance = 4.0

        cold = await _insert(
            store, "matteo", "old trivia", importance=2.0, created_offset_days=200, idle_days=200
        )
        recent = await _insert(store, "matteo", "fresh fact", importance=2.0, created_offset_days=1)
        important = await _insert(
            store, "matteo", "old but key", importance=9.0, created_offset_days=200, idle_days=200
        )

        n = await store._archive_cold_memories()

        assert n == 1
        assert (await _row(store, cold))["archived"] == 1
        assert (await _row(store, recent))["archived"] == 0
        assert (await _row(store, important))["archived"] == 0

    async def test_reinforce_bumps_counters(self, store):
        rid = await _insert(store, "matteo", "a fact")
        await store._reinforce([rid])
        row = await _row(store, rid)
        assert row["access_count"] == 1
        assert row["last_accessed"] is not None

    async def test_update_reinforces_importance(self, store):
        rid = await _insert(store, "matteo", "uses a desk", importance=5.0)
        llm = _StubLLM(
            json.dumps({"operation": "UPDATE", "id": rid, "content": "uses a standing desk"})
        )

        op = await store.update_memory(
            llm, "m", {"category": "work", "subject": "matteo", "content": "standing desk"}
        )

        assert op == "UPDATE"
        row = await _row(store, rid)
        assert row["importance"] == pytest.approx(6.0)
        assert row["content"] == "uses a standing desk"


# -- Tier 4: hygiene --


class TestHygiene:
    async def test_cluster_groups_similar(self, store):
        rows = [
            {"id": 1, "subject": "matteo", "content": "uses a standing desk", "embedding": None},
            {
                "id": 2,
                "subject": "matteo",
                "content": "has a standing desk at work",
                "embedding": None,
            },
            {"id": 3, "subject": "simge", "content": "plays the violin", "embedding": None},
        ]
        clusters = store._cluster_long_term(rows)
        # The two desk facts cluster; the violin fact is a singleton (excluded).
        assert len(clusters) == 1
        ids = {r["id"] for r in clusters[0]}
        assert ids == {1, 2}

    async def test_hygiene_pass_merges_duplicates(self, store):
        keep = await _insert(store, "matteo", "uses a standing desk")
        dup = await _insert(store, "matteo", "has a standing desk at work")
        llm = _StubLLM(
            json.dumps(
                {
                    "updates": [
                        {
                            "id": keep,
                            "category": "work",
                            "subject": "matteo",
                            "content": "uses a standing desk at work",
                        }
                    ],
                    "deletes": [dup],
                }
            )
        )

        removed = await store._hygiene_pass(llm, "m")

        assert removed == 1
        assert (await _row(store, dup)) == {}
        assert (await _row(store, keep))["content"] == "uses a standing desk at work"

    async def test_hygiene_pass_noop_when_nothing_similar(self, store):
        await _insert(store, "matteo", "lives in zurich")
        await _insert(store, "simge", "plays the violin")
        llm = _StubLLM(json.dumps({"updates": [], "deletes": []}))

        removed = await store._hygiene_pass(llm, "m")

        assert removed == 0
        assert llm.calls == 0  # no cluster formed → no LLM call

    async def test_hygiene_malformed_plan_is_safe(self, store):
        keep = await _insert(store, "matteo", "uses a standing desk")
        dup = await _insert(store, "matteo", "has a standing desk at work")
        llm = _StubLLM("not json")

        removed = await store._hygiene_pass(llm, "m")

        assert removed == 0
        assert await _row(store, keep)
        assert await _row(store, dup)


# -- consolidation summary + migration --


class TestConsolidationSummary:
    async def test_summary_has_tier_keys(self, store):
        llm = _StubLLM(json.dumps({"updates": [], "deletes": []}))
        result = await store.consolidate_and_cleanup(llm, "m")
        assert set(result) >= {
            "active_reviewed",
            "promoted_to_long_term",
            "expired_deleted",
            "hygiene_merged",
            "archived",
        }


class TestMigration:
    async def test_legacy_db_is_migrated_in_place(self, tmp_path):
        db_path = str(tmp_path / "legacy.db")
        # Create the original (pre-Tier-2/3) long_term table.
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "CREATE TABLE long_term ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT NOT NULL, "
                "subject TEXT NOT NULL, content TEXT NOT NULL, source TEXT, "
                "confidence TEXT DEFAULT 'stated', "
                "created_at DATETIME DEFAULT (datetime('now')), "
                "updated_at DATETIME DEFAULT (datetime('now')))"
            )
            await db.execute(
                "INSERT INTO long_term (category, subject, content) "
                "VALUES ('fact', 'matteo', 'old')"
            )
            await db.commit()

        store = MemoryStore(db_path=db_path)
        await store._ensure_schema()

        async with aiosqlite.connect(db_path) as db:
            cur = await db.execute("PRAGMA table_info(long_term)")
            cols = {row[1] for row in await cur.fetchall()}
        assert {"embedding", "importance", "last_accessed", "access_count", "archived"} <= cols

        # The legacy row survives and is readable with sane defaults.
        rows = await store.get_long_term()
        assert any(r["content"] == "old" for r in rows)


async def _row_all(store: MemoryStore) -> list[dict]:
    async with aiosqlite.connect(store.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM long_term ORDER BY id")
        return [dict(r) for r in await cur.fetchall()]
