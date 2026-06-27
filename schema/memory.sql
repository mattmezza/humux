-- Memory database schema (data/memory.db)
-- Two-tier memory: long-term (permanent) + short-term (expiring)

CREATE TABLE IF NOT EXISTS long_term (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    subject TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT,
    confidence TEXT DEFAULT 'stated',
    created_at DATETIME DEFAULT (datetime('now')),
    updated_at DATETIME DEFAULT (datetime('now')),
    -- Tier 2: cached embedding vector (packed float32 blob; NULL until computed)
    embedding BLOB,
    -- Tier 3: forgetting / importance / reinforcement
    importance REAL NOT NULL DEFAULT 5.0,
    last_accessed DATETIME,
    access_count INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    -- Two-tier scoped memory (#42): '' = shared (owner-level, visible to every
    -- persona + the default identity), '<persona>' = private to that persona.
    scope TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS short_term (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    context TEXT,
    expires_at DATETIME NOT NULL,
    created_at DATETIME DEFAULT (datetime('now')),
    -- See long_term.scope (#42).
    scope TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_lt_category ON long_term(category);
CREATE INDEX IF NOT EXISTS idx_lt_subject ON long_term(subject);
CREATE INDEX IF NOT EXISTS idx_st_expires ON short_term(expires_at);
-- idx_lt_archived / idx_lt_scope / idx_st_scope are created in the
-- MemoryStore._migrate_* methods, after their columns are guaranteed to exist
-- (so legacy DBs that predate those columns migrate cleanly).
