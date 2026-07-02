-- Task reflections schema
-- Stores lessons learned from completed tasks.

CREATE TABLE IF NOT EXISTS reflections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_summary TEXT NOT NULL,          -- one-line summary of the task
    outcome     TEXT NOT NULL DEFAULT 'success',  -- success | partial | failure
    lesson      TEXT NOT NULL,           -- what was learned
    tool_issues TEXT,                    -- JSON array of tool/API issues encountered
    category    TEXT NOT NULL DEFAULT 'general',  -- general | tool | api | planning | communication
    created_at  DATETIME NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_reflections_category ON reflections(category);
CREATE INDEX IF NOT EXISTS idx_reflections_created ON reflections(created_at);
