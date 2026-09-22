-- Session memory schema.
--
-- The embedding column is created with a placeholder dimension and altered
-- to the real one at startup, once the embedding model has reported it.
-- That keeps the model swappable without a hand-written migration.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS memory_facts (
    id            UUID PRIMARY KEY,
    -- 'global' for the single-user case; a chat id when scoping per thread.
    scope         TEXT        NOT NULL DEFAULT 'global',
    fact          TEXT        NOT NULL,
    -- person | relationship | history | pattern | preference | goal | general
    category      TEXT        NOT NULL DEFAULT 'general',
    salience      REAL        NOT NULL DEFAULT 0.5,
    embedding     vector(1024),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS memory_facts_scope_idx    ON memory_facts (scope);
CREATE INDEX IF NOT EXISTS memory_facts_category_idx ON memory_facts (category);
CREATE INDEX IF NOT EXISTS memory_facts_seen_idx     ON memory_facts (last_seen_at DESC);

-- Turn-level audit, kept separate from Open WebUI's own chat storage.
-- Useful for the conflict-pattern analysis in a later phase, and for
-- checking what the agent actually retrieved when an answer looks wrong.
CREATE TABLE IF NOT EXISTS turns (
    id                UUID PRIMARY KEY,
    scope             TEXT        NOT NULL DEFAULT 'global',
    role              TEXT        NOT NULL,
    content           TEXT        NOT NULL,
    retrieval_query   TEXT,
    passage_ids       TEXT[],
    safety_category   TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS turns_scope_created_idx ON turns (scope, created_at DESC);
