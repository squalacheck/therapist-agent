-- Running profile — the "session notes" layer.
--
-- memory_facts is precise but atomised, and it is recalled by semantic
-- similarity to the current question. That works well once a conversation
-- has a subject, and not at all when a new chat opens with "hey" — there
-- is nothing to match on, so eight relevant facts come back as zero and
-- the agent greets you as a stranger.
--
-- This table holds one rolling narrative per scope, injected on EVERY
-- turn regardless of the query. Facts answer "what is her name"; the
-- profile answers "where were we, and what is this person working on".
-- They are complementary, not redundant.
CREATE TABLE IF NOT EXISTS memory_profile (
    scope             TEXT PRIMARY KEY,
    summary           TEXT        NOT NULL DEFAULT '',
    -- How many exchanges since the summary was last rewritten. Rewriting
    -- every turn would double the cost of the conversation for very
    -- little gain, so it is batched.
    turns_since_write INTEGER     NOT NULL DEFAULT 0,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
