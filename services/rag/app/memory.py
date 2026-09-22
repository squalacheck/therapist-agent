"""Persistent session memory.

The point of this is that the agent does not start from zero every time.
Without it, a therapeutic assistant is a stranger you re-brief at every
sitting, which is exactly the thing that makes it useless for anything
ongoing.

Facts are stored in Postgres with a pgvector embedding and recalled by a
blend of semantic similarity, salience and recency. Extraction runs after
the response has finished streaming, so it never costs the user latency.

Scoping: memory is keyed by `chat_id` when Open WebUI supplies one, and
falls back to a single global scope otherwise. Single-user machine, so a
global scope is a reasonable default rather than a leak.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid

import structlog
from sqlalchemy import text as sql
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from .config import get_settings, load_prompt
from .embedding import embed_texts, embedding_dim
from .llm import get_llm
from .schemas import ChatMessage, MemoryFact

log = structlog.get_logger(__name__)

GLOBAL_SCOPE = "global"

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker | None = None


def _get_sessionmaker() -> async_sessionmaker:
    global _engine, _sessionmaker
    if _sessionmaker is None:
        s = get_settings()
        _engine = create_async_engine(s.database_url, pool_size=5, max_overflow=5)
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _sessionmaker


def get_sessionmaker() -> async_sessionmaker:
    """Shared connection pool.

    Public because daily.py needs the same pool rather than a second one —
    two engines against the same small Postgres is how you run out of
    connections for no reason.
    """
    return _get_sessionmaker()


async def init_schema() -> None:
    """Create the vector index once the embedding dimension is known.

    The table itself is created by infra/postgres/init/001_schema.sql; the
    index needs the runtime dimension, which is why it lives here.
    """
    if not get_settings().memory_enabled:
        return
    dim = embedding_dim()
    async with _get_sessionmaker()() as session:
        await session.execute(
            sql(f"ALTER TABLE memory_facts ALTER COLUMN embedding TYPE vector({dim})")
        )
        await session.execute(
            sql(
                "CREATE INDEX IF NOT EXISTS memory_facts_embedding_idx "
                "ON memory_facts USING hnsw (embedding vector_cosine_ops)"
            )
        )
        await session.commit()
    log.info("memory.schema_ready", dim=dim)


async def recall(query: str, scope: str = GLOBAL_SCOPE) -> list[MemoryFact]:
    """Pull the facts most worth putting in front of the model this turn."""
    s = get_settings()
    if not s.memory_enabled or not query.strip():
        return []

    try:
        vector = (await embed_texts([query], is_query=True))[0]
    except Exception as exc:  # noqa: BLE001
        log.warning("memory.embed_failed", error=str(exc))
        return []

    # Similarity dominates, with a gentle nudge for salience and for facts
    # seen recently. Pure similarity surfaces stale detail; pure recency
    # surfaces whatever was said last, which is rarely what matters.
    query_sql = sql(
        """
        SELECT id, fact, category, salience,
               extract(epoch FROM created_at)   AS created_at,
               extract(epoch FROM last_seen_at) AS last_seen_at,
               1 - (embedding <=> CAST(:vec AS vector)) AS similarity
        FROM memory_facts
        WHERE scope = :scope
        ORDER BY (1 - (embedding <=> CAST(:vec AS vector))) * 0.75
               + salience * 0.15
               + (1.0 / (1.0 + EXTRACT(EPOCH FROM (now() - last_seen_at)) / 2592000.0)) * 0.10
               DESC
        LIMIT :k
        """
    )
    try:
        async with _get_sessionmaker()() as session:
            rows = (
                await session.execute(
                    query_sql,
                    {"vec": json.dumps(vector), "scope": scope, "k": s.memory_recall_k},
                )
            ).mappings().all()
    except Exception as exc:  # noqa: BLE001
        log.warning("memory.recall_failed", error=str(exc))
        return []

    facts = [
        MemoryFact(
            id=str(r["id"]),
            text=r["fact"],
            category=r["category"],
            salience=float(r["salience"]),
            created_at=float(r["created_at"]),
            last_seen_at=float(r["last_seen_at"]),
        )
        for r in rows
    ]
    if facts:
        asyncio.create_task(_touch([f.id for f in facts if f.id]))
    log.info("memory.recalled", count=len(facts))
    return facts


async def _touch(ids: list[str]) -> None:
    """Mark recalled facts as seen, so recency reflects use, not just age."""
    if not ids:
        return
    try:
        async with _get_sessionmaker()() as session:
            await session.execute(
                sql("UPDATE memory_facts SET last_seen_at = now() WHERE id = ANY(:ids)"),
                {"ids": ids},
            )
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        log.debug("memory.touch_failed", error=str(exc))


async def extract_and_store(
    messages: list[ChatMessage],
    reply: str,
    scope: str = GLOBAL_SCOPE,
) -> None:
    """Distil durable facts from the exchange. Runs after the response.

    "Durable" is doing real work in the prompt: the point is to remember
    that someone's sister is called Ruth and that the recurring conflict is
    about money, not to transcribe the conversation back into the database.
    """
    if not get_settings().memory_enabled:
        return

    last_user = next((m.text() for m in reversed(messages) if m.role == "user"), "")
    if not last_user:
        return

    exchange = f"User: {last_user}\n\nAssistant: {reply[:2000]}"

    try:
        result = await get_llm().complete_json(
            [
                {"role": "system", "content": load_prompt("memory_extract")},
                {"role": "user", "content": exchange},
            ],
            max_tokens=600,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("memory.extract_failed", error=str(exc))
        return

    if not result:
        return
    raw_facts = result.get("facts") or []
    if not isinstance(raw_facts, list) or not raw_facts:
        return

    candidates: list[tuple[str, str, float]] = []
    for item in raw_facts[:10]:
        if not isinstance(item, dict):
            continue
        fact = str(item.get("fact", "")).strip()
        if len(fact) < 8:
            continue
        candidates.append(
            (
                fact,
                str(item.get("category", "general")),
                min(max(float(item.get("salience", 0.5) or 0.5), 0.0), 1.0),
            )
        )
    if not candidates:
        return

    try:
        vectors = await embed_texts([c[0] for c in candidates])
    except Exception as exc:  # noqa: BLE001
        log.warning("memory.embed_store_failed", error=str(exc))
        return

    stored = 0
    try:
        async with _get_sessionmaker()() as session:
            for (fact, category, salience), vector in zip(candidates, vectors, strict=True):
                # Near-duplicates get their salience and recency bumped
                # instead of piling up as separate rows. Without this the
                # store fills with fifteen phrasings of the same thing.
                existing = (
                    await session.execute(
                        sql(
                            "SELECT id FROM memory_facts "
                            "WHERE scope = :scope "
                            "  AND 1 - (embedding <=> CAST(:vec AS vector)) > 0.92 "
                            "LIMIT 1"
                        ),
                        {"scope": scope, "vec": json.dumps(vector)},
                    )
                ).scalar()

                if existing:
                    await session.execute(
                        sql(
                            "UPDATE memory_facts "
                            "SET last_seen_at = now(), "
                            "    salience = LEAST(1.0, salience + 0.05) "
                            "WHERE id = :id"
                        ),
                        {"id": existing},
                    )
                    continue

                await session.execute(
                    sql(
                        "INSERT INTO memory_facts "
                        "  (id, scope, fact, category, salience, embedding, created_at, last_seen_at) "
                        "VALUES (:id, :scope, :fact, :category, :salience, CAST(:vec AS vector), now(), now())"
                    ),
                    {
                        "id": str(uuid.uuid4()),
                        "scope": scope,
                        "fact": fact,
                        "category": category,
                        "salience": salience,
                        "vec": json.dumps(vector),
                    },
                )
                stored += 1
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("memory.store_failed", error=str(exc))
        return

    log.info("memory.stored", new=stored, considered=len(candidates))


async def get_profile(scope: str = GLOBAL_SCOPE) -> str:
    """The running notes for this person, injected on every turn.

    Unlike recall(), this takes no query and does no similarity search. It
    is unconditional by design: a new chat that opens with "hey" gives
    recall() nothing to match on, so it returns nothing, and the agent
    greets someone it has talked to for months as a stranger. This is the
    layer that stops that happening.
    """
    if not get_settings().memory_enabled:
        return ""
    try:
        async with _get_sessionmaker()() as session:
            row = (
                await session.execute(
                    sql("SELECT summary FROM memory_profile WHERE scope = :scope"),
                    {"scope": scope},
                )
            ).scalar_one_or_none()
        return (row or "").strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("memory.profile_read_failed", error=str(exc))
        return ""


async def update_profile(
    messages: list[ChatMessage],
    reply: str,
    scope: str = GLOBAL_SCOPE,
    *,
    force: bool = False,
) -> None:
    """Rewrite the running notes, batched every N exchanges.

    Runs after the response has streamed, like fact extraction, so it
    never costs the user latency. Batched because rewriting the whole
    profile on every turn doubles the model calls per exchange to keep a
    summary that rarely changes that fast.
    """
    s = get_settings()
    if not s.memory_enabled:
        return

    try:
        async with _get_sessionmaker()() as session:
            row = (
                await session.execute(
                    sql(
                        """
                        INSERT INTO memory_profile (scope, turns_since_write)
                        VALUES (:scope, 1)
                        ON CONFLICT (scope) DO UPDATE
                            SET turns_since_write = memory_profile.turns_since_write + 1
                        RETURNING summary, turns_since_write
                        """
                    ),
                    {"scope": scope},
                )
            ).mappings().one()
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("memory.profile_counter_failed", error=str(exc))
        return

    current, pending = (row["summary"] or "").strip(), row["turns_since_write"]
    # Always write the first one — otherwise a short first conversation
    # leaves the profile empty and the next chat starts cold anyway.
    # `force` is the session close-out: batching every N exchanges is a
    # sensible economy mid-conversation and the wrong thing to do on the way
    # out, where up to N-1 exchanges would otherwise never reach the notes
    # and the next session would open having forgotten how this one ended.
    if pending < s.profile_update_every and current and not force:
        return

    turns = [m for m in messages if m.role in ("user", "assistant")][-6:]
    transcript = "\n".join(f"{m.role}: {m.text()}" for m in turns)
    transcript += f"\nassistant: {reply}"

    try:
        updated = await get_llm().complete(
            [
                {"role": "system", "content": load_prompt("profile_update")},
                {
                    "role": "user",
                    "content": (
                        f"Current notes:\n{current or '(none yet)'}\n\n"
                        f"Recent exchange:\n{transcript[:6000]}"
                    ),
                },
            ],
            max_tokens=s.profile_max_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("memory.profile_write_failed", error=str(exc))
        return

    updated = updated.strip().strip('"')
    # A model that returns almost nothing has misunderstood the task; keep
    # what we had rather than destroying a good profile with an empty one.
    if len(updated) < 40 and current:
        log.info("memory.profile_unchanged", reason="model returned too little")
        return

    try:
        async with _get_sessionmaker()() as session:
            await session.execute(
                sql(
                    """
                    UPDATE memory_profile
                       SET summary = :summary, turns_since_write = 0, updated_at = now()
                     WHERE scope = :scope
                    """
                ),
                {"summary": updated, "scope": scope},
            )
            await session.commit()
        log.info("memory.profile_written", chars=len(updated), scope=scope)
    except Exception as exc:  # noqa: BLE001
        log.warning("memory.profile_persist_failed", error=str(exc))


def format_memory(facts: list[MemoryFact]) -> str:
    """Render recalled facts for the prompt."""
    if not facts:
        return ""
    now = time.time()
    lines = []
    for f in facts:
        days = int((now - f.created_at) / 86400)
        when = "today" if days < 1 else f"{days}d ago" if days < 60 else "a while back"
        lines.append(f"- ({f.category}, first mentioned {when}) {f.text}")
    return "\n".join(lines)
