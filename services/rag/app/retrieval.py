"""Hybrid retrieval over the corpus: dense + sparse, then reranked.

The shape is deliberately cast-wide-then-cut: pull 40 candidates with cheap
search, then spend the reranker on narrowing to 8. Dense search alone misses
exact terminology ("DARVO", "Four Horsemen", a specific instrument name);
sparse alone misses paraphrase. Together with a reranker on top they cover
each other's failure modes, and `make eval-retrieval` measures whether that
is actually true on your corpus rather than taking it on faith.
"""

from __future__ import annotations

import structlog
from qdrant_client import AsyncQdrantClient, models

from .config import get_settings
from .embedding import embed_query, rerank, sparse_query
from .llm import get_llm
from .schemas import ChatMessage, Passage

log = structlog.get_logger(__name__)

_client: AsyncQdrantClient | None = None


def get_qdrant() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(url=get_settings().qdrant_url, prefer_grpc=False)
    return _client


async def build_query(messages: list[ChatMessage]) -> str:
    """Condense the conversation into a standalone retrieval query.

    Necessary because the literal last turn is often unsearchable on its
    own — "what should I say to them about that?" retrieves nothing useful.
    Resolving it against recent context is what makes retrieval work in a
    real conversation rather than a one-shot Q&A demo.
    """
    turns = [m for m in messages if m.role in ("user", "assistant")]
    last_user = next((m.text() for m in reversed(turns) if m.role == "user"), "")
    if not last_user:
        return ""

    # A short, self-contained question needs no rewriting; skip the round trip.
    if len(turns) <= 1 or len(last_user.split()) > 12:
        recent = turns[-6:]
    else:
        recent = turns[-6:]

    from .config import load_prompt

    transcript = "\n".join(f"{m.role}: {m.text()}" for m in recent)
    try:
        rewritten = await get_llm().complete(
            [
                {"role": "system", "content": load_prompt("query_rewrite")},
                {"role": "user", "content": transcript},
            ],
            max_tokens=128,
        )
        query = rewritten.strip().strip('"')
        return query or last_user
    except Exception as exc:  # noqa: BLE001
        log.warning("retrieval.rewrite_failed", error=str(exc))
        return last_user


async def search(query: str) -> list[Passage]:
    """Hybrid search, then rerank down to the passages worth showing."""
    if not query.strip():
        return []

    s = get_settings()
    client = get_qdrant()

    try:
        vector = await embed_query(query)
    except Exception as exc:  # noqa: BLE001
        log.error("retrieval.embed_failed", error=str(exc))
        return []

    try:
        # Server-side fusion of dense and sparse results. Qdrant's RRF is
        # good enough that hand-rolled score blending is not worth the code.
        result = await client.query_points(
            collection_name=s.qdrant_collection,
            prefetch=[
                models.Prefetch(
                    query=vector,
                    using="dense",
                    limit=s.retrieve_top_k,
                ),
                models.Prefetch(
                    # Encoded here, not as models.Document — the OSS qdrant
                    # image has no inference service and answers 500 to a
                    # Document. Same reason as services/ingest/ingest/index.py.
                    query=sparse_query(query),
                    using="sparse",
                    limit=s.retrieve_top_k,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=s.retrieve_top_k,
            with_payload=True,
        )
        points = result.points
    except Exception as exc:  # noqa: BLE001
        log.error("retrieval.search_failed", error=str(exc))
        return []

    candidates = [
        Passage(
            chunk_id=str(p.id),
            doc_id=p.payload.get("doc_id"),
            text=p.payload.get("text", ""),
            score=float(p.score or 0.0),
            title=p.payload.get("title"),
            authors=p.payload.get("authors"),
            year=p.payload.get("year"),
            journal=p.payload.get("journal"),
            doi=p.payload.get("doi"),
            url=p.payload.get("url"),
            section=p.payload.get("section"),
            license=p.payload.get("license"),
            source=p.payload.get("source"),
            modality_tags=p.payload.get("modality_tags", []) or [],
        )
        for p in points
        if p.payload
    ]
    if not candidates:
        return []

    try:
        scores = await rerank(query, [c.text for c in candidates])
        for passage, score in zip(candidates, scores, strict=True):
            passage.rerank_score = score
        candidates.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)

        # min_rerank_score is UNCALIBRATED until you look at this line.
        # Qwen3-Reranker scores through a LogitScore head with an Identity
        # activation, so `predict` returns a raw value derived from the
        # true/false token logits — not a 0..1 probability. If those scores
        # come out negative, the default threshold of 0.15 rejects every
        # passage; the `kept or candidates` fallback below then silently
        # hands back the unfiltered top-k and the threshold does nothing at
        # all. That failure is invisible without this log line.
        # Read it, then set MIN_RERANK_SCORE from the distribution.
        # `scores` is in CANDIDATE order, not rank order — the sort above
        # reorders `candidates`, not this list. Reading scores[0] as "top"
        # reported whatever Qdrant happened to return first, which is why
        # this line has been printing a top BELOW its own median. Calibrating
        # a threshold from that number is worse than not having it.
        ranked = sorted(scores, reverse=True)
        log.info(
            "retrieval.rerank_scores",
            top=round(ranked[0], 4) if ranked else None,
            p90=round(ranked[max(0, len(ranked) // 10 - 1)], 4) if ranked else None,
            median=round(ranked[len(ranked) // 2], 4) if ranked else None,
            bottom=round(ranked[-1], 4) if ranked else None,
            kept_at_threshold=sum(1 for x in scores if x >= s.min_rerank_score),
            threshold=s.min_rerank_score,
        )

        kept = [c for c in candidates if (c.rerank_score or 0.0) >= s.min_rerank_score]
        # If the threshold rejects everything the question is probably off-corpus.
        # Keep the top few anyway and let the model say it does not know.
        candidates = _diversify(kept or candidates, s.max_passages_per_doc, s.rerank_top_k)
    except Exception as exc:  # noqa: BLE001
        log.warning("retrieval.rerank_failed", error=str(exc))
        candidates = candidates[: s.rerank_top_k]

    log.info("retrieval.done", query=query[:120], returned=len(candidates))
    return candidates


def _diversify(candidates: list[Passage], per_doc: int, limit: int) -> list[Passage]:
    """Take the best `limit`, but never more than `per_doc` from one document.

    Rank order is preserved; a passage is skipped only when its document has
    already contributed its share.

    The cap is hard, and returning fewer than `limit` is the correct
    outcome when the corpus has nothing else to offer. The first version of
    this backfilled from the skipped passages rather than come up short,
    which sounds generous and defeats the entire purpose: on the Four
    Horsemen question only three distinct documents cleared the threshold,
    so the backfill refilled the set to five copies of the same PDF — the
    exact result the cap exists to prevent.

    Five passages from three papers beat eight passages from one. Adjacent
    chunks of a single document are largely redundant with each other, so
    the extra slots buy very little evidence and cost the thing that makes
    a citation list checkable.
    """
    if per_doc <= 0:
        return candidates[:limit]

    picked: list[Passage] = []
    seen: dict[str, int] = {}

    for candidate in candidates:
        key = candidate.doc_id or candidate.chunk_id
        if seen.get(key, 0) >= per_doc:
            continue
        seen[key] = seen.get(key, 0) + 1
        picked.append(candidate)
        if len(picked) >= limit:
            break

    return picked


def format_passages(passages: list[Passage]) -> str:
    """Render passages for the prompt, numbered so they can be cited."""
    if not passages:
        return ""
    blocks = []
    for i, p in enumerate(passages, 1):
        header = f"[{i}] {p.citation()}"
        if p.section:
            header += f" — section: {p.section}"
        blocks.append(f"{header}\n{p.text.strip()}")
    return "\n\n".join(blocks)


def format_sources(passages: list[Passage]) -> str:
    """The reference list appended after a grounded answer."""
    if not passages:
        return ""
    lines = [f"[{i}] {p.citation()}" for i, p in enumerate(passages, 1)]
    return "\n\n---\n**Sources**\n" + "\n".join(lines)
