"""Embedding and reranking models, loaded once and shared.

Both live in-process rather than behind their own service. That is a
deliberate simplification: they are small, they are only ever called by
this service, and a separate container would add a network hop plus one
more thing that can fail to build on sm_120.

If they will not load on the GPU, set EMBED_DEVICE=cpu. Query-time
embedding of a single string on a 9800X3D is comfortably fast; only batch
ingestion genuinely wants the card, and that runs in a separate container
while vLLM is stopped.
"""

from __future__ import annotations

import asyncio
import threading

import structlog

from .config import get_settings

log = structlog.get_logger(__name__)

# Qwen3-Embedding is instruction-tuned: queries get a task prefix, stored
# documents do not. Skipping this asymmetry silently costs real recall.
QUERY_INSTRUCTION = (
    "Instruct: Given a question about psychotherapy, relationships or trauma "
    "research, retrieve passages from the scholarly literature that answer it.\nQuery: "
)

_lock = threading.Lock()
_embedder = None
_reranker = None


def _load_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder
    with _lock:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer

            s = get_settings()
            log.info("embedder.loading", model=s.embed_model, device=s.embed_device)
            _embedder = SentenceTransformer(s.embed_model, device=s.embed_device)
            log.info("embedder.ready", dim=_embedder.get_sentence_embedding_dimension())
    return _embedder


def _load_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker
    with _lock:
        if _reranker is None:
            from sentence_transformers import CrossEncoder

            s = get_settings()
            log.info("reranker.loading", model=s.rerank_model, device=s.embed_device)
            _reranker = CrossEncoder(s.rerank_model, device=s.embed_device)
            log.info("reranker.ready")
    return _reranker


def embedding_dim() -> int:
    return _load_embedder().get_sentence_embedding_dimension()


async def embed_query(text: str) -> list[float]:
    """Embed a search query, with the instruction prefix applied."""
    return (await embed_texts([QUERY_INSTRUCTION + text], is_query=True))[0]


async def embed_texts(texts: list[str], *, is_query: bool = False) -> list[list[float]]:
    """Embed a batch. Runs in a thread so it never blocks the event loop."""

    def _run() -> list[list[float]]:
        model = _load_embedder()
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=8 if is_query else 32,
        )
        return [v.tolist() for v in vectors]

    return await asyncio.to_thread(_run)


async def rerank(query: str, passages: list[str]) -> list[float]:
    """Score each passage against the query. Higher is better."""
    if not passages:
        return []

    def _run() -> list[float]:
        model = _load_reranker()
        scores = model.predict(
            [(query, p) for p in passages],
            show_progress_bar=False,
            batch_size=16,
        )
        return [float(s) for s in scores]

    return await asyncio.to_thread(_run)


async def warmup() -> None:
    """Load both models at startup rather than on the first user turn.

    Otherwise the first question of the day takes an extra thirty seconds
    and looks like the whole stack is broken.
    """
    try:
        await asyncio.to_thread(_load_embedder)
        await asyncio.to_thread(_load_reranker)
    except Exception as exc:  # noqa: BLE001 — startup must not hard-fail
        log.error(
            "embedding.warmup_failed",
            error=str(exc),
            hint="If this is a CUDA/sm_120 problem, set EMBED_DEVICE=cpu in .env",
        )


_bm25_model = None


def sparse_query(text: str):
    """BM25 sparse vector for a query, computed in-process.

    Qdrant's `models.Document(model="Qdrant/bm25")` asks the SERVER to do
    this, and the OSS qdrant image has no inference service — it answers
    500 "InferenceService is not initialized". The sparse half of hybrid
    search would have failed on every single query, and `search()` catches
    the exception and returns [], so it would have looked like an empty
    corpus rather than a broken query.

    It is a tokeniser plus an IDF table, not a neural model — cheap enough
    to run on CPU next to everything else.
    """
    from qdrant_client import models

    global _bm25_model
    if _bm25_model is None:
        from fastembed import SparseTextEmbedding

        _bm25_model = SparseTextEmbedding("Qdrant/bm25")
    v = next(iter(_bm25_model.query_embed(text)))
    return models.SparseVector(indices=v.indices.tolist(), values=v.values.tolist())
