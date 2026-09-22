"""Embed chunks and write them to Qdrant.

The collection carries two vectors per point: a dense one from the embedding
model, and a sparse BM25 one computed locally by fastembed. Retrieval fuses
them with RRF, which is what lets the corpus answer both "what does the
research say about contempt" and "DARVO" — one is a paraphrase problem, the
other is an exact-term problem, and neither method handles both.
"""

from __future__ import annotations

import os
import uuid

import structlog
from qdrant_client import QdrantClient, models

from .models import Chunk

log = structlog.get_logger(__name__)

# Fixed namespace, so a given chunk always maps to the same point id across
# runs and across machines.
_CHUNK_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

_embedder = None


_bm25_model = None


def _bm25():
    """BM25 sparse encoder, loaded once. CPU-only and cheap — it is a
    tokeniser plus IDF table, not a neural model."""
    global _bm25_model
    if _bm25_model is None:
        from fastembed import SparseTextEmbedding

        _bm25_model = SparseTextEmbedding("Qdrant/bm25")
    return _bm25_model


def _load_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        model = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
        device = os.getenv("EMBED_DEVICE", "cuda")
        log.info("index.embedder_loading", model=model, device=device)
        _embedder = SentenceTransformer(model, device=device)
        # Belt to the chunker's braces. Qwen3-Embedding will accept tens of
        # thousands of tokens, and a batch pads to its longest member, so a
        # single oversized chunk decides the memory cost of the whole batch.
        # Chunks target 800 tokens; anything claiming to need four times
        # that is malformed, and truncating it is the right answer.
        _embedder.max_seq_length = int(os.getenv("EMBED_MAX_SEQ_LEN", "1024"))
    return _embedder


def get_client() -> QdrantClient:
    return QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant:6333"), timeout=120)


def ensure_collection(client: QdrantClient, name: str, dim: int) -> None:
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )
    # Indexed so later phases can filter retrieval by modality or source
    # without a full scan.
    for field, schema in (
        ("modality_tags", models.PayloadSchemaType.KEYWORD),
        ("source", models.PayloadSchemaType.KEYWORD),
        ("doc_id", models.PayloadSchemaType.KEYWORD),
        ("year", models.PayloadSchemaType.INTEGER),
    ):
        client.create_payload_index(collection_name=name, field_name=field, field_schema=schema)
    log.info("index.collection_created", name=name, dim=dim)


def existing_doc_ids(client: QdrantClient, name: str) -> set[str]:
    """Which documents are already indexed, so re-runs are cheap."""
    if not client.collection_exists(name):
        return set()
    seen: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name,
            limit=1000,
            offset=offset,
            with_payload=["doc_id"],
            with_vectors=False,
        )
        seen.update(p.payload["doc_id"] for p in points if p.payload and p.payload.get("doc_id"))
        if offset is None:
            break
    return seen


def index_chunks(chunks: list[Chunk], collection: str, batch_size: int = 32) -> int:
    if not chunks:
        return 0

    model = _load_embedder()
    client = get_client()
    ensure_collection(client, collection, model.get_sentence_embedding_dimension())

    written = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors = model.encode(
            [c.text for c in batch],
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        sparse = list(_bm25().embed([c.text for c in batch]))
        client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    # Qdrant only accepts an unsigned int or a real UUID, so
                    # the chunk hash is folded into one deterministically.
                    # Stable id means re-indexing overwrites in place rather
                    # than duplicating the corpus.
                    id=str(uuid.uuid5(_CHUNK_NAMESPACE, chunk.chunk_id)),
                    vector={
                        "dense": vector.tolist(),
                        # Computed here, not passed as models.Document.
                        # A Document asks the SERVER to embed it, and the
                        # OSS qdrant image has no inference service — it
                        # answers 500 "InferenceService is not initialized"
                        # on every upsert. qdrant-client can resolve
                        # Documents locally, but relying on that silently
                        # made the whole index fail. Explicit is safer.
                        "sparse": models.SparseVector(
                            indices=sp.indices.tolist(), values=sp.values.tolist()
                        ),
                    },
                    payload=chunk.payload(),
                )
                for chunk, vector, sp in zip(batch, vectors, sparse, strict=True)
            ],
        )
        written += len(batch)
        if written % 320 == 0:
            log.info("index.progress", written=written, total=len(chunks))

    log.info("index.done", chunks=written, collection=collection)
    return written


def index_documents(
    docs: list,
    collection: str,
    *,
    batch_size: int = 32,
    max_tokens: int = 800,
    overlap: float = 0.15,
    on_complete=None,
) -> tuple[int, int]:
    """Chunk, embed and upsert one document at a time.

    Document-at-a-time is what makes an interrupted run resumable, and the
    flat shape it replaces is why the first run could not be resumed. That
    one chunked the whole corpus, then embedded it in batches of 32 that
    straddled document boundaries — so a run killed partway through left
    documents half-indexed, and because the resume check asks Qdrant "have
    I already seen this doc_id", a half-indexed document looked finished
    and was never completed.

    `on_complete(doc, n_chunks)` is called only once every chunk of that
    document is durably in Qdrant, so the caller's ledger cannot claim a
    document the index does not actually hold. Point ids are deterministic
    (uuid5 over the chunk hash), so re-running over a document that was
    partially written overwrites in place rather than duplicating it.
    """
    from .chunk import chunk_document

    model = _load_embedder()
    client = get_client()
    ensure_collection(client, collection, model.get_sentence_embedding_dimension())
    bm25 = _bm25()

    total_chunks = 0
    done_docs = 0

    for doc in docs:
        chunks = chunk_document(doc, max_tokens=max_tokens, overlap=overlap)
        if not chunks:
            # Metadata-only records with too short an abstract chunk to
            # nothing. They are still "done" — skipping the ledger entry
            # would make every later run retry them forever.
            if on_complete:
                on_complete(doc, 0)
            done_docs += 1
            continue

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectors = model.encode(
                [c.text for c in batch],
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )
            sparse = list(bm25.embed([c.text for c in batch]))
            client.upsert(
                collection_name=collection,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid5(_CHUNK_NAMESPACE, chunk.chunk_id)),
                        vector={
                            "dense": vector.tolist(),
                            "sparse": models.SparseVector(
                                indices=sp.indices.tolist(), values=sp.values.tolist()
                            ),
                        },
                        payload=chunk.payload(),
                    )
                    for chunk, vector, sp in zip(batch, vectors, sparse, strict=True)
                ],
            )

        total_chunks += len(chunks)
        done_docs += 1
        if on_complete:
            on_complete(doc, len(chunks))

        if done_docs % 25 == 0:
            log.info(
                "index.progress",
                documents=done_docs,
                of=len(docs),
                chunks=total_chunks,
            )

    log.info("index.done", documents=done_docs, chunks=total_chunks, collection=collection)
    return done_docs, total_chunks


def stats(collection: str) -> dict:
    client = get_client()
    if not client.collection_exists(collection):
        return {"exists": False}
    info = client.get_collection(collection)
    return {
        "exists": True,
        "chunks": info.points_count,
        "documents": len(existing_doc_ids(client, collection)),
    }
