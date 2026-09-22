"""Section-aware chunking.

Chunks respect section boundaries and split on sentences, never mid-sentence.
Each one is prefixed with its paper's title and section so that the embedding
carries context — a chunk reading "this effect was not replicated" is
meaningless in isolation and embeds to nothing useful.
"""

from __future__ import annotations

import hashlib
import re

from .models import Chunk, Document

SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
CHARS_PER_TOKEN = 4.0


def _tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def chunk_document(doc: Document, *, max_tokens: int = 800, overlap: float = 0.15) -> list[Chunk]:
    if not doc.has_content():
        return []

    parts: list[tuple[str, str]] = []
    if doc.abstract:
        parts.append(("Abstract", doc.abstract))
    parts.extend(doc.sections.items())

    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    overlap_chars = int(max_chars * overlap)

    chunks: list[Chunk] = []
    for section, text in parts:
        for position, body in enumerate(_split(text, max_chars, overlap_chars)):
            # The prefix is what makes an isolated chunk interpretable, both
            # to the embedding model and to you when reading eval output.
            prefixed = f"{doc.title}\n[{section}]\n\n{body}"
            chunks.append(
                Chunk(
                    # Position is part of the key, not just the text. Without
                    # it, chunks whose opening sentences repeat — common in
                    # methods sections and agency documents — hash identically
                    # and silently overwrite each other at upsert time.
                    chunk_id=hashlib.sha1(
                        f"{doc.doc_id()}:{section}:{position}:{body[:200]}".encode()
                    ).hexdigest(),
                    doc_id=doc.doc_id(),
                    text=prefixed,
                    section=section,
                    title=doc.title,
                    authors=doc.author_string(),
                    year=doc.year,
                    journal=doc.journal,
                    doi=doc.doi,
                    url=doc.url,
                    license=doc.license,
                    source=doc.source,
                    modality_tags=doc.tags,
                )
            )
    return chunks


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Break a run of text that has no sentence boundaries to break on.

    Full text scraped from journals is not all prose. Reference lists,
    results tables, poster-abstract books and author affiliations arrive as
    thousands of characters with no `.!?` in sight, so SENTENCE.split gives
    back one enormous "sentence" and the caller below, which only starts a
    new chunk when it already has one, has no choice but to emit it whole.

    That produced a single 18,422-character chunk in this corpus. Two things
    go wrong with one of those. It embeds to mush — one vector averaging
    four thousand tokens of unrelated material retrieves nothing well. And
    it is the reason batch embedding wedged the GPU twice: the batch pads to
    its longest member, so one such chunk makes the other thirty-one cost
    the same as it does.

    Split on whitespace so words stay intact; fall back to a blunt slice for
    text with no spaces either.
    """
    words = text.split(" ")
    if len(words) == 1:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    out: list[str] = []
    current: list[str] = []
    size = 0
    for word in words:
        if size + len(word) + 1 > max_chars and current:
            out.append(" ".join(current))
            current, size = [], 0
        current.append(word)
        size += len(word) + 1
    if current:
        out.append(" ".join(current))
    return out


def _split(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if _tokens(text) > 30 else []

    sentences: list[str] = []
    for sentence in SENTENCE.split(text):
        # Do this before the packing loop, so the invariant below is simply
        # "every sentence fits in a chunk" and the loop stays readable.
        if len(sentence) > max_chars:
            sentences.extend(_hard_split(sentence, max_chars))
        else:
            sentences.append(sentence)
    out: list[str] = []
    current: list[str] = []
    size = 0

    for sentence in sentences:
        sentence_len = len(sentence) + 1
        if size + sentence_len > max_chars and current:
            out.append(" ".join(current).strip())
            # Carry the tail forward so a claim split across a boundary is
            # still retrievable from either side.
            tail: list[str] = []
            tail_size = 0
            for previous in reversed(current):
                if tail_size + len(previous) > overlap_chars:
                    break
                tail.insert(0, previous)
                tail_size += len(previous) + 1
            current, size = tail, tail_size
        current.append(sentence)
        size += sentence_len

    if current and _tokens(" ".join(current)) > 30:
        out.append(" ".join(current).strip())
    return out
