"""Shared types for the ingestion pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class Document:
    """One harvested source document, before chunking."""

    # Stable identity, in preference order. Dedup runs on doc_id().
    doi: str | None = None
    pmcid: str | None = None
    pmid: str | None = None
    url: str | None = None

    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    abstract: str = ""

    # Section name -> text. Empty when only metadata was available.
    sections: dict[str, str] = field(default_factory=dict)

    license: str | None = None
    source: str = ""
    tags: list[str] = field(default_factory=list)
    full_text: bool = False

    def doc_id(self) -> str:
        for candidate in (self.doi, self.pmcid, self.pmid, self.url):
            if candidate:
                return hashlib.sha1(candidate.encode()).hexdigest()[:16]
        return hashlib.sha1((self.title + str(self.year)).encode()).hexdigest()[:16]

    def author_string(self) -> str:
        return "; ".join(self.authors[:8])

    def has_content(self) -> bool:
        return bool(self.sections) or len(self.abstract) > 200


@dataclass
class Chunk:
    """One indexable passage."""

    chunk_id: str
    doc_id: str
    text: str
    section: str

    title: str = ""
    authors: str = ""
    year: int | None = None
    journal: str | None = None
    doi: str | None = None
    url: str | None = None
    license: str | None = None
    source: str = ""
    modality_tags: list[str] = field(default_factory=list)

    def payload(self) -> dict:
        return {
            "text": self.text,
            "doc_id": self.doc_id,
            "section": self.section,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "journal": self.journal,
            "doi": self.doi,
            "url": self.url,
            "license": self.license,
            "source": self.source,
            "modality_tags": self.modality_tags,
        }
