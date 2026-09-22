"""OpenAI-compatible request/response shapes, plus internal types.

Only the subset Open WebUI actually sends is modelled. Unknown fields are
accepted and ignored rather than rejected, because frontends add new ones
without warning and a 422 in the middle of a conversation is a bad way to
find out.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None

    def text(self) -> str:
        """Flatten content to plain text.

        Open WebUI sends a list of parts when attachments are involved.
        """
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(
            part.get("text", "") for part in self.content if part.get("type") == "text"
        ).strip()


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    # Open WebUI puts its conversation id here when configured to; we fall
    # back to a header or a generated id. This is the key session memory
    # is scoped by.
    chat_id: str | None = None
    user: str | None = None


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage | None = None
    delta: dict[str, Any] | None = None
    finish_reason: str | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "local"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


# ── Internal ───────────────────────────────────────────────────────────


class Passage(BaseModel):
    """A retrieved chunk, carrying everything needed to cite it."""

    chunk_id: str
    # The document this chunk came from. Several chunks of one paper share
    # it, which is what lets retrieval cap how many slots any single source
    # may take — see retrieval._diversify.
    doc_id: str | None = None
    text: str
    score: float
    rerank_score: float | None = None

    title: str | None = None
    authors: str | None = None
    year: int | None = None
    journal: str | None = None
    doi: str | None = None
    url: str | None = None
    section: str | None = None
    license: str | None = None
    source: str | None = None
    modality_tags: list[str] = Field(default_factory=list)

    def citation(self) -> str:
        """A short human-readable reference for the citation list."""
        bits: list[str] = []
        if self.authors:
            first = self.authors.split(";")[0].split(",")[0].strip()
            bits.append(f"{first} et al." if ";" in self.authors else first)
        if self.year:
            bits.append(f"({self.year})")
        if self.title:
            bits.append(self.title)
        if self.journal:
            bits.append(f"*{self.journal}*")
        if self.doi:
            bits.append(f"https://doi.org/{self.doi}")
        elif self.url:
            bits.append(self.url)
        return " ".join(bits) or self.chunk_id


class MemoryFact(BaseModel):
    id: str | None = None
    text: str
    category: str = "general"
    salience: float = 0.5
    created_at: float = Field(default_factory=time.time)
    last_seen_at: float = Field(default_factory=time.time)


class SafetyVerdict(BaseModel):
    triggered: bool = False
    # suicidality | self_harm | immediate_danger | abuse_in_progress | none
    category: str = "none"
    confidence: float = 0.0
    rationale: str = ""
