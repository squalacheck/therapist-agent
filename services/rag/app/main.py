"""OpenAI-compatible API in front of the agent.

Presenting the agent as a model is what lets Open WebUI drive it without
knowing anything about retrieval, memory or safety — and what lets a custom
frontend replace Open WebUI later without touching any of this.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import daily, embedding, memory, pipeline, session
from .config import get_settings
from .llm import ModelUnavailable, get_llm
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ModelCard,
    ModelList,
    Usage,
)

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(logging, get_settings().log_level.upper(), logging.INFO)
    ),
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
)
log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    log.info(
        "startup",
        retrieval=s.retrieval_enabled,
        memory=s.memory_enabled,
        safety=s.safety_enabled,
        embed_device=s.embed_device,
    )
    if s.retrieval_enabled or s.memory_enabled:
        await embedding.warmup()
    if s.memory_enabled:
        try:
            await memory.init_schema()
        except Exception as exc:  # noqa: BLE001
            log.error("startup.memory_schema_failed", error=str(exc))
        if s.daily_enabled:
            try:
                await daily.init_schema()
            except Exception as exc:  # noqa: BLE001
                log.error("startup.daily_schema_failed", error=str(exc))
    yield
    await get_llm().aclose()


app = FastAPI(title="Therapist Agent", version="0.1.0", lifespan=lifespan)

# Streaming flush cadence. 120ms is about the point where a reader stops
# perceiving the text as arriving in chunks; below ~60ms the markdown
# re-render cost starts showing up as flicker again.
_FLUSH_INTERVAL_S = 0.12
_FLUSH_CHARS = 160


@app.get("/health")
async def health() -> dict[str, object]:
    s = get_settings()
    return {
        "status": "ok",
        "model_backend": await get_llm().healthy(),
        "retrieval": s.retrieval_enabled,
        "memory": s.memory_enabled,
        "profile": s.profile_enabled,
        "safety": s.safety_enabled,
        "daily": s.daily_enabled,
    }


@app.get("/v1/models")
async def list_models() -> ModelList:
    """Two entries, differing only in whether they touch memory.

    `therapist` carries everything across every conversation.
    `therapist-daily` opens on today's practice list and then talks about it.
    `therapist-fresh` reads nothing and writes nothing — for thinking
    about someone else's situation, or asking a research question that
    should not become part of your own history. Same model, same corpus,
    same stance; the difference is entirely in this service.
    """
    s = get_settings()
    cards = [ModelCard(id=s.public_model_name)]
    if s.memory_enabled and s.fresh_model_name:
        cards.append(ModelCard(id=s.fresh_model_name))
    if s.memory_enabled and s.daily_enabled and s.daily_model_name:
        cards.append(ModelCard(id=s.daily_model_name))
    return ModelList(data=cards)


def _scope(req: ChatCompletionRequest, header_chat_id: str | None) -> str:
    """Where this conversation's memory lives.

    Single user on a single machine, so a global scope is the useful
    default: continuity across conversations is the whole point. Switch to
    the per-chat id here if you ever want threads kept separate.
    """
    return memory.GLOBAL_SCOPE


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    request: Request,
    x_openwebui_chat_id: str | None = Header(default=None),
):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    settings = get_settings()
    # The requested model decides whether this turn is remembered. Anything
    # that is not the fresh variant gets full continuity, so an unfamiliar
    # model name fails toward the normal behaviour rather than silently
    # dropping someone's memory on the floor.
    remember = req.model != settings.fresh_model_name
    # Echoed back as-is when it is one of ours, so the daily and fresh
    # entries are labelled correctly; anything unfamiliar is normalised to
    # the main model, which is also the variant it is about to be served by.
    known = {settings.public_model_name, settings.fresh_model_name, settings.daily_model_name}
    model_name = req.model if req.model in known else settings.public_model_name
    scope = _scope(req, x_openwebui_chat_id)

    # Any system message the frontend injected is dropped. The therapeutic
    # stance is ours and lives in prompts/system.md; letting Open WebUI's
    # default prompt override it would quietly undo the whole design.
    messages: list[ChatMessage] = [m for m in req.messages if m.role != "system"]
    if not messages:
        raise HTTPException(status_code=400, detail="no user or assistant messages")

    # "Let's end for now" — matched deterministically on the raw turn,
    # before any model sees it. This writes the profile, closes the session
    # and asks the host to stop the stack, and none of that should hinge on
    # whether a 27B agreed that you meant it.
    last_user = next((m.text() for m in reversed(messages) if m.role == "user"), "")
    if settings.time_awareness and session.is_end_request(last_user):
        summary = await session.close_out(messages, scope)
        text = session.farewell(summary)
        if req.stream:
            return StreamingResponse(
                _sse(_once(text), model_name),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return ChatCompletionResponse(
            model=model_name,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(),
        )

    if settings.daily_enabled and req.model == settings.daily_model_name:
        stream = pipeline.run_daily(
            messages,
            scope=scope,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
    else:
        stream = pipeline.run(
            messages,
            scope=scope,
            remember=remember,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )

    if req.stream:
        return StreamingResponse(
            _sse(stream, model_name),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        content = "".join([chunk async for chunk in stream])
    except Exception as exc:  # noqa: BLE001
        log.error("completion.failed", error=str(exc))
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(exc), "type": "upstream_error"}},
        )

    return ChatCompletionResponse(
        model=model_name,
        choices=[
            Choice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=Usage(),
    )


@app.get("/v1/daily")
async def daily_today(generate: bool = True) -> JSONResponse:
    """Today's plan as JSON.

    `generate=false` reads without creating one, which is what a status
    check or a widget wants — asking for the list should not be the thing
    that silently spends a minute of GPU time generating it.
    """
    s = get_settings()
    if not (s.daily_enabled and s.memory_enabled):
        raise HTTPException(status_code=404, detail="daily practice is disabled")
    if generate:
        plan, created = await daily.ensure_plan(memory.GLOBAL_SCOPE)
    else:
        plan, created = await daily.get_plan(memory.GLOBAL_SCOPE), False
    if not plan:
        return JSONResponse(status_code=404, content={"detail": "no plan for today"})
    return JSONResponse(content={"generated": created, **plan})


@app.post("/v1/daily/regenerate")
async def daily_regenerate() -> JSONResponse:
    """Throw today's list away and build a new one.

    Completed items go with it — that is deliberate. A regenerated day is a
    statement that the list was wrong, and carrying yesterday's ticks into
    it would make the record say something that did not happen.
    """
    s = get_settings()
    if not (s.daily_enabled and s.memory_enabled):
        raise HTTPException(status_code=404, detail="daily practice is disabled")
    plan = await daily.generate(memory.GLOBAL_SCOPE)
    if not plan:
        return JSONResponse(status_code=503, content={"detail": "generation failed"})
    return JSONResponse(content=plan)


@app.post("/v1/daily/item/{item_id}")
async def daily_set_status(item_id: str, status: str, outcome: str = "") -> JSONResponse:
    s = get_settings()
    if not (s.daily_enabled and s.memory_enabled):
        raise HTTPException(status_code=404, detail="daily practice is disabled")
    if status not in daily.STATUSES:
        raise HTTPException(status_code=400, detail=f"status must be one of {daily.STATUSES}")
    if not await daily.set_status(item_id, status, outcome):
        raise HTTPException(status_code=404, detail="no such item")
    return JSONResponse(content=await daily.get_plan(memory.GLOBAL_SCOPE))


async def _once(text: str) -> AsyncIterator[str]:
    """A one-shot stream, for replies this service writes itself."""
    yield text


async def _sse(stream: AsyncIterator[str], model: str) -> AsyncIterator[str]:
    """Wrap the pipeline's text deltas in OpenAI's streaming envelope."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def frame(delta: dict, finish: str | None = None) -> str:
        return "data: " + json.dumps(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        ) + "\n\n"

    yield frame({"role": "assistant", "content": ""})

    # Coalesce deltas before sending them.
    #
    # vLLM emits one delta per token. Forwarding each one straight through
    # makes Open WebUI re-parse and re-render the entire message as markdown
    # ~30 times a second, and a half-written construct renders differently
    # from a finished one: `**bold` is literal asterisks until the closing
    # `**` arrives, a `|` is text until the row completes, `---` is a
    # paragraph until it is a rule. Every one of those flips the layout on
    # the next token. That is the flashing — it is not a network or model
    # problem, it is re-rendering faster than the content stabilises.
    #
    # Flushing on a short interval cuts the re-render rate by roughly 4x
    # while staying well under the threshold where streaming stops feeling
    # live. Flush early on a large buffer so a fast burst is not held back.
    buffer: list[str] = []
    last_flush = time.monotonic()

    def take() -> str:
        text = "".join(buffer)
        buffer.clear()
        return text

    try:
        async for chunk in stream:
            buffer.append(chunk)
            now = time.monotonic()
            if now - last_flush >= _FLUSH_INTERVAL_S or sum(map(len, buffer)) >= _FLUSH_CHARS:
                last_flush = now
                yield frame({"content": take()})
        if buffer:
            yield frame({"content": take()})
    except ModelUnavailable as exc:
        log.warning("stream.model_unavailable", error=str(exc))
        if buffer:
            yield frame({"content": take()})
        # Never show the raw transport error. Someone mid-sentence about
        # something difficult should not be handed "[Errno -5] No address
        # associated with hostname" — that is what this stack looks like
        # while vLLM restarts, and it reads as the tool breaking on them.
        yield frame({"content": ModelUnavailable.MESSAGE})
    except Exception as exc:  # noqa: BLE001
        log.error("stream.failed", error=str(exc))
        if buffer:
            yield frame({"content": take()})
        # Surface the failure in the chat rather than dying silently — a
        # response that just stops mid-sentence is impossible to debug.
        yield frame({"content": f"\n\n_[the agent hit an error: {exc}]_"})
    yield frame({}, finish="stop")
    yield "data: [DONE]\n\n"
