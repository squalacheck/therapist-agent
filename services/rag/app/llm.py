"""Thin async client for the vLLM OpenAI-compatible endpoint.

Two call shapes are needed and no more: a streaming chat completion for the
user-facing turn, and a short blocking completion for the internal helper
calls (query rewriting, safety classification, memory extraction).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from .config import get_settings

log = structlog.get_logger(__name__)


class LLMError(RuntimeError):
    pass


class ModelUnavailable(LLMError):
    """The model server is not reachable — stopped, restarting, or still
    loading. Distinguished from other failures because it is the one the
    person in the chat is most likely to meet, and a raw
    "[Errno -5] No address associated with hostname" is a terrible thing
    to put in front of someone mid-sentence."""

    MESSAGE = (
        "I can't reach the model right now — it's usually restarting or "
        "still loading, which takes a couple of minutes after a restart. "
        "Nothing you wrote was lost. Try again shortly."
    )


class LLMClient:
    def __init__(self) -> None:
        s = get_settings()
        self._base = s.vllm_base_url.rstrip("/")
        self._model = s.vllm_model
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(s.vllm_timeout_s, connect=10.0),
            # A 27B model in eager mode is not fast. Generous limits here,
            # short ones would surface as mysterious truncation mid-answer.
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def healthy(self) -> bool:
        try:
            r = await self._client.get(f"{self._base}/models", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    # ── Blocking helper calls ──────────────────────────────────────

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        """One short, deterministic completion. Used for internal reasoning
        steps where we want the same answer every time."""
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            # Thinking off for internal calls. With --reasoning-parser qwen3,
            # anything the model thinks lands in `reasoning_content` and never
            # in `content` — so a helper call with a small max_tokens can spend
            # its entire budget reasoning and return an empty string. The
            # callers all have fallbacks, which is exactly why this would go
            # unnoticed: safety and query rewriting would quietly degrade to
            # their no-op paths on every turn.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if stop:
            payload["stop"] = stop

        r = await self._client.post(f"{self._base}/chat/completions", json=payload)
        if r.status_code != 200:
            raise LLMError(f"vLLM returned {r.status_code}: {r.text[:500]}")
        data = r.json()
        return (data["choices"][0]["message"].get("content") or "").strip()

    async def complete_json(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 512,
    ) -> dict[str, Any] | None:
        """Same, but expect JSON back.

        Returns None rather than raising on malformed output. Every caller
        of this has a sane fallback, and a helper call failing should never
        take down the user's turn.
        """
        raw = await self.complete(messages, max_tokens=max_tokens, temperature=0.0)
        text = raw.strip()
        # Models like to wrap JSON in fences even when told not to.
        if text.startswith("```"):
            text = text.split("```")[1] if "```" in text[3:] else text[3:]
            text = text.removeprefix("json").strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            log.warning("llm.json_parse_failed", raw=raw[:200])
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            log.warning("llm.json_decode_failed", raw=raw[:200])
            return None

    # ── Streaming user turn ────────────────────────────────────────

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> AsyncIterator[str]:
        """Yield content deltas as they arrive."""
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": get_settings().enable_thinking},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p

        # httpx raises the connection error on __aenter__, not on stream(),
        # so the whole block has to be inside the try.
        try:
            async with self._client.stream(
                "POST", f"{self._base}/chat/completions", json=payload
            ) as r:
                if r.status_code == 503:
                    # vLLM answers 503 while the engine is still loading.
                    raise ModelUnavailable("model server is still starting")
                if r.status_code != 200:
                    body = (await r.aread()).decode()[:500]
                    raise LLMError(f"vLLM returned {r.status_code}: {body}")
                async for chunk in self._iter_sse(r):
                    yield chunk
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
            raise ModelUnavailable(str(exc)) from exc

    @staticmethod
    async def _iter_sse(r: httpx.Response) -> AsyncIterator[str]:
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                return
            try:
                delta = json.loads(chunk)["choices"][0].get("delta", {})
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if content := delta.get("content"):
                yield content


_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
