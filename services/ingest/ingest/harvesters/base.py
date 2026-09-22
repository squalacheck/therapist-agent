"""Shared HTTP behaviour for harvesters.

Every source here is someone else's free service. Rate limits are respected,
the client identifies itself, and failures back off rather than hammering.
"""

from __future__ import annotations

import asyncio
import os
import random
from typing import Any

import httpx
import structlog

log = structlog.get_logger(__name__)

USER_AGENT = (
    "therapist-agent-corpus/0.1 (local research assistant; "
    f"contact: {os.getenv('CONTACT_EMAIL', 'unset')})"
)


class RateLimiter:
    """Simple minimum-interval limiter."""

    def __init__(self, per_second: float) -> None:
        self._interval = 1.0 / per_second if per_second > 0 else 0.0
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self) -> None:
        if self._interval <= 0:
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            elapsed = loop.time() - self._last
            if elapsed < self._interval:
                await asyncio.sleep(self._interval - elapsed)
            self._last = loop.time()


class HttpClient:
    def __init__(self, per_second: float = 3.0, timeout: float = 60.0) -> None:
        self._limiter = RateLimiter(per_second)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=15.0),
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        retries: int = 3,
    ) -> httpx.Response | None:
        """GET with backoff. Returns None rather than raising — one dead
        source should never abort a whole harvest."""
        for attempt in range(retries):
            await self._limiter.wait()
            try:
                response = await self._client.get(url, params=params)
            except httpx.HTTPError as exc:
                log.warning("http.error", url=url[:120], attempt=attempt, error=str(exc))
            else:
                if response.status_code == 200:
                    return response
                if response.status_code in (429, 500, 502, 503, 504):
                    log.warning("http.retryable", url=url[:120], status=response.status_code)
                else:
                    log.warning("http.failed", url=url[:120], status=response.status_code)
                    return None
            await asyncio.sleep(2**attempt + random.random())
        return None
