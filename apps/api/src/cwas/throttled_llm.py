"""Client-side rate limiting for one hosted provider (e.g. Gemini's free-tier requests/minute)."""

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from cwas.config import Tier
from cwas.llm import LLM, Effort, LLMRateLimitError

WINDOW_SECONDS = 60.0


class ThrottledLLM:
    """Keeps a provider under its requests-per-minute and concurrency limits. A call that would
    have to wait longer than `max_wait` raises LLMRateLimitError instead, so a FallbackLLM can
    hand it to the next provider rather than stall."""

    def __init__(
        self,
        inner: LLM,
        requests_per_minute: int,
        max_concurrent: int = 4,
        max_wait: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.inner = inner
        self._rpm = requests_per_minute
        self._max_wait = max_wait
        self._clock = clock
        self._sleep = sleep
        self._started: deque[float] = deque()
        self._lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(max_concurrent)

    async def structured[T: BaseModel](
        self, *, tier: Tier, system: str, prompt: str, schema: type[T], effort: Effort = "medium"
    ) -> T:
        await self._reserve()
        async with self._slots:
            return await self.inner.structured(
                tier=tier, system=system, prompt=prompt, schema=schema, effort=effort
            )

    async def text(
        self,
        *,
        tier: Tier,
        system: str,
        prompt: str,
        effort: Effort = "medium",
        max_tokens: int = 16000,
    ) -> str:
        await self._reserve()
        async with self._slots:
            return await self.inner.text(
                tier=tier, system=system, prompt=prompt, effort=effort, max_tokens=max_tokens
            )

    async def _reserve(self) -> None:
        async with self._lock:
            while True:
                now = self._clock()
                while self._started and now - self._started[0] >= WINDOW_SECONDS:
                    self._started.popleft()
                if len(self._started) < self._rpm:
                    self._started.append(now)
                    return
                wait = WINDOW_SECONDS - (now - self._started[0])
                if wait > self._max_wait:
                    raise LLMRateLimitError(
                        f"local limit of {self._rpm} requests/min reached", retry_after=wait
                    )
                await self._sleep(wait)
