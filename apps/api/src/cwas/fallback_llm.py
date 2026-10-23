"""Chains LLM backends: each call goes to the first provider that answers."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence

from pydantic import BaseModel

from cwas.config import Tier
from cwas.llm import LLM, Effort, LLMError, LLMRateLimitError

logger = logging.getLogger(__name__)


class FallbackLLM:
    """Tries providers in order. A rate-limited provider sits out its cooldown, but is still
    tried last rather than failing the call outright. When every provider is rate-limited, the
    call waits once for the soonest to recover (up to max_retry_wait) and tries again."""

    def __init__(
        self,
        providers: Sequence[tuple[str, LLM]],
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        default_cooldown: float = 30.0,
        max_retry_wait: float = 60.0,
    ) -> None:
        if not providers:
            raise ValueError("FallbackLLM needs at least one provider")
        self._providers = list(providers)
        self._clock = clock
        self._sleep = sleep
        self._default_cooldown = default_cooldown
        self._max_retry_wait = max_retry_wait
        self._cooling_until: dict[str, float] = {}

    @property
    def provider_names(self) -> list[str]:
        return [name for name, _ in self._providers]

    async def structured[T: BaseModel](
        self, *, tier: Tier, system: str, prompt: str, schema: type[T], effort: Effort = "medium"
    ) -> T:
        return await self._first(
            lambda llm: llm.structured(
                tier=tier, system=system, prompt=prompt, schema=schema, effort=effort
            )
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
        return await self._first(
            lambda llm: llm.text(
                tier=tier, system=system, prompt=prompt, effort=effort, max_tokens=max_tokens
            )
        )

    async def _first[R](self, call: Callable[[LLM], Awaitable[R]]) -> R:
        errors: list[str] = []
        for is_retry in (False, True):
            errors, pauses = [], []
            now = self._clock()
            ready = [p for p in self._providers if self._cooling_until.get(p[0], 0.0) <= now]
            order = ready + [p for p in self._providers if p not in ready]
            for position, (name, llm) in enumerate(order):
                try:
                    return await call(llm)
                except LLMError as exc:
                    if isinstance(exc, LLMRateLimitError):
                        pause = exc.retry_after or self._default_cooldown
                        self._cooling_until[name] = self._clock() + pause
                        pauses.append(pause)
                    errors.append(f"{name}: {exc}")
                    if position + 1 < len(order):
                        logger.warning(
                            "%s failed (%s); falling back to %s", name, exc, order[position + 1][0]
                        )
            if is_retry or len(pauses) < len(order) or min(pauses) > self._max_retry_wait:
                break
            logger.warning("every LLM provider is rate-limited; waiting %.1fs", min(pauses))
            await self._sleep(min(pauses))
        raise LLMError("every LLM provider failed: " + "; ".join(errors))
