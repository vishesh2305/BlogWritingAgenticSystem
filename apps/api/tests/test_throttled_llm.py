import asyncio
from typing import Any

import pytest
from pydantic import BaseModel

from cwas.llm import LLMRateLimitError
from cwas.throttled_llm import ThrottledLLM


class Answer(BaseModel):
    value: str


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 3))
        self.now += seconds


class CountingLLM:
    def __init__(self, gate: asyncio.Event | None = None) -> None:
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self._gate = gate

    async def _call(self) -> None:
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        if self._gate:
            await self._gate.wait()
        self.in_flight -= 1

    async def text(self, **_: Any) -> str:
        await self._call()
        return "ok"

    async def structured(self, *, schema: type[BaseModel], **_: Any) -> BaseModel:
        await self._call()
        return schema(value="ok")


def throttled(inner: CountingLLM, clock: Clock, **kwargs: Any) -> ThrottledLLM:
    options: dict[str, Any] = {"requests_per_minute": 2, "max_concurrent": 4, "max_wait": 5.0}
    options.update(kwargs)
    return ThrottledLLM(inner, clock=clock, sleep=clock.sleep, **options)  # type: ignore[arg-type]


async def test_calls_under_the_limit_go_straight_through() -> None:
    clock, inner = Clock(), CountingLLM()
    llm = throttled(inner, clock)

    await llm.text(tier="opus", system="s", prompt="p")
    await llm.structured(tier="opus", system="s", prompt="p", schema=Answer)

    assert inner.calls == 2
    assert clock.sleeps == []


async def test_waits_for_a_free_slot_when_the_wait_is_short() -> None:
    clock, inner = Clock(), CountingLLM()
    llm = throttled(inner, clock, requests_per_minute=1, max_wait=60.0)

    await llm.text(tier="opus", system="s", prompt="p")
    clock.now = 57.0
    await llm.text(tier="opus", system="s", prompt="p")

    assert clock.sleeps == [3.0]
    assert inner.calls == 2


async def test_a_long_wait_raises_rate_limit_so_the_fallback_can_take_over() -> None:
    clock, inner = Clock(), CountingLLM()
    llm = throttled(inner, clock, requests_per_minute=1, max_wait=5.0)
    await llm.text(tier="opus", system="s", prompt="p")
    clock.now = 20.0

    with pytest.raises(LLMRateLimitError, match="1 requests/min") as caught:
        await llm.text(tier="opus", system="s", prompt="p")

    assert caught.value.retry_after == pytest.approx(40.0)
    assert inner.calls == 1


async def test_old_requests_leave_the_window() -> None:
    clock, inner = Clock(), CountingLLM()
    llm = throttled(inner, clock, requests_per_minute=1)

    await llm.text(tier="opus", system="s", prompt="p")
    clock.now = 60.0
    await llm.text(tier="opus", system="s", prompt="p")

    assert clock.sleeps == []


async def test_limits_concurrent_calls() -> None:
    gate = asyncio.Event()
    clock, inner = Clock(), CountingLLM(gate)
    llm = throttled(inner, clock, requests_per_minute=100, max_concurrent=2)

    calls = asyncio.gather(*(llm.text(tier="opus", system="s", prompt="p") for _ in range(5)))
    await asyncio.sleep(0)
    gate.set()
    await calls

    assert inner.calls == 5
    assert inner.max_in_flight == 2
