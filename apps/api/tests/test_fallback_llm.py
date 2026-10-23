import logging
from typing import Any

import pytest
from pydantic import BaseModel

from cwas.fallback_llm import FallbackLLM
from cwas.llm import LLMError, LLMRateLimitError, LLMRefusalError


class Answer(BaseModel):
    value: str


class ScriptedLLM:
    """Raises the scripted error, or returns `value`; counts calls."""

    def __init__(self, value: str = "ok", error: Exception | None = None) -> None:
        self.value = value
        self.error = error
        self.calls = 0

    async def structured(self, *, schema: type[BaseModel], **_: Any) -> BaseModel:
        self.calls += 1
        if self.error:
            raise self.error
        return schema(value=self.value)

    async def text(self, **_: Any) -> str:
        self.calls += 1
        if self.error:
            raise self.error
        return self.value


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def chain(primary: ScriptedLLM, fallback: ScriptedLLM, clock: Clock | None = None) -> FallbackLLM:
    clock = clock or Clock()
    return FallbackLLM(
        [("gemini", primary), ("groq", fallback)],  # type: ignore[list-item]
        clock=clock,
        sleep=clock.sleep,
        default_cooldown=30.0,
    )


async def test_primary_success_never_touches_the_fallback() -> None:
    primary, fallback = ScriptedLLM("fast"), ScriptedLLM("slow")

    assert await chain(primary, fallback).text(tier="opus", system="s", prompt="p") == "fast"
    assert fallback.calls == 0


@pytest.mark.parametrize(
    "error", [LLMError("HTTP 503"), LLMRefusalError("SAFETY"), LLMRateLimitError("429", None)]
)
async def test_primary_failure_falls_back_and_logs_a_warning(
    error: LLMError, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    primary, fallback = ScriptedLLM(error=error), ScriptedLLM("rescued")

    llm = chain(primary, fallback)
    assert await llm.text(tier="opus", system="s", prompt="p") == "rescued"
    result = await llm.structured(tier="sonnet", system="s", prompt="p", schema=Answer)
    assert result == Answer(value="rescued")
    assert "gemini failed" in caplog.text
    assert "groq" in caplog.text


async def test_when_every_provider_fails_the_error_names_each_one() -> None:
    llm = chain(ScriptedLLM(error=LLMError("quota")), ScriptedLLM(error=LLMError("too large")))

    with pytest.raises(LLMError, match=r"gemini: quota.*groq: too large"):
        await llm.text(tier="opus", system="s", prompt="p")


async def test_rate_limited_provider_is_skipped_until_its_cooldown_ends() -> None:
    clock = Clock()
    primary = ScriptedLLM(error=LLMRateLimitError("429", retry_after=20.0))
    fallback = ScriptedLLM("rescued")
    llm = chain(primary, fallback, clock)

    await llm.text(tier="opus", system="s", prompt="p")
    assert primary.calls == 1

    primary.error = None
    clock.now += 19
    await llm.text(tier="opus", system="s", prompt="p")
    assert primary.calls == 1  # still cooling down

    clock.now += 2
    assert await llm.text(tier="opus", system="s", prompt="p") == "ok"
    assert primary.calls == 2


async def test_rate_limit_without_retry_after_uses_the_default_cooldown() -> None:
    clock = Clock()
    primary = ScriptedLLM(error=LLMRateLimitError("429", retry_after=None))
    llm = chain(primary, ScriptedLLM("rescued"), clock)

    await llm.text(tier="opus", system="s", prompt="p")
    primary.error = None
    clock.now += 29
    await llm.text(tier="opus", system="s", prompt="p")
    assert primary.calls == 1
    clock.now += 2
    await llm.text(tier="opus", system="s", prompt="p")
    assert primary.calls == 2


async def test_cooling_providers_are_still_tried_last_rather_than_failing() -> None:
    clock = Clock()
    primary = ScriptedLLM(error=LLMRateLimitError("429", retry_after=60.0))
    fallback = ScriptedLLM(error=LLMError("down"))
    llm = chain(primary, fallback, clock)

    with pytest.raises(LLMError):
        await llm.text(tier="opus", system="s", prompt="p")

    primary.error = None
    clock.now += 1
    assert await llm.text(tier="opus", system="s", prompt="p") == "ok"
    assert fallback.calls == 2


async def test_non_llm_errors_are_not_swallowed() -> None:
    llm = chain(ScriptedLLM(error=ValueError("bug")), ScriptedLLM("rescued"))

    with pytest.raises(ValueError, match="bug"):
        await llm.text(tier="opus", system="s", prompt="p")


def test_needs_at_least_one_provider() -> None:
    with pytest.raises(ValueError, match="provider"):
        FallbackLLM([])


class RecoveringLLM(ScriptedLLM):
    """Rate-limited until the clock reaches `ready_at`."""

    def __init__(self, clock: Clock, ready_at: float, retry_after: float) -> None:
        super().__init__("recovered")
        self.clock, self.ready_at, self.retry_after = clock, ready_at, retry_after

    async def text(self, **kwargs: Any) -> str:
        limited = self.clock.now < self.ready_at
        self.error = LLMRateLimitError("429", self.retry_after) if limited else None
        return await super().text(**kwargs)


async def test_when_every_provider_is_rate_limited_it_waits_for_the_soonest_and_retries() -> None:
    clock = Clock()
    llm = chain(
        RecoveringLLM(clock, ready_at=1007, retry_after=7.0),
        RecoveringLLM(clock, ready_at=1003, retry_after=3.0),
        clock,
    )

    assert await llm.text(tier="opus", system="s", prompt="p") == "recovered"
    assert clock.sleeps == [3.0]


async def test_rate_limit_retry_happens_only_once() -> None:
    clock = Clock()
    llm = chain(
        ScriptedLLM(error=LLMRateLimitError("429", 2.0)),
        ScriptedLLM(error=LLMRateLimitError("429", 4.0)),
        clock,
    )

    with pytest.raises(LLMError, match="every LLM provider failed"):
        await llm.text(tier="opus", system="s", prompt="p")
    assert clock.sleeps == [2.0]
