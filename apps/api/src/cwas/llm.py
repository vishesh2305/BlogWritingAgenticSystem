"""Thin, testable wrapper over the Anthropic SDK used by every agent."""

from collections.abc import Mapping
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from cwas.config import Tier

Effort = Literal["low", "medium", "high", "xhigh", "max"]

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Server-side refusal fallbacks exist for Opus and Sonnet; Haiku has none.
_FALLBACK_TIERS: frozenset[Tier] = frozenset({"opus", "sonnet"})


class LLMError(Exception):
    pass


class LLMRefusalError(LLMError):
    pass


class LLM(Protocol):
    async def structured[T: BaseModel](
        self, *, tier: Tier, system: str, prompt: str, schema: type[T], effort: Effort = "medium"
    ) -> T: ...

    async def text(
        self,
        *,
        tier: Tier,
        system: str,
        prompt: str,
        effort: Effort = "medium",
        max_tokens: int = 16000,
    ) -> str: ...


def _raise_for_stop(response: Any) -> None:
    if response.stop_reason == "refusal":
        details = response.stop_details
        category = getattr(details, "category", None) or "unspecified"
        raise LLMRefusalError(f"model declined the request (category: {category})")
    if response.stop_reason == "max_tokens":
        raise LLMError("output truncated: stop_reason=max_tokens")


class AnthropicLLM:
    def __init__(self, client: Any, models: Mapping[Tier, str]) -> None:
        self._client = client
        self._models = dict(models)

    def _request(
        self, tier: Tier, system: str, prompt: str, effort: Effort, max_tokens: int
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self._models[tier],
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": effort},
        }
        if tier in _FALLBACK_TIERS:
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = "default"
        return request

    async def structured[T: BaseModel](
        self, *, tier: Tier, system: str, prompt: str, schema: type[T], effort: Effort = "medium"
    ) -> T:
        response = await self._client.beta.messages.parse(
            **self._request(tier, system, prompt, effort, 16000), output_format=schema
        )
        _raise_for_stop(response)
        parsed = response.parsed_output
        if parsed is None:
            raise LLMError(f"model returned no {schema.__name__}")
        return parsed  # type: ignore[no-any-return]

    async def text(
        self,
        *,
        tier: Tier,
        system: str,
        prompt: str,
        effort: Effort = "medium",
        max_tokens: int = 16000,
    ) -> str:
        response = await self._client.beta.messages.create(
            **self._request(tier, system, prompt, effort, max_tokens)
        )
        _raise_for_stop(response)
        return "".join(block.text for block in response.content if block.type == "text")
