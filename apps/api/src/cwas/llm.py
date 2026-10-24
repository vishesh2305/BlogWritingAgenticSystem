"""Thin, testable wrapper over the Anthropic SDK used by every agent."""

from collections.abc import Mapping
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from cwas.config import Tier
from cwas.usage import record_usage

Effort = Literal["low", "medium", "high", "xhigh", "max"]

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Server-side refusal fallbacks exist for Opus and Sonnet; Haiku has none.
_FALLBACK_TIERS: frozenset[Tier] = frozenset({"opus", "sonnet"})


class LLMError(Exception):
    pass


class LLMRefusalError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    def __init__(self, message: str, retry_after: float | None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def retry_after_seconds(response: Any) -> float | None:
    """How long a 429 asks us to wait: the Retry-After header, or Google's RetryInfo detail
    (`"retryDelay": "47s"`), which Gemini sends instead of the header."""
    try:
        return float(response.headers.get("retry-after", ""))
    except ValueError:
        pass
    try:
        error = response.json().get("error")
    except ValueError:
        return None
    details = error.get("details", []) if isinstance(error, dict) else []
    for detail in details:
        if str(detail.get("@type", "")).endswith("RetryInfo"):
            try:
                return float(str(detail.get("retryDelay", "")).rstrip("s"))
            except ValueError:
                return None
    return None


def raise_for_http_status(provider: str, response: Any) -> None:
    """Map an HTTP error from a hosted provider (an httpx.Response) to an LLMError."""
    if not response.is_error:
        return
    message = f"{provider} returned HTTP {response.status_code}: {response.text[:500]}"
    # 503 means the model is overloaded ("high demand"): as temporary as a rate limit.
    if response.status_code in (429, 503):
        raise LLMRateLimitError(message, retry_after_seconds(response))
    raise LLMError(message)


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


def _record_usage(response: Any) -> None:
    usage = getattr(response, "usage", None)
    record_usage(getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0))


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
        _record_usage(response)
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
        _record_usage(response)
        _raise_for_stop(response)
        return "".join(block.text for block in response.content if block.type == "text")
