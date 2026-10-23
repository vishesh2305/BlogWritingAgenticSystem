"""LLM backend for Google's Gemini API (generateContent). Fast hosted models; needs a key."""

from collections.abc import Mapping
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from cwas.config import Tier
from cwas.llm import Effort, LLMError, LLMRefusalError, raise_for_http_status
from cwas.usage import record_usage

# Speed first: each effort level thinks one notch less than its name suggests.
_THINKING_LEVEL: dict[Effort, str] = {
    "low": "low",
    "medium": "low",
    "high": "medium",
    "xhigh": "high",
    "max": "high",
}
_BLOCKED = frozenset(
    {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"}
)


class GeminiLLM:
    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        models: Mapping[Tier, str],
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        max_attempts: int = 2,
        timeout: float = 300.0,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._models = dict(models)
        self._base_url = base_url.rstrip("/")
        self._max_attempts = max_attempts
        self._timeout = timeout

    async def structured[T: BaseModel](
        self, *, tier: Tier, system: str, prompt: str, schema: type[T], effort: Effort = "medium"
    ) -> T:
        contents = [_turn("user", prompt)]
        for _ in range(self._max_attempts):
            content = await self._generate(
                tier, system, contents, effort, 16000, schema=schema.model_json_schema()
            )
            try:
                return schema.model_validate_json(content)
            except ValidationError as exc:
                contents = [
                    *contents,
                    _turn("model", content),
                    _turn(
                        "user",
                        f"That JSON did not match the required schema:\n{exc}\n"
                        "Reply again with only the corrected JSON.",
                    ),
                ]
        raise LLMError(
            f"Gemini returned no valid {schema.__name__} after {self._max_attempts} attempts"
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
        return await self._generate(tier, system, [_turn("user", prompt)], effort, max_tokens)

    async def _generate(
        self,
        tier: Tier,
        system: str,
        contents: list[dict[str, Any]],
        effort: Effort,
        max_tokens: int,
        schema: dict[str, Any] | None = None,
    ) -> str:
        config: dict[str, Any] = {
            "maxOutputTokens": max_tokens,
            "thinkingConfig": {"thinkingLevel": _THINKING_LEVEL[effort]},
        }
        if schema is not None:
            config["responseMimeType"] = "application/json"
            config["responseJsonSchema"] = schema
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": config,
        }
        url = f"{self._base_url}/models/{self._models[tier]}:generateContent"

        try:
            # The key goes in a header so it never shows up in logged URLs.
            response = await self._client.post(
                url, json=body, headers={"x-goog-api-key": self._api_key}, timeout=self._timeout
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"Gemini request failed: {exc!r}") from exc
        raise_for_http_status("Gemini", response)

        data = response.json()
        usage = data.get("usageMetadata", {})
        record_usage(
            usage.get("promptTokenCount", 0),
            usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0),
        )
        block_reason = data.get("promptFeedback", {}).get("blockReason")
        if block_reason:
            raise LLMRefusalError(f"Gemini blocked the prompt (reason: {block_reason})")
        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMError("Gemini returned no candidates")

        candidate = candidates[0]
        finish = candidate.get("finishReason", "STOP")
        if finish in _BLOCKED:
            raise LLMRefusalError(f"Gemini declined the request (finishReason: {finish})")
        if finish == "MAX_TOKENS":
            raise LLMError("output truncated (finishReason=MAX_TOKENS)")
        parts = candidate.get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()


def _turn(role: str, text: str) -> dict[str, Any]:
    return {"role": role, "parts": [{"text": text}]}
