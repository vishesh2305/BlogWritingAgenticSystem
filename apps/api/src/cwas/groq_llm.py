"""LLM backend for Groq's OpenAI-compatible chat API. Used as the fallback behind Gemini."""

from collections.abc import Mapping
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from cwas.config import Tier
from cwas.llm import Effort, LLMError, raise_for_http_status
from cwas.usage import record_usage

_REASONING_EFFORT: dict[Effort, str] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


class _SchemaRejected(Exception):
    """Groq's server-side check refused a generation that did not match the schema."""

    def __init__(self, message: str, generation: str) -> None:
        super().__init__(message)
        self.generation = generation


class GroqLLM:
    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        models: Mapping[Tier, str],
        base_url: str = "https://api.groq.com/openai/v1",
        max_attempts: int = 2,
        timeout: float = 300.0,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._models = dict(models)
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._max_attempts = max_attempts
        self._timeout = timeout

    async def structured[T: BaseModel](
        self, *, tier: Tier, system: str, prompt: str, schema: type[T], effort: Effort = "medium"
    ) -> T:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        }
        for _ in range(self._max_attempts):
            try:
                content = await self._chat(tier, messages, effort, 16000, response_format)
            except _SchemaRejected as exc:
                error, content = str(exc), exc.generation
            else:
                try:
                    return schema.model_validate_json(content)
                except ValidationError as exc:
                    error = str(exc)
            messages = [
                *messages,
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": f"That JSON did not match the required schema:\n{error}\n"
                    "Reply again with only the corrected JSON.",
                },
            ]
        raise LLMError(
            f"Groq returned no valid {schema.__name__} after {self._max_attempts} attempts"
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
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        return await self._chat(tier, messages, effort, max_tokens)

    async def _chat(
        self,
        tier: Tier,
        messages: list[dict[str, str]],
        effort: Effort,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        model = self._models[tier]
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = _REASONING_EFFORT[effort]
        if response_format is not None:
            body["response_format"] = response_format

        try:
            response = await self._client.post(
                self._url,
                json=body,
                headers={"authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"Groq request failed: {exc!r}") from exc
        if response.status_code == 400:
            error = response.json().get("error", {})
            if error.get("code") == "json_validate_failed":
                raise _SchemaRejected(error.get("message", ""), error.get("failed_generation", ""))
        raise_for_http_status("Groq", response)

        data = response.json()
        usage = data.get("usage") or {}
        record_usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise LLMError("output truncated (finish_reason=length)")
        return str(choice["message"].get("content") or "").strip()
