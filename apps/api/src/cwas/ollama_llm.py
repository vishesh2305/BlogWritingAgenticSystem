"""LLM backend that runs a local model through Ollama's chat API. Free; no API key."""

import re
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from cwas.config import Tier
from cwas.llm import Effort, LLMError

# Reasoning models sometimes inline their chain of thought; it must not reach the article.
_INLINE_THINKING = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class OllamaLLM:
    """One local model serves every tier; `effort` has no Ollama equivalent and is ignored."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        model: str,
        base_url: str = "http://localhost:11434",
        num_ctx: int = 16384,
        max_attempts: int = 2,
        timeout: float = 900.0,
    ) -> None:
        self._client = client
        self._model = model
        self._url = f"{base_url.rstrip('/')}/api/chat"
        self._num_ctx = num_ctx
        self._max_attempts = max_attempts
        self._timeout = timeout

    async def structured[T: BaseModel](
        self, *, tier: Tier, system: str, prompt: str, schema: type[T], effort: Effort = "medium"
    ) -> T:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        for _ in range(self._max_attempts):
            content = await self._chat(messages, schema=schema.model_json_schema())
            try:
                return schema.model_validate_json(content)
            except ValidationError as exc:
                # Small local models occasionally miss a field; show them the error once more.
                messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": f"That JSON did not match the required schema:\n{exc}\n"
                        "Reply again with only the corrected JSON.",
                    },
                ]
        raise LLMError(
            f"local model returned no valid {schema.__name__} after {self._max_attempts} attempts"
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
        return await self._chat(messages, max_tokens=max_tokens)

    async def _chat(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        options: dict[str, Any] = {"num_ctx": self._num_ctx}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if schema is not None:
            body["format"] = schema

        try:
            response = await self._client.post(self._url, json=body, timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Ollama request failed (is Ollama running at {self._url}?): {exc!r}"
            ) from exc
        if response.is_error:
            raise LLMError(f"Ollama returned HTTP {response.status_code}: {response.text}")

        data = response.json()
        if data.get("done_reason") == "length":
            raise LLMError("output truncated (done_reason=length); raise OLLAMA_NUM_CTX")
        return _INLINE_THINKING.sub("", data["message"]["content"]).strip()
