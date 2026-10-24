"""Text embedders: Voyage AI or Gemini for production, a hashing embedder for offline/dev runs."""

import asyncio
import hashlib
import math
import re
from collections.abc import Awaitable, Callable
from typing import Literal, Protocol

import httpx

from cwas.llm import retry_after_seconds

InputType = Literal["document", "query"]


class EmbeddingError(Exception):
    pass


class Embedder(Protocol):
    async def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]: ...


class VoyageEmbedder:
    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        model: str = "voyage-3.5",
        batch_size: int = 128,
        base_url: str = "https://api.voyageai.com/v1",
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._model = model
        self._batch_size = batch_size
        self._base_url = base_url.rstrip("/")

    async def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(
                await self._embed_batch(texts[start : start + self._batch_size], input_type)
            )
        return vectors

    async def _embed_batch(self, batch: list[str], input_type: InputType) -> list[list[float]]:
        try:
            response = await self._client.post(
                f"{self._base_url}/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"input": batch, "model": self._model, "input_type": input_type},
                timeout=60.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Voyage embedding failed: {exc}") from exc
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in data]


class GeminiEmbedder:
    """gemini-embedding-2 takes the task as a text prefix rather than a task-type field.

    The free tier allows 100 texts a minute, and every text in a batch counts, so a 429 is
    expected on large runs: the embedder waits the delay the server asks for (up to
    `max_retry_wait`) and retries the batch, at most `max_retries` times."""

    _PREFIX: dict[InputType, str] = {
        "document": "title: none | text: ",
        "query": "task: search result | query: ",
    }

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        model: str = "gemini-embedding-2",
        dimensions: int = 768,
        batch_size: int = 100,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        max_retries: int = 2,
        max_retry_wait: float = 60.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._max_retry_wait = max_retry_wait
        self._sleep = sleep
        self._url = f"{base_url.rstrip('/')}/models/{model}:batchEmbedContents"

    async def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(
                await self._embed_batch(texts[start : start + self._batch_size], input_type)
            )
        return vectors

    async def _embed_batch(self, batch: list[str], input_type: InputType) -> list[list[float]]:
        requests = [
            {
                "model": f"models/{self._model}",
                "content": {"parts": [{"text": self._PREFIX[input_type] + text}]},
                "outputDimensionality": self._dimensions,
            }
            for text in batch
        ]
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(
                    self._url,
                    headers={"x-goog-api-key": self._api_key},
                    json={"requests": requests},
                    timeout=60.0,
                )
            except httpx.HTTPError as exc:
                raise EmbeddingError(f"Gemini embedding failed: {exc}") from exc
            if response.status_code == 429 and attempt < self._max_retries:
                delay = retry_after_seconds(response)
                if delay is not None and delay <= self._max_retry_wait:
                    await self._sleep(delay)
                    continue
            break
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(f"Gemini embedding failed: {exc}") from exc
        embeddings = response.json().get("embeddings", [])
        if len(embeddings) != len(batch):
            raise EmbeddingError(
                f"Gemini returned {len(embeddings)} vectors for {len(batch)} texts"
            )
        return [item["values"] for item in embeddings]


_TOKEN = re.compile(r"\w+")


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder. No network; for tests and keyless dev runs."""

    def __init__(self, dimensions: int = 512) -> None:
        self._dimensions = dimensions

    async def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self._dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector] if norm else vector
