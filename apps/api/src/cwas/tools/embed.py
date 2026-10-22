"""Text embedders: Voyage AI for production, a hashing embedder for offline/dev runs."""

import hashlib
import math
import re
from typing import Literal, Protocol

import httpx

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
