"""Vector store interface and an in-memory implementation (pgvector arrives with Docker infra)."""

import math
from typing import Protocol

from cwas.models import Chunk, ScoredChunk, SourceType


class VectorStore(Protocol):
    async def add(
        self, collection: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> int: ...

    async def search(
        self,
        collection: str,
        vector: list[float],
        k: int,
        source_type: SourceType | None = None,
    ) -> list[ScoredChunk]: ...

    async def drop(self, collection: str) -> None: ...


def _cosine(a: list[float], b: list[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm if norm else 0.0


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._collections: dict[str, list[tuple[Chunk, list[float]]]] = {}
        self._seen: dict[str, set[tuple[str, str]]] = {}

    async def add(self, collection: str, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        rows = self._collections.setdefault(collection, [])
        seen = self._seen.setdefault(collection, set())
        added = 0
        for chunk, vector in zip(chunks, vectors, strict=True):
            key = (chunk.source_url, chunk.text)
            if key in seen:
                continue
            seen.add(key)
            rows.append((chunk, vector))
            added += 1
        return added

    async def search(
        self,
        collection: str,
        vector: list[float],
        k: int,
        source_type: SourceType | None = None,
    ) -> list[ScoredChunk]:
        hits = [
            ScoredChunk(chunk=chunk, score=_cosine(vector, stored))
            for chunk, stored in self._collections.get(collection, [])
            if source_type is None or chunk.source_type == source_type
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:k]

    async def drop(self, collection: str) -> None:
        self._collections.pop(collection, None)
        self._seen.pop(collection, None)
