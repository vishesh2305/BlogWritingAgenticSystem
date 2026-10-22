"""Knowledge base facade: chunk -> embed -> store, and query-time retrieval."""

import hashlib

from cwas.kb.chunker import chunk_text
from cwas.kb.store import VectorStore
from cwas.models import Chunk, ScoredChunk, ScrapedPage, SourceType
from cwas.tools.embed import Embedder


class KnowledgeBase:
    def __init__(
        self, embedder: Embedder, store: VectorStore, max_chars: int = 1200, overlap: int = 150
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._max_chars = max_chars
        self._overlap = overlap

    async def ingest(self, collection: str, pages: list[ScrapedPage]) -> int:
        chunks = [
            Chunk(
                id=hashlib.sha1(f"{page.url}#{i}".encode()).hexdigest(),
                text=text,
                source_url=page.url,
                source_title=page.title,
                source_type=page.source_type,
                index=i,
            )
            for page in pages
            for i, text in enumerate(chunk_text(page.text, self._max_chars, self._overlap))
        ]
        if not chunks:
            return 0
        vectors = await self._embedder.embed([c.text for c in chunks], "document")
        return await self._store.add(collection, chunks, vectors)

    async def retrieve(
        self, collection: str, query: str, k: int = 8, source_type: SourceType | None = None
    ) -> list[ScoredChunk]:
        [vector] = await self._embedder.embed([query], "query")
        return await self._store.search(collection, vector, k, source_type)

    async def drop(self, collection: str) -> None:
        await self._store.drop(collection)
